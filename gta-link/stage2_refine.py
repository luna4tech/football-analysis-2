"""
Stage 2 (refine) entrypoint: a thin single-video wrapper around GtaLink's
``refine_tracklets.py``.  It consumes the ONE ``tracklets.pkl`` Stage 1 produced
and writes the contract ``refined.txt``, reusing refine_tracklets' algorithm
functions UNCHANGED, with caching + profiling.

The refinement is always the FULL pipeline: split ID-switched tracklets, then
connect/merge fragmented ones.

Artifact contract
-----------------
    <artifacts>/<video_stem>/01_track/tracklets.pkl   (input, from Stage 1)
    <artifacts>/<video_stem>/02_refine/refined.txt    (output)
    <artifacts>/<video_stem>/profiles/02_refine.json  (profiling)

Run with CWD = ``gta-link`` so that ``import refine_tracklets`` and
``import Tracklet`` both resolve to the sibling files in this directory::

    cd gta-link
    python stage2_refine.py --video /path/to/clip.mp4

``--video`` is used ONLY to locate the artifact directory (via the same
``get_artifact_paths`` Stage 1 uses); Stage 2 does NOT read the video itself.

How the pickle unpickles here
-----------------------------
``tracklets.pkl`` is ``{track_id: Tracklet}``, pickled in Stage 1 with the class
registered as the top-level module ``"Tracklet"`` (see
``stage1_assembly.load_tracklet_class``), so the pickle references
``Tracklet.Tracklet``.  With CWD = ``gta-link``, importing ``refine_tracklets``
runs its ``from Tracklet import Tracklet`` line, which imports
``gta-link/Tracklet.py`` as the module ``"Tracklet"`` — exactly the name the
pickle needs.  We therefore do NOT re-register or shadow the ``Tracklet`` module
name; the natural import resolves it.

Connect/merge step
------------------
The connect step uses ``_fast_connect`` — an exact, batched replacement for
refine_tracklets' ``get_distance_matrix`` + ``merge_tracklets`` (same output up
to float rounding, orders of magnitude faster).  The split component and the
spatial-constraint gate are reused from ``refine_tracklets.py`` UNCHANGED.
"""

from __future__ import annotations

import argparse
import os.path as osp
import pickle
import sys

# --- import path setup ------------------------------------------------------
# CWD is expected to be gta-link, so `import refine_tracklets` and
# `import Tracklet` resolve to the sibling files.  We do NOT add gta-link/ to
# sys.path beyond what CWD already provides, and we do NOT re-register the
# Tracklet module name (importing refine_tracklets does that naturally via its
# `from Tracklet import Tracklet`).
_THIS_DIR = osp.dirname(osp.abspath(__file__))

# Repo root on path so `import pipeline` works (CWD is gta-link).
_REPO_ROOT = osp.abspath(osp.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline.artifacts import get_artifact_paths, ensure_dirs
from pipeline.cache import should_run
from pipeline.profiling import profile_stage


# ---------------------------------------------------------------------------
# Step-decision orchestration (dependency-injected for CPU testing)
# ---------------------------------------------------------------------------
class RefineDeps:
    """Bundle of the refine_tracklets callables ``run_refine`` depends on.

    Injecting these lets ``run_refine`` be unit-tested on a CPU-only machine
    with stubs, without importing ``refine_tracklets`` (which pulls in
    torch / sklearn / scipy / matplotlib / seaborn / loguru).

    Attributes mirror the signatures of the real refine_tracklets functions:

      * ``get_spatial_constraints(tid2track, factor) -> (max_x, max_y)``
      * ``split_tracklets(tmp, eps, max_k, min_samples, len_thres) -> dict``
      * ``check_spatial_constraints(trk1, trk2, max_x, max_y) -> bool`` — the
        spatial gate used by the batched connect step (``_fast_connect``).
      * ``save_results(out_path, tracklets) -> None``
    """

    def __init__(
        self,
        get_spatial_constraints,
        split_tracklets,
        check_spatial_constraints,
        save_results,
    ) -> None:
        self.get_spatial_constraints = get_spatial_constraints
        self.split_tracklets = split_tracklets
        self.check_spatial_constraints = check_spatial_constraints
        self.save_results = save_results


def _build_deps() -> RefineDeps:
    """Import refine_tracklets (heavy deps) and bundle its reusable functions.

    Done lazily inside ``main()`` so that importing *this* module (e.g. from the
    CPU unit test) does NOT trigger refine_tracklets' torch/sklearn/scipy/
    matplotlib/seaborn/loguru imports.  The functions themselves are reused
    UNCHANGED — refine_tracklets.py is never modified.
    """
    import refine_tracklets  # noqa: WPS433 — intentional lazy import

    return RefineDeps(
        get_spatial_constraints=refine_tracklets.get_spatial_constraints,
        split_tracklets=refine_tracklets.split_tracklets,
        check_spatial_constraints=refine_tracklets.check_spatial_constraints,
        save_results=refine_tracklets.save_results,
    )


def _fast_connect(tracklets, deps, max_x_range, max_y_range, merge_dist_thres):
    """Exact, batched replacement for ``get_distance_matrix`` + ``merge_tracklets``.

    Produces the SAME merged ``{tid: Tracklet}`` as the original per-pair path
    (up to float rounding), but:

      * builds the all-pairs cosine-distance matrix in ONE batched matmul
        instead of ~N**2 per-pair ``get_distance`` calls (each of which moves
        two feature tensors to the GPU and back — the actual bottleneck), and
      * updates a merged tracklet's row with a single vector-matrix product
        instead of N per-pair GPU calls.

    Why it is EXACT, not an approximation
    -------------------------------------
    The per-pair ``get_distance(A, B)`` for non-overlapping tracklets is
    ``mean over (i in A, j in B) of (1 - cosine(f_i, g_j))``.  Because the dot
    product is bilinear, the average of the pairwise cosine *similarities*
    equals the dot product of the per-tracklet means of the L2-normalized
    features::

        mean_{i,j} (f_i/|f_i|) . (g_j/|g_j|) = (mean_i f_i/|f_i|) . (mean_j g_j/|g_j|)

    so ONE mean-of-normalized-features embedding per tracklet reproduces the
    full pairwise mean exactly.  A merge uses a frame-count-weighted mean, so
    the merged embedding equals what the per-pair path computes from the
    concatenated feature lists.  Overlapping tracklets get distance 1 (max),
    matching ``get_distance``'s ``set(times) & set(times)`` short-circuit —
    here via a batched occupancy product.

    The greedy hierarchical merge order, the spatial-constraint gate, and the
    "block this pair" branch are reproduced exactly so the merge SEQUENCE (and
    therefore the output) matches the original ``merge_tracklets``; only the
    distance *arithmetic* is reorganized.  ``check_spatial_constraints`` and
    ``save_results`` read only ``times`` / ``bboxes`` (never ``features``), so
    this skips concatenating the (large) per-tracklet feature lists.
    """
    import numpy as np

    tids = list(tracklets.keys())
    n = len(tids)
    if n == 0:
        return tracklets

    # --- per-tracklet mean of L2-normalized features; weight = frame count ---
    feat_dim = np.asarray(tracklets[tids[0]].features[0], dtype=np.float64).size
    means = np.zeros((n, feat_dim), dtype=np.float64)  # mean-normalized embedding
    counts = np.zeros(n, dtype=np.float64)             # weights (frame counts)
    for i, tid in enumerate(tids):
        feats = np.stack(
            [np.asarray(f, dtype=np.float64).ravel() for f in tracklets[tid].features]
        )  # (Li, D)
        feats /= np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12)
        means[i] = feats.mean(axis=0)
        counts[i] = feats.shape[0]

    # --- batched temporal-overlap matrix (share >=1 frame -> True) ----------
    max_frame = max(int(max(t.times)) for t in tracklets.values())
    occ = np.zeros((n, max_frame + 1), dtype=np.float32)
    for i, tid in enumerate(tids):
        occ[i, np.asarray(tracklets[tid].times, dtype=np.int64)] = 1.0
    overlap = (occ @ occ.T) > 0.5  # (n, n) bool: True iff they share a frame

    # --- all-pairs distance in ONE matmul -----------------------------------
    Dist = 1.0 - (means @ means.T)
    Dist[overlap] = 1.0
    np.fill_diagonal(Dist, np.inf)  # diagonal excluded from the argmin

    idx2tid = {i: tid for i, tid in enumerate(tids)}

    # --- same greedy hierarchical merge as merge_tracklets ------------------
    # argmin over the full matrix (diagonal = inf) picks the first off-diagonal
    # minimum in row-major order — identical pair (and tie-break) to the
    # original's argmin over the masked off-diagonal array. For a symmetric
    # matrix that first hit is the upper-triangle one, so t1 < t2 always (t1
    # therefore keeps its index after t2's row/col is deleted).
    while True:
        t1, t2 = np.unravel_index(int(np.argmin(Dist)), Dist.shape)
        if Dist[t1, t2] >= merge_dist_thres:
            break
        track1 = tracklets[idx2tid[t1]]
        track2 = tracklets[idx2tid[t2]]
        if deps.check_spatial_constraints(track1, track2, max_x_range, max_y_range):
            # merge track2 -> track1 (times + bboxes only; embedding via means)
            track1.times += track2.times
            track1.bboxes += track2.bboxes
            tracklets.pop(idx2tid[t2])

            # frame-count-weighted mean of normalized embeddings (exact)
            new_count = counts[t1] + counts[t2]
            means[t1] = (means[t1] * counts[t1] + means[t2] * counts[t2]) / new_count
            counts[t1] = new_count
            overlap[t1, :] |= overlap[t2, :]
            overlap[:, t1] = overlap[t1, :]

            # drop t2's row/col from every index-aligned structure
            Dist = np.delete(np.delete(Dist, t2, axis=0), t2, axis=1)
            means = np.delete(means, t2, axis=0)
            counts = np.delete(counts, t2, axis=0)
            overlap = np.delete(np.delete(overlap, t2, axis=0), t2, axis=1)
            idx2tid = {i: tid for i, tid in enumerate(tracklets.keys())}

            # recompute ONLY the merged tracklet's row/col (single matvec)
            new_row = 1.0 - (means @ means[t1])
            new_row[overlap[t1]] = 1.0
            Dist[t1, :] = new_row
            Dist[:, t1] = new_row
            Dist[t1, t1] = np.inf
        else:
            # block this pair (== thres -> never < thres again), like the original
            Dist[t1, t2] = Dist[t2, t1] = merge_dist_thres

    return tracklets


def run_refine(tmp, refined_txt, params, deps, seq_name):
    """Refine ONE pkl's tracklets: split, then connect/merge.

    Mirrors refine_tracklets.main()'s per-seq body (split THEN connect) for a
    single video's ``{tid: Tracklet}``, factored out and dependency-injected so
    it is unit-testable on CPU with stubs.  The connect step uses the exact
    batched ``_fast_connect``.

    Equivalent original code (refine_tracklets.main inner body)::

        max_x, max_y = get_spatial_constraints(tmp, spatial_factor)
        split = split_tracklets(tmp, ...)
        Dist  = get_distance_matrix(split)
        out   = merge_tracklets(split, {}, Dist, ...)
        save_results(out_path, out)

    Here, ``get_distance_matrix`` + ``merge_tracklets`` are replaced by the
    exact batched ``_fast_connect``.

    Parameters
    ----------
    tmp:
        ``{track_id: Tracklet}`` loaded from ``tracklets.pkl`` (mutated in place
        by ``split_tracklets`` / ``_fast_connect``, exactly as the original).
    refined_txt:
        Output path for the refined MOT txt (the contract ``refined.txt``).
    params:
        Dict with keys: ``min_len``, ``eps``, ``min_samples``, ``max_k``,
        ``spatial_factor``, ``merge_dist_thres``.
    deps:
        A :class:`RefineDeps` bundle of the algorithm callables.
    seq_name:
        Sequence name (the video stem); kept for parity / logging.

    Returns
    -------
    dict
        IO counts for profiling: ``n_tracklets_in``, ``n_tracklets_after_split``,
        ``n_tracklets_out``.
    """
    n_tracklets_in = len(tmp)

    # Spatial constraints come from the ORIGINAL tracklets (factor-scaled
    # extent), matching the original ordering: computed before splitting.
    max_x_range, max_y_range = deps.get_spatial_constraints(
        tmp, params["spatial_factor"]
    )

    # --- split component (split ID-switched tracklets) ---
    split = deps.split_tracklets(
        tmp,
        eps=params["eps"],
        max_k=params["max_k"],
        min_samples=params["min_samples"],
        len_thres=params["min_len"],
    )
    n_tracklets_after_split = len(split)

    # --- connect/merge component (exact batched _fast_connect) ---
    # The prints flush so the otherwise SILENT stretch shows progress in a
    # non-TTY (subprocess) log.
    print(
        "[stage2] connect: batched distance + merge over {} tracklets...".format(
            n_tracklets_after_split
        ),
        flush=True,
    )
    out = _fast_connect(
        split,
        deps,
        max_x_range=max_x_range,
        max_y_range=max_y_range,
        merge_dist_thres=params["merge_dist_thres"],
    )
    print(
        "[stage2] connect: done ({} tracklets after merge).".format(len(out)),
        flush=True,
    )
    n_tracklets_out = len(out)

    deps.save_results(refined_txt, out)

    return {
        "n_tracklets_in": n_tracklets_in,
        "n_tracklets_after_split": n_tracklets_after_split,
        "n_tracklets_out": n_tracklets_out,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def make_parser() -> argparse.ArgumentParser:
    """Stage-2 CLI: contract path args + refine params (re-declared).

    The refine params are RE-DECLARED here with the same names + defaults as
    ``refine_tracklets.parse_args`` (we do NOT reuse that parser because it has
    unrelated required args ``--dataset / --tracker / --track_src``).  We do NOT
    use the original's param-named output-dir scheme; output is the contract
    ``refined.txt``.
    """
    parser = argparse.ArgumentParser(
        description="Stage 2 — GtaLink tracklet refinement (single video, "
        "split + connect, with caching + profiling)."
    )

    # --- contract path resolution (identical to Stage 1) ---
    parser.add_argument(
        "--video",
        required=True,
        type=str,
        help="path to the input video — used ONLY to locate the artifacts dir "
        "(Stage 2 does NOT read the video).",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        type=str,
        help="base directory for artifacts (default: ./artifacts)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="force recompute even if cached outputs are up-to-date",
    )

    # --- refine params (same names + defaults as refine_tracklets.parse_args) ---
    parser.add_argument(
        "--min_len",
        type=int,
        default=100,
        help="Minimum length for a tracklet required for splitting.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.6,
        help="For DBSCAN clustering, the maximum distance between two samples "
        "for one to be considered as in the neighborhood of the other.",
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=10,
        help="The number of samples in a neighborhood for a point to be "
        "considered as a core point.",
    )
    parser.add_argument(
        "--max_k",
        type=int,
        default=3,
        help="Maximum number of clusters/subtracklets output by splitting.",
    )
    parser.add_argument(
        "--spatial_factor",
        type=float,
        default=1.0,
        help="Factor to adjust spatial distances.",
    )
    parser.add_argument(
        "--merge_dist_thres",
        type=float,
        default=0.4,
        help="Minimum cosine distance between two tracklets for merging.",
    )
    return parser


def main(args) -> None:
    paths = get_artifact_paths(args.video, base_dir=args.artifacts_dir)
    ensure_dirs(paths)

    tracklets_pkl = paths.tracklets_pkl
    refined_txt = paths.refined_txt

    # --- caching: skip if refined.txt is present + newer than tracklets.pkl ---
    if not should_run(refined_txt, [tracklets_pkl], force=args.force):
        print(
            "Stage 2 cached (refined.txt newer than tracklets.pkl); skipping. "
            "Use --force to recompute. Output: {}".format(refined_txt)
        )
        return

    if not osp.isfile(tracklets_pkl):
        raise FileNotFoundError(
            "Stage 1 output not found: {} (run Stage 1 first).".format(
                tracklets_pkl
            )
        )

    # Lazy-import refine_tracklets (heavy deps); functions reused UNCHANGED.
    deps = _build_deps()

    # Load the {tid: Tracklet} pickle.  With CWD = gta-link, importing
    # refine_tracklets above already imported `Tracklet`, so the pickle's
    # `Tracklet.Tracklet` reference resolves without us re-registering anything.
    with open(tracklets_pkl, "rb") as pkl_f:
        tmp = pickle.load(pkl_f)

    seq_name = osp.splitext(osp.basename(args.video))[0]  # video stem

    params = {
        "min_len": args.min_len,
        "eps": args.eps,
        "min_samples": args.min_samples,
        "max_k": args.max_k,
        "spatial_factor": args.spatial_factor,
        "merge_dist_thres": args.merge_dist_thres,
    }

    io_counts = {
        "process": "Split+Connect",
        "n_tracklets_in": 0,
        "n_tracklets_after_split": 0,
        "n_tracklets_out": 0,
        "n_output_rows": 0,
        # key params, recorded for the profile.
        "eps": args.eps,
        "min_samples": args.min_samples,
        "max_k": args.max_k,
        "merge_dist_thres": args.merge_dist_thres,
        "min_len": args.min_len,
        "spatial_factor": args.spatial_factor,
    }

    with profile_stage("02_refine", paths, extra=io_counts):
        counts = run_refine(tmp, str(refined_txt), params, deps, seq_name)
        io_counts.update(counts)
        # n_output_rows: total MOT rows written (sum of tracklet lengths in
        # `out`).  After run_refine, `tmp`/split were mutated in place; the
        # written file is authoritative, so count its lines.
        io_counts["n_output_rows"] = _count_lines(refined_txt)

    print("Stage 2 done: {}".format(io_counts))


def _count_lines(path) -> int:
    """Count rows in the written refined.txt (one MOT row per line)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


if __name__ == "__main__":
    main(make_parser().parse_args())

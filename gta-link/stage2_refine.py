"""
Stage 2 (refine) entrypoint: a thin single-video wrapper around GtaLink's
``refine_tracklets.py``.  It consumes the ONE ``tracklets.pkl`` Stage 1 produced
and writes the contract ``refined.txt``, reusing refine_tracklets' algorithm
functions UNCHANGED, with caching + profiling.

Artifact contract
-----------------
    <artifacts>/<video_stem>/01_track/tracklets.pkl   (input, from Stage 1)
    <artifacts>/<video_stem>/02_refine/refined.txt    (output)
    <artifacts>/<video_stem>/profiles/02_refine.json  (profiling)

Run with CWD = ``gta-link`` so that ``import refine_tracklets`` and
``import Tracklet`` both resolve to the sibling files in this directory::

    cd gta-link
    python stage2_refine.py --video /path/to/clip.mp4 --use_split --use_connect

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

DELIBERATE DEVIATION from refine_tracklets.main()
-------------------------------------------------
The original ``refine_tracklets.main()`` computes the distance matrix and calls
``merge_tracklets`` **unconditionally** — it does NOT gate merging on
``--use_connect`` (it only gates *splitting* on ``--use_split``).  This wrapper
INTENTIONALLY gates the merge step on ``--use_connect`` (see ``run_refine``) so
the documented flag semantics (README: "--use_connect: use the connecting
component") actually hold.  Consequences:

  * Full pipeline (``--use_split --use_connect``) is IDENTICAL to the original.
  * ``--use_split`` only -> split, then NO merge (original would also merge).
  * ``--use_connect`` only -> NO split, then merge (matches original).
  * neither -> raises (matches original).

The algorithm functions in ``refine_tracklets.py`` are reused verbatim; only the
step-decision orchestration differs, and only in the ``--use_split``-only case.
This is flagged for ratification in the task report.
"""

from __future__ import annotations

import argparse
import os
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
      * ``get_distance_matrix(tracklets) -> ndarray``
      * ``merge_tracklets(tracklets, seq2Dist, Dist, seq_name, max_x_range,
                          max_y_range, merge_dist_thres) -> dict``
      * ``save_results(out_path, tracklets) -> None``
    """

    def __init__(
        self,
        get_spatial_constraints,
        split_tracklets,
        get_distance_matrix,
        merge_tracklets,
        save_results,
    ) -> None:
        self.get_spatial_constraints = get_spatial_constraints
        self.split_tracklets = split_tracklets
        self.get_distance_matrix = get_distance_matrix
        self.merge_tracklets = merge_tracklets
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
        get_distance_matrix=refine_tracklets.get_distance_matrix,
        merge_tracklets=refine_tracklets.merge_tracklets,
        save_results=refine_tracklets.save_results,
    )


def run_refine(tmp, refined_txt, params, deps, seq_name):
    """Mirror refine_tracklets.main()'s per-seq body for ONE pkl's tracklets.

    This is the step-decision core, factored out and dependency-injected so it
    is unit-testable on CPU with stubs.  It reproduces the original per-sequence
    pipeline EXCEPT that the merge step is gated on ``params["use_connect"]``
    (see the module docstring's "DELIBERATE DEVIATION" note).

    Equivalent original code (refine_tracklets.main inner body)::

        max_x, max_y = get_spatial_constraints(tmp, spatial_factor)
        split = split_tracklets(tmp, ...) if use_split else tmp
        Dist = get_distance_matrix(split)          # original: ALWAYS
        out  = merge_tracklets(split, {}, Dist...)  # original: ALWAYS
        save_results(out_path, out)

    Here, ``get_distance_matrix`` + ``merge_tracklets`` run only when
    ``use_connect`` is set.

    Parameters
    ----------
    tmp:
        ``{track_id: Tracklet}`` loaded from ``tracklets.pkl`` (mutated in place
        by ``split_tracklets`` / ``merge_tracklets``, exactly as the original).
    refined_txt:
        Output path for the refined MOT txt (the contract ``refined.txt``).
    params:
        Dict with keys: ``use_split``, ``use_connect``, ``min_len``, ``eps``,
        ``min_samples``, ``max_k``, ``spatial_factor``, ``merge_dist_thres``.
    deps:
        A :class:`RefineDeps` bundle of the algorithm callables.
    seq_name:
        Sequence name passed to ``merge_tracklets`` (the video stem).

    Returns
    -------
    dict
        IO counts for profiling: ``n_tracklets_in``, ``n_tracklets_after_split``,
        ``n_tracklets_out``.
    """
    use_split = params["use_split"]
    use_connect = params["use_connect"]

    # Require at least one component, exactly like the original main() does.
    if not use_split and not use_connect:
        raise ValueError(
            "Both use_split and use_connect are false, must at least use one "
            "of --use_split / --use_connect."
        )

    n_tracklets_in = len(tmp)

    # Spatial constraints come from the ORIGINAL tracklets (factor-scaled
    # extent), matching the original ordering: computed before splitting.
    max_x_range, max_y_range = deps.get_spatial_constraints(
        tmp, params["spatial_factor"]
    )

    # --- split component (gated on use_split, same as original) ---
    if use_split:
        split = deps.split_tracklets(
            tmp,
            eps=params["eps"],
            max_k=params["max_k"],
            min_samples=params["min_samples"],
            len_thres=params["min_len"],
        )
    else:
        split = tmp
    n_tracklets_after_split = len(split)

    # --- connect/merge component (gated on use_connect — the DEVIATION) ---
    # Original main() runs get_distance_matrix + merge_tracklets UNCONDITIONALLY;
    # we gate them on use_connect so --use_split-only does not also merge.
    if use_connect:
        Dist = deps.get_distance_matrix(split)
        # merge_tracklets mutates `split` and takes a seq2Dist debug dict; pass
        # an empty dict and the video stem as seq_name (per the task brief).
        out = deps.merge_tracklets(
            split,
            {},
            Dist,
            seq_name=seq_name,
            max_x_range=max_x_range,
            max_y_range=max_y_range,
            merge_dist_thres=params["merge_dist_thres"],
        )
    else:
        out = split
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
        "contract CLI with caching + profiling)."
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
        "--use_split",
        action="store_true",
        help="If using split component.",
    )
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
        "--use_connect",
        action="store_true",
        help="If using connecting component.",
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
    # Determine the process label (also validates at-least-one-flag, like the
    # original main()).
    if args.use_split and args.use_connect:
        process = "Split+Connect"
    elif args.use_split:
        process = "Split"
    elif args.use_connect:
        process = "Connect"
    else:
        raise ValueError(
            "Both use_split and use_connect are false, must at least use one "
            "of --use_split / --use_connect."
        )

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
        "use_split": args.use_split,
        "use_connect": args.use_connect,
        "min_len": args.min_len,
        "eps": args.eps,
        "min_samples": args.min_samples,
        "max_k": args.max_k,
        "spatial_factor": args.spatial_factor,
        "merge_dist_thres": args.merge_dist_thres,
    }

    io_counts = {
        "process": process,
        "n_tracklets_in": 0,
        "n_tracklets_after_split": 0,
        "n_tracklets_out": 0,
        "n_output_rows": 0,
        # key params, recorded for the profile.
        "use_split": args.use_split,
        "use_connect": args.use_connect,
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

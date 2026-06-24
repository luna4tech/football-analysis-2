"""
Stage 3 (team) entrypoint: assign player tracks to two teams via ReID
clustering, then emit the class+team output the renderer/eval consume.

It consumes the ONE ``refined_tracklets.pkl`` Stage 2 produced (whose ids match
``02_refine/refined.txt``) and writes:

  * ``03_team/refined.txt`` — ``02_refine/refined.txt`` with the per-frame
    ``team_id`` written to MOT col 9 (cols 1-8 kept byte-identical; only col 9
    is rewritten, looked up by the row's track id).
  * ``03_team/track_attributes.json`` — ``{ "<id>": {"class", "team", "gk"} }``
    (one entry per track, the aggregated class + assigned team + gk flag).

Artifact contract
-----------------
    <artifacts>/<video_stem>/02_refine/refined.txt              (input)
    <artifacts>/<video_stem>/02_refine/refined_tracklets.pkl    (input)
    <artifacts>/<video_stem>/03_team/refined.txt                (output)
    <artifacts>/<video_stem>/03_team/track_attributes.json      (output)
    <artifacts>/<video_stem>/profiles/03_team.json              (profiling)

CWD-independent imports / pickle resolution
-------------------------------------------
Unlike Stage 2 (run with CWD = ``gta-link``), this stage is CWD-independent: it
computes the repo root from ``__file__`` and prepends BOTH the repo root (so
``import pipeline.*`` resolves) AND ``<repo>/gta-link`` (so ``import Tracklet``
registers the top-level module name the pickle references) to ``sys.path``
before ``pickle.load``.  The pkl is ``{id: Tracklet}`` pickled with the class as
the top-level module ``"Tracklet"`` (see ``stage1_assembly.load_tracklet_class``
/ ``stage2_refine``), so the pickle references ``Tracklet.Tracklet``.

CPU-light import / dependency injection
---------------------------------------
The heavy clustering (sklearn KMeans) is isolated behind ``_kmeans_cluster`` and
lazy-imports sklearn only when called.  All surrounding logic (class
aggregation, deterministic naming, gating, txt/json writing) takes the cluster
function as an INJECTED parameter (mirroring stage2_refine's ``RefineDeps``
style), so the module imports on a CPU-only box and the logic is unit-testable
with a fake cluster function.

Run with::

    python pipeline/team_assignment.py --video /path/to/clip.mp4

``--video`` is used ONLY to locate the artifact directory; Stage 3 does NOT read
the video itself.
"""

from __future__ import annotations

import argparse
import json
import os.path as osp
import pickle
import sys

# --- import path setup (CWD-independent) ------------------------------------
# Add the repo root (for `import pipeline.*`) AND <repo>/gta-link (so the
# pickle's `Tracklet.Tracklet` reference resolves to gta-link/Tracklet.py)
# BEFORE pickle.load.  Done at import time so this module works from any CWD.
_THIS_DIR = osp.dirname(osp.abspath(__file__))
_REPO_ROOT = osp.abspath(osp.join(_THIS_DIR, ".."))
_GTA_LINK_DIR = osp.join(_REPO_ROOT, "gta-link")
for _p in (_REPO_ROOT, _GTA_LINK_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.artifacts import ensure_dirs, get_artifact_paths
from pipeline.cache import should_run
from pipeline.profiling import profile_stage

# Canonical class ids (convention): 0=player, 1=goalkeeper, 2=referee, -1=unknown.
PLAYER_CLASS = 0
GOALKEEPER_CLASS = 1
REFEREE_CLASS = 2
UNKNOWN_TEAM = -1

# Fixed clustering config (k is fixed at 2; deterministic seed / n_init).
N_CLUSTERS = 2
RANDOM_SEED = 0
N_INIT = 10


# ---------------------------------------------------------------------------
# Per-track class aggregation
# ---------------------------------------------------------------------------
def aggregate_class(class_ids, scores=None):
    """Return the confidence-weighted mode of ``class_ids``.

    Each per-frame class id is weighted by its detection ``score``; the class
    with the greatest total weight wins.  Falls back to an unweighted count
    when ``scores`` is absent, empty, or shorter than ``class_ids`` (so a
    short/missing score array never silently down-weights frames).  Ties break
    on the LOWEST class id (deterministic).  Empty input -> ``-1`` (unknown).
    """
    if not class_ids:
        return -1

    use_scores = scores is not None and len(scores) >= len(class_ids)

    weights = {}
    for idx, cid in enumerate(class_ids):
        w = float(scores[idx]) if use_scores else 1.0
        weights[cid] = weights.get(cid, 0.0) + w

    # Max total weight; tie-break on the lowest class id for determinism.
    best_cid = None
    best_w = None
    for cid in sorted(weights):
        if best_w is None or weights[cid] > best_w:
            best_cid = cid
            best_w = weights[cid]
    return best_cid


def _mean_embedding(features):
    """Return the L2-normalized mean of a track's per-frame features.

    Returns ``None`` if the track has no features (cannot be clustered).
    """
    import numpy as np

    if not features:
        return None
    feats = np.stack([np.asarray(f, dtype=np.float64).ravel() for f in features])
    mean = feats.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm < 1e-12:
        return mean
    return mean / norm


# ---------------------------------------------------------------------------
# Clustering (heavy dep isolated + injectable for CPU testing)
# ---------------------------------------------------------------------------
def _kmeans_cluster(embeddings, k, seed=RANDOM_SEED, n_init=N_INIT):
    """Cluster L2-normalized embeddings with k-means; lazy-imports sklearn.

    ``embeddings`` is an ``(n, d)`` array of L2-normalized vectors (so euclidean
    k-means is equivalent to cosine k-means).  Returns a length-``n`` list of
    integer RAW cluster labels in ``[0, k)``.

    Heavy imports (numpy/sklearn) happen HERE so the module stays CPU-light;
    callers inject a fake for unit tests (mirroring stage2_refine's deps style).
    """
    import numpy as np
    from sklearn.cluster import KMeans  # noqa: WPS433 — intentional lazy import

    X = np.asarray(embeddings, dtype=np.float64)
    km = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
    labels = km.fit_predict(X)
    return [int(lbl) for lbl in labels]


# ---------------------------------------------------------------------------
# Deterministic team naming
# ---------------------------------------------------------------------------
def _name_teams(player_ids, raw_labels):
    """Map RAW cluster labels to deterministic ``team_id`` 0/1.

    The LARGER cluster gets ``team_id 0`` (the smaller gets ``1``); ties break
    by the lowest minimum track id in the cluster, so the names are stable
    across runs.  Returns ``{track_id: team_id}`` for the given player ids.

    Parameters
    ----------
    player_ids:
        Track ids being clustered (aligned 1:1 with ``raw_labels``).
    raw_labels:
        The raw cluster label per player id (e.g. from k-means).
    """
    # Gather, per raw label, the track ids in that cluster.
    clusters = {}
    for tid, lbl in zip(player_ids, raw_labels):
        clusters.setdefault(lbl, []).append(tid)

    # Sort clusters by (-size, min_track_id): larger first, lowest-min wins ties.
    ordered = sorted(
        clusters.items(),
        key=lambda kv: (-len(kv[1]), min(kv[1])),
    )

    team_of = {}
    for team_id, (_lbl, tids) in enumerate(ordered):
        for tid in tids:
            team_of[tid] = team_id
    return team_of


# ---------------------------------------------------------------------------
# Core assignment logic (cluster fn injected; CPU-light + unit-testable)
# ---------------------------------------------------------------------------
def assign_teams(tracklets, cluster_fn=_kmeans_cluster):
    """Aggregate class + assign teams for ``{track_id: Tracklet}``.

    Steps (see the spec / Task 003):

    1. Per track, aggregate class = confidence-weighted mode of ``class_ids``;
       ``gk = (class == goalkeeper)``.
    2. Player tracks = aggregated class ``player``; per player track the mean
       embedding = L2-normalize(mean of its features).
    3. With >= 2 player tracks (that have usable embeddings): cluster the mean
       embeddings with the injected ``cluster_fn`` (k=2), then remap raw labels
       to deterministic team ids (larger cluster -> 0, tie-break lowest min id).
    4. Players get team 0/1; goalkeepers and referees get -1.  With < 2 player
       tracks, every track gets team -1.

    Returns
    -------
    dict
        ``{track_id: {"class": int, "team": int, "gk": bool}}`` — one entry per
        track id in ``tracklets``.
    """
    attributes = {}
    player_ids = []
    player_embeddings = []

    for tid in sorted(tracklets.keys()):
        track = tracklets[tid]
        cls = aggregate_class(
            getattr(track, "class_ids", None) or [],
            getattr(track, "scores", None),
        )
        attributes[tid] = {
            "class": int(cls),
            "team": UNKNOWN_TEAM,
            "gk": bool(cls == GOALKEEPER_CLASS),
        }
        if cls == PLAYER_CLASS:
            emb = _mean_embedding(getattr(track, "features", None) or [])
            if emb is not None:
                player_ids.append(tid)
                player_embeddings.append(emb)

    # Cluster only when >= 2 player tracks have usable embeddings.
    if len(player_ids) >= N_CLUSTERS:
        raw_labels = cluster_fn(player_embeddings, N_CLUSTERS)
        team_of = _name_teams(player_ids, raw_labels)
        for tid, team_id in team_of.items():
            attributes[tid]["team"] = int(team_id)

    return attributes


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_team_txt(refined_txt, team_refined_txt, attributes):
    """Write ``03_team/refined.txt`` = ``02_refine/refined.txt`` with col 9 set.

    Reads each MOT row of ``refined_txt``, looks up the row's track team by its
    col-2 id, and rewrites ONLY col 9 (``team_id``) to that team.  Cols 1-8 and
    col 10 are kept byte-identical to the input.  Tracks not in ``attributes``
    (should not happen given pkl ids == refined.txt ids) keep col 9 unchanged.

    Returns the number of rows written.
    """
    n_rows = 0
    with open(refined_txt, "r", encoding="utf-8", newline="") as src, open(
        team_refined_txt, "w", encoding="utf-8", newline=""
    ) as dst:
        for line in src:
            # Preserve the line's exact terminator (rewrite only col 9's token).
            stripped = line.rstrip("\n")
            had_newline = line.endswith("\n")
            if not stripped:
                dst.write(line)
                continue
            cols = stripped.split(",")
            tid = int(cols[1])
            attrs = attributes.get(tid)
            if attrs is not None:
                cols[8] = str(attrs["team"])
            out = ",".join(cols)
            dst.write(out + "\n" if had_newline else out)
            n_rows += 1
    return n_rows


def write_track_attributes(track_attributes_json, attributes):
    """Write ``03_team/track_attributes.json`` ({id: {class, team, gk}}).

    JSON keys are STRINGS (every track id is included: players, GKs, referees,
    unknowns).
    """
    out = {
        str(tid): {
            "class": attrs["class"],
            "team": attrs["team"],
            "gk": attrs["gk"],
        }
        for tid, attrs in sorted(attributes.items())
    }
    with open(track_attributes_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def run_team_assignment(
    refined_pkl, refined_txt, team_refined_txt, track_attributes_json, cluster_fn
):
    """Load the refined pkl, assign teams, and write both Stage-3 outputs.

    Dependency-injected (``cluster_fn``) so the full flow is unit-testable on a
    CPU-only box with a fake cluster function.  Returns IO counts for profiling.
    """
    with open(refined_pkl, "rb") as f:
        tracklets = pickle.load(f)

    attributes = assign_teams(tracklets, cluster_fn=cluster_fn)
    n_rows = write_team_txt(refined_txt, team_refined_txt, attributes)
    write_track_attributes(track_attributes_json, attributes)

    n_players = sum(1 for a in attributes.values() if a["class"] == PLAYER_CLASS)
    teamed = sum(1 for a in attributes.values() if a["team"] != UNKNOWN_TEAM)
    return {
        "n_tracks": len(attributes),
        "n_players": n_players,
        "n_team_0": sum(1 for a in attributes.values() if a["team"] == 0),
        "n_team_1": sum(1 for a in attributes.values() if a["team"] == 1),
        "n_teamed": teamed,
        "n_output_rows": n_rows,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def make_parser() -> argparse.ArgumentParser:
    """Stage-3 CLI: contract path args (mirrors stage2_refine's CLI)."""
    parser = argparse.ArgumentParser(
        description="Stage 3 — team assignment via ReID clustering "
        "(single video, k=2 deterministic, with caching + profiling)."
    )
    parser.add_argument(
        "--video",
        required=True,
        type=str,
        help="path to the input video — used ONLY to locate the artifacts dir "
        "(Stage 3 does NOT read the video).",
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
    return parser


def main(args) -> None:
    paths = get_artifact_paths(args.video, base_dir=args.artifacts_dir)
    ensure_dirs(paths)

    refined_pkl = paths.refined_tracklets_pkl
    refined_txt = paths.refined_txt
    team_refined_txt = paths.team_refined_txt
    track_attributes_json = paths.track_attributes_json

    # --- caching: skip only if BOTH outputs are present + newer than the pkl ---
    if (
        not should_run(team_refined_txt, [refined_pkl], force=args.force)
        and osp.isfile(track_attributes_json)
    ):
        print(
            "Stage 3 cached (refined.txt + track_attributes.json up-to-date); "
            "skipping. Use --force to recompute. Output: {}".format(team_refined_txt)
        )
        return

    if not osp.isfile(refined_pkl):
        raise FileNotFoundError(
            "Stage 2 output not found: {} (run Stage 2 first).".format(refined_pkl)
        )
    if not osp.isfile(refined_txt):
        raise FileNotFoundError(
            "Stage 2 output not found: {} (run Stage 2 first).".format(refined_txt)
        )

    io_counts = {
        "process": "TeamAssignment",
        "k": N_CLUSTERS,
        "seed": RANDOM_SEED,
    }

    with profile_stage("03_team", paths, extra=io_counts):
        counts = run_team_assignment(
            str(refined_pkl),
            str(refined_txt),
            str(team_refined_txt),
            str(track_attributes_json),
            cluster_fn=_kmeans_cluster,
        )
        io_counts.update(counts)

    print("Stage 3 done: {}".format(io_counts))


if __name__ == "__main__":
    main(make_parser().parse_args())

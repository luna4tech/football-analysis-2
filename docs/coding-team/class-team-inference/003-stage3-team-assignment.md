# Task 003 — Stage 3 team assignment

## Context
Stage 2 (done) writes `02_refine/refined.txt` (per-frame class in col 8) and
`02_refine/refined_tracklets.pkl` = `{id: Tracklet}` whose ids match `refined.txt` and whose
per-frame arrays (`times`/`bboxes`/`features`/`scores`/`class_ids`) are all aligned. Features
are L2-normalized 512-D ReID embeddings.

Canonical class ids (convention): `0=player, 1=goalkeeper, 2=referee`, unknown `-1`.

This task adds Stage 3: assign player tracks to two teams via ReID clustering and emit the
class+team output the renderer/eval will consume.

## Objective
New `pipeline/team_assignment.py` (with CLI), wired into the orchestrator after Stage 2, that
produces:
- `03_team/refined.txt`: `02_refine/refined.txt` with per-frame `team_id` written to col 9.
- `03_team/track_attributes.json`: `{ "<track_id>": {"class": int, "team": int, "gk": bool} }`.

## Algorithm
1. Load `refined_tracklets.pkl`.
2. Per track, aggregate class = **confidence-weighted mode** of `class_ids` (weight by `scores`;
   fall back to unweighted count if scores absent/short). `gk = (class == 1)`.
3. Player tracks = aggregated class `0`. Per player track, mean embedding =
   L2-normalize(mean of its `features`).
4. If **≥ 2** player tracks: cluster their mean embeddings with **k=2 cosine k-means**
   (euclidean k-means on L2-normalized vectors is equivalent). Use a fixed seed / `n_init` for
   reproducibility.
   - **Deterministic naming**: the **larger** cluster → `team_id 0`, the smaller → `team_id 1`;
     tie-break by lowest minimum track_id in the cluster. Remap raw cluster labels accordingly.
5. `team_id`: players get `0`/`1`; goalkeepers and referees get `-1`. If fewer than 2 player
   tracks exist, every track gets `team_id = -1` (still emit both outputs).
6. `03_team/refined.txt`: read `02_refine/refined.txt` and rewrite **only col 9** to the row's
   track's `team_id` (look up by col-2 id). Leave cols 1-8 byte-identical. (Row parity is
   guaranteed: pkl ids == refined.txt ids.)
7. `03_team/track_attributes.json`: one entry per track with aggregated `class`, `team`, `gk`.

## Scope
- `pipeline/team_assignment.py`: the logic above + a CLI mirroring `stage2_refine.py`
  (`--video`, `--artifacts-dir`, `--force`), with self-caching (skip when both outputs exist and
  are newer than the input pkl) and a `profile_stage("03_team", paths, ...)` block so the run
  appears in `summary.md`.
  - **Pickle resolution / imports (CWD-independent)**: compute repo root from `__file__`; add
    repo root (for `import pipeline.*`) AND `<repo>/gta-link` (so `import Tracklet` registers the
    module name the pickle references) to `sys.path` before `pickle.load`.
  - **Isolate the clustering** behind a small function so the module imports CPU-light and the
    surrounding logic is unit-testable by injecting a fake cluster function (mirror the
    `RefineDeps` dependency-injection style). Lazy-import sklearn inside the real cluster fn.
- `pipeline/artifacts.py`: add `team_dir` (`03_team`), `team_refined_txt`
  (`03_team/refined.txt`), `track_attributes_json` (`03_team/track_attributes.json`); add
  `team_dir` to `ensure_dirs`; update the module docstring tree.
- `pipeline/orchestrator.py`: add `STAGE3_CWD` (= repo root) / `STAGE3_SCRIPT`
  (`pipeline/team_assignment.py`), a `build_stage3_command(...)` pure builder, `force_stage3` on
  `RunOptions` (and `--force-all` should set it), run Stage 3 after Stage 2 (abort/raise
  `StageError` on non-zero), and add `stage3_seconds` to `RunResult`.
- `pipeline/__main__.py`: if it wires `--force-all`/forwards force flags, extend it for
  `force_stage3` so `--force-all` re-runs Stage 3 too. (Check the existing pattern; keep it
  consistent. A dedicated `assign-teams` subcommand is optional, not required.)

## Non-goals / Later
- No rendering changes (Task 004). No eval changes (Task 005).
- No jersey-color/Approach-A, no GK→team assignment, no home/away semantics.
- Do not modify Stage 1/2 code or the `Tracklet` model.

## Constraints / Caveats
- Output `03_team/refined.txt` must keep cols 1-8 byte-identical to `02_refine/refined.txt`;
  only col 9 changes.
- k is fixed at 2. Clustering must be deterministic across runs (seed + the naming rule).
- Keep `pipeline/team_assignment.py` importable without sklearn/torch (lazy heavy imports), so
  the orchestrator and CPU tests stay light.
- JSON keys are strings; include every track (players, GKs, referees, unknowns).

## Acceptance criteria
- With ≥2 player tracks, players are split into teams 0/1 by the deterministic rule; GK/referee
  team = -1.
- With <2 player tracks, all team ids are -1 and both outputs are still written.
- `03_team/refined.txt` cols 1-8 == `02_refine/refined.txt`; col 9 = the track's team id.
- `track_attributes.json` has one correct entry per track id present in `refined.txt`.

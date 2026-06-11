# Task 004 — Stage 2 refine wrapper (contract CLI + caching + profiling)

## Context
Stage 2 is GtaLink tracklet refinement. `gta-link/refine_tracklets.py` today is a BATCH script: its
`main()` iterates a directory of `.pkl` files, runs split/merge, and writes param-named output dirs.
We want a thin **single-video** Stage-2 entrypoint that consumes the ONE `tracklets.pkl` Stage 1
produced and writes the contract `refined.txt`, reusing refine_tracklets' algorithm functions
UNCHANGED, with caching + profiling.

Facts to rely on:
- `tracklets.pkl` is `{track_id: Tracklet}` and was pickled (Task 2) with the class registered as
  module `"Tracklet"`, so it unpickles wherever `import Tracklet` resolves to `gta-link/Tracklet.py`.
- `refine_tracklets.py` exposes the reusable functions: `get_spatial_constraints`,
  `split_tracklets`, `get_distance_matrix`, `merge_tracklets`, `save_results`. Importing the module
  pulls in numpy/torch/sklearn/scipy/matplotlib/seaborn/loguru/tqdm (heavy; fine on Colab).

This machine can't run the heavy deps — static review + a stubbed CPU test only.

## Objective
Add `gta-link/stage2_refine.py` (run with CWD = `gta-link`, so `import refine_tracklets` and
`import Tracklet` both resolve; `sys.path`-insert the repo root to `import pipeline`). It:
1. resolves the artifact contract for a video, reads `01_track/tracklets.pkl`,
2. runs split and/or connect by reusing `refine_tracklets`' functions unmodified,
3. writes `02_refine/refined.txt`,
4. with `should_run` caching and a `profile_stage("02_refine", ...)` profile.

## Scope (do now)

### Orchestration (mirror refine_tracklets.main()'s per-seq body, for ONE pkl)
```
tmp = pickle.load(tracklets_pkl)                      # {tid: Tracklet}
max_x, max_y = get_spatial_constraints(tmp, spatial_factor)
split = split_tracklets(tmp, eps, max_k, min_samples, len_thres=min_len) if use_split else tmp
if use_connect:
    Dist = get_distance_matrix(split)
    out = merge_tracklets(split, {}, Dist, seq_name=<stem>, max_x_range=max_x,
                          max_y_range=max_y, merge_dist_thres=merge_dist_thres)
else:
    out = split
save_results(refined_txt, out)
```
- Require at least one of `--use_split`/`--use_connect` (raise like the original `main()` does).

### Deliberate behavior note (call it out in code + report)
The original `refine_tracklets.main()` computes `Dist` and calls `merge_tracklets` **unconditionally**
— it does NOT gate merging on `--use_connect`. This task INTENTIONALLY gates merge on `--use_connect`
(above) so the documented flag semantics (README: "--use_connect: use the connecting component")
actually hold. The full pipeline (`--use_split --use_connect`) is therefore identical to the
original; the only difference is `--use_split`-only no longer also merges. The algorithm functions
themselves are untouched. Flag this clearly in your report so it can be ratified.

### CLI
- Path resolution identical to Stage 1: `--video` (required, used only to locate the artifacts dir —
  Stage 2 does NOT read the video) and `--artifacts-dir` (optional) -> `get_artifact_paths(...)`.
- Refine params, same names + defaults as `refine_tracklets.parse_args`: `--use_split`,
  `--min_len` (100), `--eps` (0.6), `--min_samples` (10), `--max_k` (3), `--use_connect`,
  `--spatial_factor` (1.0), `--merge_dist_thres` (0.4). Re-declare them here; do NOT reuse
  `refine_tracklets.parse_args` (it has unrelated required args `--dataset/--tracker/--track_src`).
- `--force`.
- Do NOT reuse the param-named output-dir scheme from the original; write to the contract
  `refined_txt`.

### Caching + profiling
- `should_run(refined_txt, [tracklets_pkl], force=args.force)`; if cached, log and exit 0.
- `profile_stage("02_refine", paths, extra={...})` recording counts:
  `n_tracklets_in`, `n_tracklets_after_split`, `n_tracklets_out`, `n_output_rows`, and the key
  params (eps/min_samples/max_k/merge_dist_thres/min_len/spatial_factor, use_split, use_connect).

### Unit test (CPU-only, runnable here)
The refine functions need torch/sklearn/scipy, so don't depend on them. Factor the step-decision
orchestration so it accepts INJECTED callables for split/distance/merge/save (e.g.
`run_refine(tmp, flags, params, deps=...)`), and write a CPU test that, with stubs:
- `--use_split` only -> split called, merge NOT called, save gets the split result;
- `--use_connect` only -> split NOT called, merge called;
- both -> split then merge;
- neither -> raises;
- save target is the contract `refined_txt` path.
Optionally, if torch+sklearn+scipy happen to be importable, add a tiny end-to-end refine on a
synthetic 2-tracklet pkl that SKIPS gracefully when any dep is missing (must not fail on this host).

## Non-goals / Later
- Do NOT modify `refine_tracklets.py`, `Tracklet.py`, or anything in Deep-EIoU.
- No batch/dir mode; single pkl only.
- No orchestrator / `python -m pipeline` wrapper (Task 5).
- Keep `generate_tracklets.py` as-is (unused fallback).

## Constraints / Caveats
- Run with CWD = `gta-link`. The pkl must unpickle here via `import Tracklet`; do not re-register or
  shadow the `Tracklet` module name.
- `save_results` renumbers ids to 1..N and writes score=1, frames as-is (0-based from Stage 1) —
  that's the existing format; keep it. `refined.txt` then renders with `render_from_txt.py` WITHOUT
  `--one_indexed`.
- `merge_tracklets` mutates its inputs and takes a `seq2Dist` dict (debug display) — pass `{}` and a
  `seq_name` of the video stem.

## Acceptance criteria
- Reads `01_track/tracklets.pkl`, writes `02_refine/refined.txt`; caching + profiling wired via the
  pipeline helpers.
- Reuses refine_tracklets' functions with that file unmodified; merge gated on `--use_connect`
  (deviation documented).
- The stubbed CPU orchestration test passes here without torch/sklearn/scipy.
- Report what ran here vs deferred to Colab.

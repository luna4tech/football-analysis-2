# Task E1 — Eval scaffold + MOTChallenge materialization (no TrackEval run yet)

## Context
We are building a standalone tracking-eval tool (see `docs/coding-team/evaluation/PLAN.md`) that
scores the pipeline's `refined.txt` against ground truth using TrackEval (HOTA/MOTA/IDF1). This task
builds everything EXCEPT the actual TrackEval invocation: the CLI, the GT/pred loaders, the
0->1-based frame handling, the TrackEval on-disk layout writer, and a parser for TrackEval's text
output. All of it must be CPU-testable here (no TrackEval, no torch). Task E2 wires the real run.

## Objective
Create the `eval/` package (top-level, separate from `pipeline/`) with the building blocks and a CLI
that loads inputs and materializes the TrackEval MOTChallenge layout. The actual `trackeval` call is
a clearly-marked seam left for Task E2.

## Scope (do now)

### `eval/gt_adapters.py` — GT loader seam
- A canonical in-memory GT representation (a small dataclass or an `Nx9` numpy array with columns
  `frame,id,x,y,w,h,conf,class,visibility`).
- `load_gt(path, fmt="motchallenge") -> canonical`: for now implement `fmt="motchallenge"` (parse a
  standard MOTChallenge `gt.txt`, which is **1-based** frames). Structure it as a dispatch
  (dict of `fmt -> loader`) so the user's custom-export converter can be registered later WITHOUT
  touching the eval core. Document the seam.

### `eval/io_pred.py` (or fold into gt_adapters/evaluate) — prediction loader
- `load_pred(path) -> rows`: parse the pipeline's `refined.txt`
  (`frame,id,x,y,w,h,score,-1,-1,-1`), frames **0-based**. Keep score.

### `eval/trackeval_runner.py` — layout writer + output parser (NO trackeval import yet)
- `materialize_layout(work_dir, seq_name, gt, pred, *, tracker_name="refined", benchmark="SPORTS",
  split="eval", img_width=1920, img_height=1080) -> paths`: write the standard MOTChallenge layout
  TrackEval expects:
  ```
  <work>/gt/mot_challenge/<BENCH>-<SPLIT>/<seq>/gt/gt.txt
  <work>/gt/mot_challenge/<BENCH>-<SPLIT>/<seq>/seqinfo.ini
  <work>/gt/mot_challenge/seqmaps/<BENCH>-<SPLIT>.txt
  <work>/trackers/mot_challenge/<BENCH>-<SPLIT>/<tracker_name>/data/<seq>.txt
  ```
  - GT `gt.txt`: write the canonical GT rows in MOTChallenge format, frames AS-IS (already 1-based).
  - Pred `data/<seq>.txt`: write the pred rows in MOTChallenge format
    (`frame,id,x,y,w,h,conf,-1,-1,-1`) with **frame + 1** (0-based -> 1-based) and `conf` = score.
    Make the frame-base conversion explicit and centralized; this is the #1 silent eval bug.
  - `seqinfo.ini`: `[Sequence]` with `name`, `seqLength` = max frame across GT and the (converted,
    1-based) pred, plus `imWidth`/`imHeight` (defaults; note in a comment they don't affect IoU-based
    HOTA). Optionally `frameRate`.
  - seqmap: a file whose first line is `name` then the single `<seq>` line.
- `parse_trackeval_output(tracker_dir, class_name="pedestrian") -> dict`: parse TrackEval's text
  output for the tracker and return at least `{HOTA, DetA, AssA, MOTA, IDF1}` (floats). Target the
  per-tracker summary file (`<class>_summary.txt`: a whitespace-delimited header line of metric names
  followed by a values line) and/or the `<class>_detailed.csv` COMBINED row. Be tolerant of extra
  columns; raise a clear error naming any required metric that is absent. (Task E2 confirms the exact
  filenames/column names against a real run and adjusts if needed.)

### `eval/evaluate.py` — CLI (wires everything except the TrackEval call)
- Args: `--pred` (path to `refined.txt`) OR `--video` [+ `--artifacts-dir`] to auto-locate it via
  `pipeline.artifacts.get_artifact_paths(...)` (import the pipeline helper; insert repo root on
  `sys.path` if needed); `--gt` (path to GT, required); `--seq-name` (default from the video/pred
  stem); `--gt-format` (default `motchallenge`); `--work-dir` (default a temp dir); `--tracker-name`
  (default `refined`); optional `--img-width/--img-height`.
- Flow: `load_gt` -> `load_pred` -> `materialize_layout`. Then the actual TrackEval run is a single
  clearly-marked seam (a function `run_trackeval(...)` in `trackeval_runner.py` that for now raises
  `NotImplementedError("wired in Task E2")`), so the CLI is coherent and Task E2 only fills that hole
  + the reporting.

### Tests (CPU-only, runnable here — no trackeval/torch)
Add `eval/tests/` (plain-python runner like the pipeline tests):
- pred loader parses sample `refined.txt` rows; GT adapter parses sample MOTChallenge `gt.txt`.
- `materialize_layout` creates exactly the expected files/paths; GT `gt.txt` content matches input;
  the pred data file has **frame+1** and MOTChallenge columns; `seqinfo.ini` `seqLength` equals the
  max 1-based frame across GT and converted pred; seqmap content correct.
- `parse_trackeval_output` extracts HOTA/DetA/AssA/MOTA/IDF1 from an embedded SAMPLE fixture
  (representative of TrackEval's `*_summary.txt`), and raises a clear error when a required metric is
  missing.

## Non-goals / Later
- Do NOT import or run `trackeval` (Task E2). Do NOT compute mAP. No multi-sequence/batch. No
  baseline-vs-refined. Do NOT modify the pipeline, Deep-EIoU, or gta-link. Do NOT wire eval into the
  orchestrator. Do NOT implement the user's custom GT converter (only the seam).

## Constraints / Caveats
- GT is assumed MOTChallenge `gt.txt` (1-based frames); pred `refined.txt` is 0-based — convert pred
  to 1-based on write, GT stays as-is.
- Keep everything import-light (stdlib + numpy); must run on this Windows/no-GPU host.
- Windows path-safe.

## Acceptance criteria
- `eval/` package with the loaders, `materialize_layout`, `parse_trackeval_output`, and an
  `evaluate.py` CLI that loads + materializes; the TrackEval run is a single NotImplementedError seam.
- Pred frames are converted 0->1 exactly once, centrally; GT written as-is.
- CPU tests pass here without trackeval/torch.
- Report what ran here vs deferred to Task E2 / Colab.

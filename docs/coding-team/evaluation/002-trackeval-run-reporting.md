# Task E2 — Wire the TrackEval run + reporting + eval README

## Context
Task E1 built `eval/` (GT/pred loaders, `materialize_layout`, `parse_trackeval_output`) with the
actual run left as a seam: `eval/trackeval_runner.run_trackeval(...)` currently raises
`NotImplementedError("wired in Task E2")`. This task fills that seam, adds reporting (a metrics table
+ `metrics.json`), and writes `eval/README.md`.

This host has NO `trackeval`/`scipy`, so the real run cannot execute here — but TrackEval is CPU-only
(no GPU needed), so the user runs it in any env with the deps (Colab or local once installed). Make
the run a single thin seam and keep all extraction/reporting CPU-testable here with stubs.

## Objective
Implement `run_trackeval` using TrackEval as a LIBRARY over the layout E1 already materializes,
extract the headline metrics (HOTA, DetA, AssA, MOTA, IDF1), print a table, write `metrics.json`,
and document usage.

## Scope (do now)

### `eval/trackeval_runner.py` — implement `run_trackeval`
- Import `trackeval` (lazily, inside the function). Support an optional `trackeval_path` arg: if
  given, insert it on `sys.path` before importing (TrackEval is normally a git clone, not pip).
- Configure `trackeval.datasets.MotChallenge2DBox` to read the layout E1 wrote:
  point `GT_FOLDER`/`TRACKERS_FOLDER` at our `<work>/gt/mot_challenge` and
  `<work>/trackers/mot_challenge`; set `BENCHMARK`/`SPLIT_TO_EVAL` to the same `SPORTS`/`eval` used by
  `materialize_layout`; `TRACKERS_TO_EVAL=[tracker_name]`; `CLASSES_TO_EVAL=['pedestrian']`;
  point the seqmap at the file we wrote. **Set `do_preproc=False`** (sports setting — no MOT17
  pedestrian distractor removal). Disable TrackEval's own plotting; let it write its output files.
- Metrics: `[trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]`.
- Run `trackeval.Evaluator(eval_config).evaluate(dataset_list, metrics_list)` and RETURN the raw
  result structure (the nested dict). Keep this function thin — it's the un-testable seam.

### `eval/trackeval_runner.py` — `extract_metrics(result, *, tracker_name, class_name="pedestrian")`
Pure, CPU-testable. Pull the COMBINED-over-sequences entry
(`res[...][tracker]['COMBINED_SEQ'][class]`) and return floats `{HOTA, DetA, AssA, MOTA, IDF1}`:
- **GOTCHA**: TrackEval's HOTA/DetA/AssA are arrays over alpha thresholds — the headline scalar is
  the MEAN over alphas (`float(np.mean(arr))`). MOTA (from CLEAR) and IDF1 (from Identity) are
  scalars. Handle both. Make the result-key path the single point a Colab run can correct if a key
  name differs, and raise a clear error if an expected key is absent.
- Keep E1's file-based `parse_trackeval_output` as a documented FALLBACK for users who run
  TrackEval's `scripts/run_mot_challenge.py` manually instead of via the library.

### `eval/evaluate.py` — reporting
- Wire: `materialize_layout` -> `run_trackeval` (INJECTABLE, like the orchestrator's runner, so a
  CPU test can pass a stub returning a fake result) -> `extract_metrics` -> print a clean table and
  write `metrics.json`.
- `metrics.json` (via `--out`, default = the pred file's directory): `{seq_name, tracker_name,
  metrics:{HOTA,DetA,AssA,MOTA,IDF1}}`. Print the same as an aligned table to stdout.
- Add `--trackeval-path` (optional) forwarded into `run_trackeval`.
- Eval is standalone: do NOT add pipeline profiling here.

### `eval/README.md`
- What it does + that it's separate from the pipeline (reads `refined.txt` post-hoc; CPU-only).
- Install/clone TrackEval (+ scipy) and pass `--trackeval-path` (or pip).
- Run: both `--pred refined.txt --gt gt.txt` and `--video … --artifacts-dir …` forms.
- GT expectation: MOTChallenge `gt.txt`, 1-based; pred is auto-converted 0->1. Note the GT-adapter
  seam for adding the user's custom export later.
- Sports config note (`do_preproc=False`).
- Where `metrics.json` lands; how to read HOTA/DetA/AssA/MOTA/IDF1, and the practical tip: for judging
  GtaLink's refinement, watch IDF1 + AssA (association), not MOTA.

### Tests (CPU-only, runnable here — no trackeval/scipy)
- `extract_metrics` on a FAKE result dict: HOTA/DetA/AssA returned as the mean of their alpha arrays;
  MOTA/IDF1 as scalars; clear error when an expected key is missing.
- `metrics.json` writer + table formatter produce the documented shape.
- `evaluate` end-to-end with an INJECTED stub `run_trackeval` (returns a fake result): asserts
  `metrics.json` is written with the 5 metrics and the table prints — no trackeval import triggered.

## Non-goals / Later
- No mAP, no multi-seq/batch, no baseline-vs-refined, no custom-GT converter (seam only). Do not
  modify the pipeline/Deep-EIoU/gta-link or wire eval into the orchestrator.

## Constraints / Caveats
- `trackeval` imported lazily ONLY inside `run_trackeval`; the rest of `eval/` stays import-light so
  it runs on this host.
- Exact TrackEval result-dict key names + `MotChallenge2DBox` config field names must be confirmed in
  the real Colab run; keep them to one clearly-marked location each so that's a 1-line fix.
- Reuse the same `BENCHMARK`/`SPLIT`/seqmap constants `materialize_layout` uses (don't hardcode a
  second, divergent copy).

## Acceptance criteria
- `run_trackeval` implemented as a thin library call over the E1 layout with `do_preproc=False`;
  `extract_metrics` returns the 5 headline metrics with correct array-mean handling.
- `evaluate.py` prints a metrics table and writes `metrics.json`; `--trackeval-path` supported.
- `eval/README.md` covers run/verify/interpret + the IDF1/AssA tip.
- CPU tests for extraction + reporting + injected-stub end-to-end pass here without trackeval/scipy.
- Report what ran here vs the real TrackEval run deferred to a deps-installed env.

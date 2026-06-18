# Task E3 — Fix GT class/visibility dropped by the TrackEval layout writer

## Context
Real GT samples (`test-data/gt_mot_*.txt`) are MOTChallenge-style, 1-based, with
`conf=1` but `class=-1` and `visibility=-1` on every row. The GT loader already coerces
`-1 -> 1` for conf/class/visibility (`eval/gt_adapters.py` `_parse_mot_rows` /
`_field_or_default`), so the in-memory canonical GT has `class=1`.

BUT `eval/trackeval_runner.py::_format_mot_line` hardcodes the trailing columns as
`...,conf,-1,-1,-1`, so the `gt.txt` written for TrackEval has `class=-1`. TrackEval's
MotChallenge2DBox pedestrian eval keeps only GT rows with class id 1, so it would find ZERO
valid GT and report ~0 for every metric. This bug only manifests at TrackEval runtime (not
covered by the CPU tests).

## Objective
Make the materialized GT `gt.txt` carry the real `class` and `visibility` from the canonical
rows (i.e. `class=1` for the sample data), so TrackEval's pedestrian eval sees the GT. No new
GT-adapter/format, no changes to the data files.

## Scope (do now)
- Fix `_format_mot_line` (or split GT vs tracker formatting) in `eval/trackeval_runner.py` so the
  written line uses the row's `class` (canonical col index 7) and `visibility` (index 8) instead of
  hardcoded `-1`. Result columns for GT: `frame,id,x,y,w,h,conf,class,visibility` (the order
  TrackEval's MotChallenge2DBox reads: class at index 7, visibility at index 8).
  - The prediction/tracker file must remain valid for TrackEval's tracker reader (it reads
    `frame,id,bbox,conf` at indices 0-6 and ignores class/visibility). Writing the canonical
    class/vis for the pred too is harmless; just don't break the first 7 columns. Keep `conf` =
    the pred score.
- Fix the now-inaccurate `_format_mot_line` docstring (the E1 nit: it said "(>=10)" and described
  `-1,-1,-1`).
- Frame/id stay integer-formatted; box + conf keep their float precision.

## Tests (CPU-only, runnable here)
- Add/extend `eval/tests/test_eval.py`: build a canonical GT whose source rows have `class=-1`
  (like the real data), run it through `load_gt`(motchallenge) semantics + `materialize_layout`,
  read back the written `gt.txt`, and assert **every GT row's class column == 1** (and visibility ==
  1). Also assert the pred/tracker file's first 7 columns are unchanged/valid (frame+1, conf=score).
- Keep all existing eval + pipeline tests green.

## Non-goals / Later
- No new GT format/adapter; no edits to `test-data/`. No changes to pipeline/Deep-EIoU/gta-link.
- Don't touch the 0->1 pred frame conversion (that's correct and centralized).

## Constraints / Caveats
- Stay import-light (no trackeval/scipy/torch); must run on this host.
- Do not regress the pred line's TrackEval-tracker compatibility (indices 0-6).
- This is the GT that drives metric correctness — the column order (class@7, vis@8) must match what
  TrackEval's MotChallenge2DBox expects.

## Acceptance criteria
- Materialized `gt.txt` carries class=1 / vis=1 for the sample data (verified by reading the file
  back in a CPU test); the bug that wrote class=-1 is gone.
- Their `test-data/gt_mot_*.txt` then work via the default `--gt-format motchallenge` (no converter,
  no data edits).
- All CPU tests pass here.

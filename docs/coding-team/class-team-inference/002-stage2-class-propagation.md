# Task 002 — Stage 2 class propagation

## Context
Stage 1 (Task 001, done) now stores per-frame `class_ids` on each `Tracklet`, aligned 1:1 with
`times`/`scores`/`bboxes`/`features`, and persists `01_track/tracklets.pkl`.

Stage 2 refines tracklets (split ID-switches, then connect/merge) and writes
`02_refine/refined.txt`. The pipeline actually runs `gta-link/stage2_refine.py`:
`run_refine` → `split_tracklets` (reused from `refine_tracklets.py`) → `_fast_connect`
(local batched merge) → `save_results` (reused from `refine_tracklets.py`).

Stage 3 (next task) will read the refined tracklets and needs, per track: complete per-frame
`features`, `scores`, and `class_ids`, all aligned. Today they are NOT:
- `split_tracklets` rebuilds sub-tracklets without `class_ids` (class lost on split).
- `_fast_connect` merge concatenates only `times`/`bboxes`; `features`/`scores`/`class_ids`
  are left stale/incomplete on merged tracks.
- `save_results` hardcodes MOT col 8 to `-1`.
- There is no `refined_tracklets.pkl` output.

## Objective
Preserve per-frame `class_ids` through split and merge, keep all per-frame parallel arrays
aligned on merged tracks, write per-frame class to `refined.txt` col 8, and export
`02_refine/refined_tracklets.pkl` with IDs matching `refined.txt`.

## Scope
- `gta-link/refine_tracklets.py`
  - `split_tracklets`: when emitting a sub-tracklet, mask `class_ids` by `clusters == label`
    (same mask used for embs/frames/bboxes/scores) and pass it to the `Tracklet` constructor.
  - `merge_tracklets` (reference per-pair impl): on merge add `track1.scores += track2.scores`
    and `track1.class_ids += track2.class_ids` (it already concatenates `features`). Keep parity
    with `_fast_connect`.
  - `save_results`: write per-frame class into MOT col 8 →
    `frame,id,x,y,w,h,score,class_id,-1,-1`. Use the track's `class_ids[instance_idx]`
    (fallback `-1` if absent/short). Do NOT change the existing id-renumbering or cols 1-7.
- `gta-link/stage2_refine.py`
  - `_fast_connect`: on each accepted merge, also concatenate
    `track1.features += track2.features`, `track1.scores += track2.scores`,
    `track1.class_ids += track2.class_ids` so the merged track's parallel arrays stay aligned
    with `times`/`bboxes`. (This re-adds feature concatenation the optimization skipped — the
    merge SEQUENCE/distances are computed from `means` and are unaffected, so output is
    unchanged; only the per-track stored arrays become complete for Stage 3.)
  - Export `02_refine/refined_tracklets.pkl` as `{new_id: Tracklet}` using the **same**
    renumbering `save_results` applies (`new_id = i+1` over `sorted(out.keys())`); also set each
    Tracklet's `.track_id` to its `new_id` so dict key and attribute agree and both match
    `refined.txt`.
- `pipeline/artifacts.py`: add a `refined_tracklets_pkl` path at
  `<artifacts>/<stem>/02_refine/refined_tracklets.pkl` (mirror the existing `tracklets_pkl`/
  `refined_txt` conventions).
- Caching (`stage2_refine.main`): treat the new pkl as a required output too — re-run when the
  pkl is missing even if `refined.txt` is up to date (so Stage 3 never sees a missing pkl).

## Non-goals / Later
- No team assignment / clustering (Task 003). No rendering/eval changes.
- Do not change split/merge association logic, the merge sequence, or output boxes/ids.
- Do not alter MOT cols 1-7.

## Constraints / Caveats
- Invariant for the exported pkl: for every track, `len(times) == len(bboxes) ==
  len(features) == len(scores) == len(class_ids)`. Add a cheap assertion or test for this.
- The exported pkl ids MUST equal `refined.txt` ids (same sort + `i+1`).
- Keep the CPU-testability of `stage2_refine.py` intact (it's tested with stubbed `RefineDeps`;
  don't introduce heavy top-level imports).
- `class_ids` may be absent on genuinely old pkls — Task 001 added `Tracklet.__setstate__`
  backfill, so unpickled inputs are safe; still guard the `save_results` index defensively.

## Acceptance criteria
- After a split+merge, every track in `refined_tracklets.pkl` has all five parallel arrays equal
  in length.
- `refined.txt` carries per-frame canonical class in col 8; ids and cols 1-7 unchanged.
- `refined_tracklets.pkl` keys == the set of ids in `refined.txt`.

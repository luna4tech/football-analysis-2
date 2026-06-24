# Task 001 — Stage 1 class preservation

## Context
YOLOv11 emits 7-col detections `[x1,y1,x2,y2,score,class_conf,class_id]`
(`Deep-EIoU/Deep-EIoU/tools/stage1_assembly.py` `ultralytics_result_to_yolox_output`).
The tracker `Deep_EIoU.update()` reads only cols 0-4 and drops `class_id`. We need the class
to survive from detection into the per-frame tracklet data and into `01_track/tracks.txt`.

Detailed behavior spec: `docs/class-team-tracking-spec.md`.

## Canonical class IDs (project-wide convention)
`0 = player, 1 = goalkeeper, 2 = referee`. Unknown/legacy = `-1`.
These three packages run as separate subprocesses with no shared imports, so this is a
**documented convention**, not a shared module — use local literals/constants per package.

## Objective
Carry the detector class, mapped to canonical IDs, from detection → `STrack` → `Tracklet`
(per frame) → `tracks.txt` col 8.

## Scope
- `tools/stage1_assembly.py` `ultralytics_result_to_yolox_output`: resolve each detection's
  raw class via the Ultralytics result's `names` dict (id→name string) and remap to the
  canonical ID by **name** (`player`/`goalkeeper`/`referee`, case-insensitive). Put the
  canonical ID in the output array's class column. If a name doesn't match, use `-1`.
- `tracker/Deep_EIoU.py`: add a `class_id` field to `STrack` (default `-1`). For detection
  STracks, populate it from the detection's class column. On track update/re-activation, copy
  the matched detection's `class_id` onto the track so `online_targets` expose the latest
  matched class.
- `tools/stage1_track.py` `track_consume`: pass the track's `class_id` into `assembler.add(...)`.
- `tools/stage1_assembly.py` `TrackletAssembler.add` + `format_mot_line`: accept the per-frame
  class; append it to a per-frame `class_ids` list on the `Tracklet`; write `tracks.txt` rows as
  `frame,id,x,y,w,h,score,class_id,-1,-1`.
- `gta-link/Tracklet.py`: add a `class_ids` list, aligned 1:1 with `times`/`scores`/`bboxes`/
  `features`; ensure the constructor and the per-frame append path keep it aligned.

## Non-goals / Later
- No Stage 2 split/merge changes (Task 002). No team logic. No rendering/eval changes.
- Do not change association/matching logic — only attach and carry the class.
- Do not touch the first 7 MOT columns.

## Constraints / Caveats
- The 7-col detection path multiplies `score*class_conf`; leave that untouched — only the
  class column's *value* changes (raw → canonical).
- Keep legacy behavior safe: when class is unknown, everything must still work with `-1`.
- `Tracklet` is shared on disk (pickle) with Stage 2; the new `class_ids` field must be
  constructed even when not provided (default empty/`-1`-filled) so older code paths don't break.

## Acceptance criteria
- A track's per-frame `class_ids` length matches its `times`/`bboxes` length.
- `tracks.txt` carries the canonical class in column 8; cols 1-7 unchanged.

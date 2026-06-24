# Plan: Class-Aware Tracking, Team Assignment, Rendering & Eval

Detailed spec: `docs/class-team-tracking-spec.md` (source of truth for behavior).

## Goal
Preserve detector class through tracking, assign players to teams via ReID clustering,
render class/team via box color + labels, and evaluate class/team correctness against
extended MOT ground truth.

## Locked decisions
- **Class mapping**: resolve by name from the detector's `names` dict at the detection
  boundary → canonical `{player:0, goalkeeper:1, referee:2}`. Persist canonical IDs only.
- **Team assignment**: post-Stage-2, players-only, k=2 cosine K-means on each track's mean
  ReID embedding. GK/referee team = `-1`. Deterministic naming (larger cluster = team 0).
- **GK→team**: unknown by design (not assigned).
- **Output**: per-frame `class_id` in MOT col 8; per-frame `team_id` in col 9 (Stage 3 only);
  per-track attributes in `03_team/track_attributes.json`. First 7 MOT columns unchanged.
- **GT layout** (`test-data/gt_mot_*.txt`): `frame,id,x,y,w,h,conf,class_id,team_id,-1`
  (col 8 class_id 0/1/2, col 9 team_id 0/1 for players else -1).
- **Eval**: keep TrackEval HOTA/MOTA all-person (materialized `gt.txt` forces pedestrian
  class=1); add separate attribute metrics; degrade gracefully when GT lacks class/team.

## Constraints
- No team logic inside `Deep-EIoU/` or `gta-link/` association code.
- Keep MOT cols 1-7 intact (TrackEval compatibility). Legacy txt/GT must still work.
- Stage 3 is post-processing only; it must not affect tracking or refinement.

## Tasks
1. **001 — Stage 1 class preservation**: name-based class resolution → canonical IDs; carry
   latest matched class on `STrack`; per-frame `class_ids` on `Tracklet` (aligned with
   times/scores/bboxes/features); write `01_track/tracks.txt` with class in col 8.
2. **002 — Stage 2 class propagation**: preserve `class_ids` through `append_det`, split,
   merge, extract; export `02_refine/refined_tracklets.pkl` (IDs matching `refined.txt`);
   write `02_refine/refined.txt` with class in col 8, team left `-1`.
3. **003 — Stage 3 team assignment**: `pipeline/team_assignment.py` + orchestrator wiring +
   CLI; players-only k=2 cosine K-means on mean embedding; deterministic naming; emit
   `03_team/refined.txt` (class col 8, team col 9) and `03_team/track_attributes.json`;
   `team_id=-1` for GK/referee and when fewer than two player tracks.
4. **004 — Rendering**: category→color lookup (team0/team1/GK/referee/fallback) replacing
   `get_color(track_id)`; labels = track ID only with `gk:` prefix; consume `03_team/refined.txt`
   when present; legacy txt still renders.
5. **005 — Eval attribute metrics**: parse semantic class/team from GT cols 8/9; reuse
   TrackEval matching (IoU fallback) to transfer GT attributes onto predictions; class
   accuracy + confusion matrix; team ARI/NMI + best-permutation accuracy (players only);
   within-track consistency; write `eval/attributes_metrics.json`; graceful degradation.

## Non-goals
- Color/jersey clustering (Approach A), CLIP/SigLIP embeddings.
- GK→team assignment; home/away semantics; multi-camera; halftime kit changes.
- Changing tracking/refinement association behavior.

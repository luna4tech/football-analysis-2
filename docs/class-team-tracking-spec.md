# Class-Aware Tracking, Team Assignment, Rendering & Eval — Implementation Spec

## Scope
Carry detector class through the pipeline, assign players to teams via ReID clustering,
render class/team via box color and labels, and evaluate class + team correctness against
existing MOT ground truth.

## Detector classes
The YOLOv11 detector emits three classes: `player`, `goalkeeper`, `referee`. Use the numeric
class indices exactly as produced by the trained model — read them from the model, do not
re-map. Treat any non-detected/legacy class as `-1` (unknown).

---

## 1. Class-aware output

### Carry class through tracking
- `Deep-EIoU/Deep-EIoU/tracker/Deep_EIoU.py`: `update()` already receives the 7-column array
  `[x1,y1,x2,y2,score,class_conf,class_id]`. Read `class_id` (col 6) for the matched detection
  and store it on the corresponding `STrack` (latest matched class).
- `Deep-EIoU/Deep-EIoU/tools/stage1_track.py` (`track_consume`): pass the track's class into
  `assembler.add(...)`.
- `Deep-EIoU/Deep-EIoU/tools/stage1_assembly.py` (`TrackletAssembler.add`): store a per-frame
  `class_ids` list on the `Tracklet`, aligned 1:1 with `times` / `scores` / `bboxes` / `features`.

### Tracklet data model
- `gta-link/Tracklet.py`: add an optional per-detection `class_ids` list, kept aligned with the
  existing parallel arrays. Preserve it through `append_det`, split, and merge so each resulting
  tracklet keeps its own per-frame class history (no post-hoc bbox/IoU re-matching).

### Per-track class aggregation
- Final class per track = **confidence-weighted mode** of its per-frame `class_ids`.

### Output schema
- `01_track/tracks.txt` and `02_refine/refined.txt`: write **per-frame `class_id` in MOT column 8**:
  `frame,id,x,y,w,h,score,class_id,-1,-1`. Keep the first 7 columns unchanged so TrackEval still parses.
- Stage 2 wrapper: export final refined tracklets to `02_refine/refined_tracklets.pkl` with IDs
  normalized to match `refined.txt`.

---

## 2. Team assignment (ReID, post-Stage-2)

Run as a new step after Stage 2 refinement (e.g. `pipeline/team_assignment.py`, invoked by the
orchestrator after refine). Consume `02_refine/refined_tracklets.pkl`.

- Select tracks whose aggregated class is `player`. **Exclude goalkeepers and referees from clustering.**
- Per selected track, compute one feature = **temporal mean of its 512-D ReID embeddings**.
- Cluster with **k=2 cosine K-means** → `team_id ∈ {0, 1}`.
- **Goalkeeper team = unknown (`-1`)** — do not assign GKs to a team.
- **Referee team = unknown (`-1`)**.
- **Deterministic team naming:** assign `team_id = 0` to the larger cluster (tie-broken by lowest
  min track_id) so team labels are stable across runs.
- If fewer than two valid player tracks exist, leave all `team_id = -1`.

### Output
- `03_team/refined.txt`: `frame,id,x,y,w,h,score,class_id,team_id,-1`.
- `03_team/track_attributes.json`: `{ track_id: { "class": <id>, "team": <id or -1>, "gk": <bool> } }`
  (per-track aggregated class + team; `gk` true when class == goalkeeper).

Downstream render/eval consume `03_team/refined.txt` when present.

---

## 3. Rendering

Update `Deep-EIoU/Deep-EIoU/tools/render_from_txt.py` (parse `class_id`/`team_id`) and the draw
path in `Deep-EIoU/Deep-EIoU/yolox/utils/visualize.py` (`plot_tracking`). Replace the
`get_color(track_id)` color with a **category → color** lookup.

| Category | Box color | Label |
|---|---|---|
| player, team 0 | C1 | `<id>` |
| player, team 1 | C2 | `<id>` |
| goalkeeper | C_gk (single distinct color) | `gk:<id>` |
| referee | C_ref (distinct) | `<id>` |
| unknown / fallback | default | `<id>` |

- Labels show **track ID only**; goalkeepers get a `gk:` prefix.
- Legacy txt files (no class/team columns) must still render via the fallback category.

---

## 4. Evaluation

Extend the existing MOT GT and TrackEval flow; keep HOTA/DetA/AssA/MOTA/IDF1 unchanged.

### Ground truth
- Add per-frame **class** to GT column 8 and a per-track **team** mapping.
- Keep the materialized TrackEval GT class field as pedestrian class `1` for all rows so existing
  tracking metrics stay all-person. Use the semantic class separately (below).

### Metrics (write to `eval/attributes_metrics.json`)
- Link predicted tracks to GT tracks by **reusing TrackEval's predicted↔GT identity matching**
  (already computed for HOTA/IDF1) to transfer GT class/team onto predicted tracks. Use IoU
  matching where IDs are not guaranteed to align.
- **Class:** per-track accuracy + confusion matrix over the three classes (watch goalkeeper↔player
  and referee↔player).
- **Team (players only):** ARI / NMI + best-permutation accuracy (handles arbitrary cluster labels);
  exclude referees and goalkeepers.
- **Consistency:** within-track team/class switch rate; cross-fragment agreement after merges.

### Guardrail (no GT)
- Per-video silhouette score / intra-vs-inter cluster cosine distance, as a regression signal.

---

## Tests
- Stage 1: class propagation onto `STrack` and extended MOT rows.
- Stage 2: class metadata preserved through split / merge / extract; `refined_tracklets.pkl` IDs
  match `refined.txt`.
- Team: player-only k=2 clustering; deterministic team naming; `team_id=-1` for GK/referee and when
  fewer than two player tracks.
- Render: legacy and extended txt; track-ID-only labels; `gk:` prefix; category color selection.
- Eval: class metrics + confusion matrix; team ARI/NMI/accuracy; JSON output shape.

## Assumptions
- Exactly two teams; single video feed; single half / single kit per clip.
- Team IDs are arbitrary per video (no home/away semantics).
- Team assignment is post-processing only; it does not affect tracking or refinement.
- One shared goalkeeper color; referees use no label prefix.

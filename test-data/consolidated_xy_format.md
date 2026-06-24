# `consolidated_xy/` — Data Organization Reference

Reference for the consolidated tracking export under `test-data/consolidated_xy/`, and how
to turn it into MOT ground-truth text files (`gt_mot_<name>.txt`).

> All facts below were derived empirically from the data in this folder (chunk files +
> `track_index.json`), cross-checked for consistency. Where a value is inferred, the
> supporting evidence is given.

---

## 1. Purpose

`consolidated_xy/` is the **final, human-corrected tracking output for one full match**
(both halves, `H1` + `H2`, per `consolidation_manifest.json`). It contains, for every
tracked object (players, goalkeeper, referees, ball) in every frame:

- the image-space bounding box,
- the detector **class** (ball / goalkeeper / player / referee),
- the **team** assignment,
- a stable consolidated **track id**, and the assigned **player (jersey) id**,
- a normalized pitch coordinate (`xy`).

The two `gt_mot_video*.txt` files in `test-data/` are **format samples only** — they cover
a *different* clip than this export, so their rows do not correspond 1:1 to these chunks.

---

## 2. Directory layout

```
consolidated_xy/
├── chunks/                     # per-second detection data — the main payload
│   ├── 04508.json              # 3,493 files, one per second
│   ├── 04509.json
│   └── ... 08000.json
├── track_index.json            # canonical per-track metadata (class, team, player, active intervals)
├── track_map.json              # consolidation merge map (consolidated id -> original ids) + audit
├── player_map.json             # track id -> jersey/player number assignments + audit
├── ball_fix_state.json         # manual ball-trajectory correction segments (global integer frames)
├── consolidation_manifest.json # provenance: match id, half/player/ball editor uids, checksums
└── history_log.jsonl           # append-only edit history (one JSON object per line)
```

---

## 3. Time / frame model  (important)

- **Frame rate: 25 fps.**
- Chunk files are named by **whole second**: `04508.json` … `08000.json`
  (zero-padded to 5 digits). The range `4508–8000` is **contiguous** (3,493 files,
  `8000 − 4508 + 1 = 3493`) — there is **no halftime gap** in the numbering.
- Inside each chunk, detections are keyed by a `<second>.<frame>` timestamp string:
  `"4508.04"`, `"4508.08"`, … `"4508.96"` — step `0.04` (= 1/25 s). File `N.json`
  holds keys `N.00`‥`N.96` (up to 25 keys).
- **Global integer frame index** for any key:  `global_frame = round(key × 25)`.
  - `4508.04 → 112701`, `4508.96 → 112724`, `4509.00 → 112725`, `8000.96 → 200024`.
  - This matches the integer frames used in `ball_fix_state.json` (e.g. `112700/112701`).
- **Data span:** first key `4508.04` (global 112701) → last key `8000.96` (global 200024),
  i.e. ~87,324 frames ≈ 3,493 s ≈ 58 min.
- **Gaps are normal.** Some sub-frames are absent (e.g. `4508.48` is missing) — that frame
  simply had no recorded detections. A frame with no objects yields no MOT rows.

### Mapping to MOT frame numbers
The sample `gt_mot_video2780-2960.txt` runs frames `1..4500` (= 180 s × 25 fps), i.e. the
clip is **re-indexed from 1 at its start**. We follow the same convention, with the range
treated as **half-open `[start, end)`** (start-inclusive, end-exclusive — like video
trimming, so a range yields exactly `(end − start) × 25` frames):

> **`offset = round(start × 25) − 1`;  `mot_frame = round(key × 25) − offset`**, keeping a
> frame while `round(start × 25) ≤ round(key × 25) < round(end × 25)`.

- `start` / `end` are seconds and may be **fractional** (e.g. `4600.40`). Frame `1` is the
  frame **at** `start` (`--start 4600.40` → frame `1` is the `4600.40` detection).
- By default (no range) the first available frame (`4508.04`) becomes frame `1`, and the
  last data frame is included.
- Gaps are preserved: a missing/empty frame leaves a hole in the numbering and contributes
  no rows.

---

## 4. `chunks/NNNNN.json` — detection records

Top level is a JSON object: `{ "<second.frame>": [ <detection>, ... ], ... }`.

Each detection object:

| field        | type        | meaning |
|--------------|-------------|---------|
| `bbox`       | `[x1,y1,x2,y2]` | **image-pixel corner coords** (top-left `x1,y1`, bottom-right `x2,y2`). Integers for people, floats for the ball. |
| `class_id`   | int `0..3`  | detector class — see §5 |
| `team`       | int `-1/0/1`| team — see §6 |
| `track_id`   | string      | consolidated track id — see §7 |
| `player_id`  | int         | assigned jersey/player number. **Absent for the ball** (class 0). |
| `xy`         | `[x,y]`     | normalized pitch coordinate, roughly `[-1, 1]` on each axis (pitch-space, not image-space). Not used for image-space MOT. |

**Bounding-box conversion to MOT** (`bb_left, bb_top, bb_w, bb_h`):
`bb_left = x1`, `bb_top = y1`, `bb_w = x2 − x1`, `bb_h = y2 − y1`.

---

## 5. `class_id` mapping  ← answer to "what are the class mappings?"

Indices come straight from the **YOLOv11 detector** (the spec says: read indices from the
model, do not re-map). Empirically, across all 120 tracks and a frame-level census, the
mapping is unambiguous:

| `class_id` | label        | # tracks | team(s)        | `player_id`? | ~ per frame | evidence |
|:----------:|--------------|:--------:|----------------|:------------:|:-----------:|----------|
| **0** | **ball**       | 88 | `-1` only            | no  | 0–1 (~0.9) | every class-0 id is `b_*`; 1:1 with the `b_`-prefixed ids |
| **1** | **goalkeeper** | 1  | `0`                  | yes | 0–1 (~0.4) | single track `"20"`, `player_id 1`; off-screen often |
| **2** | **player**     | 28 | `0` (15) / `1` (13)  | yes | 18–21 (~20)| the outfield players of both teams |
| **3** | **referee**    | 3  | `-1` only            | yes | exactly 2  | tracks `"22","23","24"` (ref + assistants) |

This is the standard Roboflow *football-players-detection* ordering
(`0 ball, 1 goalkeeper, 2 player, 3 referee`).

> ⚠️ Note: the unit-test fixture `_FOOTBALL_NAMES = {0:"Referee",1:"Player",2:"GoalKeeper",3:"ball"}`
> in `pipeline/tests/test_stage1_assembly.py` is an **arbitrary test stub**, *not* the real
> model mapping. Trust the data / the model, per the spec.

Per-detection `class_id` and `team` were verified to match the canonical
`track_index.json` values **exactly** (0 mismatches over 29,398 detections), so either
source is equivalent — use the per-detection values.

---

## 6. `team` mapping

| `team` | meaning |
|:------:|---------|
| `0`  | team A (outfield + the goalkeeper) |
| `1`  | team B (outfield) |
| `-1` | unknown / not team-assigned — used for **ball** and **referees**; goalkeepers are *not* `-1` here (the lone GK is team `0`) |

Per the spec, team is assigned to **players only** via ReID clustering; goalkeepers and
referees are normally left `-1`. In this export the single GK happens to carry team `0`.

---

## 7. `track_id` encoding

- Numeric strings for people: `"20"`, `"21"`, … and large "consolidated" ids like
  `"1000176"`, `"2000299"`, `"3000182"`. The large ids encode merge/half provenance
  (`1xxxxxx`, `2xxxxxx`, `3xxxxxx` families) and appear verbatim in the sample GT files,
  confirming **MOT column 2 (id) = `int(track_id)`**.
- Ball ids are **`b_<n>`** (e.g. `"b_15"`) — string, *not* MOT-integer-friendly. The ball
  is excluded from the MOT output (matches the sample GT, which contains no ball rows), so
  this needs no integer mapping.
- `track_map.json["map"]` maps each consolidated id → the list of raw tracker ids it
  absorbed (or `["__deleted__"]`). `player_map.json["assignments"]` maps consolidated id →
  jersey number.

### `track_index.json`
`{"tracks": { "<track_id>": { "classId", "team", "playerId", "isPlayerAssigned",
"intervals": [ {"begin","end"}, … ] } }}` — `intervals` are the active `<second.frame>`
spans for that track (when it is present on screen).

---

## 8. Support files (provenance / editing state — not needed for MOT)

| file | content |
|------|---------|
| `consolidation_manifest.json` | `match_id`, consolidation version/time, per-half editor uids, input/output checksums |
| `track_map.json`   | `map` (consolidated→raw ids), `audit` (e.g. `swap_teams` actions), `splitChildren` |
| `player_map.json`  | `assignments` (track→jersey), `audit` (assignment history) |
| `ball_fix_state.json` | ball-trajectory `segments` with control points, in **global integer frames**; manual ball fix-up state |
| `history_log.jsonl`| append-only edit log, one JSON object per line |

---

## 9. Target MOT output format

Standard 10-column MOT (comma-separated, one detection per line), with the **final three
fields repurposed** from `-1,-1,-1` to `class_id,team_id,-1`:

```
frame, id, bb_left, bb_top, bb_w, bb_h, conf, class_id, team_id, -1
```

| col | source |
|:---:|--------|
| 1 `frame`   | re-indexed-from-1 MOT frame number (§3) |
| 2 `id`      | `int(track_id)` |
| 3 `bb_left` | `bbox[0]` |
| 4 `bb_top`  | `bbox[1]` |
| 5 `bb_w`    | `bbox[2] − bbox[0]` |
| 6 `bb_h`    | `bbox[3] − bbox[1]` |
| 7 `conf`    | `1` (constant, as in the samples) |
| 8 `class_id`| detection `class_id` (0–3), written **as-is** |
| 9 `team_id` | detection `team` (`-1/0/1`), written **as-is** |
| 10          | `-1` (constant) |

Example sample line (original): `1,21,427.0,529.0,17.0,51.0,1,-1,-1,-1`
→ new form: `1,21,427.0,529.0,17.0,51.0,1,2,1,-1` (class=player, team=1).

### Generation rules (agreed)
- **Range input is in seconds** and accepts **fractional** values (e.g. `4600.40`);
  omit for **all frames** (default). Bounds are half-open `[start, end)`.
- **Frame numbers re-index from 1** at the start of the range (frame 1 = the frame at `start`).
- **Ball excluded** (`class_id == 0` / `b_*` ids dropped) — matches the sample GT.
- **`class_id` / `team_id` written raw** (0–3 and -1/0/1), `team_id = -1` kept for referees.
- Bbox values formatted like the samples (floats, e.g. `427.0`).

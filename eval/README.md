# eval — Standalone Tracking Evaluation (HOTA / MOTA / IDF1)

Scores the pipeline's `refined.txt` against ground truth using
[TrackEval](https://github.com/JonathonLuiten/TrackEval) (HOTA + CLEAR +
Identity metrics). **Completely separate from the pipeline**: reads the
`refined.txt` artifact post-hoc; no video, GPU, or pipeline runtime required.

---

## What it does

1. Loads your ground-truth `gt.txt` (MOTChallenge format, 1-based frames).
2. Loads the pipeline's `refined.txt` (0-based frames — auto-converted to
   1-based exactly once, centrally, when writing the TrackEval layout).
3. Materializes the standard MOTChallenge on-disk layout TrackEval expects.
4. Runs TrackEval (`HOTA`, `CLEAR`, `Identity` metrics; sports config:
   `do_preproc=False` — no MOT17 pedestrian distractor removal).
5. Extracts the 5 headline scalars, prints an aligned table, and writes
   `metrics.json`.

---

## Prerequisites

### Install TrackEval

TrackEval is normally used as a git clone rather than a pip package:

```bash
git clone https://github.com/JonathonLuiten/TrackEval.git
pip install scipy   # TrackEval's only heavy dep
```

Pass the path to the clone via `--trackeval-path`:

```bash
python -m eval.evaluate --pred refined.txt --gt gt.txt \
    --trackeval-path /path/to/TrackEval
```

Or, if you `pip install trackeval` (unofficial), the `--trackeval-path` flag
can be omitted.

### Ground-truth format

The GT file must be a standard MOTChallenge `gt.txt`:
```
frame,id,x,y,w,h,conf,class,visibility
```
with **1-based frames**. The prediction is auto-converted 0 → 1 before
evaluation (never double-shifted; this is the #1 silent eval bug — we handle
it centrally so you don't have to).

**Custom GT export seam**: if your tool exports GT in a different format,
register a converter via `eval.gt_adapters.register_gt_loader("my_fmt", fn)`.
The converter receives the raw file path and must return a `Tracks` with
`frame_base=1`. See `eval/gt_adapters.py` for the seam documentation.

---

## Running the evaluation

### Using the prediction path directly

```bash
# From the repo root
python -m eval.evaluate \
    --pred artifacts/clip/02_refine/refined.txt \
    --gt   /path/to/gt.txt \
    --seq-name M59 \
    --trackeval-path /path/to/TrackEval
```

### Auto-locating refined.txt from the video path

```bash
python -m eval.evaluate \
    --video /path/to/clip.mp4 \
    --artifacts-dir artifacts \
    --gt /path/to/gt.txt \
    --trackeval-path /path/to/TrackEval
```

The sequence name defaults to the video/pred stem if `--seq-name` is omitted.

### Common options

| Flag | Default | Description |
|---|---|---|
| `--pred` / `--video` | — | Prediction source (mutually exclusive, one required) |
| `--gt` | — | Ground-truth file (required) |
| `--seq-name` | video/pred stem | Sequence name used in TrackEval layout |
| `--gt-format` | `motchallenge` | GT adapter format |
| `--trackeval-path` | — | Path to a TrackEval git clone |
| `--work-dir` | temp dir | Where the TrackEval layout is materialized |
| `--tracker-name` | `refined` | Tracker label in the layout |
| `--out` | pred dir | Output path for `metrics.json` |
| `--img-width` / `--img-height` | 1920/1080 | Written to `seqinfo.ini` (does not affect IoU-based metrics) |

### Sports config note

`do_preproc=False` is set unconditionally — this disables MOT17's pedestrian
distractor removal, which is appropriate for sports footage (SportsMOT and
SoccerNet evaluate this way). Do **not** enable `do_preproc` for sports data.

---

## Output

### Console table

```
[eval] Tracking metrics:
Metric         Value
------------  -------
HOTA          72.345
DetA          68.100
AssA          77.900
MOTA          65.500
IDF1          81.250
```

### `metrics.json`

Written alongside `refined.txt` by default (override with `--out`):

```json
{
  "seq_name": "M59",
  "tracker_name": "refined",
  "metrics": {
    "HOTA": 72.345,
    "DetA": 68.100,
    "AssA": 77.900,
    "MOTA": 65.500,
    "IDF1": 81.250
  }
}
```

---

## Interpreting the metrics

| Metric | What it measures | What to watch for GtaLink refinement |
|---|---|---|
| **HOTA** | Balanced detection + association | Overall quality headline |
| **DetA** | Detection accuracy (sub-metric of HOTA) | How well boxes are found |
| **AssA** | Association accuracy (sub-metric of HOTA) | How well identities are tracked |
| **MOTA** | Multi-Object Tracking Accuracy (CLEAR) | Detection-heavy; penalises FP/FN/ID-switches |
| **IDF1** | Identity F1 (ID-based metric) | Long-track coherence |

**Tip**: for judging GtaLink's refinement effect, focus on **IDF1** and
**AssA** (association quality) rather than MOTA. MOTA is detection-heavy and
will not change between `tracks.txt` and `refined.txt` if the set of
detections is unchanged; IDF1/AssA directly reflect whether the tracklet
connecting step produced cleaner identities.

---

## Verifying outputs (Colab)

1. Run the full pipeline first:
   ```bash
   python -m pipeline run --video clip.mp4
   ```

2. Check track counts (expect fewer IDs after refinement):
   ```bash
   python count_tracks.py artifacts/clip/01_track/tracks.txt
   python count_tracks.py artifacts/clip/02_refine/refined.txt
   ```

3. Render both for visual inspection (`--one_indexed` is NOT needed;
   frames are 0-based):
   ```bash
   python render_from_txt.py artifacts/clip/01_track/tracks.txt
   python render_from_txt.py artifacts/clip/02_refine/refined.txt
   ```

4. Evaluate:
   ```bash
   python -m eval.evaluate \
       --pred artifacts/clip/02_refine/refined.txt \
       --gt   /path/to/gt.txt \
       --trackeval-path /path/to/TrackEval
   ```

---

## Package layout

```
eval/
  evaluate.py          # CLI entrypoint + reporting (format_metrics_table,
                       #   write_metrics_json, run_evaluation)
  gt_adapters.py       # GT loader seam (MOTChallenge built-in; custom fmt
                       #   plugs in via register_gt_loader / @gt_loader)
  trackeval_runner.py  # MOTChallenge layout writer (materialize_layout),
                       #   TrackEval library run (run_trackeval),
                       #   result extraction (extract_metrics),
                       #   file-based parser fallback (parse_trackeval_output)
  tests/
    test_eval.py       # CPU-only tests (no trackeval/torch/scipy)
```

### File-based parser fallback

If you run TrackEval's own `scripts/run_mot_challenge.py` manually rather than
via this tool's library call, `parse_trackeval_output(tracker_dir)` reads the
`pedestrian_summary.txt` (or `pedestrian_detailed.csv`) TrackEval writes and
extracts the same 5 metrics. Use this as a fallback when the library API
breaks across TrackEval versions.

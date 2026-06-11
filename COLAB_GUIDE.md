# Running the GTA tracking pipeline on Google Colab

Step-by-step instructions to run the two-stage pipeline (DeepEIoU tracking → GtaLink
refinement) on a Colab GPU, reading input videos from Google Drive and persisting all outputs
back to Drive.

- **Input videos:** `Colab Notebook/football-analysis-project/input-videos/`
- **Outputs:** `Colab Notebook/football-analysis-project/output/gta-track/`

> For the meaning of each stage, flag, and artifact, see [USER_GUIDE.md](USER_GUIDE.md). This
> document is only about *running it on Colab*.

---

## Before you start (one-time, on your machine)

The code lives on the `pipeline-refactor` branch. Push it so Colab can clone it:

```bash
git push -u origin pipeline-refactor
```

You also need two model checkpoints (they are **not** in the repo — they are gitignored):

| File | Used by | Where it must end up |
|------|---------|----------------------|
| `best_ckpt.pth.tar` | YOLOX detector (Stage 1) | `Deep-EIoU/Deep-EIoU/checkpoints/best_ckpt.pth.tar` |
| `sports_model.pth.tar-60` | OSNet ReID (Stage 1) | `Deep-EIoU/Deep-EIoU/checkpoints/sports_model.pth.tar-60` |

Download them from the original Deep-EIoU / GtaLink release pages (see those projects' READMEs),
then upload both into a Drive folder you control, e.g.:

```
Colab Notebook/football-analysis-project/checkpoints/best_ckpt.pth.tar
Colab Notebook/football-analysis-project/checkpoints/sports_model.pth.tar-60
```

> **Folder name note:** Drive's auto-created folder is usually **`Colab Notebooks`** (plural).
> You wrote `Colab Notebook` (singular). Use whichever actually exists in your Drive and keep it
> consistent in the `DRIVE_BASE` variable below.

---

## Colab notebook — paste each block into its own cell

### 0. Use a GPU runtime
`Runtime → Change runtime type → Hardware accelerator → GPU (T4 or better)`. Then verify:

```python
!nvidia-smi
```

### 1. Mount Drive and define paths

```python
from google.colab import drive
drive.mount('/content/drive')

import os
from pathlib import Path

# --- EDIT THIS if your Drive folder is "Colab Notebooks" (plural) ---
DRIVE_BASE = "/content/drive/MyDrive/Colab Notebook/football-analysis-project"

INPUT_DIR   = f"{DRIVE_BASE}/input-videos"
OUTPUT_DIR  = f"{DRIVE_BASE}/output/gta-track"
CKPT_DIR    = f"{DRIVE_BASE}/checkpoints"
REPO        = "/content/football-analysis-2"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pick the clip to process (must exist under INPUT_DIR):
VIDEO = f"{INPUT_DIR}/M59-5min-1.mp4"     # <-- change to your file
STEM  = Path(VIDEO).stem

# Export to the shell environment so the `!` cells below can use them:
os.environ.update(INPUT_DIR=INPUT_DIR, OUTPUT_DIR=OUTPUT_DIR,
                  CKPT_DIR=CKPT_DIR, REPO=REPO, VIDEO=VIDEO, STEM=STEM)

print("Input videos available:")
!ls -la "$INPUT_DIR"
```

### 2. Clone the code

```python
!git clone -b pipeline-refactor https://github.com/luna4tech/football-analysis-2.git "$REPO"
%cd $REPO
!git log --oneline -3
```

> Keeping the repo on Colab's fast local disk (`/content`) and writing only outputs to Drive is
> the recommended setup. (If you'd rather not push to GitHub, copy/unzip the project into
> `/content/football-analysis-2` from Drive instead of cloning.)

### 3. Install dependencies
Colab already has `torch`/`torchvision`. Install the rest of the runtime deps for both stages:

```python
!pip install -q opencv-python loguru lap scikit-learn scipy matplotlib seaborn tqdm thop tabulate Pillow cython pycocotools ninja
```

> The `yolox` and `reid` packages are used **in place** (each stage runs from its own directory),
> so there is nothing to `pip install` or compile for them — these runtime libraries are enough
> for inference.

### 4. Link the checkpoints into the location Stage 1 expects

```python
!mkdir -p "$REPO/Deep-EIoU/Deep-EIoU/checkpoints"
!ln -sf "$CKPT_DIR/best_ckpt.pth.tar"        "$REPO/Deep-EIoU/Deep-EIoU/checkpoints/best_ckpt.pth.tar"
!ln -sf "$CKPT_DIR/sports_model.pth.tar-60"  "$REPO/Deep-EIoU/Deep-EIoU/checkpoints/sports_model.pth.tar-60"
!ls -la "$REPO/Deep-EIoU/Deep-EIoU/checkpoints"
```

### 5. Run the full pipeline → outputs straight to Drive

```python
%cd $REPO
!python -m pipeline run \
    --video "$VIDEO" \
    --artifacts-dir "$OUTPUT_DIR" \
    --device gpu
```

This runs Stage 1 (tracking) then Stage 2 (refine) and writes everything under
`OUTPUT_DIR/<STEM>/` on Drive (see the layout at the bottom). The orchestrator streams each
stage's log live and prints a profiling summary at the end.

**Faster perception (optional):** add the opt-in parallel path (batched detect+ReID + prefetch
decode). Lower `--batch-size` if you hit GPU out-of-memory:

```python
!python -m pipeline run \
    --video "$VIDEO" \
    --artifacts-dir "$OUTPUT_DIR" \
    --device gpu \
    --parallel --batch-size 8
```

> **Cache caveat:** stages skip work when their output is newer than their input. Re-running the
> same video is therefore instant. Changing a *parameter* (e.g. `--eps`) does **not** bust the
> cache — pass `--force-all`, `--force-stage1`, or `--force-stage2` to recompute. Example: retune
> the refine step only:
> ```python
> !python -m pipeline run --video "$VIDEO" --artifacts-dir "$OUTPUT_DIR" \
>     --eps 0.5 --merge_dist_thres 0.3 --force-stage2
> ```

### 6. (Optional) Render an annotated video to Drive
The pipeline outputs MOT `.txt` files, not a video. Rebuild a watchable overlay from the refined
result (CPU-only, no GPU needed). Frames are 0-based, so do **not** pass `--one_indexed`:

```python
%cd $REPO/Deep-EIoU/Deep-EIoU
!python tools/render_from_txt.py \
    --path "$VIDEO" \
    --txt  "$OUTPUT_DIR/$STEM/02_refine/refined.txt" \
    --save_path "$OUTPUT_DIR/$STEM/${STEM}_refined.mp4"
%cd $REPO
```

### 7. Verify the outputs

```python
# Track-count before vs after refinement (refined should have fewer / cleaner IDs):
%cd $REPO/Deep-EIoU/Deep-EIoU
!python tools/count_tracks.py "$OUTPUT_DIR/$STEM/01_track/tracks.txt"
!python tools/count_tracks.py "$OUTPUT_DIR/$STEM/02_refine/refined.txt"
%cd $REPO

# Profiling summary (per-stage time / GPU mem / CPU mem / IO counts):
!cat "$OUTPUT_DIR/$STEM/profiles/summary.md"
```

**(Optional) Confirm `--parallel` matches sequential.** Run each mode into a *separate* artifacts
dir, then diff the two `tracks.txt` with the tolerant comparator (parity is exact only up to
floating-point noise):

```python
!python -m pipeline run --video "$VIDEO" --artifacts-dir /content/_seq --device gpu --force-stage1
!python -m pipeline run --video "$VIDEO" --artifacts-dir /content/_par --device gpu --parallel --batch-size 8 --force-stage1
!python -m pipeline compare \
    --a "/content/_seq/$STEM/01_track/tracks.txt" \
    --b "/content/_par/$STEM/01_track/tracks.txt" \
    --tol 1.0
```

---

## Output layout on Drive

After a run, `Colab Notebook/football-analysis-project/output/gta-track/<STEM>/` contains:

```
<STEM>/
  01_track/
    tracks.txt        # raw DeepEIoU tracking (MOT, 0-based frames)
    tracklets.pkl     # per-id tracklets WITH reused ReID features (large)
  02_refine/
    refined.txt       # GtaLink split+connect output (the final result)
  profiles/
    01_track.json     # Stage 1 profile (time / GPU / CPU / counts)
    02_refine.json    # Stage 2 profile
    summary.json
    summary.md
  <STEM>_refined.mp4  # only if you ran the optional render step
```

---

## Tips & troubleshooting

- **GPU out of memory** in `--parallel`: lower `--batch-size` (e.g. 4 or 2), or drop `--parallel`
  to run sequentially.
- **`FileNotFoundError` for a checkpoint**: re-check Step 4 — the symlinks must resolve to real
  files in your Drive `checkpoints/` folder, and `sports_model.pth.tar-60` must keep that exact
  name.
- **Writing to Drive is slow / flaky on long videos** (the `tracklets.pkl` can be hundreds of MB):
  run with `--artifacts-dir /content/work` (fast local disk), then copy results to Drive at the
  end: `!cp -r "/content/work/$STEM" "$OUTPUT_DIR/"`.
- **Re-tuning a parameter seems ignored**: that's the cache — add the matching `--force-*` flag
  (see Step 5).
- **Path with a space** (`Colab Notebook`): always keep the `"$VAR"` quotes in the `!` cells.
- **Wrong/blank detections**: ensure the detector `best_ckpt.pth.tar` matches the YOLOX exp
  (`yolox/yolox_x_ch_sportsmot.py`, the Stage 1 default).
- **Running stages independently** (e.g. only refine, from a cached `tracklets.pkl`) is documented
  in [USER_GUIDE.md](USER_GUIDE.md) — the Colab commands are the same, just `cd gta-link` and call
  `stage2_refine.py` with the shared `--artifacts-dir`.

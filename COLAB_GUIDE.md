# Running the GTA tracking pipeline on Google Colab

Step-by-step instructions to run the two-stage pipeline (DeepEIoU tracking → GtaLink
refinement) on a Colab GPU, reading input videos from Google Drive and persisting all outputs
back to Drive.

- **Input videos:** `Colab Notebooks/football-analysis-project/input-videos/`
- **Outputs:** `Colab Notebooks/football-analysis-project/output/gta-track/`

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
| `yolov11l.pt` | YOLOv11 detector (Stage 1 default) | `Deep-EIoU/Deep-EIoU/checkpoints/yolov11l.pt` |
| `sports_model.pth.tar-60` | OSNet ReID (Stage 1) | `Deep-EIoU/Deep-EIoU/checkpoints/sports_model.pth.tar-60` |

Put your custom `yolov11l.pt` in your Drive `checkpoints/` folder. The ReID model lives in the public Google Drive folder linked from the Deep-EIoU readme:
**<https://drive.google.com/drive/folders/1wItcb0yeGaxOS08_G9yRWBTnpVf0vZ2w>**

Step 4 below fetches the ReID model with `gdown` into your
Drive `checkpoints/` folder (once) and links them into place. Put your custom `yolov11l.pt` there first.

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

# EDIT this if your project folder lives elsewhere in Drive:
DRIVE_BASE = "/content/drive/MyDrive/Colab Notebooks/football-analysis-project"

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
Colab already has `torch`/`torchvision`. Install GtaLink's `requirements.txt` (it's the
comprehensive set that covers Stage 2's refine libraries **and** the import-time deps of the
bundled torchreid that Stage 1's ReID needs), then add the libraries it doesn't list —
`cython_bbox` (the tracker's IoU), `tqdm`, and `ultralytics`:

```python
%cd $REPO
!pip install -q -r gta-link/requirements.txt
!pip install -q cython_bbox tqdm ultralytics
!pip install -q -e Deep-EIoU/Deep-EIoU/reid
```

> The last line installs the **vendored** `reid` package (Deep-EIoU's bundled torchreid) — this is
> the Deep-EIoU README's `cd reid && python setup.py develop` step, done the modern `pip install -e`
> way (so it avoids the `setuptools<60` caveat the gta-link README mentions). It is required:
> Stage 1's ReID self-imports by absolute name (`from torchreid import …`), so the vendored copy
> must be registered as the top-level `torchreid`.
>
> **Do NOT `pip install torchreid` from PyPI** — the current PyPI release uses a `torchreid.reid.*`
> layout that is incompatible with the vendored code's `torchreid.utils` imports (you'll get
> `ModuleNotFoundError: No module named 'torchreid.utils'`).
>
> You do **not** need Deep-EIoU's *other* `setup.py` (the top-level `yolox` one) — it only compiles
> `yolox._C`, used solely by the COCO mAP evaluator, not by detection/tracking. `requirements.txt`
> also pulls a few training/dev-only packages (`tb-nightly`, `flake8`, `yapf`) the pipeline doesn't
> use — harmless, just slower. If `pip install cython_bbox` ever fails to build, run
> `!pip install -q cython` first and retry.

#### One environment, two working directories

Everything runs in **one Colab environment** — you `pip install` once (above) and that single set of
packages serves both stages. The stages are not separate environments; the *only* thing that differs
is the **working directory**, which the orchestrator (`python -m pipeline run`) sets per stage when
it launches each as a subprocess:

| Stage | Working dir | ReID |
|-------|-------------|------|
| 1 (track) | `Deep-EIoU/Deep-EIoU` | `from reid.torchreid.utils import FeatureExtractor` → the vendored `Deep-EIoU/Deep-EIoU/reid/` package installed editable above (so top-level `torchreid` resolves to it) |
| 2 (refine) | `gta-link` | does **not** import `reid`/`torchreid` — it reuses Stage 1's cached embeddings |

### 4. Fetch/link the checkpoints
Downloads the ReID model from the public Drive folder into your Drive `checkpoints/` folder
the first time, then reuses it on later sessions. Your custom `yolov11l.pt` must already be in
that Drive folder. Finally it symlinks both files into the directory Stage 1 expects.

```python
import glob, os

os.makedirs(CKPT_DIR, exist_ok=True)
CKPT_FOLDER_URL = "https://drive.google.com/drive/folders/1wItcb0yeGaxOS08_G9yRWBTnpVf0vZ2w"

# Download only the ReID model if it is not already in Drive.
have = {os.path.basename(p) for p in glob.glob(f"{CKPT_DIR}/**/*", recursive=True)}
if "sports_model.pth.tar-60" not in have:
    !pip install -q gdown
    !gdown --folder "{CKPT_FOLDER_URL}" -O "$CKPT_DIR"

# Link each checkpoint into Deep-EIoU/Deep-EIoU/checkpoints/ (robust to any sub-folder gdown made).
dst_dir = f"{REPO}/Deep-EIoU/Deep-EIoU/checkpoints"
os.makedirs(dst_dir, exist_ok=True)
for name in ("yolov11l.pt", "sports_model.pth.tar-60"):
    matches = glob.glob(f"{CKPT_DIR}/**/{name}", recursive=True)
    assert matches, f"{name} not found under {CKPT_DIR}"
    dst = f"{dst_dir}/{name}"
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(matches[0], dst)
    print("linked", matches[0], "->", dst)

!ls -la "{dst_dir}"
```

> If `gdown --folder` ever fails (Drive quota / auth), open the folder link in a browser, download
> `sports_model.pth.tar-60` manually, and drop it into your Drive `checkpoints/` folder. Your
> custom `yolov11l.pt` must also be in that folder before you run this cell.

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

**Faster perception (optional):** add `--fp16` (half-precision detector inference; the main
throughput lever, ~1.5–2× on GPU) and optionally `--fuse`:

```python
!python -m pipeline run \
    --video "$VIDEO" \
    --artifacts-dir "$OUTPUT_DIR" \
    --device gpu \
    --fp16 --fuse
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

---

## Output layout on Drive

After a run, `Colab Notebooks/football-analysis-project/output/gta-track/<STEM>/` contains:

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

- **GPU out of memory**: try `--fp16` (roughly halves activation memory), or process shorter clips.
- **`FileNotFoundError` for a checkpoint**: re-check Step 4 — the symlinks must resolve to real
  files in your Drive `checkpoints/` folder, and `yolov11l.pt` plus
  `sports_model.pth.tar-60` must keep those exact names.
- **Writing to Drive is slow / flaky on long videos** (the `tracklets.pkl` can be hundreds of MB):
  run with `--artifacts-dir /content/work` (fast local disk), then copy results to Drive at the
  end: `!cp -r "/content/work/$STEM" "$OUTPUT_DIR/"`.
- **Re-tuning a parameter seems ignored**: that's the cache — add the matching `--force-*` flag
  (see Step 5).
- **Path with a space** (`Colab Notebooks`): always keep the `"$VAR"` quotes in the `!` cells.
- **Wrong/blank detections**: ensure `yolov11l.pt` is the custom Ultralytics YOLOv11 checkpoint
  you intend to use.
- **Running stages independently** (e.g. only refine, from a cached `tracklets.pkl`) is documented
  in [USER_GUIDE.md](USER_GUIDE.md) — the Colab commands are the same, just `cd gta-link` and call
  `stage2_refine.py` with the shared `--artifacts-dir`.

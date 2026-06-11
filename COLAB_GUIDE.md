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

Both live in the public Google Drive folder linked from the Deep-EIoU readme:
**<https://drive.google.com/drive/folders/1wItcb0yeGaxOS08_G9yRWBTnpVf0vZ2w>**

You don't need to download them by hand — **Step 4 below fetches them with `gdown`** into your
Drive `checkpoints/` folder (once) and links them into place. Nothing to do here.

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
Colab already has `torch`/`torchvision`. Install GtaLink's `requirements.txt` (it's the
comprehensive set that covers Stage 2's refine libraries **and** the import-time deps of the
bundled torchreid that Stage 1's ReID needs), then add the two libraries it doesn't list —
`cython_bbox` (the tracker's IoU) and `tqdm`:

```python
%cd $REPO
!pip install -q -r gta-link/requirements.txt
!pip install -q cython_bbox tqdm
```

> **You do NOT need to run the READMEs' build steps.** The `yolox` and `reid` packages are used
> **in place**, so:
> - Deep-EIoU's `python setup.py develop` is skipped — it only compiles `yolox._C`, which is used
>   solely by the COCO mAP evaluator, not by detection/tracking (NMS uses torchvision).
> - `reid/setup.py develop` is skipped — it only builds an optional Cython ranking metric that has
>   a Python fallback; `FeatureExtractor` doesn't use it.
> - GtaLink's separate `pip install torchreid` is skipped — see the box below.
>
> `requirements.txt` also pulls a few training/dev-only packages (`tb-nightly`, `flake8`, `yapf`)
> the pipeline doesn't use — harmless, just slower to install. If `pip install cython_bbox` ever
> fails to build, run `!pip install -q cython` first and retry.

#### One environment, two working directories — where `reid` comes from

Everything here runs in **one Colab environment**: you `pip install` once (above) and that single
set of packages serves both stages. The stages are not separate environments — the *only* thing
that differs between them is the **working directory**, which the orchestrator
(`python -m pipeline run`) sets per stage when it launches each as a subprocess:

| Stage | Working dir | ReID / `reid` source |
|-------|-------------|----------------------|
| 1 (track) | `Deep-EIoU/Deep-EIoU` | `from reid.torchreid.utils import FeatureExtractor` → the **bundled** `Deep-EIoU/Deep-EIoU/reid/` copy, used **in place** (never pip-installed) |
| 2 (refine) | `gta-link` | does **not** import `reid`/`torchreid` at all — it reuses Stage 1's cached embeddings |

So **`reid` is not installed from anywhere** — it's the in-repo folder, found because Stage 1's CWD
is `Deep-EIoU/Deep-EIoU`. Do **not** `pip install torchreid`: it would add an unused `torchreid` to
the env, and Stage 1 imports `reid.torchreid` (namespaced under `reid`), not the bare `torchreid`
that pip provides — so it wouldn't be used anyway. (The two bundled torchreid copies under
`Deep-EIoU/.../reid/` and `gta-link/reid/` never clash because only Stage 1 imports one of them, and
only from its own CWD.) The libraries `requirements.txt` installed are simply the leaf dependencies
that bundled `reid` needs at import time.

### 4. Fetch the checkpoints (once, to Drive) and link them into place
Downloads the two model files from the public Drive folder into your Drive `checkpoints/` folder
the first time, then reuses them on later sessions. Finally it symlinks them into the directory
Stage 1 expects.

```python
import glob, os

os.makedirs(CKPT_DIR, exist_ok=True)
CKPT_FOLDER_URL = "https://drive.google.com/drive/folders/1wItcb0yeGaxOS08_G9yRWBTnpVf0vZ2w"

# Download only if not already in Drive (the files are large — avoid re-downloading each session).
have = {os.path.basename(p) for p in glob.glob(f"{CKPT_DIR}/**/*", recursive=True)}
if not {"best_ckpt.pth.tar", "sports_model.pth.tar-60"} <= have:
    !pip install -q gdown
    !gdown --folder "{CKPT_FOLDER_URL}" -O "$CKPT_DIR"

# Link each checkpoint into Deep-EIoU/Deep-EIoU/checkpoints/ (robust to any sub-folder gdown made).
dst_dir = f"{REPO}/Deep-EIoU/Deep-EIoU/checkpoints"
os.makedirs(dst_dir, exist_ok=True)
for name in ("best_ckpt.pth.tar", "sports_model.pth.tar-60"):
    matches = glob.glob(f"{CKPT_DIR}/**/{name}", recursive=True)
    assert matches, f"{name} not found under {CKPT_DIR} — check the gdown download"
    dst = f"{dst_dir}/{name}"
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(matches[0], dst)
    print("linked", matches[0], "->", dst)

!ls -la "{dst_dir}"
```

> If `gdown --folder` ever fails (Drive quota / auth), open the folder link in a browser, download
> the two files manually, and drop them into your Drive `checkpoints/` folder — then re-run this
> cell (it will skip the download and just link them).

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

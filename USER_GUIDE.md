# User Guide — Tracking Pipeline (DeepEIoU + GtaLink + Team Assignment)

This pipeline turns a single football clip into clean, refined, team-labelled
player tracks in three cached, profileable stages:

```
clip.mp4 ─▶ Stage 1 (DeepEIoU)         ─▶ Stage 2 (GtaLink)      ─▶ Stage 3 (Team)
            detect + ReID + track          split + connect/merge      cluster → team
            tracks.txt + tracklets.pkl ───▶ refined.txt + .pkl ───▶ refined.txt + attrs
```


# Part A — Quickstart

## A1. Prerequisites & from-scratch setup

Colab preinstalls most of this; a bare GPU VM does not. From scratch you need:

1. **A CUDA GPU + NVIDIA driver.** Stage 1 (detection + ReID) needs it; Stages 2-3
   are CPU-bound. Verify the driver:
   ```bash
   nvidia-smi
   ```
2. **Python 3.10 or 3.11 + a virtualenv.** Use 3.10/3.11 — the versions this stack is
   validated on (Colab runs ~3.10/3.11). Avoid 3.13: some pinned deps have no 3.13
   wheels and fall back to (often failing) source builds. With conda:
   ```bash
   conda create -n football python=3.11 -y && conda activate football
   ```
   or a venv (needs the deadsnakes PPA if 3.11 isn't your system python):
   ```bash
   python3.11 -m venv .venv && source .venv/bin/activate
   ```
3. **System build tools + OpenCV libs** (commonly missing on a headless VM):
   ```bash
   sudo apt-get install -y build-essential python3-dev libgl1 libglib2.0-0
   ```
   `build-essential` + `python3-dev` compile the `reid` package's Cython extension in
   step 5 (which installs with `--no-build-isolation`); `libgl1` / `libglib2.0-0` are
   OpenCV's runtime libraries.
4. **`torch` / `torchvision`** — Colab ships these; on a bare VM install a CUDA
   build yourself **before** step 5 (otherwise step 5 pulls a CPU-only torch).
   Check the CUDA version your driver supports — the top-right of `nvidia-smi`
   ("CUDA Version: …") — then install the matching wheel from PyTorch's index:
   ```bash
   # pick the cuXXX matching your driver (cu121 is a safe default on recent
   # drivers; cu124 for newer, cu118 for older):
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
   Verify the GPU is visible before continuing:
   ```bash
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"  # expect: <ver> True
   ```
5. **The pipeline dependencies**, in this order:
   ```bash
   pip install -r gta-link/requirements.txt
   pip install cython_bbox tqdm ultralytics
   pip install -e Deep-EIoU/Deep-EIoU/reid --no-build-isolation
   ```
   `gta-link/requirements.txt` is the comprehensive set (it covers Stage 2's refine
   libraries **and** the import-time deps of the bundled torchreid that Stage 1's
   ReID needs); the second line adds the libraries it omits — `cython_bbox` (the
   tracker's IoU), `tqdm`, `ultralytics` (YOLOv11). If `cython_bbox` ever fails to
   build, `pip install cython` first and retry.

   > **Vendored `reid` / torchreid caveat.** The last line installs the **vendored**
   > `reid` package (Deep-EIoU's bundled torchreid) as editable. Stage 1's ReID
   > self-imports by absolute name (`from torchreid import …`), so the vendored copy
   > must register as the top-level `torchreid`. **Do NOT `pip install torchreid`
   > from PyPI** — its `torchreid.reid.*` layout is incompatible with the vendored
   > code (you'll get `ModuleNotFoundError: No module named 'torchreid.utils'`).
6. **Checkpoints** — two model files (gitignored) must sit in
   `Deep-EIoU/Deep-EIoU/checkpoints/`:
   - `yolov11l.pt` — YOLOv11 detector (Stage 1 default)
   - `sports_model.pth.tar-60` — OSNet ReID (Stage 1)

   Stages 2 and 3 need **no** checkpoint — they reuse the ReID embeddings Stage 1
   saved into `tracklets.pkl`.

All commands below assume your shell's CWD is the **repo root**.

## A2. Minimal end-to-end run

```bash
python -m pipeline run --video clip.mp4 --artifacts-dir ./out --fp16 --fuse
```

This chains Stage 1 (tracking) → Stage 2 (refine) → Stage 3 (team assignment),
streaming each stage's live log, and writes a profiling summary. Everything for the
clip lands under `./out/<video_stem>/` (full layout in §B2); the final result is
`03_team/refined.txt` (refined MOT rows with a per-row `team_id`).

## A3. Key options

| Option | What it does |
|---|---|
| `--artifacts-dir DIR` | Where artifacts land (default `<repo>/artifacts`). |
| `--device gpu\|cpu` | Stage 1 device (`cpu` works but is slow — debug only). |
| `--fp16` | Half-precision detector inference — the main throughput lever (~1.5-2×; see §B7). |
| `--fuse` | Fuse detector conv+BN; small free gain. |
| `--detector-ckpt PATH` | Custom YOLOv11 checkpoint (default `checkpoints/yolov11l.pt`). |
| `--force-all` / `--force-stage1` / `--force-stage2` / `--force-stage3` | Force recompute (params alone do NOT bust the cache — see §B5). |

Full list: `python -m pipeline run --help`. Exhaustive reference in §B4.

## A4. Azure storage to download and upload artifacts

Pull inputs and checkpoints from Blob, run on local VM, then push the outputs back.
The repo ships `azure_blob.py` (repo root) for this — a tiny helper that needs only
the connection string (no `az login`). Install its one dependency on the VM:

```bash
pip install azure-storage-blob
```

Authenticate once via connection string (`azure_blob.py` reads this env var):

```bash
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"
```

Then pull → run → push:

```bash
# Pull the clip and the checkpoints from Blob
python azure_blob.py download     videos clip.mp4 ./clip.mp4
python azure_blob.py download-dir  checkpoints "" Deep-EIoU/Deep-EIoU/checkpoints

# Run on local disk
python -m pipeline run --video ./clip.mp4 --artifacts-dir ./out --device gpu --fp16 --fuse

# Push the artifact tree back to Blob
python azure_blob.py upload-dir    outputs <stem> ./out/<stem>
```

`az storage blob` / `azcopy` (via SAS, for very large transfers) remain alternatives.


# Part B — Reference

## B1. Architecture

Three stages run as separate **subprocesses, each in its own working directory**. The
two vendored projects (DeepEIoU, GtaLink) bundle their own incompatible `torchreid`, so
importing both into one interpreter would clash — hence the subprocess-per-stage design.
The orchestrator (`python -m pipeline`) chains them, resolves **one** absolute video
path + **one** absolute artifacts dir, and hands the **same** absolute paths to every
stage (a relative path would resolve to a different place per CWD).

You `pip install` once (§A1); that single set of packages serves all stages. The stages
are **not** separate environments — only the working directory differs, set per stage by
the orchestrator:

| Stage | Working dir | ReID |
|-------|-------------|------|
| 1 (track) | `Deep-EIoU/Deep-EIoU` | `from reid.torchreid.utils import FeatureExtractor` → the vendored `Deep-EIoU/Deep-EIoU/reid/` package installed editable in §A1 (top-level `torchreid` resolves to it) |
| 2 (refine) | `gta-link` | none — reuses Stage 1's cached embeddings from `tracklets.pkl` |
| 3 (team) | `<repo>` (CWD-independent) | none — reuses the embeddings in `refined_tracklets.pkl` |

Stage 1 detects players, computes a ReID embedding per box, and runs the online tracker.
Stage 2 **reuses Stage 1's embeddings** (never re-reads frames, loads **no** ReID model)
to split ID-switched tracklets and merge fragmented ones. Stage 3 clusters per-track mean
embeddings into two teams and writes the team label into the MOT output.

## B2. Artifact contract

Everything for one video lives under `<artifacts-dir>/<video_stem>/`:

```
<video_stem>/
  01_track/tracks.txt              # MOT, 0-based frames        (Stage 1)
  01_track/tracklets.pkl           # {id: Tracklet}, features = reused ReID embeddings
  02_refine/refined.txt            # refined MOT                (Stage 2)
  02_refine/refined_tracklets.pkl  # {id: Tracklet}, ids match refined.txt (slim by default)
  03_team/refined.txt              # refined MOT + team_id in col 9  (Stage 3, final result)
  03_team/track_attributes.json    # {id: {class, team, gk}} per-track aggregation
  profiles/01_track.json           # per-stage profile (time / GPU / CPU / IO)
  profiles/02_refine.json
  profiles/03_team.json
  profiles/summary.{json,md}       # aggregated across stages
```

Dataflow (one line):

```
video ─▶ [S1] tracks.txt + tracklets.pkl ─▶ [S2] refined.txt + refined_tracklets.pkl ─▶ [S3] 03_team/refined.txt + track_attributes.json ─▶ profiles/summary.md
```

**Caching:** a stage is skipped when its output exists and is newer than its inputs
(mtime-based). Stage 1 keys on the **video** mtime; Stage 2 on `tracklets.pkl`; Stage 3
on `refined_tracklets.pkl`. Force a recompute with `--force-*` (see §B5).

## B3. Run each stage independently

To re-run just one stage (e.g. re-tune refinement without re-tracking), call the stage
scripts directly. Each runs **in its own CWD** and takes the **same** `--video` /
`--artifacts-dir` so they share the artifact tree.

**Stage 1 (tracking)** — CWD `Deep-EIoU/Deep-EIoU`:
```bash
cd Deep-EIoU/Deep-EIoU
python tools/stage1_track.py \
    --video /abs/path/clip.mp4 \
    --artifacts-dir /abs/path/out \
    --device gpu --fp16
```

**Stage 2 (refine)** — CWD `gta-link` (always runs split + connect):
```bash
cd gta-link
python stage2_refine.py \
    --video /abs/path/clip.mp4 \
    --artifacts-dir /abs/path/out
```

**Stage 3 (team)** — CWD-independent, but conventionally the repo root:
```bash
python pipeline/team_assignment.py \
    --video /abs/path/clip.mp4 \
    --artifacts-dir /abs/path/out
```

> Use **absolute** `--video` / `--artifacts-dir` here. Because each script runs in a
> different CWD, a relative path resolves to a different place per stage. (The
> orchestrator does this for you automatically.)

Stages 2 and 3 use `--video` **only** to locate the artifact directory — they never read
the video. Stage 2 reuses Stage 1's embeddings from `tracklets.pkl` (no ReID model);
Stage 3 reuses the embeddings in `refined_tracklets.pkl`.

To force a single direct stage: pass `--force` to that stage script.

## B4. Exhaustive options

### Orchestrator — `python -m pipeline run`

| Flag | Default | Meaning |
|---|---|---|
| `--video PATH` | (required) | Input video. |
| `--artifacts-dir DIR` | `<repo>/artifacts` | Base artifacts dir. |
| `--device gpu\|cpu` | `gpu` | Stage 1 detection + ReID device. |
| `--detector-ckpt PATH` | `checkpoints/yolov11l.pt` | Stage 1 YOLOv11 checkpoint. |
| `--fp16` | off | Half-precision detector inference (main throughput lever). |
| `--fuse` | off | Fuse detector conv+BN (small free gain). |
| `--batch-size INT` | `16` | Stage 1 perception batch window — frames to detect + ReID per batch before replaying the tracker frame-by-frame. Tracking stays sequential; only detection/ReID are batched. Lower it if a high-res batch overflows VRAM. |
| `--eps FLOAT` | `0.6` | Stage 2 DBSCAN eps (split). |
| `--min_samples INT` | `10` | Stage 2 DBSCAN min_samples. |
| `--max_k INT` | `3` | Stage 2 max subtracklets per split. |
| `--min_len INT` | `100` | Stage 2 min tracklet length to split. |
| `--spatial_factor FLOAT` | `1.0` | Stage 2 spatial distance factor. |
| `--merge_dist_thres FLOAT` | `0.4` | Stage 2 max cosine distance to merge. |
| `--force-all` | off | Force recompute of all stages. |
| `--force-stage1` | off | Force Stage 1 only. |
| `--force-stage2` | off | Force Stage 2 only. |
| `--force-stage3` | off | Force Stage 3 only. |

### Stage-script-only flags (NOT on `python -m pipeline run`)

These exist only on the individual stage scripts; the orchestrator always uses their
**defaults**. To change them you must run the stage directly (§B3).

| Flag | Stage script | Default | Meaning |
|---|---|---|---|
| `--keep-features` | `stage2_refine.py` | off | Export the **full** `refined_tracklets.pkl` (per-detection features / times / bboxes intact). The default is a slim pkl — a per-track mean embedding + scores + class_ids only (Stage 3, the sole consumer, reads just those), which drops the ~4 GB of features. |

Stage 1 also inherits DeepEIoU's `demo.py` tracker/detector knobs (`--conf`, `--nms`,
`--tsize`, `--min_box_area`, `--track_buffer`, `--match_thresh`, …); see
`python tools/stage1_track.py --help`. Each stage script also takes `--force`.

## B5. Force recompute — and the param/cache caveat

By default each stage skips work when its output is newer than its inputs:

```bash
python -m pipeline run --video clip.mp4 --force-all       # all stages
python -m pipeline run --video clip.mp4 --force-stage1    # tracking only
python -m pipeline run --video clip.mp4 --force-stage2    # refine only
python -m pipeline run --video clip.mp4 --force-stage3    # team assignment only
```

(Directly: pass `--force` to `stage1_track.py` / `stage2_refine.py` / `team_assignment.py`.)

> ### ⚠️ The cache keys on input MTIME, **not** on parameters
>
> Caching compares **file modification times** only. Changing a **parameter** (a
> tracking threshold, `--eps`, `--min_samples`, `--merge_dist_thres`, …) does **not**
> change any input file's mtime, so it does **NOT** invalidate the cache — a re-run with
> new params is **silently skipped** and you keep the old output.
>
> **When re-tuning, always add the matching force flag:**
> - changed Stage-1 params → `--force-stage1` (or `--force-all`)
> - changed Stage-2 refine params → `--force-stage2` (or `--force-all`)
>
> ```bash
> python -m pipeline run --video clip.mp4 --eps 0.5 --merge_dist_thres 0.35 --force-stage2
> ```

## B6. Inspect profiling

Each stage writes `profiles/<stage>.json`; the orchestrator aggregates them into
`profiles/summary.json` + `profiles/summary.md`.

Per-stage JSON fields:

| field | meaning |
|---|---|
| `stage` | stage id (`01_track` / `02_refine` / `03_team`) |
| `wall_time_s` | wall-clock seconds for the stage body |
| `gpu_peak_mb` | peak CUDA memory (MB); `null` without a GPU |
| `cpu_rss_start_mb` / `cpu_rss_end_mb` | process RSS (MB) before/after (`null` if `psutil`/`resource` unavailable) |
| `cpu_peak_mb` | peak RSS (MB) when available, else `null` |
| `io` | caller metadata: frame counts, track counts, `emb_dim`, detector, refine params, team counts, … |

```bash
cat /abs/path/out/<stem>/profiles/summary.md            # human-readable table
python -c "import json,sys; print(json.load(open(sys.argv[1]))['totals'])" \
    /abs/path/out/<stem>/profiles/summary.json          # totals (wall, gpu peak)
```

`summary.md` has one row per stage plus a **TOTAL** row (summed wall time, max GPU peak).
Use the `io` block to inspect per-stage counts and the refine params used.

## B7. Throughput: `--fp16`

Stage 1 detector inference is usually the bottleneck. Half-precision is the main lever:

```bash
python -m pipeline run --video clip.mp4 --device gpu --fp16
python -m pipeline run --video clip.mp4 --device gpu --fp16 --fuse   # + conv/BN fuse
```

On a modern GPU (T4/A100) `--fp16` typically gives **~1.5–2× Stage-1 throughput** and
roughly halves activation memory, with no meaningful accuracy change for inference.

## B8. Verify outputs

### Count tracks before vs after refinement

Refinement should yield **fewer, cleaner** IDs (ID-switches split & re-merged):

```bash
cd Deep-EIoU/Deep-EIoU
python tools/count_tracks.py /abs/path/out/<stem>/01_track/tracks.txt
python tools/count_tracks.py /abs/path/out/<stem>/02_refine/refined.txt
# ignore short-lived (noise) tracks:
python tools/count_tracks.py /abs/path/out/<stem>/02_refine/refined.txt --min_len 5
```

Expect the `unique tracks` count on `refined.txt` to be **lower / cleaner** than on
`tracks.txt`.

### Eyeball the tracks (render an annotated video)

The pipeline outputs MOT `.txt`, not video. Rebuild a watchable overlay (CPU-only, no
GPU). Frames are **0-based**, so do **not** pass `--one_indexed`:

```bash
cd Deep-EIoU/Deep-EIoU
python tools/render_from_txt.py --path /abs/path/clip.mp4 \
    --txt /abs/path/out/<stem>/01_track/tracks.txt
python tools/render_from_txt.py --path /abs/path/clip.mp4 \
    --txt /abs/path/out/<stem>/03_team/refined.txt
```

Each writes `<txt>_rendered.mp4` next to the `.txt` (or pass `--save_path`).

## B9. Evaluate (HOTA / MOTA / IDF1 + class/team)

Score the pipeline's `refined.txt` against ground truth. The evaluator is
**standalone and CPU-only** (stdlib + numpy; no GPU or pipeline runtime) and reports
**HOTA / DetA / AssA / MOTA / IDF1** via TrackEval, plus class/team attribute metrics.

**Setup — TrackEval** (not pip-installed; clone it and pass its path):

```bash
git clone https://github.com/JonathonLuiten/TrackEval.git /path/to/TrackEval
# TrackEval still uses the removed np.float / np.int / np.bool aliases; on modern
# numpy, patch them to the builtins once:
grep -rl 'np\.float\|np\.int\|np\.bool' /path/to/TrackEval/trackeval \
  | xargs -r sed -i 's/np\.float\b/float/g; s/np\.int\b/int/g; s/np\.bool\b/bool/g'
```

**Run:**

```bash
python -m eval.evaluate \
    --pred /abs/path/out/<stem>/03_team/refined.txt \
    --gt   /abs/path/gt_mot_<stem>.txt \
    --seq-name <stem> \
    --trackeval-path /path/to/TrackEval \
    --out  /abs/path/out/<stem>/eval/metrics.json
```

Or pass `--video clip.mp4` (with `--artifacts-dir`) instead of `--pred` to auto-locate
`refined.txt` (prefers `03_team/`, falls back to `02_refine/`).

- **GT format:** MOTChallenge `gt.txt` (**1-based** frames); the pipeline's **0-based**
  output is converted automatically (`--gt-format` defaults to `motchallenge`).
- **Class label:** TrackEval scores under `--class-name` (default `pedestrian`); override
  only if your GT uses a different label.
- **Outputs** (next to `--out`, i.e. `<stem>/eval/`):
  - `metrics.json` — the 5 headline metrics + the full per-family dump (HOTA / CLEAR /
    Identity / Count).
  - `attributes_metrics.json` — class/team attribute accuracy. This step is **non-fatal**:
    if it fails, the HOTA/MOTA results above still stand.

## B10. Dependency rationale & troubleshooting

**Why these deps.** `gta-link/requirements.txt` is the comprehensive set because it
covers both Stage 2's refine libraries and the import-time deps of the vendored torchreid
Stage 1 uses; `cython_bbox` / `tqdm` / `ultralytics` are the libraries it doesn't list.
The editable `reid` install is the Deep-EIoU README's `cd reid && python setup.py
develop` step done the modern `pip install -e` way (avoiding the `setuptools<60` caveat
the gta-link README mentions). You do **not** need Deep-EIoU's *other* `setup.py` (the
top-level `yolox` one) — it only compiles `yolox._C`, used solely by the COCO mAP
evaluator, not by detection/tracking. `requirements.txt` also pulls a few training/dev-only
packages (`tb-nightly`, `flake8`, `yapf`) the pipeline doesn't use — harmless, just slower.

**Troubleshooting.**
- **`ModuleNotFoundError: No module named 'torchreid.utils'`** — you installed PyPI
  `torchreid`. Uninstall it and `pip install -e Deep-EIoU/Deep-EIoU/reid` instead (§A1).
- **GPU out of memory** — add `--fp16` (roughly halves activation memory); lower
  `--batch-size` — now passable straight to `python -m pipeline run --batch-size N`
  (§B4); or process shorter clips.
- **`FileNotFoundError` for a checkpoint** — `yolov11l.pt` and `sports_model.pth.tar-60`
  must sit in `Deep-EIoU/Deep-EIoU/checkpoints/` under those exact names (§A1).
- **Writing artifacts is slow / flaky on long videos** (`tracklets.pkl` can be hundreds
  of MB) — write to fast local disk (`--artifacts-dir /local/work`), then sync to your
  network/Blob store afterwards (§A4).
- **Re-tuning a parameter seems ignored** — that's the cache; add the matching `--force-*`
  flag (§B5).
- **Wrong / blank detections** — ensure `yolov11l.pt` is the custom Ultralytics YOLOv11
  checkpoint you intend to use.
- **Running stages independently** (e.g. only refine, from a cached `tracklets.pkl`) is in
  §B3 — `cd` to the stage's CWD and call its script with the shared `--artifacts-dir`.

# User Guide — Two-Stage Tracking Pipeline (DeepEIoU + GtaLink)

This pipeline turns a single football clip into clean, refined player tracks in
two cached, profileable stages:

```
clip.mp4 ──▶ Stage 1 (DeepEIoU)            ──▶ Stage 2 (GtaLink)        ──▶ refined.txt
            detect + ReID + online track       split + connect/merge        (clean IDs)
            tracks.txt  +  tracklets.pkl  ─────▶ reuses tracklets.pkl
```

Stage 1 detects players, computes a ReID embedding per box, and runs the online
tracker. Stage 2 **reuses Stage 1's embeddings** (it never re-reads frames and
loads **no** ReID model) to split ID-switched tracklets and merge fragmented
ones.

The two projects each bundle their own (incompatible) `torchreid`, so each stage
runs as its **own subprocess in its own working directory**. The orchestrator
(`python -m pipeline`) chains them, resolving one absolute video path + one
absolute artifacts directory and handing the **same** absolute paths to both
stages.

---

## 1. Artifact contract

Everything for one video lives under `artifacts/<video_stem>/`:

```
artifacts/<video_stem>/
  01_track/tracks.txt        # MOT, 0-based frames  (Stage 1 output)
  01_track/tracklets.pkl     # {id: Tracklet}, features = reused ReID embeddings
  02_refine/refined.txt      # refined MOT          (Stage 2 output)
  profiles/01_track.json     # per-stage profile (time / GPU / CPU / IO)
  profiles/02_refine.json
  profiles/summary.{json,md} # aggregated across stages
```

Dataflow (one line):

```
video ─▶ [Stage 1] tracks.txt + tracklets.pkl ─▶ [Stage 2] refined.txt ─▶ profiles/summary.md
```

**Caching:** a stage is skipped when its output exists and is newer than its
inputs (mtime-based). Stage 1 keys on the **video** mtime; Stage 2 keys on
`tracklets.pkl` mtime. Force a recompute with `--force-*` (see §5).

---

## 2. Prerequisites

- **One shared Python env** with the DeepEIoU + GtaLink dependencies (torch,
  torchvision, opencv-python, numpy, scipy, scikit-learn, loguru, matplotlib,
  seaborn, yolox, …). The orchestrator launches both stages with
  `sys.executable`, so they run in **this same env**.
- **A CUDA GPU** for Stage 1 (detection + ReID). Stage 2 is CPU-bound and small.
  (`--device cpu` works for Stage 1 but is slow — debugging only.)
- **Two model checkpoints**, placed where Stage 1 looks for them (paths are
  relative to the Stage 1 CWD `Deep-EIoU/Deep-EIoU/`):
  - **Detector:** `Deep-EIoU/Deep-EIoU/checkpoints/best_ckpt.pth.tar`
  - **ReID:** `Deep-EIoU/Deep-EIoU/checkpoints/sports_model.pth.tar-60`

  Stage 2 needs **no** checkpoint — it reuses the embeddings already saved in
  `tracklets.pkl`.

All commands below assume your shell's CWD is the **repo root**.

---

## 3. Run the FULL pipeline

```bash
python -m pipeline run --video clip.mp4
```

This runs Stage 1 then Stage 2 with the **full** refinement (both `--use_split`
and `--use_connect` on by default), streaming each stage's live progress, and
finally writes `profiles/summary.md`.

Common options:

```bash
# Choose where artifacts land (default: <repo>/artifacts)
python -m pipeline run --video clip.mp4 --artifacts-dir /content/artifacts

# Stage-1 throughput: --fp16 is the main lever (~1.5-2x on GPU). See §8.
python -m pipeline run --video clip.mp4 --fp16 --fuse

# --parallel adds decode/compute overlap (small extra gain; NOT a substitute
# for --fp16 — batching alone does not speed up the saturated detector).
python -m pipeline run --video clip.mp4 --fp16 --parallel --batch-size 16

# Drop one refine component (at least one must remain)
python -m pipeline run --video clip.mp4 --no-split      # connect/merge only
python -m pipeline run --video clip.mp4 --no-connect     # split only

# Tune refine params (see the cache caveat in §5 — pair these with --force-stage2)
python -m pipeline run --video clip.mp4 --eps 0.5 --merge_dist_thres 0.35 --force-stage2
```

Full option list: `python -m pipeline run --help`.

---

## 4. Run EACH STAGE independently (from cached intermediates)

Sometimes you want to re-run just one stage (e.g. re-tune refinement without
re-tracking). Call the stage scripts directly — each runs **in its own CWD** and
takes the **same** `--video` / `--artifacts-dir` so they share the artifact tree.

**Stage 1 (tracking)** — CWD `Deep-EIoU/Deep-EIoU`:

```bash
cd Deep-EIoU/Deep-EIoU
python tools/stage1_track.py \
    --video /abs/path/clip.mp4 \
    --artifacts-dir /abs/path/artifacts \
    --device gpu --fp16          # --fp16 = main throughput lever (see §8)
# parallel variant (decode/compute overlap on top of --fp16):
python tools/stage1_track.py --video /abs/path/clip.mp4 \
    --artifacts-dir /abs/path/artifacts --fp16 --parallel --batch-size 16
```

**Stage 2 (refine)** — CWD `gta-link`:

```bash
cd gta-link
python stage2_refine.py \
    --video /abs/path/clip.mp4 \
    --artifacts-dir /abs/path/artifacts \
    --use_split --use_connect
```

> Use **absolute** `--video` / `--artifacts-dir` here. Because each script runs
> in a different CWD, a relative path resolves to a different place per stage.
> (The orchestrator does this for you automatically.)

Stage 2 reuses Stage 1's embeddings from `tracklets.pkl`: it loads **no** ReID
model and **never re-reads the video frames** — `--video` is used only to locate
the artifact directory.

---

## 5. Force recompute — and the param/cache caveat

By default each stage skips work when its output is newer than its inputs. To
force a recompute:

```bash
python -m pipeline run --video clip.mp4 --force-all       # both stages
python -m pipeline run --video clip.mp4 --force-stage1    # tracking only
python -m pipeline run --video clip.mp4 --force-stage2    # refine only
```

(Directly: pass `--force` to `stage1_track.py` / `stage2_refine.py`.)

> ### ⚠️ The cache keys on input MTIME, **not** on parameters
>
> Caching compares **file modification times** only. Changing a **parameter**
> (a tracking threshold, `--eps`, `--min_samples`, `--merge_dist_thres`, …) does
> **not** change any input file's mtime, so it does **NOT** invalidate the
> cache — a re-run with new params will be **silently skipped** and you'll keep
> the old output.
>
> **When re-tuning, always add the matching force flag:**
> - changed Stage-1 params → `--force-stage1` (or `--force-all`)
> - changed Stage-2 refine params → `--force-stage2` (or `--force-all`)

---

## 6. Verify outputs (Colab)

### 6a. Count tracks before vs after refinement

Refinement should yield **fewer, cleaner** IDs (ID-switches split & re-merged):

```bash
cd Deep-EIoU/Deep-EIoU
python tools/count_tracks.py /abs/path/artifacts/<stem>/01_track/tracks.txt
python tools/count_tracks.py /abs/path/artifacts/<stem>/02_refine/refined.txt
# ignore short-lived (noise) tracks:
python tools/count_tracks.py /abs/path/artifacts/<stem>/02_refine/refined.txt --min_len 5
```

Expect the `unique tracks` count on `refined.txt` to be **lower / cleaner** than
on `tracks.txt`.

### 6b. Eyeball the tracks (render an annotated video)

Both `tracks.txt` and `refined.txt` use **0-based** frames (do **not** pass
`--one_indexed`):

```bash
cd Deep-EIoU/Deep-EIoU
python tools/render_from_txt.py --path /abs/path/clip.mp4 \
    --txt /abs/path/artifacts/<stem>/01_track/tracks.txt
python tools/render_from_txt.py --path /abs/path/clip.mp4 \
    --txt /abs/path/artifacts/<stem>/02_refine/refined.txt
```

Each writes `<txt>_rendered.mp4` next to the `.txt`. (`render_from_txt.py` is
cv2+numpy only — no GPU.)

### 6c. Parallel == Sequential equivalence check

Run Stage 1 both ways into separate artifact dirs, then diff the two `tracks.txt`
**tolerantly** (parity is floating-point nondeterministic, so a strict byte-diff
is the *wrong* tool):

```bash
# sequential
python -m pipeline run --video clip.mp4 --artifacts-dir /tmp/seq --force-stage1
# parallel
python -m pipeline run --video clip.mp4 --artifacts-dir /tmp/par --parallel --batch-size 16 --force-stage1

python -m pipeline compare \
    --a /tmp/seq/<stem>/01_track/tracks.txt \
    --b /tmp/par/<stem>/01_track/tracks.txt \
    --tol 1.0
```

`compare` matches rows by `(frame, id)` and reports key-set differences plus the
max/mean absolute box-coordinate difference. It prints **EQUIVALENT** (exit code
0) when the key sets match and the max coord diff is ≤ `--tol` (default 1.0 px),
otherwise **NOT EQUIVALENT** (exit code 1). A handful of px of jitter is expected
and fine; large coord diffs or differing key sets indicate a real divergence.

---

## 7. Inspect profiling

Each stage writes `profiles/<stage>.json`; the orchestrator aggregates them into
`profiles/summary.json` + `profiles/summary.md`.

Per-stage JSON fields:

| field | meaning |
|---|---|
| `stage` | stage id (`01_track` / `02_refine`) |
| `wall_time_s` | wall-clock seconds for the stage body |
| `gpu_peak_mb` | peak CUDA memory (MB); `null` without a GPU |
| `cpu_rss_start_mb` / `cpu_rss_end_mb` | process RSS (MB) before/after (`null` if `psutil`/`resource` unavailable) |
| `cpu_peak_mb` | peak RSS (MB) when available, else `null` |
| `io` | caller metadata: frame counts, track counts, `emb_dim`, `mode` (sequential/parallel), `batch_size`, refine params, … |

```bash
cat /abs/path/artifacts/<stem>/profiles/summary.md       # human-readable table
python -c "import json,sys; print(json.load(open(sys.argv[1]))['totals'])" \
    /abs/path/artifacts/<stem>/profiles/summary.json     # totals (wall, gpu peak)
```

`summary.md` has one row per stage plus a **TOTAL** row (summed wall time, max
GPU peak). Use the `io` block to compare a sequential vs a parallel run (same
`n_output_rows`/`n_unique_tracks`, different `wall_time_s`).

---

## 8. Throughput: `--fp16` first, then parallelization

### 8a. `--fp16` is the main throughput lever

The Stage-1 detector (YOLOX-x, ~99M params / ~793 GFLOPs **per frame**) is the
bottleneck, and at this input size it **saturates the GPU at batch size 1**.
Half-precision inference is the single biggest win:

```bash
python -m pipeline run --video clip.mp4 --device gpu --fp16
# combine with --fuse (fuses conv+BN; small free gain):
python -m pipeline run --video clip.mp4 --device gpu --fp16 --fuse
```

On a modern GPU (T4/A100) `--fp16` typically gives **~1.5–2× Stage-1 throughput**
and roughly halves activation memory, with no meaningful accuracy change for
inference. Try this **before** reaching for `--parallel`.

### 8b. What `--parallel` actually buys (and what it doesn't)

Stage 1 has two paths producing the **same artifact contract**:

- **sequential** (default): one frame at a time — decode → detect → ReID → track,
  all on one thread.
- **`--parallel --batch-size N`**: a **3-stage pipeline** — a prefetch thread
  decodes frames, a producer thread runs **batched** detect+ReID on the GPU, and
  the calling thread runs the tracker in strict frame order. So the GPU computes
  batch *N+1* while the tracker drains batch *N* (decode ∥ GPU ∥ track).

**What it does NOT do — batching does not speed up a compute-saturated detector.**
Because one frame already maxes out the GPU's compute, batching N frames runs the
same total FLOPs with no idle time to fill: throughput stays flat while VRAM
grows ~N×. The pipeline's win is **overlap**, not batching — it hides the
per-frame CPU tail (decode + tracking + postprocess) behind GPU compute instead
of running it serially.

How big is that overlap win? It equals the share of wall time spent **off** the
GPU. On a T4 in FP32 the detector dominates (GPU ~80% busy), so there is only
~20% tail to hide and the early measured gain was small (~5%). **Under `--fp16`
the detector gets ~2× cheaper but the CPU tail does not, so the tail becomes a
much larger share — and the pipeline overlap matters more.** Rule of thumb:
the lower your `sm%` during a run, the more `--parallel` buys you.

> The "fps" printed in parallel mode counts only the cheap CPU tracking step, not
> the GPU work — it is **not** the pipeline rate. Judge speed by the
> **Stage 1 wall** line at the end of the run.

**Use `--parallel` when** there is a non-trivial off-GPU tail to hide — i.e.
`sm%` is well under 100 (you're running `--fp16`, a fast GPU, high-res decode,
or heavy tracking) and you have VRAM to spare. If `sm%` is pinned near 100,
the GPU is the wall and overlap has nothing to reclaim; reach for `--fp16` first.

**Choosing `--batch-size` against VRAM:** batch memory scales roughly linearly
with batch size. Start at `8`, raise toward `16` only while watching
`gpu_peak_mb` in the profile and `nvidia-smi`; if you hit OOM, halve it. Batches
larger than what keeps the GPU continuously fed add latency-to-first-output and
VRAM with **no** throughput gain.

**Prefer sequential when:**
- **debugging** — simplest control flow, deterministic ordering, easiest to read.
- **very short clips** — the prefetch/batching overhead isn't amortized.
- **low VRAM** — batching can OOM; sequential has the smallest footprint.
- **strict reproducibility** — sequential is the behavior reference.

> **FP-nondeterminism caveat:** the parallel path is equal to sequential only
> *up to floating-point nondeterminism*. Batched GPU matmul/NMS kernels can
> differ from single-image ones in the last FP bits, which can **rarely** flip a
> detection sitting exactly on a confidence/NMS threshold. So `--parallel` output
> is **not** bit-identical to sequential. Verify equivalence with the tolerant
> `python -m pipeline compare` check (§6c), **not** a strict diff. If you need
> exact reproducibility, use the sequential path.

---

## 9. End-to-end GPU smoke (manual, Colab)

The authoritative correctness check runs on a GPU box (this repo's host has no
GPU). On Colab, with the two checkpoints in place:

```bash
# full pipeline on a short clip
python -m pipeline run --video clip.mp4

# verify
python Deep-EIoU/Deep-EIoU/tools/count_tracks.py artifacts/clip/01_track/tracks.txt
python Deep-EIoU/Deep-EIoU/tools/count_tracks.py artifacts/clip/02_refine/refined.txt
python Deep-EIoU/Deep-EIoU/tools/render_from_txt.py --path clip.mp4 --txt artifacts/clip/02_refine/refined.txt

# parallel == sequential
python -m pipeline run --video clip.mp4 --artifacts-dir /tmp/seq --force-stage1
python -m pipeline run --video clip.mp4 --artifacts-dir /tmp/par --parallel --batch-size 16 --force-stage1
python -m pipeline compare --a /tmp/seq/clip/01_track/tracks.txt --b /tmp/par/clip/01_track/tracks.txt
```

# Task 002 — Consolidate & restructure the docs

## Context
Two run-docs exist: `USER_GUIDE.md` (general, CLI-focused) and `COLAB_GUIDE.md` (Colab runbook).
The Colab runbook is now redundant with the self-contained `run_pipeline_colab.ipynb`, but
COLAB_GUIDE carries **environment-agnostic** caveats the notebook omits. Decision: collapse to
**two artifacts** — `USER_GUIDE.md` (canonical reference) + the notebook — and delete
`COLAB_GUIDE.md`, folding its general content into USER_GUIDE.

Read both `USER_GUIDE.md` and `COLAB_GUIDE.md` fully before starting. Preserve every factual/
technical detail (cache caveat, `--fp16` numbers, artifact contract, per-stage CWD rules,
torchreid caveat). The owner dislikes redundancy — reorganize, don't bloat.

## Objective
Rewrite `USER_GUIDE.md` into two clearly delimited parts, absorb COLAB_GUIDE's general content,
add a from-scratch-VM + Azure section, then delete `COLAB_GUIDE.md` and fix any links to it.

## Part A — Quickstart (the "just run it" path)
1. **Prerequisites & from-scratch setup** (this is new — Colab preinstalls most of it, a bare VM
   does not):
   - A CUDA GPU + NVIDIA driver; verify with `nvidia-smi`.
   - Python 3 + a virtualenv.
   - System libs commonly missing on headless Linux: `libgl1` (and `libglib2.0-0`) for OpenCV —
     `sudo apt-get install -y libgl1 libglib2.0-0`.
   - `torch` / `torchvision` (Colab has them preinstalled; from scratch, install a CUDA build).
   - The exact pip sequence (from COLAB_GUIDE §3):
     ```
     pip install -r gta-link/requirements.txt
     pip install cython_bbox tqdm ultralytics
     pip install -e Deep-EIoU/Deep-EIoU/reid
     ```
   - **The vendored-`reid`/torchreid caveat** (absorb from COLAB_GUIDE): install the editable
     vendored `reid` package; do NOT `pip install torchreid` from PyPI (incompatible layout →
     `ModuleNotFoundError: torchreid.utils`).
   - **Checkpoints**: `yolov11l.pt` + `sports_model.pth.tar-60` must sit in
     `Deep-EIoU/Deep-EIoU/checkpoints/`.
2. **Minimal end-to-end run** + where outputs land:
   ```
   python -m pipeline run --video clip.mp4 --artifacts-dir ./out --device gpu --fp16 --fuse
   ```
3. **Key options that matter** (short list): `--artifacts-dir`, `--device`, `--fp16` / `--fuse`,
   `--detector-ckpt`, `--force-all/-stage1/-stage2`. Full list → `python -m pipeline run --help`.
4. **Run on a fresh GPU VM (e.g. Azure)** subsection — NO bespoke script:
   - Auth via connection string: `export AZURE_STORAGE_CONNECTION_STRING="..."`.
   - Pull inputs/checkpoints/GT from Blob, run on **local disk**, push outputs back. Use
     `az storage blob download` / `download-batch` / `upload-batch` (they honor
     `AZURE_STORAGE_CONNECTION_STRING`). Example:
     ```
     az storage blob download       --container-name videos -n clip.mp4 -f ./clip.mp4
     az storage blob download-batch  -s checkpoints -d Deep-EIoU/Deep-EIoU/checkpoints
     python -m pipeline run --video ./clip.mp4 --artifacts-dir ./out --device gpu --fp16 --fuse
     az storage blob upload-batch    -d outputs/<stem> -s ./out/<stem>
     ```
   - Note: artifacts go to fast local disk during the run, then sync to Blob (matches the I/O
     design); long runs survive disconnects via `tmux`/`nohup`.

## Part B — Reference / deep-dive
Reorganize the current USER_GUIDE technical content (drop Colab-specific framing — that lives in
the notebook):
- **Architecture**: two stages run as separate subprocesses in their own CWDs; include the
  "one environment, two working directories" table absorbed from COLAB_GUIDE.
- **Artifact contract** (current §1).
- **Run each stage independently** (current §4) — keep the absolute-path caveat.
- **Exhaustive options**: the full orchestrator flag set, PLUS the per-stage-only flags that are
  NOT exposed on `python -m pipeline run`:
  - `--batch-size` (Stage 1, `tools/stage1_track.py`) — perception batch window, default 16.
  - `--keep-features` (Stage 2, `stage2_refine.py`) — export the full `refined_tracklets.pkl`
    instead of the default slim one.
  Mark these clearly as **stage-script flags** (the orchestrator uses their defaults). Verify
  every documented flag against the actual parsers (`pipeline/__main__.py`,
  `tools/stage1_track.py`, `stage2_refine.py`) — do NOT document flags that don't exist.
- **Cache caveat** (current §5) — mtime, not params; matching `--force-*`.
- **Profiling** (current §7) — summary fields table.
- **`--fp16` throughput** (current §8).
- **Verify outputs** (current §6) — `count_tracks.py`, `render_from_txt.py`; drop the "(Colab)"
  framing, it's environment-agnostic.
- **Dependency rationale + troubleshooting** absorbed from COLAB_GUIDE (the env/CWD explanation,
  GPU-OOM, checkpoint FileNotFound, slow-disk→local-artifacts tip, cache-ignored, wrong
  detections). Drop Colab-only items (drive.mount, gdown public folder, "Colab Notebooks" path
  quoting).

## Delete COLAB_GUIDE.md
- `git rm COLAB_GUIDE.md`.
- `grep -rn "COLAB_GUIDE" .` and fix/remove every reference (the notebook, README, other docs) so
  no dangling link remains. If a reference pointed Colab users to COLAB_GUIDE, repoint to the
  notebook or the relevant USER_GUIDE section.

## Non-goals / Later
- Do NOT add a script or a new guide.
- Do NOT change pipeline code. (If you notice `--batch-size`/`--keep-features` aren't reachable
  from the orchestrator, just document them as stage-script flags — wiring them through is a
  separate code task, out of scope.)
- Do NOT touch the notebook's cells (Task 001 already made it Drive-only); only fix a
  COLAB_GUIDE link in it if one exists.

## Acceptance criteria
- `USER_GUIDE.md` has a clear Quickstart part (prereqs/from-scratch + minimal run + key options +
  fresh-VM/Azure subsection) and a Reference part (exhaustive options incl. the per-stage flags,
  architecture, contract, cache, profiling, fp16, verify, troubleshooting).
- All environment-agnostic content from COLAB_GUIDE is preserved in USER_GUIDE.
- `COLAB_GUIDE.md` is deleted and `grep -rn "COLAB_GUIDE" .` returns nothing (no dangling links).
- Every documented CLI flag exists in the corresponding parser.

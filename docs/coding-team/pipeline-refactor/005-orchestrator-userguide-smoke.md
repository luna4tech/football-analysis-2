# Task 005 — Orchestrator + verify tooling + USER_GUIDE + CPU smoke

## Context
Tasks 1–4 delivered the pipeline helpers, Stage 1 (`Deep-EIoU/Deep-EIoU/tools/stage1_track.py`,
sequential + `--parallel`), and Stage 2 (`gta-link/stage2_refine.py`). Each stage self-caches and
writes a profile JSON. This task adds the top-level orchestrator that chains them, a tolerant
tracks-diff tool for the parallel-vs-sequential check, the user guide, and CPU-runnable tests.

Architecture reminder: stages run as SUBPROCESSES in their own CWD (Stage 1 in
`Deep-EIoU/Deep-EIoU`, Stage 2 in `gta-link`) so the two bundled `torchreid` copies never clash.
The orchestrator runs from the repo root.

## Objective
`python -m pipeline run --video clip.mp4 [...]` runs Stage 1 then Stage 2 over the shared artifact
contract, honoring caching/force, and writes an aggregated profile summary. Plus
`python -m pipeline compare ...` for the equivalence check, and a `USER_GUIDE.md`.

## Scope (do now)

### `pipeline/orchestrator.py`
- `run(video, opts, runner=<subprocess default>)`:
  - Resolve an ABSOLUTE artifacts base and an absolute video path up front, and compute
    `paths = get_artifact_paths(video, base_dir)` so the orchestrator knows `profiles_dir`. Pass the
    SAME absolute `--video` and `--artifacts-dir` to BOTH stages — do NOT rely on each stage's
    relative default (it resolves differently under each stage's CWD). `ensure_dirs(paths)`.
  - Build + run Stage 1: `[sys.executable, "tools/stage1_track.py", "--video", <abs>,
    "--artifacts-dir", <abs>, ...forwarded]` with `cwd=<repo>/Deep-EIoU/Deep-EIoU`.
    Forward: `--device`, and if parallel `--parallel --batch-size N`; add `--force` when stage 1 is
    forced.
  - Build + run Stage 2: `[sys.executable, "stage2_refine.py", "--video", <abs>,
    "--artifacts-dir", <abs>, ...refine params]` with `cwd=<repo>/gta-link`.
    Forward the refine params (use_split/use_connect/eps/min_samples/max_k/min_len/spatial_factor/
    merge_dist_thres); add `--force` when stage 2 is forced.
  - Stages inherit the parent's stdout/stderr (stream live; do NOT capture) so the user sees
    progress. After each, check the return code; if non-zero, ABORT before the next stage and raise
    a clear error naming the failed stage. (Stage 1 failure must not run Stage 2.)
  - On success, `write_summary(paths.profiles_dir)` and print a short summary (per-stage wall time +
    the summary.md path).
  - The `runner` must be INJECTABLE (default = a thin wrapper over `subprocess.run`) so tests can
    assert the built commands without launching anything.
- Default to the FULL pipeline: enable BOTH `--use_split` and `--use_connect` for Stage 2 unless the
  user disables one (Stage 2 requires at least one).

### Force semantics
- `--force-all` → pass `--force` to both stages.
- `--force-stage1` / `--force-stage2` → force only that stage.
- IMPORTANT caveat to encode + document: the per-stage cache keys on input-file MTIME, so changing
  PARAMS (tracking thresholds, eps, etc.) does NOT invalidate the cache. Re-tuning therefore requires
  the matching `--force-*`. Surface this in `--help` and prominently in the user guide.

### `pipeline/__main__.py`
- `python -m pipeline run ...` (the orchestrator CLI: `--video` required, `--artifacts-dir`,
  `--device`, `--parallel`, `--batch-size`, `--no-split`, `--no-connect`, refine params, the force
  flags above).
- `python -m pipeline compare --a A.txt --b B.txt [--tol 1.0]` → see below.

### `pipeline/compare_tracks.py` (pure stdlib+numpy, torch/cv2-free)
Tolerant MOT diff for the parallel-vs-sequential equivalence check. Parse two MOT `.txt` files,
match rows by `(frame, id)`, and report: row-count match, set of `(frame,id)` keys only in A / only in
B, and the max/mean absolute box-coordinate difference over common keys; return "equivalent" when
key sets match and max coord diff ≤ `tol` (default 1.0 px). Expose `compare_tracks(a, b, tol)` and a
CLI entry used by `python -m pipeline compare`. (Parity is FP-nondeterministic, so a strict diff is
the wrong tool; this tolerant one is correct.)

### `USER_GUIDE.md` (repo root)
Must cover, with copy-paste commands (Colab/GPU oriented):
- The artifact contract + a one-line dataflow diagram.
- Prerequisites: shared env, GPU, the model checkpoints each stage needs and where they live
  (detector `checkpoints/best_ckpt.pth.tar`, ReID `checkpoints/sports_model.pth.tar-60`).
- Run the FULL pipeline: `python -m pipeline run --video clip.mp4` (+ common options).
- Run EACH STAGE independently from cached intermediates (the two stage scripts directly, with their
  CWD and the shared `--artifacts-dir`); note Stage 2 reuses Stage 1's embeddings, so it loads no
  ReID model and never re-reads frames.
- Force recompute: `--force-all` / `--force-stage1` / `--force-stage2`, AND the params-don't-bust-the-
  cache caveat.
- Verify outputs (Colab): `count_tracks.py` on `tracks.txt` vs `refined.txt` (expect fewer/cleaner
  IDs); `render_from_txt.py` to eyeball both (no `--one_indexed`, frames are 0-based); the
  parallel==sequential check via `python -m pipeline compare`.
- Inspect profiling: the `profiles/*.json` fields (time/GPU/CPU/IO) and `summary.md`.
- WHEN to use parallelization: throughput vs latency; choosing `--batch-size` against VRAM; when
  sequential is better (debugging, very short clips, low VRAM, strict reproducibility); expected wins
  and the FP-nondeterminism caveat.

### Tests (CPU-only, runnable here)
- Orchestrator: with an injected fake runner, assert (a) Stage 1 command, cwd, and forwarded args
  (incl. `--parallel --batch-size`) are correct; (b) Stage 2 command/cwd/refine-params correct;
  (c) both get absolute `--video`/`--artifacts-dir`; (d) `--force-all` adds `--force` to both, and
  `--force-stage2` only to Stage 2; (e) a non-zero Stage 1 return aborts before Stage 2; (f)
  `write_summary` is invoked on success (use fake profile JSONs to check aggregation).
- `compare_tracks`: identical files → equivalent; within-tol coord jitter → equivalent; out-of-tol or
  mismatched keys → not equivalent.

## Non-goals / Later
- No in-process stage execution (subprocess only). No new stage logic. No changes to
  Deep-EIoU/gta-link source. The real GPU end-to-end smoke is a documented manual Colab step, not run
  here.

## Constraints / Caveats
- Absolute paths everywhere crossing the CWD boundary.
- Use `sys.executable` for the subprocess interpreter (shared env).
- Keep `compare_tracks` and the orchestrator import-light (no torch) so they run on this host.

## Acceptance criteria
- `python -m pipeline run` chains both stages as subprocesses with correct cwd, absolute paths,
  forwarded args, and force propagation; aggregates the profile summary; aborts on a stage failure.
- `python -m pipeline compare` does a tolerant MOT diff.
- `USER_GUIDE.md` covers every bullet above.
- The orchestrator-wiring and compare CPU tests pass here without torch/cv2.
- Report what ran here vs the deferred real Colab smoke.

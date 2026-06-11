# Task 001 — Scaffold `pipeline/` package: profiling + artifact/caching helpers

## Context
We are building a thin orchestration layer over two existing, untouched subprojects
(`Deep-EIoU/`, `gta-link/`). This task lays the shared foundation only — no stage logic, no
changes to DeepEIoU/GtaLink. See `docs/coding-team/pipeline-refactor/PLAN.md` for the full
plan and the artifact contract.

Hard reality: this machine is Windows with no GPU and the heavy deps (torch, OpenCV, the two
torchreid copies) are likely not installed. So **everything you write in this task must
import and unit-test without torch, CUDA, or a GPU.** Anything torch-related must be guarded
behind `torch.cuda.is_available()`-style checks and a lazy/optional import.

## Objective
Create a new top-level `pipeline/` Python package providing the cross-cutting utilities every
stage will use: per-stage profiling, the artifact directory layout, and mtime-based caching.

## Scope (do now)
Create `pipeline/` at the repo root with roughly:

- `pipeline/__init__.py`
- `pipeline/artifacts.py` — the on-disk contract helpers:
  - resolve the artifacts root for a given video path (default `artifacts/<video_stem>/`,
    overridable via an `--artifacts-dir` style argument later; for now a function that takes
    a video path + optional base dir and returns a small paths object/dataclass exposing:
    `track_dir`, `tracks_txt`, `tracklets_pkl`, `refine_dir`, `refined_txt`, `profiles_dir`,
    and `profile_json(stage_name)`).
  - `ensure_dirs(...)` to create the directories.
- `pipeline/cache.py` — caching helpers:
  - `is_stale(output_path, input_paths) -> bool`: True if output is missing or older than any
    input (mtime based). Missing inputs should be handled sanely (document the choice).
  - a small helper that, given `(output_path, inputs, force)`, returns whether the stage
    should run, so stages share identical caching semantics.
- `pipeline/profiling.py` — profiling:
  - A context manager (e.g. `profile_stage(name, artifacts, extra=None)`) that measures:
    - wall time (seconds),
    - peak GPU memory in MB — only if torch is importable AND `torch.cuda.is_available()`;
      call `reset_peak_memory_stats()` at entry and `max_memory_allocated()` at exit;
      otherwise record `null`/"N/A",
    - CPU memory: process RSS at entry/exit and peak if available, via `psutil` if importable,
      else fall back to `tracemalloc` or `resource`, else `null`. Guard the import.
    - caller-supplied IO metadata: allow the caller to attach arbitrary key→value (e.g.
      `{"n_frames": 300, "emb_shape": [25, 512]}`) that lands in the JSON.
  - On exit, write a JSON profile to `artifacts/.../profiles/<name>.json` with at least:
    stage name, wall_time_s, gpu_peak_mb, cpu_rss_start_mb/cpu_rss_end_mb (and peak if
    available), and the IO dict. Use a fixed, documented schema.
  - `write_summary(profiles_dir)` — read all `*.json` profiles in the dir (excluding
    `summary.json`) and write `summary.json` + a human-readable `summary.md` table.
- `pipeline/tests/` — pure-Python unit tests (plain `python`-runnable, no pytest dependency
  required; a `if __name__ == "__main__"` runner or simple asserts is fine) covering:
  - artifact path layout is correct for a sample video path,
  - `is_stale` true/false cases using temp files with controlled mtimes,
  - `profile_stage` produces a valid JSON file with the documented keys when run on CPU
    (no torch/psutil present must still work),
  - `write_summary` aggregates two fake profile JSONs into summary.{json,md}.

Keep it small and readable. Prefer the standard library + numpy only.

## Non-goals / Later
- No edits to `Deep-EIoU/` or `gta-link/`.
- No stage entrypoints, no orchestrator CLI, no `argparse` wiring of the real stages — those
  are Tasks 2–5.
- No subprocess launching yet.
- Do not add torch or psutil to any install requirement; treat both as optional at runtime.

## Constraints / Caveats
- Must import and run on CPU-only with neither torch nor psutil installed.
- Windows path-safe (use `os.path`/`pathlib`, no hardcoded `/`).
- The JSON profile schema you choose here is the contract the later stages and the user guide
  depend on — keep it stable and document the keys in a module docstring.
- GPU memory and CPU memory must degrade to a clearly-marked "unavailable" value rather than
  crashing when the measurement backend is missing.

## Acceptance criteria
- The package imports with only the standard library + numpy available.
- The unit tests run and pass under a plain `python` interpreter with no torch/psutil/GPU.
- After review, report to the architect exactly which tests you executed and any you could
  not (and why).

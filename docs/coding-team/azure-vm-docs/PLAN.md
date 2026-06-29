# Azure-VM support & doc consolidation — Plan

Goal: make the pipeline easy to run on a from-scratch GPU VM (e.g. Azure) **without** a bespoke
script or a third guide, and tidy the docs to a clean two-artifact world.

## Decisions (locked with the owner)
- **No Azure runner script.** `python -m pipeline run` is environment-agnostic; the only
  Azure-specific step is moving files to/from Blob, done with documented `az storage blob`
  commands. Auth = connection string. Artifacts written to local VM disk, synced to Blob.
- **Two docs only:** `USER_GUIDE.md` (canonical reference) + the self-contained
  `run_pipeline_colab.ipynb`. `COLAB_GUIDE.md` is deleted, its general content folded into
  USER_GUIDE.
- Commit after each task with a readable message; proceed without pausing for review.
- Branch: `class-team-inference`.

## Tasks
1. **001 — Revert Azure from `run_pipeline_colab.ipynb`.** Restore the notebook to its pre-Azure
   state (`1e68ad5`): pure Colab + Google Drive, no `VIDEO_BACKEND` / `AZURE_*` / download-upload
   branches / Azure markdown. Only the notebook changes; `render_from_txt.py` and `.gitignore`
   stay. (Mechanical git revert performed by the architect + verified.)
2. **002 — Consolidate & restructure the docs.** Rewrite `USER_GUIDE.md` into a **Quickstart**
   part (from-scratch prerequisites incl. the vendored-`reid`/torchreid caveat absorbed from
   COLAB_GUIDE, minimal end-to-end run, key options, and a short "run on a fresh GPU VM / Azure
   via `az` CLI" subsection) and a **Reference / deep-dive** part (artifact contract, per-stage
   runs, exhaustive options, cache caveat, profiling, `--fp16`, verify). Delete `COLAB_GUIDE.md`
   and fix any links to it.

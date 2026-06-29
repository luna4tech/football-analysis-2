# Task 001 — Revert Azure from `run_pipeline_colab.ipynb`

## Context
Two commits added an optional Azure backend to the Colab notebook:
- `c9b04c6` "support azure for video input" — notebook + `render_from_txt.py` (line-thickness
  defaults, NOT azure) + `.gitignore` (`output-colab/`, NOT azure).
- `db54144` "rendered video upload to azure" — notebook only.

These are the ONLY two commits to touch `run_pipeline_colab.ipynb` since `1e68ad5`, so its
pre-Azure state is exactly the notebook at `1e68ad5`.

## Objective
`run_pipeline_colab.ipynb` becomes pure Colab + Google Drive again: no `VIDEO_BACKEND` switch, no
`AZURE_*` config / connection-string prompt, no download/upload branches (step 1 + step 6b), no
Azure markdown.

## How
`git checkout 1e68ad5 -- run_pipeline_colab.ipynb`, then verify the diff vs HEAD shows ONLY Azure
removal (the round-trip restores the known-good Drive-only notebook).

## Non-goals
- Do NOT revert `render_from_txt.py` (line-thickness 1→2) or `.gitignore` (`output-colab/`) —
  unrelated to Azure, merely bundled in `c9b04c6`.
- No content changes beyond removing Azure.

## Acceptance criteria
- The notebook has no Azure references and runs Drive-only end to end.
- `git diff 1e68ad5 -- run_pipeline_colab.ipynb` is empty after the revert.

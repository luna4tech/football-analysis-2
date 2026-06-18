# Task E4 — Add eval cells to the Colab notebook (GT from Drive) + gitignore test-data

## Context
`run_pipeline_colab.ipynb` (repo root) runs the pipeline on Colab: mount Drive → set paths → clone →
deps → checkpoints → run pipeline → render → verify → (final markdown) "Output layout". It uses a
`!`-shell + `$ENV` style and these vars (set in the paths cell and exported to `os.environ`):
`DRIVE_BASE`, `INPUT_DIR`, `OUTPUT_DIR`, `CKPT_DIR`, `REPO`, `VIDEO`, `STEM`.

We now have a standalone eval (`eval/`): `python -m eval.evaluate` scores `refined.txt` vs a
MOTChallenge `gt.txt` using TrackEval (HOTA/MOTA/IDF1). It is CPU-only and reads files post-hoc.
The user's GT lives on Google Drive and their GT has `class=-1` — which the eval already handles
(loader coerces `-1->1`; the E3 fix writes `class=1` into the gt.txt TrackEval reads). So the default
`--gt-format motchallenge` works; NO `--gt-format` flag and NO data edits are needed.

The user will NOT commit any sample data — `test-data/` must be gitignored.

## Objective
Add an "Evaluate" section to `run_pipeline_colab.ipynb` that installs TrackEval, points at the GT on
Drive, runs the eval, and shows the metrics — and add `test-data/` to `.gitignore`.

## Scope (do now)

### 1. `.gitignore`
Append an entry so the local sample data is never committed:
```
# Local sample data (videos + GT), never committed
test-data/
```

### 2. `run_pipeline_colab.ipynb` — insert a new "## 8. Evaluate (HOTA / MOTA / IDF1)" section
Insert AFTER the current section 7 "Verify" code cell and BEFORE the final "## Output layout"
markdown cell. Match the existing notebook style (markdown + code cells; `!` shell using `$ENV`
vars; keep it runnable top-to-bottom in the same GPU session — eval itself is CPU-only but reuses
the session). Cells:

- **8.0 (markdown)**: Eval is standalone + CPU-only; scores `refined.txt` against ground truth from
  Drive via TrackEval (HOTA/MOTA/IDF1 + DetA/AssA). Note GT must be a MOTChallenge `gt.txt`
  (1-based); the pipeline's 0-based output is converted automatically; `class=-1` GT is handled
  (no flag needed). Tip: for judging GtaLink's refinement, watch IDF1 + AssA, not MOTA.

- **8.1 (code)** — install TrackEval (clone, not pip) + scipy:
  ```python
  !git clone -q https://github.com/JonathonLuiten/TrackEval.git /content/TrackEval
  !pip install -q scipy
  ```

- **8.2 (code)** — point at the GT on Drive (user edits the path; GT naming may not match the video
  stem, so make it explicit + helpful):
  ```python
  GT_DIR = f"{DRIVE_BASE}/ground-truth"
  GT = f"{GT_DIR}/gt_mot_{STEM}.txt"   # <-- EDIT to your GT file for THIS clip (MOTChallenge gt.txt)
  os.environ["GT"] = GT
  if not os.path.exists(GT):
      print("GT not found:", GT, "\nAvailable in", GT_DIR, ":")
      !ls -la "$GT_DIR" 2>/dev/null || echo "  (folder missing — create it and upload your gt.txt)"
      raise FileNotFoundError(GT)
  print("Using GT:", GT)
  ```

- **8.3 (code)** — run the eval and show the metrics:
  ```python
  %cd $REPO
  !python -m eval.evaluate \
      --pred "$OUTPUT_DIR/$STEM/02_refine/refined.txt" \
      --gt   "$GT" \
      --seq-name "$STEM" \
      --trackeval-path /content/TrackEval \
      --out  "$OUTPUT_DIR/$STEM/eval"
  import json
  print(json.dumps(json.load(open(f"{OUTPUT_DIR}/{STEM}/eval/metrics.json")), indent=2))
  ```

- **8.4 (markdown + code, optional)** — quantify GtaLink's gain (baseline vs refined; watch
  IDF1/AssA). Eval `01_track/tracks.txt` into `…/eval_baseline` and print both `metrics.json`.
  Mark clearly as optional.

### 3. Update the final "## Output layout" markdown
Add the eval output to the tree, e.g.:
```
  eval/metrics.json         # HOTA / DetA / AssA / MOTA / IDF1 (after step 8)
```

## Non-goals / Later
- No new GT format/adapter (default motchallenge works). No edits to `eval/` code, the pipeline,
  Deep-EIoU, or gta-link. Do NOT add `--gt-format`. Do NOT commit `test-data/`.

## Constraints / Caveats
- The notebook must remain valid (loads with `nbformat`/`json`); don't corrupt existing cells or
  their ids/structure. Only ADD the new cells + the one Output-layout edit.
- Use the exact eval CLI flags that exist: `--pred`, `--gt`, `--seq-name`, `--trackeval-path`,
  `--out` (verify against `eval/evaluate.py`). `--class-name` is only needed if the GT class label
  isn't pedestrian — not needed here; mention it as a comment only.
- Keep the cells copy-paste runnable with the notebook's existing env vars.

## Acceptance criteria
- `.gitignore` ignores `test-data/`.
- The notebook has a working Evaluate section (install TrackEval+scipy, GT-from-Drive with a clear
  edit point + existence check, run eval, print metrics.json, optional baseline-vs-refined) and the
  Output-layout tree mentions `eval/metrics.json`.
- The notebook still parses as valid JSON/nbformat; existing cells unchanged.
- Verify the eval CLI flags used match `eval/evaluate.py`.

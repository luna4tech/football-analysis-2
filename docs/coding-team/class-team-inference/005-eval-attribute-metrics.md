# Task 005 — Attribute (class + team) evaluation metrics

## Context
The `eval/` module scores `refined.txt` vs a MOT GT via TrackEval (HOTA/DetA/AssA/MOTA/IDF1) and
writes `metrics.json`. It is import-light (stdlib+numpy; TrackEval lazily imported / injectable).

The extended GT and our pred now carry semantic `class_id` in MOT col 8 and `team_id` in col 9.
Two existing behaviors collide with that:
- `eval/trackeval_runner.py::_format_gt_line` writes the canonical **class** (col idx 7) and
  **visibility** (idx 8) into the materialized `gt.txt`. TrackEval's MOTChallenge pedestrian eval
  keeps ONLY class==1 rows — so an extended GT (class 0/1/2) would drop all players (0) and
  referees (2) and break HOTA/MOTA.
- `eval/gt_adapters.py::_field_or_default` coerces `-1`→1 for class/visibility, and team (col 9)
  is parsed into the canonical **visibility** slot — so the canonical rows CANNOT be trusted for
  semantic class/team (team=-1 becomes 1, class default is 1). The TrackEval path only needs
  frame/id/bbox/conf, so this is fine for HOTA but useless for attribute eval.

Canonical class ids: `0=player, 1=goalkeeper, 2=referee`, unknown `-1`.

## Objective
Add class/team/consistency metrics computed against the extended GT, written to
`eval/.../attributes_metrics.json`, WITHOUT changing the existing TrackEval HOTA/MOTA results
(which must stay all-person). Degrade gracefully when GT/pred lack semantic class or team.

## Scope
- `eval/trackeval_runner.py::_format_gt_line`: write **class=1 and visibility=1** (constant
  pedestrian, fully visible) into the materialized `gt.txt`, decoupling it from the semantic
  class/team now present in the canonical rows. (For the current all-`-1` sample GT this is
  identical to today; for extended GT it is the fix that keeps HOTA all-person.) Update the
  docstring. Keep `_format_pred_line` as-is.
- New `eval/attributes.py` (pure; stdlib+numpy; no TrackEval/torch; sklearn only via lazy import):
  - Raw attribute parser: read a MOT txt → per-detection `(frame, id, x, y, w, h, class, team)`
    with **NO -1 coercion** (missing/blank → -1). Do not reuse `_field_or_default` for class/team.
  - Frame alignment: pred is 0-based, GT 1-based — shift pred frames +1 (same convention as
    `trackeval_runner._pred_to_motchallenge_rows`).
  - Per-frame greedy IoU matching (default IoU ≥ 0.5) of pred↔GT detections; aggregate matches to
    a per-pred-track → GT-track association by majority co-occurrence. (We use our own IoU
    matching, per the spec's "use IoU matching where ids are not guaranteed to align" — we do NOT
    extract TrackEval's internal matching.)
  - Pred per-track class/team: prefer the sibling `track_attributes.json` (the authoritative
    Stage-3 per-track values); fall back to per-frame majority (class) / constant (team) from the
    pred txt when the json is absent.
  - GT per-track class/team: majority of per-frame class; team per track (constant).
  - **Class metrics**: per-track accuracy + 3×3 confusion matrix over {player, goalkeeper,
    referee}, across pred tracks that matched a GT track. (Watch GK↔player, referee↔player.)
  - **Team metrics (players only)**: over pred player tracks matched to GT player tracks, compute
    best-of-2-permutation accuracy (pure) and, when sklearn is importable, ARI/NMI; exclude GK and
    referee.
  - **Consistency** (pred-only, always computable): per-track class purity (fraction of frames ==
    the track's aggregated class) and class switch count; note team is constant per track.
  - **Graceful degradation**: emit class metrics only when GT has semantic class (not all -1);
    team metrics only when both GT and pred have teams; matched metrics only when IoU matches
    exist. Skip the rest with a logged `reason`. Always include counts (matched tracks, players,
    etc.).
  - `run_attribute_eval(gt_path, pred_path, out_path, *, iou_thresh=0.5) -> dict` that writes the
    JSON and returns the metrics dict.
- `eval/evaluate.py`:
  - After the TrackEval run in `run_evaluation`, call `run_attribute_eval` and write
    `attributes_metrics.json` next to `metrics.json`. Make it **non-fatal** (a failure here must
    not break the HOTA path — log and continue).
  - Pred auto-location (`_resolve_pred_path`): prefer `paths.team_refined_txt` when it exists,
    else `paths.refined_txt` (explicit `--pred` always honored). Boxes/ids are identical between
    02_refine and 03_team, so HOTA is unaffected; this just makes class/team available.
- `run_pipeline_colab.ipynb`: point the eval cell's prediction at `03_team/refined.txt`
  (only that cell; leave the `count_tracks` cell alone).

## Non-goals / Later
- Do not change Stage 1/2/3 code, output formats, or the canonical `Tracks` shape (9 cols).
- Do not extract TrackEval's internal id matching; own IoU matching is the chosen approach.
- No silhouette/unsupervised guardrail in this task (optional future work).

## Constraints / Caveats
- The materialized `gt.txt` must keep HOTA/MOTA all-person — verify existing eval tests stay green
  after the `_format_gt_line` change (sample GT is all -1 → class=1, unchanged).
- Attribute parser must NOT coerce -1 (it is meaningful: unknown class / no team).
- Keep `eval/attributes.py` import-light and CPU-testable; lazy-import sklearn only for ARI/NMI.
- Team metrics must be permutation-invariant (k-means/GT team labels are arbitrary).

## Acceptance criteria
- With extended GT, HOTA/MOTA/IDF1 are unchanged vs. the all-person baseline (no rows dropped).
- `attributes_metrics.json` reports class accuracy + confusion matrix, players-only team
  accuracy (+ ARI/NMI when sklearn present), and per-track class consistency — or documented
  skip reasons when GT/pred lack the data.
- With the current all-`-1` sample GT, attribute eval skips class/team gracefully and still emits
  consistency + counts; the eval command exits 0 and `metrics.json` is unchanged.

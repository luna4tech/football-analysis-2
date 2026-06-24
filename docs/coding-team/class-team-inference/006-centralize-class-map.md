# Task 006 — Centralize the class-id map and switch to the GT scheme

## Context
Class ids are currently the convention `0=player, 1=goalkeeper, 2=referee` (unknown `-1`),
re-declared as local literals in four separate-subprocess packages (no shared imports), plus a
docstring and many test fixtures. The user's GT files use a different scheme:
**`1=goalkeeper, 2=player, 3=referee`**. Predictions must share the GT's numbering for the
attribute eval to compare correctly. The detector is resolved BY NAME (confirmed names:
`goalkeeper`/`player`/`referee`), so the predicted numbers are ours to choose.

## Objective
Introduce ONE shared config as the single source of truth and switch the whole pipeline to the
GT scheme `goalkeeper=1, player=2, referee=3` (unknown `-1`). Replace every hardcoded class-id
literal/constant with imports from the config. No behavior change other than the id values.

## Scope
- New repo-root module `class_map.py` (pure, zero dependencies):
  ```python
  CLASS_NAME_TO_ID = {"goalkeeper": 1, "player": 2, "referee": 3}
  UNKNOWN_CLASS = -1
  GOALKEEPER_CLASS = 1
  PLAYER_CLASS = 2
  REFEREE_CLASS = 3
  CLASS_ID_TO_NAME = {v: k for k, v in CLASS_NAME_TO_ID.items()}
  # ordered for the eval confusion matrix (keep the existing row/col order: player, gk, ref)
  CLASS_LABELS = (PLAYER_CLASS, GOALKEEPER_CLASS, REFEREE_CLASS)
  CLASS_NAMES = ("player", "goalkeeper", "referee")
  ```
- Repoint the four sites to import from `class_map` (each adds repo root to `sys.path` if not
  already on it — compute from `__file__`):
  - `Deep-EIoU/Deep-EIoU/tools/stage1_assembly.py`: replace `CLASS_UNKNOWN` and
    `_NAME_TO_CANONICAL` with the config (`CLASS_NAME_TO_ID`, `UNKNOWN_CLASS`). Name resolution
    already lowercases — keep that; keys in the config are lowercase.
  - `pipeline/team_assignment.py`: replace `PLAYER_CLASS/GOALKEEPER_CLASS/REFEREE_CLASS`.
  - `Deep-EIoU/Deep-EIoU/tools/render_from_txt.py`: replace the same three constants. Must stay
    importable WITHOUT cv2 (config is pure, fine).
  - `eval/attributes.py`: replace `PLAYER_CLASS/GOALKEEPER_CLASS/REFEREE_CLASS` and the
    `CLASS_LABELS`/`CLASS_NAMES`/unknown sentinel with the config. Keep confusion row/col order
    (player, gk, ref).
- `gta-link/Tracklet.py`: update the docstring (`class_ids` line) to the new scheme
  `1=goalkeeper, 2=player, 3=referee, -1=unknown`. No logic change.
- Update all test fixtures that hardcode `0/1/2` semantic class to the new scheme (e.g.
  `test_stage1_assembly.py` name→id expectations, `test_team_assignment.py`,
  `test_render_from_txt.py`, `eval/tests/test_eval.py`). Prefer referencing the config constants
  over bare integers where practical.

## Non-goals / Later
- No change to team ids (still `0/1` players, `-1` GK/referee/unknown).
- No change to the TrackEval path: the materialized `gt.txt` still forces `class=1, visibility=1`
  (pedestrian) for every row — semantic class never reaches TrackEval, so HOTA/MOTA are
  unaffected even though goalkeeper now == 1 semantically.
- No data migration: any previously generated artifacts/pkls used the old scheme; regenerating via
  the pipeline produces the new ids. (Mention in the report; nothing to do.)

## Constraints / Caveats
- `class_map.py` must be import-light (no torch/cv2/sklearn) so every subprocess can import it
  safely without triggering the bundled-torchreid clash.
- Detector resolution stays BY NAME; only the numeric values change. Names must remain
  `goalkeeper`/`player`/`referee` (confirmed).
- Keep the eval confusion matrix's human row/col order (player, gk, ref) so output layout is
  stable; only the underlying ids change.
- All CPU suites must stay green after fixture updates.

## Acceptance criteria
- Exactly one definition of the class map (`class_map.py`); no remaining hardcoded `0=player`-era
  literals in the four source files (grep clean).
- Stage 1 maps detector names → `gk=1/player=2/referee=3`; pred `tracks.txt`/`refined.txt` col 8
  carries these ids; render/team/eval all agree.
- HOTA/MOTA path unchanged (gt.txt still all class=1).
- All pipeline + eval CPU suites pass.

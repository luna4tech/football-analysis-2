# Task 004 — Class/team-aware rendering

## Context
Stage 3 (done) writes `03_team/refined.txt` (`frame,id,x,y,w,h,score,class_id,team_id,-1`) and
`03_team/track_attributes.json` (`{ "<id>": {"class", "team", "gk"} }` — stable per-track).

Today `tools/render_from_txt.py` parses a MOT txt into `{frame: (tlwhs, ids, scores)}` and calls
`yolox/utils/visualize.py::plot_tracking`, which colors every box by `get_color(track_id)` and
labels it with the bare track id. `render_from_txt` loads `visualize.py` by file path (to avoid
the heavy `yolox` import).

Canonical class ids: `0=player, 1=goalkeeper, 2=referee`, unknown `-1`.

## Objective
Color boxes by category and prefix goalkeeper labels, driven by the per-track attributes:

| Category | Box color | Label |
|---|---|---|
| player, team 0 | C_team0 | `<id>` |
| player, team 1 | C_team1 | `<id>` |
| goalkeeper | C_gk (single distinct) | `gk:<id>` |
| referee | C_ref (distinct) | `<id>` |
| player unknown-team / unknown class / legacy | fallback (keep legacy `get_color(id)`) | `<id>` |

## Scope
- `Deep-EIoU/Deep-EIoU/tools/render_from_txt.py`
  - Make `import cv2` **lazy** (move into `main`/drawing path) so the module imports — and the
    pure functions below are unit-testable — without cv2.
  - `parse_results`: also parse per-row `class_id` (field 8) and `team_id` (field 9) when present;
    legacy rows (<8/<9 fields) → `-1`. Keep backward compatibility for existing callers.
  - Load `track_attributes.json` for **stable per-track** category: auto-detect it as a sibling of
    the txt (and/or a `--attributes` arg); build `{id: (class, team)}`.
  - Pure helper `category_color_and_label(track_id, class_id, team_id) -> (bgr, label)` implementing
    the table above, with the legacy fallback (`class==-1 and team==-1` → `get_color(id)`,
    bare id) so fully-legacy txt renders byte-for-byte as before.
  - Category resolution precedence per id: attributes.json (stable) → per-frame txt cols → legacy.
    Build per-frame `colors` + `id_texts` lists and pass them to `plot_tracking`.
  - Define the 4 category colors as documented BGR constants; ensure label text stays legible
    against the box colors.
- `Deep-EIoU/Deep-EIoU/yolox/utils/visualize.py`
  - Extend `plot_tracking` with optional `colors=None` (per-box BGR) and `id_texts=None` (per-box
    label). When provided, use them; otherwise keep the exact current behavior
    (`get_color(id)` + bare id). Do NOT break existing callers (e.g. `demo.py`).
- `run_pipeline_colab.ipynb`
  - Point the render cell's `--txt` at `03_team/refined.txt` (so the rendered video uses the
    class/team output).

## Non-goals / Later
- No eval changes (Task 005). No changes to Stage 1/2/3 logic or output formats.
- No jersey-color/Approach-A. Referee gets no label prefix; goalkeepers share one color.

## Constraints / Caveats
- Backward compatibility: a legacy txt with no class/team columns and no attributes.json must
  render identically to today (color by id, bare id label).
- Team is constant per track in `03_team` output; prefer the per-track attributes for color so the
  category never flickers frame-to-frame.
- Colors are BGR (OpenCV). Pick 4 visually distinct constants; document them.

## Acceptance criteria
- With `03_team` output present, players are colored by team, GKs share one distinct color with a
  `gk:` label prefix, referees have their own color; labels are the bare track id otherwise.
- Legacy txt (no class/team, no attributes.json) renders exactly as before.
- The pure parsing + `category_color_and_label` functions are importable and tested without cv2.

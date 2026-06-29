# Task 001 — OPT-1: Slim `refined_tracklets.pkl`

## Context
- 3-stage pipeline. Stage 2 (`gta-link/stage2_refine.py`) writes `refined_tracklets.pkl`
  (`{new_id: Tracklet}`). Stage 3 (`pipeline/team_assignment.py`) is the **only** consumer
  (renderer/eval read the `.txt` files, not this pkl — confirmed by grep).
- Stage 3 uses, per track, exactly three things:
  - the **mean embedding** via `_mean_embedding(features)` — *raw mean of per-frame features,
    then L2-normalize* (see `team_assignment.py`),
  - `class_ids` and `scores` (for `aggregate_class`).
  It never reads `times`, `bboxes`, or individual feature vectors.
- Today the pkl carries full per-detection `features`/`times`/`bboxes`/`scores`/`class_ids`
  → ~4.6 GB at 1 h. The features (~4 GB) are dead weight: Stage 3 collapses them to one mean.

## Objective
Stage 2's export stores a **precomputed per-track mean embedding** + `class_ids` + `scores`
and drops `features`/`times`/`bboxes`. Stage 3 reads the stored mean. The Stage 3 outputs
(`track_attributes.json`, `03_team/refined.txt`) stay **bit-identical**.

## Scope (files likely touched)
- `gta-link/stage2_refine.py` — the export path (`_renumber_and_export_pkl`, called from
  `run_refine`); Stage 2 CLI (`make_parser`/`main`) for the `--keep-features` flag.
- `pipeline/team_assignment.py` — `assign_teams` reads the stored mean when present.
- `pipeline/tests/test_stage2_refine.py`, `pipeline/tests/test_team_assignment.py` — tests.

## Implementation notes / caveats
1. **Share the mean function — do not reimplement.** Stage 2 must compute the stored mean with
   `pipeline.team_assignment._mean_embedding` (the exact function Stage 3 uses) so clustering is
   bit-identical. Stage 2 already puts the repo root on `sys.path`, so
   `from pipeline.team_assignment import _mean_embedding` works. **Do NOT** reuse
   `_fast_connect`'s `means` — that is a mean-of-*normalized* features, a different computation.
2. **`mean` is order-independent**, so the post-merge feature concatenation order does not matter:
   computing the mean at export from the merged `track.features` equals what Stage 3 would compute
   from the same stored list. That is what makes this bit-identical.
3. **Ordering of export steps.** `save_results` (writes `refined.txt` from `times`/`bboxes`) runs
   *before* `_renumber_and_export_pkl`, so stripping `times`/`bboxes`/`features` in the export is
   safe and does not affect `refined.txt`.
4. **Slim representation.** Keep emitting `Tracklet` objects (Stage 3 unpickles them; the class is
   registered as the top-level module `Tracklet`). In slim mode, per track: compute
   `mean = _mean_embedding(track.features)`, set a new attribute `track.mean_emb = mean`
   (np.ndarray or `None`), then set `track.features = []`, `track.times = []`,
   `track.bboxes = []`. Keep `track.scores` and `track.class_ids`. No change to `Tracklet.py` is
   required (the attribute round-trips via `__dict__`); Stage 3 reads it with `getattr(.., None)`.
5. **The alignment assertion** in `_renumber_and_export_pkl`
   (`len(times)==len(bboxes)==len(features)==len(scores)==len(class_ids)`) must run on the FULL
   arrays *before* slimming (validate alignment, then compute the mean, then strip). Do not assert
   the per-frame invariant after slimming.
6. **Stage 3 read path** (`assign_teams`): for a player track, use
   `m = getattr(track, "mean_emb", None)`; if `m is not None` use it directly as the player
   embedding, else fall back to `_mean_embedding(getattr(track, "features", None) or [])`
   (back-compat with old/full pkls and `--keep-features` output). A track whose mean is `None`
   (no features) is skipped from clustering, exactly as today. `aggregate_class(class_ids, scores)`
   is unchanged.
7. **`--keep-features` flag** on Stage 2 (default `False` = slim). When set, export the full
   `Tracklet` exactly as today (no stripping, no `mean_emb`). Thread it from `main` → `run_refine`
   → `_renumber_and_export_pkl`.

## Non-goals / Later
- Do **not** precompute `aggregate_class` in Stage 2 or drop `scores`/`class_ids` — keep them.
- Do **not** change any algorithm, threshold, or `refined.txt` content.
- Do **not** add new `times`/`bboxes` consumers.

## Acceptance criteria
- Slim-mode `refined_tracklets.pkl` carries no per-detection `features`/`times`/`bboxes`;
  it carries `mean_emb` + `class_ids` + `scores`.
- A CPU unit test proves Stage 3's `assign_teams` (and the written `track_attributes.json`)
  is **identical** whether it consumes the slim pkl or the full pkl — i.e. build a small
  `{id: Tracklet}`, run Stage 3 on the full version and on the slimmed-then-exported version,
  assert equal attributes.
- `--keep-features` reproduces today's full pkl (per-detection arrays intact, no `mean_emb`
  dependency in Stage 3).
- Existing tests stay green.

# Pipeline Optimizations — Plan

Implements OPT-1, OPT-3, OPT-2 from [OPTIMIZATION_NOTES.md](OPTIMIZATION_NOTES.md),
in that order. OPT-4/5/6 and float16 are out of scope.

## Ground rules
- **Accuracy-preserving.** OPT-1 and OPT-3 are bit-identical (provable on CPU). OPT-2 is
  *numerically equivalent* — batched GPU inference may differ in the last FP bits and rarely
  flip a boundary detection; we verify aggregate metrics, not byte-for-byte `tracks.txt`.
- **Verification.** Developer proves equivalence with CPU unit tests + code review. The real
  end-to-end diff against committed sample artifacts runs on Colab (owner-run) before merge.
  OPT-2's GPU path cannot be exercised on the dev box — its unit tests cover only the
  pure-Python regrouping/ordering.
- **Commits.** One focused commit per task (stage only that task's files), then proceed.
- Branch: `class-team-inference`.

## Tasks
1. **001 — OPT-1: Slim `refined_tracklets.pkl`.** Stage 2 stores each output track's mean
   embedding (via the *shared* `_mean_embedding`) + `class_ids` + `scores`, drops
   `features`/`times`/`bboxes`. Stage 3 reads the stored mean (falls back to features for old
   pkls). `--keep-features` flag retained as a safety hatch. ~4.6 GB → ~25 MB; Stage 3 RAM
   5–6 GB → <0.5 GB.
2. **002 — OPT-3: Sparse overlap matrix in `_fast_connect`.** Replace the dense `occ` (n×F)
   build with a `scipy.sparse` occupancy matmul; materialize only the n×n bool. Identical
   booleans; merge loop untouched. −~2 GB Stage 2 RAM at 1 h.
3. **003 — OPT-2: Batch Stage 1 detect + ReID.** Batch `model.predict` over a frame window,
   concat crops across frames into one ReID call and split back, replay `track_consume` in
   strict frame order. Numerically equivalent. Stage 1 ~1.5–2.5× faster.

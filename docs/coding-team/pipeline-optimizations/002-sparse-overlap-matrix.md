# Task 002 — OPT-3: Sparse overlap matrix in `_fast_connect`

## Context
- `gta-link/stage2_refine.py` → `_fast_connect` builds an all-pairs temporal-overlap matrix to
  decide which tracklet pairs can merge (overlapping pairs are forced to distance 1).
- Today it does this with a **dense occupancy matrix**:
  ```python
  max_frame = max(int(max(t.times)) for t in tracklets.values())
  occ = np.zeros((n, max_frame + 1), dtype=np.float32)   # n × F  -> ~2.5 GB at 1 h
  for i, tid in enumerate(tids):
      occ[i, np.asarray(tracklets[tid].times, dtype=np.int64)] = 1.0
  overlap = (occ @ occ.T) > 0.5   # (n, n) bool
  ```
  At 1 h, `F` ≈ 90,000 and `n` (tracklets after split) can reach a few thousand, so the dense
  `occ` is the ~2 GB Stage-2 RAM spike. The merge loop afterwards only ever uses/updates the
  **n×n** bool `overlap`; the dense n×F `occ` exists solely to build it.

## Objective
Build the same `n×n` boolean `overlap` **without ever materializing the dense n×F matrix**, using
a `scipy.sparse` occupancy matrix and a sparse matmul. Materialize only the `n×n` bool. Output
must be **bit-identical booleans** to the current path; the merge loop is untouched.

## Scope
- `gta-link/stage2_refine.py` — only the initial `overlap` construction inside `_fast_connect`.
- `pipeline/tests/test_stage2_refine.py` — add a dense-vs-sparse equivalence test.

## Implementation notes
1. Replace the dense `occ` build + `(occ @ occ.T) > 0.5` with:
   ```python
   from scipy.sparse import csr_matrix   # lazy import inside _fast_connect, like `import numpy`
   # rows = tracklet index, cols = frame index, data = 1
   occ = csr_matrix((data, (rows, cols)), shape=(n, max_frame + 1))
   overlap = (occ @ occ.T).toarray() > 0   # dense n×n bool
   ```
   Integer counts of shared frames; `> 0` is identical to the old float `> 0.5` (counts are
   non-negative integers, so `>0 ⟺ ≥1 ⟺ >0.5`). `max_frame` is still needed for `shape`.
2. **`overlap` must remain a writable dense numpy bool ndarray** — the merge loop does
   `overlap[t1,:] |= overlap[t2,:]` and `np.delete(overlap, t2, ...)`. `.toarray() > 0` returns
   exactly that, so no change to the loop.
3. Recommended: factor the construction into a small helper, e.g.
   `_build_overlap(tracklets, tids, n) -> np.ndarray (n×n bool)`, so it can be unit-tested in
   isolation against the dense reference.
4. Use `scipy` — it is already a Stage-2 dependency (importing `refine_tracklets` pulls in
   torch/sklearn/scipy/...). Keep the import lazy/local to match the module's CPU-light style.

## Non-goals / Later
- Do **not** touch the greedy merge loop, the distance arithmetic, `means`/`counts`, the spatial
  gate, or the incremental `overlap` row/col updates.
- Do **not** change `_fast_connect`'s output or any merge sequencing.

## Caveats
- Tracklet `times` are unique frames per track, so no duplicate `(i, frame)` entries are expected;
  even if a duplicate occurred, `csr_matrix` sums it, which only inflates a positive count and
  cannot change the `> 0` boolean — harmless.
- The sparse product is bounded by `n²` nonzeros (worst case ~hundreds of MB at 1 h, typically far
  sparser) and `.toarray()` yields the ~49 MB n×n bool — both well under the ~2.5 GB dense n×F it
  replaces.
- `n == 0` is already short-circuited earlier in `_fast_connect`; `n == 1` yields a 1×1 overlap
  (diagonal is set to `inf` in `Dist` regardless).

## Acceptance criteria
- A CPU unit test builds tracklets with mixed temporal relationships (disjoint, identical-frames,
  partial overlap, single shared frame, no overlap) and asserts the sparse-built `overlap` equals
  the dense `(occ @ occ.T) > 0.5` reference **element-for-element**.
- `_fast_connect`'s merged output is unchanged (existing `_fast_connect` tests stay green).
- No dense n×F array is allocated anywhere in `_fast_connect`.

# Task 003 — OPT-2: Batch Stage 1 detector + ReID

## Context
- Stage 1 (`Deep-EIoU/Deep-EIoU/tools/stage1_track.py`) is ~90% of pipeline wall time. Its
  `run_stage1` loop reads ONE frame, runs `perceive(frame)` (a single `model.predict(frame)` +
  crops + a per-frame ReID `extractor(crops)` call), then `track_consume(frame_id, ...)` — all
  per frame. The per-call overhead of ~90,000 `predict` launches dominates; GPU is under-utilized.
- The producer/consumer split is already clean:
  - `perceive(frame, detector, extractor, width, height) -> (det, embs)` — pure per-frame, no
    cross-frame state. Returns `(None, None)` when the detector output is None; `(det, empty_embs)`
    when det has 0 surviving rows.
  - `track_consume(frame_id, det, embs, tracker, assembler, min_box_area)` — carries the tracker's
    cross-frame state; MUST be called in strictly increasing `frame_id` order.

## Objective
Batch the **perception** (detection + ReID) across a window of frames, then **replay
`track_consume` per frame in strict order**. Tracking stays sequential (it can't batch).
Acceptance bar: **numerically equivalent** output (NOT byte-identical) — batched GPU inference may
differ in the last FP bits and rarely flip a boundary detection; that is accepted. Verify the
pure-Python regrouping/ordering on CPU; the GPU numerical equivalence is validated by the owner on
Colab.

## Scope
- `Deep-EIoU/Deep-EIoU/tools/stage1_track.py`:
  - Add `YOLOv11Detector.infer_many(frames: list) -> list` (one YOLOX-shaped output or None per
    frame, in order).
  - Add `perceive_batch(frames, detector, extractor, width, height) -> list[(det, embs)]`.
  - Rewrite `run_stage1`'s loop to read frames in windows of `batch_size`, call `perceive_batch`,
    then loop the window calling `track_consume(frame_id, det, embs, ...)` in order.
  - Add a CLI arg `--batch-size` (default 16).
- Tests: `pipeline/tests/test_stage1_assembly.py` (or a sibling CPU test module) — equivalence of
  `perceive_batch` vs looping `perceive` using fakes.

## Implementation notes
1. **`infer_many`**: call the existing `_predict` with a LIST source
   (`self.model.predict(source=frames, ...)` via `_predict`); Ultralytics returns a list of
   `Results` in input order. Map each with `ultralytics_result_to_yolox_output`. Return a list of
   length `len(frames)` (None entries allowed). Use the SAME kwargs (`conf`/`iou`/`imgsz`/`half`/
   `device`) as `infer_one` — do not diverge per-frame post-processing.
2. **`perceive_batch`** must reproduce `perceive`'s per-frame contract EXACTLY (so the only change
   is batching, not semantics):
   - For each frame `k`: `det_k, crops_k = det_and_crops_from_output(outputs[k], frames[k], width, height)`.
   - Concatenate `crops_k` across ALL frames (in frame order, and within a frame in det-row order)
     into one list; call `extractor(all_crops)` ONCE; `.cpu().detach().numpy()`; split the result
     back by per-frame crop counts (`np.split` on cumulative counts, or slice by offsets).
   - Per-frame result: `det_k is None -> (None, None)`; `len(crops_k) == 0 -> (det_k, np.empty((0,0), dtype=np.float32))`;
     else `(det_k, embs_slice_k)`. This matches `perceive` exactly (note: `len(crops)==len(det)`
     always when det is not None).
   - If `all_crops` is empty (no crops in the whole window), skip the extractor call entirely.
3. **`run_stage1` loop**: read up to `batch_size` frames (stop at video end; handle a final partial
   window), call `perceive_batch`, then `for offset, (det, embs) in enumerate(window): track_consume(frame_id, det, embs, ...); frame_id += 1`. Preserve the `emb_dim` capture (set from any
   frame's non-empty `embs.shape[1]`). Keep a progress log (per-batch or every N frames is fine —
   logging cadence is not output-critical).
4. **Memory**: holding `batch_size` frames + their crops is small (16×~6 MB ≈ 100 MB); fine. Keep
   `batch_size` modest (default 16) so a high-res batch fits T4 VRAM.

## Non-goals / Later
- Do NOT modify the tracker, `TrackletAssembler`, `det_and_crops_from_output`, or any per-frame
  detection/crop/ReID math. Batching must not change per-frame semantics.
- Do NOT change the `tracks.txt` / `tracklets.pkl` formats.
- Keep `perceive` (single-frame) in place — it documents the contract `perceive_batch` must match
  and stays unit-tested.
- Do NOT attempt to run Stage 1 / GPU / a real video on this dev box.

## Constraints / caveats
- **Ordering is the correctness crux.** Crops must be concatenated and split back in exact
  (frame-order, then det-row-order) sequence so each embedding maps to its det row; `track_consume`
  must see frames in strict ascending `frame_id`.
- Byte-identical `tracks.txt` is NOT required (GPU batch nondeterminism is accepted). Do not add
  hacks chasing bit-exactness.
- Match the module's heavy docstring/comment style; explain the batching and the ordering guarantee.

## Acceptance criteria
- A CPU unit test (fakes for detector + extractor) proves `perceive_batch(frames, ...)` returns,
  for every frame, the SAME `(det, embs)` as calling `perceive(frame, ...)` one-by-one — with the
  fakes returning identical per-frame values. Cover: normal frames, a `None`-output frame, a
  zero-det frame, frames with differing crop counts, and a partial final window.
- `run_stage1` feeds `track_consume` in strict frame order; the assembled `tracks.txt`/`tracklets`
  are unchanged given identical per-frame perception (proven via the fakes at the `run_stage1`
  level if feasible, else via the `perceive_batch` equivalence test).
- `--batch-size` is wired (default 16).
- Existing Stage 1 tests stay green.

# Task 003 — Stage 1 (parallel): batched detect+ReID + prefetch decode, alongside sequential

## Context
Task 2 built the sequential Stage-1 entrypoint `Deep-EIoU/Deep-EIoU/tools/stage1_track.py` with a
clean producer/consumer split: stateless `perceive(frame) -> (det, embs)` and stateful
`track_consume(frame_id, det, embs, ...)` that must receive frames in strict order. This task adds
an **opt-in parallel perception path** that overlaps video decode with GPU compute and batches the
detector + ReID across frames. Per-frame detection/ReID are independent across frames; only tracking
carries state — so only the perception side changes. The tracking consumer stays UNTOUCHED.

## Objective
Add a `--parallel --batch-size N` mode to `stage1_track.py` that produces the **same** Stage-1
artifact contract as the sequential mode, by:
1. decoding frames in a background prefetch thread (overlap decode with GPU compute), and
2. running detection and ReID on **batches of frames**, then
3. feeding the tracking consumer in strict frame order (reusing `track_consume` verbatim).

Sequential remains the **default** and the behavior reference. Parallel is opt-in.

## Scope (do now)

Edit `Deep-EIoU/Deep-EIoU/tools/stage1_track.py` (and `stage1_assembly.py` only if a pure helper
belongs there). Do not touch `demo.py`, the tracker, the detector, or any gta-link file.

### Guarantee parity by sharing the per-frame post-detection code
Factor the sequential per-frame "detection output -> (det, crops)" logic (rescale
`det /= scale`, edge-removal `det[:,0:4] < 1`, clamp/zero-area-drop/crop, and the
`outputs[0] is None` empty case) into ONE shared helper, and call it from BOTH the sequential and
parallel paths. This makes the two modes do identical per-frame math; only how detection forward and
ReID are *invoked* (single vs batched) differs.

### Batched perception
- `infer_batch(predictor, frames) -> list[per_frame_detection_output]`: mirror
  `Predictor.inference`'s internals but batched — `preproc_to_tensor` each frame (same letterbox;
  all frames in one video share size/ratio), stack to `(B,3,H,W)`, `predictor.model(batch)`, apply
  `predictor.decoder` if set, then `postprocess(outputs, num_classes, confthre, nmsthre)` which
  returns a length-B list (one detection set per image, each equal to single-image inference's
  `outputs[0]`). Reuse the predictor's existing attrs (`num_classes`, `confthre`, `nmsthre`,
  `decoder`, `device`, `fp16`).
- ReID across the batch: collect ALL crops across the B frames into one list, call the extractor
  ONCE, then split the resulting `(total_crops, 512)` embeddings back per-frame by crop count (a
  frame with 0 crops -> empty `(0,512)`; a frame whose detection output was `None` stays `None`).
  Put the split helper (`split_by_counts(embs, counts)`) in `stage1_assembly.py` as a pure function.

### Prefetch decode thread
- One background thread reads `cv2.VideoCapture` frames and pushes `(frame_id, frame)` onto a
  **bounded** `queue.Queue` (cap memory; depth = a small multiple of batch size, documented). A
  sentinel signals end-of-stream. The main thread pulls frames, forms batches of up to `batch_size`,
  runs batched perception, then calls `track_consume(...)` for each frame **in ascending frame_id
  order**. Handle the final partial batch.
- Keep it to a SINGLE prefetch thread + main-thread GPU/consume. Do not add multiple GPU workers,
  multiprocessing, or CUDA streams (out of scope / YAGNI).

### Strict-order + skip parity (critical)
- The consumer must be called for frames `0,1,2,...` with no reordering and no gaps.
- Preserve the sequential skip rule exactly: when a frame's detection output is `None`, the consumer
  skips `tracker.update` (so `tracker.frame_id` does NOT advance), but the loop `frame_id` still
  advances and no row is emitted — identical to sequential/demo.py. The output `frame_id` written to
  artifacts is the loop index, never `tracker.frame_id`.

### CLI / profiling
- Add `--parallel` (flag, default off) and `--batch-size` (int, default 8). No other new flags;
  derive the queue depth internally from batch size and document it.
- Keep the same `profile_stage("01_track", ...)`; add `mode` ("sequential"|"parallel") and
  `batch_size` to the profile `io` dict so the user guide can compare runs. The cache key / contract
  paths / outputs are unchanged.

### Unit test (CPU-only, runnable here)
Add tests (no torch/cv2/yolox) for the pure pieces:
- `split_by_counts`: correct per-frame split including zero-count frames and a `None` passthrough;
  total rows conserved.
- Ordering invariant: drive the batching/consume loop with a STUB batched-perceive (returns
  deterministic fake `(det, embs)` keyed by frame_id) and a fake consumer that records the
  `frame_id`s it receives; assert it receives `0..N-1` in order, exactly once each, across uneven
  batch sizes and a final partial batch, including some `None` frames.

## Non-goals / Later
- No change to the tracking/refinement algorithms or to `track_consume`'s logic.
- No multi-GPU, multiprocessing, CUDA streams, or hardware-decode backends.
- No orchestrator (Task 5).

## Constraints / Caveats
- **Parity is "equal up to floating-point nondeterminism," not bit-identical.** Batched GPU matmul
  kernels can differ from single-image ones in the last FP bits, which could rarely flip a detection
  at a threshold. The consumer/tracker path is identical; only perception batching differs. State
  this in code comments; the Colab equivalence check (Task 5) compares `tracks.txt` within tolerance.
- Assume constant frame size within a video (true for a single file); rely on it for the shared
  letterbox ratio. A defensive assert is fine.
- Bound the prefetch queue so a fast decoder cannot exhaust memory on a long video.
- Sequential mode must remain byte-for-byte what Task 2 produced (do not regress it while
  refactoring out the shared helper).

## Acceptance criteria
- `--parallel --batch-size N` runs the same contract outputs as sequential; sequential is unchanged
  and still the default.
- Per-frame post-detection math is literally shared between the two modes (one helper).
- CPU unit tests for `split_by_counts` and the strict-ordering invariant pass here without
  torch/cv2/yolox.
- Report what ran here vs deferred to Colab (real batched GPU run + tolerance diff vs sequential).

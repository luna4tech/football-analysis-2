# Task 002 — Stage 1 (sequential): emit tracks.txt + tracklets.pkl, reusing curr_feat

## Context
Stage 1 is DeepEIoU detection + ReID + online tracking. Today `Deep-EIoU/Deep-EIoU/tools/demo.py`
runs the whole loop on a video and writes only a MOT `.txt` (the ReID embeddings it computes are
discarded). We want a Stage-1 entrypoint that ALSO emits a GtaLink-compatible `tracklets.pkl` whose
per-detection features are the embeddings DeepEIoU already computed — so GtaLink never recomputes
them. See `PLAN.md` for the contract and rationale.

Verified facts you can rely on:
- The tracker (`tracker/Deep_EIoU.py`) attaches a fresh, L2-normalized `curr_feat` (a `(512,)`
  numpy array) to every track it returns from `update()`; every output row had a current-frame
  detection match, so `curr_feat` is always present and current.
- GtaLink's `Tracklet` class lives at `gta-link/Tracklet.py` and is pure-Python (no torchreid).
  `refine_tracklets.py` (Stage 2) unpickles `{track_id: Tracklet}` dicts and reads
  `.times`, `.scores`, `.bboxes`, `.features`.

This machine cannot run the GPU pipeline. Make the non-GPU assembly logic unit-testable; the GPU
run is verified later in Colab.

## Objective
Add a new Stage-1 entrypoint `Deep-EIoU/Deep-EIoU/tools/stage1_track.py` that runs DeepEIoU on a
single video and writes the Stage-1 artifact contract:
- `<artifacts>/01_track/tracks.txt` — MOT, **0-based** frames, byte-equivalent to what demo.py
  emits today.
- `<artifacts>/01_track/tracklets.pkl` — `{track_id: Tracklet}` with features reused from
  `curr_feat`.
Plus contract paths, caching, and profiling from the `pipeline/` package (Task 1).

**Leave `demo.py` unchanged.** Reuse its components (`Predictor`, `preproc_to_tensor`, and its
argparser defaults) by importing them; replicate only the inner per-frame logic you cannot import.

## Scope (do now)

`Deep-EIoU/Deep-EIoU/tools/stage1_track.py`, run with CWD = `Deep-EIoU/Deep-EIoU` (same as demo.py).
It must `sys.path`-insert the repo root so it can `import pipeline` (artifacts/cache/profiling).

Structure the per-frame work as a **producer/consumer split** (this is what Task 3 will parallelize;
the consumer must stay untouched there):
- `perceive(frame, predictor, extractor, width, height) -> (det, embs)`: detection forward +
  rescale + edge-removal + clamp/crop + ReID embedding. Pure per-frame, no cross-frame state.
- a tracking **consumer** that, given `(frame_id, det, embs)` IN STRICT FRAME ORDER, calls
  `tracker.update(det, embs)`, applies the `min_box_area` filter, and accumulates outputs.

For each surviving output target per frame, in one place, append to BOTH:
- the MOT results list (the line `f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"`), and
- the track's `Tracklet`: `times.append(frame_id)`, `scores.append(t.score)`,
  `bboxes.append([l,t,w,h])` (= the same `t.last_tlwh`), `features.append(t.curr_feat)`.
So `tracks.txt` rows and `tracklets.pkl` entries are produced from the identical filtered set and
stay index-aligned (one feature per row), exactly mirroring what `generate_tracklets.py` produced.

### Behavior-parity invariants (must match demo.py exactly)
- Detection rescale: `scale = min(1440/width, 800/height); det /= scale`.
- Edge removal: drop rows where `np.any(det[:, 0:4] < 1, axis=1)`.
- Box clamp + zero-area drop + crop: copy demo.py's `imageflow_demo` block verbatim.
- `min_box_area` filter on `tlwh[2]*tlwh[3]` before emitting.
- 0-based `frame_id` starting at 0; same `:.2f` formatting. Result identical to demo.py's txt.

### Pickle compatibility (critical)
The pickle must unpickle in the gta-link process where `import Tracklet` resolves to
`gta-link/Tracklet.py` (module name `"Tracklet"`). Load the class WITHOUT putting `gta-link/` on
`sys.path` (that dir also contains a `reid/` package that would clash with DeepEIoU's `reid`
package). Instead load by file path and register the module name explicitly:
```python
import importlib.util, sys
spec = importlib.util.spec_from_file_location("Tracklet", <abs path to gta-link/Tracklet.py>)
mod = importlib.util.module_from_spec(spec); sys.modules["Tracklet"] = mod
spec.loader.exec_module(mod); Tracklet = mod.Tracklet
```
This makes `Tracklet.__module__ == "Tracklet"`, so the pickle references `Tracklet.Tracklet` and
loads cleanly in Stage 2. Add a unit test asserting the pickled bytes reference module `"Tracklet"`
and round-trip.

### Device, caching, profiling, CLI
- Derive the ReID `FeatureExtractor` device from the `--device` arg (cpu/gpu) instead of demo.py's
  hardcoded `'cuda'`, so a CPU run is possible.
- Caching: use `pipeline.cache.should_run(tracklets_pkl, [video_path], force=args.force)`. If cached
  (both outputs present + newer than the video, no `--force`), log and exit 0 without recomputing.
- Profiling: wrap the run in `pipeline.profiling.profile_stage("01_track", paths, extra={...})` and
  record IO counts: `n_frames`, `n_output_rows`, `n_unique_tracks`, `emb_dim`.
- Artifacts: `pipeline.artifacts.get_artifact_paths(video, base_dir=args.artifacts_dir)` +
  `ensure_dirs`.
- CLI: base it on `demo.make_parser()` (reuse all detector/tracker/reid args + defaults) and add
  `--video` (required), `--artifacts-dir` (optional), `--force`. No annotated-video output in this
  stage (render_from_txt covers that). Do NOT add `--parallel`/`--batch-size` — that is Task 3.

### Unit test (CPU-only, runnable here)
Add a small test (alongside or under `pipeline/tests/` or a `tools/`-local test) for the pure
assembly path: feed synthetic `(frame_id, track_id, tlwh, score, feat)` records through the
Tracklet-assembly function and assert (a) per-track `times/scores/bboxes/features` are aligned and
correct, (b) the dict pickles and unpickles, (c) the pickle references module `"Tracklet"`. Do this
by factoring the assembly into a small importable function that does NOT require torch/cv2/yolox.

## Non-goals / Later
- No parallel/batched path, no decode thread (Task 3).
- No changes to `demo.py`, the tracker, the detector, or any GtaLink file.
- No orchestrator / `python -m pipeline` wrapper (Task 5).
- No annotated-video writing.

## Constraints / Caveats
- Preserve tracking behavior exactly; the only intended output difference vs demo.py is the added
  `tracklets.pkl` and the fixed contract paths/filenames (instead of timestamped ones).
- Keep frames 0-based everywhere; do not reintroduce generate_tracklets' 1-based `imgs[frame-1]`.
- `curr_feat` is already L2-normalized — store it as-is (`np.float32`), do not re-normalize.
- The torch/cv2/yolox-dependent code will not run on this machine; isolate it so the assembly unit
  test imports without those deps.

## Acceptance criteria
- `tracks.txt` is byte-equivalent to demo.py's emitted results for the same input/args (reason about
  this statically; confirmed in Colab later).
- `tracklets.pkl` entries are row-aligned with `tracks.txt` and unpickle under module `"Tracklet"`.
- The CPU assembly unit test runs and passes here without torch/cv2/yolox.
- Report which checks ran here vs deferred to Colab.

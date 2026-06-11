# Plan: Modular DeepEIoU + GtaLink tracking pipeline

## Goal
Refactor the repo into a modular, cached, profileable two-stage tracking pipeline that
combines DeepEIoU (detection + ReID + tracking) and GtaLink (tracklet refinement), while
**reusing DeepEIoU's ReID embeddings** instead of recomputing them in GtaLink.

## Key decisions (locked)
- **Integration = thin orchestrator + on-disk contract (Option A).** DeepEIoU and GtaLink
  stay intact in their own directories; each stage runs as its own process in its own CWD,
  so the two bundled `torchreid` copies never clash. Stages communicate through files.
- **Embeddings are reused, not recomputed.** Every track DeepEIoU outputs already carries a
  fresh, L2-normalized `curr_feat` (verified in `tracker/Deep_EIoU.py`). Stage 1 assembles
  the GtaLink `Tracklet` pickle directly from these, so the separate
  `generate_tracklets.py` OSNet pass is dropped from the pipeline path (the file stays on
  disk as a fallback for other trackers).
- **Stage 1 keeps BOTH a sequential (default, behavior-preserving) and a parallel (opt-in)
  path.** Parallel = frame-batched detect+ReID + prefetch decode thread. Both feed the
  tracker in frame order and produce the identical artifact contract, so they are diffable.
- **Single video** input. **One shared Python env.** Minimal change; preserve tracking and
  refinement behavior.
- **This environment cannot run the code** (Windows, no GPU). Agents do static sanity checks
  + pure-Python unit tests only. Authoritative verification is done manually in Colab,
  driven by the user guide.

## Artifact contract
```
artifacts/<video_stem>/
  01_track/tracks.txt       # MOT, 0-based frames (feeds render_from_txt / count_tracks)
  01_track/tracklets.pkl    # {id: Tracklet} with features = reused curr_feat
  02_refine/refined.txt     # refined MOT
  profiles/<stage>.json     # per-stage profile (time / GPU mem / CPU mem / IO shapes+counts)
  profiles/summary.{json,md}
```
Caching: a stage skips work when its output is newer than its inputs, unless `--force`
(orchestrator: `--force-all`).

## Tasks
1. **Scaffold `pipeline/` + profiling + artifact/caching helpers.** Pure-Python where
   possible (unit-testable without GPU). Profiler captures wall time, GPU mem (guarded by
   `torch.cuda.is_available()`), CPU mem (psutil, guarded), and caller-supplied IO
   shapes/counts; writes per-stage JSON + an aggregated summary. Artifact path layout +
   mtime-based staleness/caching helpers. No changes to DeepEIoU/GtaLink yet.
2. **Stage 1 — sequential path.** DeepEIoU emits `tracks.txt` + `tracklets.pkl` (features
   reused from `curr_feat`), with perception/tracking internally decoupled. Contract CLI +
   caching + profiling. Default mode; behavior-preserving. Tracklets assembled from the same
   `min_box_area`-filtered set used for the txt so the two artifacts stay row-consistent.
3. **Stage 1 — parallel path (alongside sequential).** `--parallel --batch-size N`: frame
   batching for detect+ReID + prefetch decode thread, bounded queue, strict frame ordering
   into the tracker. Same contract/outputs as Task 2.
4. **Stage 2 — refine wrapper.** Contract CLI + caching + profiling around
   `refine_tracklets.py`; consumes `tracklets.pkl` directly; refinement algorithm untouched.
5. **Orchestrator + user guide + smoke test.** `python -m pipeline run --video ...` chaining
   both stages with caching/force; aggregated profile report; `USER_GUIDE.md` covering: how
   to run each stage + full pipeline, how to force recompute, how to verify outputs in Colab
   (incl. the parallel==sequential equivalence diff), how to inspect profiling, and **when to
   use parallelization** (throughput vs latency, batch-size vs VRAM, when sequential is
   better). One end-to-end smoke test on a short clip.

## Verification model
- Agents (this machine): static review + pure-Python unit tests for non-GPU logic
  (artifact/caching helpers, Tracklet assembly grouping, profiler JSON shape). They report
  exactly what could and could not be executed.
- User (Colab): per-stage smoke runs, full pipeline run, and parallel-vs-sequential
  equivalence diff. Commands provided in `USER_GUIDE.md`.

## Non-goals (YAGNI)
No unified package / torchreid merge. No multi-GPU or multiprocess sharding. No CUDA-stream
micro-optimization. No batch-dataset mode. No full pytest suite (lightweight verify + one
smoke test only). No changes to the tracking or refinement algorithms.

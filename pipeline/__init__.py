"""
pipeline — thin orchestration layer over Deep-EIoU and GtaLink.

Sub-modules
-----------
artifacts       On-disk artifact path layout helpers.
cache           mtime-based staleness / skip-logic helpers.
profiling       Per-stage wall-time + memory profiling with JSON output.
orchestrator    Chains Stage 1 then Stage 2 as subprocesses (CWD-isolated).
compare_tracks  Tolerant MOT-file diff (parallel==sequential equivalence check).
__main__        ``python -m pipeline run ...`` / ``python -m pipeline compare ...``.
"""

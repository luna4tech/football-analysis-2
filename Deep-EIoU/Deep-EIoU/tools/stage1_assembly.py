"""
stage1_assembly — pure-Python Tracklet assembly for Stage 1.

This module deliberately has **no** torch / cv2 / yolox dependency so it can be
imported and unit-tested on a CPU-only machine.  It owns two responsibilities
that are independent of the GPU perception/tracking loop:

1. Loading GtaLink's ``Tracklet`` class by file path and registering it under
   the module name ``"Tracklet"`` (see ``load_tracklet_class``).  This is what
   makes the produced pickle reference ``Tracklet.Tracklet`` so it unpickles in
   the Stage-2 (gta-link) process where ``import Tracklet`` resolves to
   ``gta-link/Tracklet.py``.

2. Accumulating the per-frame *surviving* detections (the same ``min_box_area``
   filtered set that produces ``tracks.txt``) into both the MOT result lines and
   a ``{track_id: Tracklet}`` dict, kept strictly row-aligned (one feature per
   MOT row), exactly mirroring what ``gta-link/generate_tracklets.py`` produced.

The caller (``stage1_track.py``) feeds records in strict frame order; this
module performs no cross-frame inference of its own.
"""

from __future__ import annotations

import importlib.util
import queue
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np


# MOT result line format.  This is byte-for-byte the string demo.py appends to
# its ``results`` list:
#   f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
# Keeping it in one place guarantees tracks.txt parity.
def format_mot_line(frame_id: int, track_id: int, tlwh: Sequence[float], score: float) -> str:
    """Return the MOT-format line for one surviving detection.

    Matches demo.py's ``results.append(...)`` exactly: 0-based ``frame_id``,
    ``:.2f`` formatting on the four box coordinates and the score, trailing
    ``-1,-1,-1`` and a newline.
    """
    return (
        f"{frame_id},{track_id},"
        f"{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},"
        f"{score:.2f},-1,-1,-1\n"
    )


def load_tracklet_class(gta_link_dir: "str | Path") -> type:
    """Import GtaLink's ``Tracklet`` class without putting gta-link/ on sys.path.

    gta-link/ contains a ``reid/`` package that clashes with DeepEIoU's own
    ``reid`` package, so we must NOT add gta-link/ to ``sys.path``.  Instead we
    load ``Tracklet.py`` by absolute file path and register the resulting module
    under the name ``"Tracklet"`` in ``sys.modules``.

    Registering under ``"Tracklet"`` is what makes ``Tracklet.__module__ ==
    "Tracklet"``, so a pickle of ``Tracklet`` instances references
    ``Tracklet.Tracklet`` and loads cleanly in the Stage-2 process (where
    ``import Tracklet`` resolves to this same file).

    Parameters
    ----------
    gta_link_dir:
        Path to the ``gta-link`` directory containing ``Tracklet.py``.

    Returns
    -------
    type
        The ``Tracklet`` class object.
    """
    tracklet_path = Path(gta_link_dir) / "Tracklet.py"
    if not tracklet_path.is_file():
        raise FileNotFoundError(f"Tracklet.py not found at {tracklet_path}")

    spec = importlib.util.spec_from_file_location("Tracklet", str(tracklet_path))
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"could not create import spec for {tracklet_path}")
    module = importlib.util.module_from_spec(spec)
    # Register BEFORE exec so the module name is bound; this is what the pickle
    # will reference.
    sys.modules["Tracklet"] = module
    spec.loader.exec_module(module)
    return module.Tracklet


class TrackletAssembler:
    """Accumulates surviving detections into MOT lines + a ``{id: Tracklet}`` dict.

    Both artifacts are produced from the identical filtered record set, in the
    same order, so ``tracks.txt`` rows and ``tracklets.pkl`` feature lists stay
    index-aligned (exactly one feature per MOT row).

    Usage::

        Tracklet = load_tracklet_class(gta_link_dir)
        asm = TrackletAssembler(Tracklet)
        for frame_id, det, embs in consumer(...):
            for tid, tlwh, score, feat in surviving_targets:
                asm.add(frame_id, tid, tlwh, score, feat)
        results = asm.results          # list[str] -> tracks.txt
        tracklets = asm.tracklets      # {id: Tracklet} -> tracklets.pkl
    """

    def __init__(self, tracklet_cls: type) -> None:
        self._tracklet_cls = tracklet_cls
        self.results: List[str] = []
        self.tracklets: Dict[Any, Any] = {}

    def add(
        self,
        frame_id: int,
        track_id: int,
        tlwh: Sequence[float],
        score: float,
        feat: "np.ndarray",
    ) -> None:
        """Append one surviving detection to both the MOT list and its Tracklet.

        Parameters
        ----------
        frame_id:
            0-based frame index.
        track_id:
            Track identifier from the tracker.
        tlwh:
            ``[left, top, width, height]`` — the same ``last_tlwh`` demo.py
            writes to tracks.txt.
        score:
            Detection/track score.
        feat:
            The track's ``curr_feat`` — already L2-normalized.  Stored as
            ``np.float32`` and NOT re-normalized.
        """
        # MOT line (identical formatting to demo.py).
        self.results.append(format_mot_line(frame_id, track_id, tlwh, score))

        # Box stored exactly as demo.py's tracks.txt row -> [l, t, w, h].
        bbox = [float(tlwh[0]), float(tlwh[1]), float(tlwh[2]), float(tlwh[3])]
        # curr_feat is already L2-normalized; store as float32, do not renormalize.
        feat = np.asarray(feat, dtype=np.float32)

        track = self.tracklets.get(track_id)
        if track is None:
            track = self._tracklet_cls(track_id, frame_id, score, bbox)
            track.append_feat(feat)
            self.tracklets[track_id] = track
        else:
            track.append_det(frame_id, score, bbox)
            track.append_feat(feat)

    @property
    def n_rows(self) -> int:
        """Number of MOT output rows accumulated so far."""
        return len(self.results)

    @property
    def n_unique_tracks(self) -> int:
        """Number of distinct track ids accumulated so far."""
        return len(self.tracklets)


# ---------------------------------------------------------------------------
# Batched-ReID split (pure; used by the Task-3 parallel path)
# ---------------------------------------------------------------------------
def split_by_counts(
    embs: "np.ndarray | None",
    counts: Sequence["int | None"],
) -> List["np.ndarray | None"]:
    """Split a stacked ReID embedding block back into per-frame embeddings.

    The parallel path batches every crop across the frames of one batch into a
    single extractor call, producing one ``(total_crops, D)`` array.  This
    function splits that block back into one entry per frame, in order, so the
    consumer receives the same ``embs`` it would have under sequential
    per-frame ReID.

    Parameters
    ----------
    embs:
        The stacked ``(total_crops, D)`` embeddings for the whole batch (the
        concatenation of every frame's crops in frame order), or ``None`` when
        the batch contributed no crops at all (``total_crops == 0``).
    counts:
        One entry per frame, in frame order:

        * a non-negative ``int`` — the number of crops that frame contributed
          (``0`` is allowed: a frame whose detector fired but every box was
          dropped by clamp/zero-area filtering), or
        * ``None`` — the frame's detection output was ``None`` (detector
          produced nothing); it is a pass-through and stays ``None`` so the
          consumer applies the exact sequential skip rule.

    Returns
    -------
    list
        One element per entry in *counts*, in the same order:

        * ``None`` where ``counts[i] is None`` (``None`` passthrough),
        * a ``(0, D)`` empty array where ``counts[i] == 0``,
        * the matching ``(counts[i], D)`` slice of *embs* otherwise.

    Notes
    -----
    Total rows are conserved: ``sum(c for c in counts if c) == embs.shape[0]``
    (when *embs* is not ``None``).  A ``ValueError`` is raised if the integer
    counts do not sum to ``embs.shape[0]`` so a producer bug surfaces loudly
    rather than silently misaligning embeddings with detections.
    """
    int_counts = [c for c in counts if c is not None]
    total = sum(int_counts)

    if embs is None:
        if total != 0:
            raise ValueError(
                f"embs is None but counts sum to {total} (expected 0)"
            )
        # Infer the embedding width only matters for non-empty frames; with no
        # crops at all every int-count is 0, so a 0-wide empty array is fine.
        dim = 0
        out: List["np.ndarray | None"] = []
        for c in counts:
            if c is None:
                out.append(None)
            else:
                out.append(np.empty((0, dim), dtype=np.float32))
        return out

    embs = np.asarray(embs)
    if embs.shape[0] != total:
        raise ValueError(
            f"embs has {embs.shape[0]} rows but counts sum to {total}"
        )
    dim = embs.shape[1] if embs.ndim == 2 else 0

    out = []
    offset = 0
    for c in counts:
        if c is None:
            out.append(None)
            continue
        out.append(embs[offset : offset + c])
        offset += c
    return out


# ---------------------------------------------------------------------------
# Strict-order batch/consume loop (pure; the ordering core of the parallel path)
# ---------------------------------------------------------------------------
def batched_consume_loop(
    frame_source: Iterable[Tuple[int, Any]],
    batch_size: int,
    perceive_batch_fn: Callable[[List[Any]], Sequence[Tuple[Any, Any]]],
    consume_fn: Callable[[int, Any, Any], None],
) -> int:
    """Form batches from *frame_source* and drive the consumer in strict order.

    This is the ordering core of the Task-3 parallel path.  It is pure Python
    (no torch / cv2 / yolox) and accepts INJECTED ``perceive_batch_fn`` and
    ``consume_fn`` callables so it can be exercised on a CPU-only machine with
    stubs.

    Contract / invariants
    ---------------------
    * ``frame_source`` yields ``(frame_id, frame)`` in strict ascending
      ``frame_id`` order with no gaps (the prefetch thread guarantees this).
    * Frames are gathered into batches of up to ``batch_size``; the final
      partial batch is handled.
    * ``perceive_batch_fn(frames)`` returns a sequence of ``(det, embs)``
      aligned one-to-one with the frames it was given.
    * ``consume_fn(frame_id, det, embs)`` is called for EVERY frame, exactly
      once, in ascending ``frame_id`` order — ``0, 1, 2, ..., N-1`` with no
      reordering and no gaps.  The ``frame_id`` passed is always the loop index
      (decode order), never any tracker-internal counter.  When a frame's
      ``det`` is ``None``, ``consume_fn`` is still called (it applies the
      sequential skip rule itself: it must NOT advance the tracker and must emit
      no row), so the loop index keeps advancing identically to the sequential
      path.

    Returns
    -------
    int
        The number of frames processed (``N``).
    """
    n_frames = 0
    batch_ids: List[int] = []
    batch_frames: List[Any] = []

    def _flush() -> None:
        results = perceive_batch_fn(batch_frames)
        for fid, (det, embs) in zip(batch_ids, results):
            consume_fn(fid, det, embs)
        batch_ids.clear()
        batch_frames.clear()

    for frame_id, frame in frame_source:
        batch_ids.append(frame_id)
        batch_frames.append(frame)
        n_frames += 1
        if len(batch_frames) == batch_size:
            _flush()

    # Final partial batch.
    if batch_frames:
        _flush()

    return n_frames


# ---------------------------------------------------------------------------
# Pipelined batch/consume loop (overlaps GPU perceive with CPU consume)
# ---------------------------------------------------------------------------
# A drop-in replacement for ``batched_consume_loop`` that runs
# ``perceive_batch_fn`` on a SEPARATE producer thread, so the GPU can compute
# batch N+1 while the (strictly ordered, stateful) consumer processes batch N on
# the calling thread.  This reclaims the per-frame CPU tail — tracking,
# postprocess, host<->device copies — that the serial loop leaves the GPU idle
# for.  The win grows as the detector gets cheaper (e.g. under ``--fp16``), when
# that tail becomes a larger share of wall time.
#
# Correctness is identical to ``batched_consume_loop``:
#   * exactly ONE producer thread iterates ``frame_source`` in order and pushes
#     per-batch results FIFO, so batches arrive in strict frame order;
#   * ``consume_fn`` runs only on the CALLING thread and is invoked for
#     ``frame_id`` 0, 1, 2, ... exactly once each, in order;
#   * the producer touches only the GPU model/extractor (via
#     ``perceive_batch_fn``) and the consumer touches only the tracker (via
#     ``consume_fn``), so there is no shared mutable state between threads.
# Only the *timing* of perceive relative to consume changes, so the emitted
# artifacts are byte-identical to the serial parallel path (up to the same
# floating-point nondeterminism batched GPU kernels already introduce).
_PIPELINE_SENTINEL = object()  # producer pushes this once when the stream ends


def pipelined_consume_loop(
    frame_source: Iterable[Tuple[int, Any]],
    batch_size: int,
    perceive_batch_fn: Callable[[List[Any]], Sequence[Tuple[Any, Any]]],
    consume_fn: Callable[[int, Any, Any], None],
    queue_depth: int = 2,
) -> int:
    """Like :func:`batched_consume_loop`, but overlaps perceive with consume.

    Parameters
    ----------
    frame_source, batch_size, perceive_batch_fn, consume_fn:
        Same contract as :func:`batched_consume_loop`.
    queue_depth:
        Max number of *completed* batches the producer may queue ahead of the
        consumer (bounds memory).  ``2`` lets the producer compute the next batch
        while the consumer drains the current one without running unboundedly
        ahead.

    Returns
    -------
    int
        The number of frames processed (``N``).

    Notes
    -----
    A producer-thread exception is re-raised on the calling thread after the
    queue drains, so failures in detection/ReID surface loudly instead of
    hanging.  Pure-Python (``threading`` + ``queue``); no torch/cv2 import, so it
    is unit-testable on a CPU-only box with stub callables.
    """
    maxsize = max(1, queue_depth)
    results_q: "queue.Queue" = queue.Queue(maxsize=maxsize)
    error_box: List[BaseException] = []

    def _producer() -> None:
        batch_ids: List[int] = []
        batch_frames: List[Any] = []

        def _emit() -> None:
            results = perceive_batch_fn(batch_frames)
            # Copy ids/results so the next batch can reuse the working lists.
            results_q.put((list(batch_ids), list(results)))
            batch_ids.clear()
            batch_frames.clear()

        try:
            for frame_id, frame in frame_source:
                batch_ids.append(frame_id)
                batch_frames.append(frame)
                if len(batch_frames) == batch_size:
                    _emit()
            if batch_frames:  # final partial batch
                _emit()
        except BaseException as exc:  # noqa: BLE001 - surfaced to caller thread
            error_box.append(exc)
        finally:
            # Always signal end-of-stream so the consumer never blocks forever.
            results_q.put(_PIPELINE_SENTINEL)

    worker = threading.Thread(target=_producer, name="stage1-perceive", daemon=True)
    worker.start()

    n_frames = 0
    try:
        while True:
            item = results_q.get()
            if item is _PIPELINE_SENTINEL:
                break
            batch_ids, results = item
            for fid, (det, embs) in zip(batch_ids, results):
                consume_fn(fid, det, embs)
                n_frames += 1
    finally:
        # If the consumer stopped early (e.g. consume_fn raised), the producer
        # may be blocked on a full queue.  Drain so it can reach its sentinel,
        # then join with a timeout so we never hang forever.
        try:
            while results_q.get_nowait() is not _PIPELINE_SENTINEL:
                pass
        except queue.Empty:
            pass
        worker.join(timeout=5.0)

    if error_box:
        raise error_box[0]

    return n_frames

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
   MOT row).

The caller (``stage1_track.py``) feeds records in strict frame order; this
module performs no cross-frame inference of its own.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


def _array_like_to_numpy(value: Any) -> "np.ndarray":
    """Convert torch/numpy-like values to a NumPy array without importing torch."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def ultralytics_result_to_yolox_output(result: Any) -> "np.ndarray | None":
    """Convert one Ultralytics detection result to DeepEIoU's YOLOX row shape.

    Output rows are ``[x1, y1, x2, y2, score, class_conf, class_id]``.  The
    tracker multiplies ``score * class_conf`` for 7-column rows, so
    ``class_conf`` is fixed at 1.0 to preserve Ultralytics' confidence as the
    effective tracking score.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return None

    xyxy = _array_like_to_numpy(getattr(boxes, "xyxy", None))
    if xyxy.size == 0:
        return None
    xyxy = xyxy.reshape(-1, 4).astype(np.float32, copy=False)

    conf = _array_like_to_numpy(getattr(boxes, "conf", None)).reshape(-1, 1)
    cls = _array_like_to_numpy(getattr(boxes, "cls", None)).reshape(-1, 1)
    if conf.shape[0] != xyxy.shape[0] or cls.shape[0] != xyxy.shape[0]:
        raise ValueError(
            "Ultralytics result has mismatched boxes/conf/classes: "
            f"{xyxy.shape[0]} boxes, {conf.shape[0]} confidences, {cls.shape[0]} classes"
        )

    class_conf = np.ones_like(conf, dtype=np.float32)
    return np.concatenate(
        [
            xyxy,
            conf.astype(np.float32, copy=False),
            class_conf,
            cls.astype(np.float32, copy=False),
        ],
        axis=1,
    )


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

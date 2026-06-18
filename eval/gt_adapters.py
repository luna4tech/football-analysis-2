"""
eval.gt_adapters — ground-truth (and prediction) loaders for tracking eval.

Canonical representation
------------------------
Both GT and predictions are carried as a plain ``numpy.ndarray`` of shape
``(N, 9)`` and dtype ``float64`` with the MOTChallenge column order::

    frame, id, x, y, w, h, conf, class, visibility

``(x, y)`` is the top-left corner; ``(w, h)`` the box size. This is the same
column order TrackEval's MOTChallenge dataset reads, so the layout writer in
``trackeval_runner`` can emit rows almost verbatim (only the pred frame base is
shifted — see that module).

A small dataclass :class:`Tracks` wraps the array together with the original
frame base (``frame_base``: ``1`` for MOTChallenge GT, ``0`` for the pipeline's
``refined.txt``). This makes the frame convention explicit at the type level so
the 0->1 conversion can be centralized in exactly one place (the layout writer).

Frame conventions (the #1 silent eval bug)
------------------------------------------
* MOTChallenge ``gt.txt``  -> 1-based frames. Loaded AS-IS (``frame_base=1``).
* pipeline ``refined.txt`` -> 0-based frames. Loaded AS-IS (``frame_base=0``);
  the +1 shift happens once, centrally, in ``trackeval_runner.materialize_layout``.

GT loader seam
--------------
:func:`load_gt` dispatches on a ``fmt`` string through the module-level
:data:`GT_LOADERS` registry (``fmt -> loader``). The MOTChallenge loader is
registered here. A future custom-export converter plugs in WITHOUT touching the
eval core by calling :func:`register_gt_loader("my_fmt", my_loader)` (or using
the :func:`gt_loader` decorator); ``load_gt(path, fmt="my_fmt")`` then routes to
it. A registered loader must return a :class:`Tracks` with ``frame_base=1``
(i.e. it is responsible for converting its native frame base to MOTChallenge
1-based, so the rest of the eval core can treat all GT identically).

Import-light: stdlib + numpy only. No torch / TrackEval.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict

import numpy as np

# Canonical MOTChallenge column order shared by GT and predictions.
COLUMNS = ("frame", "id", "x", "y", "w", "h", "conf", "class", "visibility")
N_COLS = len(COLUMNS)

# Defaults used when a source row omits the trailing MOTChallenge columns.
_DEFAULT_CONF = 1.0
_DEFAULT_CLASS = 1.0
_DEFAULT_VISIBILITY = 1.0


@dataclass
class Tracks:
    """A canonical set of tracking rows plus their frame base.

    Attributes
    ----------
    rows:
        ``(N, 9)`` float64 array in MOTChallenge column order
        (``frame,id,x,y,w,h,conf,class,visibility``).
    frame_base:
        The frame numbering of ``rows`` AS LOADED: ``1`` for MOTChallenge
        (already 1-based), ``0`` for the pipeline's 0-based ``refined.txt``.
        The eval core converts predictions to 1-based exactly once (in the
        layout writer); this field records the source convention so that
        conversion is unambiguous and never applied twice.
    """

    rows: np.ndarray
    frame_base: int = 1

    def __post_init__(self) -> None:
        arr = np.asarray(self.rows, dtype=np.float64)
        if arr.ndim != 2 or (arr.size and arr.shape[1] != N_COLS):
            raise ValueError(
                f"Tracks.rows must be (N, {N_COLS}); got shape {arr.shape}"
            )
        if arr.size == 0:
            arr = arr.reshape(0, N_COLS)
        self.rows = arr
        if self.frame_base not in (0, 1):
            raise ValueError(f"frame_base must be 0 or 1; got {self.frame_base!r}")

    def __len__(self) -> int:
        return int(self.rows.shape[0])

    @property
    def frames(self) -> np.ndarray:
        """The frame column (as loaded, in this object's ``frame_base``)."""
        return self.rows[:, 0]

    def max_frame(self) -> int:
        """Max frame number AS LOADED (0 for an empty set)."""
        if self.rows.shape[0] == 0:
            return 0
        return int(self.frames.max())


# ---------------------------------------------------------------------------
# Low-level row parser (shared by the GT and pred loaders)
# ---------------------------------------------------------------------------


def _parse_mot_rows(path: "str | os.PathLike[str]") -> np.ndarray:
    """Parse a comma-separated MOT-style ``.txt`` into an ``(N, 9)`` array.

    Tolerant of:
      * blank lines and rows with fewer than 6 columns (skipped),
      * a trailing ``conf`` / ``class`` / ``visibility`` that may be absent,
        ``""`` or ``-1`` (defaults are filled in),
      * extra columns beyond the 9th (ignored),
      * values written as floats (e.g. ``1.0``) or ints.

    Returns rows in MOTChallenge column order; the frame column is left exactly
    as written (no base conversion happens here).
    """
    out: list[list[float]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            fields = line.split(",")
            if len(fields) < 6:
                continue

            frame = int(float(fields[0]))
            tid = int(float(fields[1]))
            x, y, w, h = (float(v) for v in fields[2:6])

            conf = _field_or_default(fields, 6, _DEFAULT_CONF)
            cls = _field_or_default(fields, 7, _DEFAULT_CLASS)
            vis = _field_or_default(fields, 8, _DEFAULT_VISIBILITY)

            out.append([float(frame), float(tid), x, y, w, h, conf, cls, vis])

    if not out:
        return np.zeros((0, N_COLS), dtype=np.float64)
    return np.asarray(out, dtype=np.float64)


def _field_or_default(fields: list[str], idx: int, default: float) -> float:
    """Return ``float(fields[idx])`` or *default* if absent / empty / ``-1``."""
    if idx >= len(fields):
        return default
    token = fields[idx].strip()
    if token in ("", "-1", "-1.0"):
        return default
    return float(token)


# ---------------------------------------------------------------------------
# GT loader seam (fmt -> loader registry)
# ---------------------------------------------------------------------------

# Registry mapping a GT format name to a loader ``(path) -> Tracks``. A loader
# MUST return a Tracks with frame_base == 1 (MOTChallenge 1-based), converting
# from its native base if necessary. This is the seam a future custom-export
# converter registers into.
GT_LOADERS: Dict[str, Callable[["str | os.PathLike[str]"], Tracks]] = {}


def register_gt_loader(
    fmt: str, loader: Callable[["str | os.PathLike[str]"], Tracks]
) -> None:
    """Register *loader* under format name *fmt* in :data:`GT_LOADERS`.

    The loader receives a path and must return a :class:`Tracks` whose
    ``frame_base`` is ``1`` (1-based MOTChallenge frames). Registering an
    already-present *fmt* overwrites it.
    """
    GT_LOADERS[fmt] = loader


def gt_loader(fmt: str):
    """Decorator form of :func:`register_gt_loader`.

    Usage (in a separate module, no edit to the eval core needed)::

        from eval.gt_adapters import gt_loader, Tracks

        @gt_loader("my_export")
        def load_my_export(path) -> Tracks:
            ...  # parse + convert to 1-based
            return Tracks(rows, frame_base=1)
    """

    def _decorate(loader):
        register_gt_loader(fmt, loader)
        return loader

    return _decorate


def load_gt(path: "str | os.PathLike[str]", fmt: str = "motchallenge") -> Tracks:
    """Load ground truth from *path* using the loader registered for *fmt*.

    Parameters
    ----------
    path:
        Path to the GT file.
    fmt:
        Format key; ``"motchallenge"`` is built in. Custom formats can be added
        via :func:`register_gt_loader` / :func:`gt_loader`.

    Returns
    -------
    Tracks
        Canonical GT with ``frame_base == 1`` (1-based frames).

    Raises
    ------
    ValueError
        If *fmt* is not registered (the message lists the known formats).
    """
    try:
        loader = GT_LOADERS[fmt]
    except KeyError:
        known = ", ".join(sorted(GT_LOADERS)) or "(none)"
        raise ValueError(
            f"unknown GT format {fmt!r}; registered formats: {known}"
        ) from None
    return loader(path)


def load_gt_motchallenge(path: "str | os.PathLike[str]") -> Tracks:
    """Load a standard MOTChallenge ``gt.txt`` (1-based frames) AS-IS.

    The file uses the column order ``frame,id,x,y,w,h,conf,class,visibility``
    and 1-based frames; we keep both exactly, returning ``frame_base=1``.
    """
    rows = _parse_mot_rows(path)
    return Tracks(rows=rows, frame_base=1)


# Register the built-in MOTChallenge GT loader.
register_gt_loader("motchallenge", load_gt_motchallenge)


# ---------------------------------------------------------------------------
# Prediction loader (pipeline refined.txt — 0-based)
# ---------------------------------------------------------------------------


def load_pred(path: "str | os.PathLike[str]") -> Tracks:
    """Load the pipeline's ``refined.txt`` (``frame,id,x,y,w,h,score,...``).

    Frames are **0-based** (matching ``demo.py`` / the pipeline output), so the
    returned :class:`Tracks` has ``frame_base == 0``. The per-detection
    ``score`` is kept in the ``conf`` column. The +1 shift to 1-based is applied
    once, centrally, by ``trackeval_runner.materialize_layout`` — NOT here.
    """
    rows = _parse_mot_rows(path)
    return Tracks(rows=rows, frame_base=0)

"""
pipeline.compare_tracks — tolerant MOT-file diff (pure stdlib + numpy).

Used for the parallel-vs-sequential equivalence check.  Stage 1's parallel path
is only equal to the sequential path "up to floating-point nondeterminism"
(batched GPU kernels differ from single-image ones in the last FP bits, which can
*rarely* flip a detection sitting exactly on a confidence/NMS threshold).  A
strict byte diff is therefore the WRONG tool — this tolerant comparator is the
right one.

What it does
------------
Parse two MOT ``.txt`` files (``frame,id,x,y,w,h,score,...``), match rows by the
``(frame, id)`` key, and report:

  * whether the row counts match,
  * the set of keys present only in A / only in B,
  * the max and mean absolute difference of the box coords (x, y, w, h) over the
    keys common to both files.

Two files are deem ``equivalent`` iff their key sets are identical AND the max
absolute box-coordinate difference over common keys is ``<= tol`` (default
``1.0`` px).

This module is import-light (no torch / cv2) so it runs on a CPU-only host.

Public API
----------
parse_mot(path) -> dict[(frame, id) -> (x, y, w, h)]
compare_tracks(a, b, tol=1.0) -> CompareResult
CompareResult                  — dataclass-like result object.
main(argv=None) -> int         — CLI entry used by ``python -m pipeline compare``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# A MOT row key is (frame, id); the value is the box (x, y, w, h).
Key = Tuple[int, int]
Box = Tuple[float, float, float, float]


def parse_mot(path: "str | Path") -> Dict[Key, Box]:
    """Parse a MOT ``.txt`` into ``{(frame, id): (x, y, w, h)}``.

    Each line is ``frame,id,x,y,w,h,score,...`` (extra trailing columns are
    ignored).  Blank lines and lines with fewer than 6 fields are skipped.

    Parameters
    ----------
    path:
        Path to a MOT-format results file.

    Returns
    -------
    dict
        Maps each ``(frame, id)`` key to its ``(x, y, w, h)`` box.  If a key
        repeats within a file the LAST occurrence wins (MOT files should not
        repeat a key within a frame, so this is just defensive).
    """
    boxes: Dict[Key, Box] = {}
    # utf-8-sig transparently strips a leading BOM if one is present (some
    # editors/tools write one); plain UTF-8 files are unaffected.
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            fields = line.split(",")
            if len(fields) < 6:
                continue
            frame = int(float(fields[0]))
            tid = int(float(fields[1]))
            x, y, w, h = (float(v) for v in fields[2:6])
            boxes[(frame, tid)] = (x, y, w, h)
    return boxes


class CompareResult:
    """Outcome of a tolerant MOT comparison.

    Attributes
    ----------
    equivalent : bool
        ``True`` iff key sets match AND ``max_coord_diff <= tol``.
    tol : float
        The tolerance used (px).
    n_a, n_b : int
        Row counts of file A / file B.
    row_count_match : bool
        ``n_a == n_b``.
    only_in_a, only_in_b : list[(frame, id)]
        Sorted keys present in exactly one file.
    n_common : int
        Number of keys present in both files.
    max_coord_diff : float | None
        Largest absolute (x|y|w|h) difference over common keys; ``None`` when
        there are no common keys.
    mean_coord_diff : float | None
        Mean absolute coord difference over common keys; ``None`` when there are
        no common keys.
    """

    def __init__(
        self,
        *,
        equivalent: bool,
        tol: float,
        n_a: int,
        n_b: int,
        only_in_a: List[Key],
        only_in_b: List[Key],
        n_common: int,
        max_coord_diff: Optional[float],
        mean_coord_diff: Optional[float],
    ) -> None:
        self.equivalent = equivalent
        self.tol = tol
        self.n_a = n_a
        self.n_b = n_b
        self.row_count_match = n_a == n_b
        self.only_in_a = only_in_a
        self.only_in_b = only_in_b
        self.n_common = n_common
        self.max_coord_diff = max_coord_diff
        self.mean_coord_diff = mean_coord_diff

    @property
    def keys_match(self) -> bool:
        """True iff neither file has any key the other lacks."""
        return not self.only_in_a and not self.only_in_b

    def report(self) -> str:
        """Render a human-readable multi-line report."""
        lines = [
            f"rows:            A={self.n_a}  B={self.n_b}  "
            f"(match={self.row_count_match})",
            f"common keys:     {self.n_common}",
            f"only in A:       {len(self.only_in_a)}",
            f"only in B:       {len(self.only_in_b)}",
        ]
        if self.max_coord_diff is None:
            lines.append("coord diff:      N/A (no common keys)")
        else:
            lines.append(
                f"coord diff:      max={self.max_coord_diff:.4f}  "
                f"mean={self.mean_coord_diff:.4f}  (tol={self.tol})"
            )
        # Show a few example offending keys to aid debugging.
        if self.only_in_a:
            lines.append(f"  e.g. only-in-A: {self.only_in_a[:5]}")
        if self.only_in_b:
            lines.append(f"  e.g. only-in-B: {self.only_in_b[:5]}")
        verdict = "EQUIVALENT" if self.equivalent else "NOT EQUIVALENT"
        lines.append(f"verdict:         {verdict}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CompareResult(equivalent={self.equivalent}, "
            f"max_coord_diff={self.max_coord_diff})"
        )


def compare_tracks(
    a: "str | Path",
    b: "str | Path",
    tol: float = 1.0,
) -> CompareResult:
    """Tolerantly compare two MOT files; see module docstring.

    Parameters
    ----------
    a, b:
        Paths to the two MOT ``.txt`` files (e.g. a sequential and a parallel
        ``tracks.txt``).
    tol:
        Max allowed absolute box-coordinate difference (px) for the files to be
        considered equivalent.  Default ``1.0``.

    Returns
    -------
    CompareResult
    """
    boxes_a = parse_mot(a)
    boxes_b = parse_mot(b)

    keys_a = set(boxes_a)
    keys_b = set(boxes_b)
    only_in_a = sorted(keys_a - keys_b)
    only_in_b = sorted(keys_b - keys_a)
    common = keys_a & keys_b

    max_diff: Optional[float] = None
    mean_diff: Optional[float] = None
    if common:
        # Stack the (x, y, w, h) of every common key into (N, 4) arrays and take
        # the elementwise absolute difference.
        arr_a = np.array([boxes_a[k] for k in common], dtype=np.float64)
        arr_b = np.array([boxes_b[k] for k in common], dtype=np.float64)
        abs_diff = np.abs(arr_a - arr_b)
        max_diff = float(abs_diff.max())
        mean_diff = float(abs_diff.mean())

    keys_match = not only_in_a and not only_in_b
    # Equivalent iff key sets identical AND every common box within tol.
    # (max_diff is None only when there are no common keys at all; if that
    # happens while keys_match is True both files are empty -> equivalent.)
    within_tol = max_diff is None or max_diff <= tol
    equivalent = keys_match and within_tol

    return CompareResult(
        equivalent=equivalent,
        tol=tol,
        n_a=len(boxes_a),
        n_b=len(boxes_b),
        only_in_a=only_in_a,
        only_in_b=only_in_b,
        n_common=len(common),
        max_coord_diff=max_diff,
        mean_coord_diff=mean_diff,
    )


def build_parser() -> argparse.ArgumentParser:
    """Argparse parser for ``python -m pipeline compare``."""
    parser = argparse.ArgumentParser(
        prog="python -m pipeline compare",
        description=(
            "Tolerant MOT-file diff. Matches rows by (frame, id) and reports "
            "key-set differences + max/mean abs box-coord diff. Files are "
            "'equivalent' iff key sets match and max coord diff <= --tol. Use "
            "this for the parallel-vs-sequential check (parity is "
            "floating-point nondeterministic, so a strict diff is wrong)."
        ),
    )
    parser.add_argument("--a", required=True, help="path to MOT file A")
    parser.add_argument("--b", required=True, help="path to MOT file B")
    parser.add_argument(
        "--tol",
        type=float,
        default=1.0,
        help="max allowed abs box-coord difference in px (default: 1.0)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry: parse args, compare, print the report, return an exit code.

    Returns ``0`` when the files are equivalent, ``1`` otherwise — so it can be
    used directly in a CI / smoke check.
    """
    args = build_parser().parse_args(argv)
    result = compare_tracks(args.a, args.b, tol=args.tol)
    print(result.report())
    return 0 if result.equivalent else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

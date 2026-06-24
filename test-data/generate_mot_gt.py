#!/usr/bin/env python3
"""Generate MOT ground-truth text files from the ``consolidated_xy`` export.

The consolidated export (see ``consolidated_xy_format.md``) stores per-second JSON
"chunk" files of corrected detections.  This script flattens them into the standard
10-column MOT format, with the trailing ``-1,-1,-1`` repurposed to ``class_id,team_id,-1``:

    frame, id, bb_left, bb_top, bb_w, bb_h, conf, class_id, team_id, -1

Column derivation (per detection):
    frame     re-indexed-from-1 frame number (see "Frame numbering" below)
    id        int(track_id)
    bb_left   bbox[0]                      bb_w  bbox[2] - bbox[0]
    bb_top    bbox[1]                      bb_h  bbox[3] - bbox[1]
    conf      1   (constant)
    class_id  detection class_id, raw      0 ball / 1 goalkeeper / 2 player / 3 referee
    team_id   detection team, raw          -1 unknown / 0 team A / 1 team B
    (col 10)  -1  (constant)

Defaults (matching the sample ``gt_mot_video*.txt`` files):
  * The **ball** (class_id 0, ``b_*`` ids) is **excluded**.  Pass ``--include-ball`` to keep it.
  * ``class_id`` / ``team_id`` are written **raw**.

Time range:
  * ``--start`` / ``--end`` are given in **seconds** and accept fractional values
    (e.g. ``4432.40``).  Omit either to default to the data start / data end.

Frame numbering (25 fps; global frame = round(timestamp * 25)):
  * Ranges are **half-open** ``[start, end)`` (start-inclusive, end-exclusive) and
    re-indexed so that frame 1 == the first frame of the range.
  * ``offset = round(start * 25) - 1``; ``mot_frame = global_frame - offset``; a frame is
    kept while ``round(start*25) <= global_frame < round(end*25)``.
  * Frame 1 is the frame AT ``--start`` (e.g. ``--start 4432.40`` -> frame 1 == 4432.40),
    and a range yields exactly ``(end - start) * 25`` frame slots -- the same convention
    used to extract the sample clips (``video2780-2960`` -> frames 1..4500).
  * With no range, frame 1 == the first frame present in the data and the last data frame
    is included; gaps are preserved (a frame with no detections contributes no rows).

Examples:
    python generate_mot_gt.py                              # all frames -> gt_mot_4508-8000.txt
    python generate_mot_gt.py --start 4600 --end 4700      # [4600, 4700) -> 2500 frames
    python generate_mot_gt.py --start 4432.40 --end 4500   # fractional start
    python generate_mot_gt.py --end 5000                   # data start .. second 5000 (excl.)
    python generate_mot_gt.py --include-ball -o /tmp/out.txt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

FPS = 25                  # frames per second; global_frame = round(timestamp * FPS)
BALL_CLASS_ID = 0         # detector class for the ball
BALL_ID_BASE = 9_000_000  # namespace offset for ball ids when --include-ball is set


def to_global_frame(timestamp_key: str) -> int:
    """Convert a chunk timestamp key (e.g. ``"4508.04"``) to its global integer frame."""
    return round(float(timestamp_key) * FPS)


def fmt_coord(value: float) -> str:
    """Format a bbox coordinate like the sample GT (``427.0``), preserving precision."""
    f = float(value)
    return f"{f:.1f}" if f.is_integer() else repr(f)


def label(value: float) -> str:
    """Filename-friendly label: ``4508`` for integers, ``4432.4`` for fractionals."""
    f = float(value)
    return str(int(f)) if f.is_integer() else repr(f)


def discover_seconds(chunks_dir: Path) -> list[int]:
    """Return the sorted list of whole seconds for which a chunk file exists."""
    seconds = []
    for path in chunks_dir.glob("*.json"):
        try:
            seconds.append(int(path.stem))
        except ValueError:
            continue  # ignore non-numeric files
    return sorted(seconds)


def resolve_offset(chunks_dir: Path, start: float | None, first_sec: int) -> int:
    """Frame-number offset so the first frame of the range becomes MOT frame 1.

    With an explicit ``--start`` we anchor to the frame AT ``start`` (start-inclusive):
    ``offset = round(start*FPS) - 1``.  Otherwise we anchor to the first frame actually
    present in the data so the default never drops a leading frame.
    """
    if start is not None:
        return round(start * FPS) - 1
    first_chunk = json.loads((chunks_dir / f"{first_sec:05d}.json").read_text())
    first_global = min(to_global_frame(k) for k in first_chunk)
    return first_global - 1


def build_rows(chunks_dir, seconds, offset, end, include_ball):
    """Read the selected chunk files and return sorted MOT rows + a small stats dict.

    ``end`` (seconds, or ``None``) is the *exclusive* upper bound: a frame is kept while
    ``mot_frame >= 1`` and ``global_frame < round(end*FPS)``.
    """
    rows = []
    end_global = round(end * FPS) if end is not None else None
    frames_with_data = set()
    for sec in seconds:
        chunk = json.loads((chunks_dir / f"{sec:05d}.json").read_text())
        for key, detections in chunk.items():
            global_frame = to_global_frame(key)
            frame = global_frame - offset
            if frame < 1:
                continue
            if end_global is not None and global_frame >= end_global:
                continue
            for det in detections:
                class_id = det["class_id"]
                track_raw = det["track_id"]
                if track_raw.startswith("b_"):          # ball
                    if not include_ball:
                        continue
                    track_id = BALL_ID_BASE + int(track_raw[2:])
                elif class_id == BALL_CLASS_ID:          # defensive: any other ball coding
                    if not include_ball:
                        continue
                    track_id = int(track_raw)
                else:
                    track_id = int(track_raw)
                x1, y1, x2, y2 = det["bbox"]
                rows.append((frame, track_id, x1, y1, x2 - x1, y2 - y1,
                             class_id, det["team"]))
                frames_with_data.add(frame)
    rows.sort(key=lambda r: (r[0], r[1]))
    stats = {
        "rows": len(rows),
        "frames_with_data": len(frames_with_data),
        "min_frame": min(frames_with_data) if frames_with_data else None,
        "max_frame": max(frames_with_data) if frames_with_data else None,
    }
    return rows, stats


def write_mot(rows, out_path: Path) -> None:
    with out_path.open("w", newline="\n") as fh:
        for frame, tid, left, top, width, height, class_id, team in rows:
            fh.write(
                f"{frame},{tid},{fmt_coord(left)},{fmt_coord(top)},"
                f"{fmt_coord(width)},{fmt_coord(height)},1,{class_id},{team},-1\n"
            )


def parse_args(argv=None):
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate MOT ground-truth from the consolidated_xy export.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--chunks-dir", type=Path,
                        default=here / "consolidated_xy" / "chunks",
                        help="directory of per-second chunk JSON files "
                             "(default: ./consolidated_xy/chunks)")
    parser.add_argument("--start", type=float, default=None,
                        help="range start in seconds (fractional ok), INCLUSIVE. "
                             "Default: data start.")
    parser.add_argument("--end", type=float, default=None,
                        help="range end in seconds (fractional ok), EXCLUSIVE. "
                             "Default: data end (inclusive).")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="output file path "
                             "(default: ./gt_mot_<start>-<end>.txt next to this script)")
    parser.add_argument("--include-ball", action="store_true",
                        help="include the ball (class_id 0); its id is namespaced as "
                             f"{BALL_ID_BASE}+n. Default: excluded, matching the sample GT.")
    return parser.parse_args(argv), here


def main(argv=None) -> int:
    args, here = parse_args(argv)

    chunks_dir = args.chunks_dir
    if not chunks_dir.is_dir():
        sys.exit(f"error: chunks directory not found: {chunks_dir}")

    all_seconds = discover_seconds(chunks_dir)
    if not all_seconds:
        sys.exit(f"error: no chunk files found in {chunks_dir}")
    data_start, data_end = all_seconds[0], all_seconds[-1]

    if args.start is not None and args.end is not None and args.start >= args.end:
        sys.exit(f"error: --start ({args.start}) must be < --end ({args.end})")

    # Whole-second chunk files we need to read (the per-frame bounds do the exact cutoff).
    lo = args.start if args.start is not None else data_start
    hi = args.end if args.end is not None else data_end
    file_lo, file_hi = math.floor(lo), math.floor(hi)
    seconds = [s for s in all_seconds if file_lo <= s <= file_hi]
    if not seconds:
        sys.exit(f"error: no chunk files in range [{lo}, {hi}] "
                 f"(data covers seconds {data_start}..{data_end})")

    offset = resolve_offset(chunks_dir, args.start, seconds[0])
    rows, stats = build_rows(chunks_dir, seconds, offset, args.end, args.include_ball)

    eff_start = args.start if args.start is not None else data_start
    eff_end = args.end if args.end is not None else data_end
    out_path = args.output or (here / f"gt_mot_{label(eff_start)}-{label(eff_end)}.txt")
    write_mot(rows, out_path)

    end_note = "exclusive" if args.end is not None else "data end, inclusive"
    print(f"Wrote {stats['rows']} rows across {stats['frames_with_data']} frames "
          f"(frame {stats['min_frame']}..{stats['max_frame']}) to {out_path}")
    print(f"Range: [{eff_start}, {eff_end}) seconds ({end_note})  |  ball "
          f"{'included' if args.include_ball else 'excluded'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

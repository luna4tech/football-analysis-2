"""Count the unique tracks in a MOT-format .txt written by demo.py.

The .txt is the standard MOT format:

    frame,id,x,y,w,h,score,-1,-1,-1

The unique track count is just the number of distinct values in the `id`
column (field 2). With --min_len you can ignore short-lived tracks (those
that appear in fewer than N frames), which are usually ID-switch noise.

Example:
    python tools/count_tracks.py YOLOX_outputs/.../2026_06_09_10_18_01.txt
    python tools/count_tracks.py results.txt --min_len 5 --verbose
"""

import argparse
import os.path as osp
from collections import Counter


def make_parser():
    parser = argparse.ArgumentParser("Count unique tracks in a MOT .txt")
    parser.add_argument("txt", help="path to the MOT-format results .txt")
    parser.add_argument(
        "--min_len",
        type=int,
        default=1,
        help="only count tracks seen in at least this many frames (default: 1)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="also print each track id and how many frames it spans",
    )
    return parser


def count_tracks(txt_path, min_len=1):
    """Return (kept_ids_sorted, frames_per_id, n_rows, frame_min, frame_max)."""
    frames_per_id = Counter()
    n_rows = 0
    frame_min = None
    frame_max = None
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.split(",")
            if len(fields) < 2:
                continue
            frame = int(float(fields[0]))
            tid = int(float(fields[1]))
            frames_per_id[tid] += 1
            n_rows += 1
            frame_min = frame if frame_min is None else min(frame_min, frame)
            frame_max = frame if frame_max is None else max(frame_max, frame)
    kept = sorted(tid for tid, c in frames_per_id.items() if c >= min_len)
    return kept, frames_per_id, n_rows, frame_min, frame_max


def main():
    args = make_parser().parse_args()
    if not osp.isfile(args.txt):
        raise FileNotFoundError("results txt not found: {}".format(args.txt))

    kept, frames_per_id, n_rows, frame_min, frame_max = count_tracks(
        args.txt, args.min_len
    )

    print("file:           {}".format(args.txt))
    print("total boxes:    {}".format(n_rows))
    if frame_min is not None:
        span = frame_max - frame_min + 1
        print("frame range:    {}..{} ({} frames)".format(frame_min, frame_max, span))
    print("unique tracks:  {}".format(len(frames_per_id)))
    if args.min_len > 1:
        print("tracks (>= {} frames): {}".format(args.min_len, len(kept)))

    if args.verbose:
        print("\n{:>10}  {:>12}".format("track_id", "n_frames"))
        for tid in sorted(frames_per_id):
            print("{:>10}  {:>12}".format(tid, frames_per_id[tid]))


if __name__ == "__main__":
    main()

"""Render an annotated tracking video from a source video + a MOT-format .txt.

This is the companion to `tools/demo.py --no_save_video`: run the (faster)
tracking pass without drawing anything, then rebuild the annotated video offline
whenever you actually need to look at it.

The .txt is the standard MOT format written by demo.py:

    frame,id,x,y,w,h,score,-1,-1,-1

where (x, y) is the top-left corner and (w, h) the box size. `frame` is 0-based,
matching demo.py's output; pass --one_indexed for 1-based files (e.g. the MOT
challenge / SportsMOT convention).

Only cv2 + numpy are needed -- no torch/GPU -- so this can run anywhere.

Example:
    python tools/render_from_txt.py \
        --path /path/to/M59-5min-1.mp4 \
        --txt  YOLOX_outputs/yolox_x_ch_sportsmot/track_vis/2026_06_09_10_18_01.txt
"""

import argparse
import importlib.util
import json
import logging
import os
import os.path as osp
from collections import defaultdict

# NOTE: cv2 is imported lazily (inside main / the drawing path) so this module
# and its pure functions (parse_results, category_color_and_label,
# load_track_attributes) import — and unit-test — without cv2 installed.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("render_from_txt")


def load_plot_tracking():
    """Load plot_tracking from yolox/utils/visualize.py directly.

    Loading the file by path avoids importing the heavy `yolox` package (which
    pulls in torch); visualize.py itself only needs cv2 + numpy.
    """
    this_dir = osp.dirname(osp.abspath(__file__))
    vis_path = osp.join(this_dir, "..", "yolox", "utils", "visualize.py")
    if osp.isfile(vis_path):
        spec = importlib.util.spec_from_file_location("deepeiou_visualize", vis_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.plot_tracking
    # Fallback: rely on the package being importable (matches demo.py).
    from yolox.utils.visualize import plot_tracking
    return plot_tracking


# Canonical class ids (must match the detector / Stage-3 output).
PLAYER_CLASS = 0
GOALKEEPER_CLASS = 1
REFEREE_CLASS = 2
UNKNOWN_CLASS = -1
UNKNOWN_TEAM = -1

# Category box colors, BGR (OpenCV order). Chosen to be visually distinct and to
# keep the red (0, 0, 255) label text legible on top of each:
#   C_TEAM0  cyan/teal      (B,G,R) = (200, 200,   0)
#   C_TEAM1  yellow         (B,G,R) = (  0, 220, 220)
#   C_GK     magenta/purple (B,G,R) = (220,   0, 220)
#   C_REF    green          (B,G,R) = (  0, 200,   0)
C_TEAM0 = (200, 200, 0)
C_TEAM1 = (0, 220, 220)
C_GK = (220, 0, 220)
C_REF = (0, 200, 0)


def legacy_color(track_id):
    """The original per-id color (mirrors visualize.get_color(abs(id))).

    Reimplemented here so the legacy fallback stays a pure function — importable
    and testable without cv2 (visualize.py imports cv2 at module load).
    """
    idx = abs(int(track_id)) * 3
    return ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)


def category_color_and_label(track_id, class_id, team_id):
    """Map a track's (class, team) to a box color (BGR) and a label string.

    Policy (see Task 004 / spec section 3):
      * player, team 0 -> C_TEAM0, label "<id>"
      * player, team 1 -> C_TEAM1, label "<id>"
      * goalkeeper      -> C_GK,    label "gk:<id>"
      * referee         -> C_REF,   label "<id>"
      * unknown / legacy (class == -1 and team == -1) -> legacy_color(id), "<id>"

    Pure function (no cv2): unit-testable on its own.
    """
    tid = int(track_id)
    cid = int(class_id)
    team = int(team_id)
    label = "{}".format(tid)

    if cid == GOALKEEPER_CLASS:
        return C_GK, "gk:{}".format(tid)
    if cid == REFEREE_CLASS:
        return C_REF, label
    if cid == PLAYER_CLASS:
        if team == 0:
            return C_TEAM0, label
        if team == 1:
            return C_TEAM1, label
        # player with unknown team -> legacy fallback.
        return legacy_color(tid), label
    # Unknown / legacy class (includes class == -1 and team == -1): byte-for-byte
    # identical to the original render (color by id, bare id label).
    return legacy_color(tid), label


def load_track_attributes(attr_path):
    """Load 03_team/track_attributes.json into {id: (class_id, team_id)}.

    The JSON shape is {"<id>": {"class": int, "team": int, "gk": bool}}. Returns
    an empty dict if the file is missing or unreadable (legacy fallback).
    """
    if not attr_path or not osp.isfile(attr_path):
        return {}
    try:
        with open(attr_path, "r") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        logger.warning("could not read track attributes %s: %s", attr_path, exc)
        return {}
    out = {}
    for key, entry in data.items():
        try:
            tid = int(key)
            cid = int(entry.get("class", UNKNOWN_CLASS))
            team = int(entry.get("team", UNKNOWN_TEAM))
        except (TypeError, ValueError):
            continue
        out[tid] = (cid, team)
    return out


def find_attributes_path(txt_path, explicit=None):
    """Resolve track_attributes.json: explicit arg, else sibling of the txt."""
    if explicit:
        return explicit
    sibling = osp.join(osp.dirname(osp.abspath(txt_path)), "track_attributes.json")
    return sibling if osp.isfile(sibling) else None


def make_parser():
    parser = argparse.ArgumentParser("Render annotated video from MOT .txt")
    parser.add_argument("--path", required=True, help="path to the source video")
    parser.add_argument("--txt", required=True, help="path to the MOT-format results .txt")
    parser.add_argument(
        "--attributes",
        default=None,
        help=(
            "path to track_attributes.json for stable per-track class/team "
            "(default: auto-detect a sibling of --txt)"
        ),
    )
    parser.add_argument(
        "--save_path",
        default=None,
        help="output video path (default: <txt_dir>/<txt_name>_rendered.mp4)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="output frame rate (default: source video fps)",
    )
    parser.add_argument(
        "--one_indexed",
        action="store_true",
        help="treat the frame column as 1-based (MOT/SportsMOT). demo.py output is 0-based.",
    )
    parser.add_argument(
        "--line_thickness",
        type=int,
        default=1,
        help="bounding-box line thickness in px (default: 1; demo.py uses 3)",
    )
    parser.add_argument(
        "--text_scale",
        type=float,
        default=1.0,
        help="id-label font scale (default: 1.0; demo.py uses 2)",
    )
    parser.add_argument(
        "--text_thickness",
        type=int,
        default=1,
        help="id-label font thickness (default: 1; demo.py uses 2)",
    )
    return parser


def _parse_int_field(fields, index):
    """Parse an optional MOT column as an int, defaulting to -1 (unknown).

    Missing, blank, or non-numeric columns -> -1, so legacy rows (no class/team
    columns) and partially-populated rows degrade to the fallback category.
    """
    if len(fields) <= index:
        return UNKNOWN_CLASS
    raw = fields[index].strip()
    if raw == "":
        return UNKNOWN_CLASS
    try:
        return int(float(raw))
    except ValueError:
        return UNKNOWN_CLASS


def parse_results(txt_path):
    """Parse a MOT .txt into {frame_index: (tlwhs, ids, scores, class_ids, team_ids)}.

    Per-row class_id (column 8 / field index 7) and team_id (column 9 / field
    index 8) are parsed when present; legacy rows without them yield -1 (unknown)
    so they render via the fallback category.
    """
    frames = defaultdict(lambda: ([], [], [], [], []))
    n_rows = 0
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.split(",")
            if len(fields) < 6:
                continue
            frame = int(float(fields[0]))
            tid = int(float(fields[1]))
            x, y, w, h = (float(v) for v in fields[2:6])
            score = float(fields[6]) if len(fields) > 6 and fields[6] not in ("", "-1") else 1.0
            class_id = _parse_int_field(fields, 7)
            team_id = _parse_int_field(fields, 8)
            tlwhs, ids, scores, class_ids, team_ids = frames[frame]
            tlwhs.append((x, y, w, h))
            ids.append(tid)
            scores.append(score)
            class_ids.append(class_id)
            team_ids.append(team_id)
            n_rows += 1
    return frames, n_rows


def main():
    # cv2 is imported here (not at module load) so the pure functions above stay
    # importable / unit-testable without cv2.
    import cv2

    args = make_parser().parse_args()

    if not osp.isfile(args.path):
        raise FileNotFoundError("source video not found: {}".format(args.path))
    if not osp.isfile(args.txt):
        raise FileNotFoundError("results txt not found: {}".format(args.txt))

    plot_tracking = load_plot_tracking()

    frames, n_rows = parse_results(args.txt)
    attr_path = find_attributes_path(args.txt, args.attributes)
    track_attrs = load_track_attributes(attr_path)
    if track_attrs:
        logger.info("loaded class/team for {} tracks from {}".format(
            len(track_attrs), attr_path))
    logger.info("loaded {} boxes across {} annotated frames from {}".format(
        n_rows, len(frames), args.txt))

    cap = cv2.VideoCapture(args.path)
    if not cap.isOpened():
        raise RuntimeError("could not open video: {}".format(args.path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = args.fps if args.fps else (src_fps if src_fps and src_fps > 0 else 30.0)

    if args.save_path:
        save_path = args.save_path
    else:
        stem = osp.splitext(osp.basename(args.txt))[0]
        save_path = osp.join(osp.dirname(osp.abspath(args.txt)), stem + "_rendered.mp4")
    os.makedirs(osp.dirname(osp.abspath(save_path)), exist_ok=True)

    writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (width, height)
    )
    logger.info("rendering -> {} ({}x{} @ {:.2f} fps)".format(save_path, width, height, out_fps))

    frame_idx = 0
    matched_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        key = frame_idx + 1 if args.one_indexed else frame_idx
        if key in frames:
            tlwhs, ids, _, class_ids, team_ids = frames[key]
            # Category resolution precedence per id: track_attributes.json (stable
            # per-track) -> per-frame txt cols 8/9 -> legacy (-1).
            colors, id_texts = [], []
            for tid, cid, team in zip(ids, class_ids, team_ids):
                if tid in track_attrs:
                    cid, team = track_attrs[tid]
                color, label = category_color_and_label(tid, cid, team)
                colors.append(color)
                id_texts.append(label)
            online_im = plot_tracking(
                frame, tlwhs, ids, frame_id=frame_idx + 1, fps=out_fps,
                line_thickness=args.line_thickness,
                text_scale=args.text_scale,
                text_thickness=args.text_thickness,
                colors=colors, id_texts=id_texts,
            )
            matched_frames += 1
        else:
            online_im = frame
        writer.write(online_im)
        frame_idx += 1
        if total > 0 and frame_idx % 100 == 0:
            logger.info("rendered {}/{} frames".format(frame_idx, total))

    cap.release()
    writer.release()

    logger.info("done: wrote {} frames ({} with annotations) to {}".format(
        frame_idx, matched_frames, save_path))
    if matched_frames == 0 and n_rows > 0:
        logger.warning(
            "no .txt frames matched any video frame -- check the frame indexing "
            "(try toggling --one_indexed) or that the .txt belongs to this video."
        )


if __name__ == "__main__":
    main()

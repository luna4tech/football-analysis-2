"""
Stage 1 entrypoint: YOLOv11 detection + ReID + online tracking, emitting the
Stage-1 artifact contract:

    <artifacts>/<video_stem>/01_track/tracks.txt      MOT, 0-based frames
    <artifacts>/<video_stem>/01_track/tracklets.pkl   {id: Tracklet}, features=curr_feat
    <artifacts>/<video_stem>/profiles/01_track.json    profiling

This reuses ``demo.py``'s ``make_parser`` (for the shared tracker/ReID args) and
replicates only the inner per-frame logic, kept behavior-identical to ``demo.py``
so ``tracks.txt`` is byte-equivalent.

Run with CWD = ``Deep-EIoU/Deep-EIoU`` (same as demo.py)::

    cd Deep-EIoU/Deep-EIoU
    python tools/stage1_track.py --video /path/to/clip.mp4

The torch/cv2/ultralytics-dependent perception + tracking loop only runs on a GPU
box (verified in Colab).  The pure-Python Tracklet assembly lives in
``tools/stage1_assembly.py`` and is unit-tested separately on CPU.

Producer/consumer split:
  * ``perceive(frame, detector, extractor, width, height) -> (det, embs)``
    is the per-frame producer: detection forward + edge-removal + clamp/crop +
    ReID embedding.  No cross-frame state.
  * ``track_consume(...)`` is the strict-frame-order tracking consumer: it calls
    ``tracker.update(det, embs)``, applies the ``min_box_area`` filter, and feeds
    surviving targets into the ``TrackletAssembler``.
"""

import os.path as osp
import pickle
import sys

import numpy as np
import cv2
import torch
from loguru import logger

# --- torch.load compatibility shim ------------------------------------------
# PyTorch >= 2.6 flipped torch.load(weights_only) to default True, which rejects
# the numpy globals stored in the trusted OSNet/YOLOv11 checkpoints (e.g.
# numpy.core.multiarray.scalar -> UnpicklingError "Weights only load failed").
# The vendored torchreid's load_checkpoint calls torch.load without this kwarg,
# so restore the legacy weights_only=False default here rather than editing
# vendored code. Affects only this Stage 1 process; the checkpoints are local.
_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat

# --- numpy alias compatibility shim -----------------------------------------
# NumPy >= 1.24 removed the np.float / np.int / np.bool aliases that the vendored
# tracker (tracker/Deep_EIoU.py, tracker/matching.py) still uses as dtypes.
# Restore them (they were just the Python builtins) so the vendored code runs on
# modern NumPy without editing it. numpy is a singleton module, so this is
# visible to the vendored modules too.
for _np_alias, _py_builtin in (
    ("float", float), ("int", int), ("bool", bool),
    ("object", object), ("str", str), ("complex", complex),
):
    if not hasattr(np, _np_alias):
        setattr(np, _np_alias, _py_builtin)

# --- import path setup ------------------------------------------------------
# CWD is Deep-EIoU/Deep-EIoU (demo.py does sys.path.append('.')); mirror that so
# `tracker`, `reid`, and sibling `tools` modules import.
sys.path.append(".")
# tools/ dir on path so we can import the sibling demo + assembly modules when
# invoked as `python tools/stage1_track.py`.
_THIS_DIR = osp.dirname(osp.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Repo root on path so `import pipeline` works (CWD is Deep-EIoU/Deep-EIoU).
_REPO_ROOT = osp.abspath(osp.join(_THIS_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from yolox.tracking_utils.timer import Timer

from tracker.Deep_EIoU import Deep_EIoU
from reid.torchreid.utils import FeatureExtractor

# Reuse demo.py's parser for the shared tracker/ReID args — do NOT reimplement.
from demo import make_parser as demo_make_parser

from stage1_assembly import (
    TrackletAssembler,
    load_tracklet_class,
    ultralytics_result_to_yolox_output,
)

from pipeline.artifacts import get_artifact_paths, ensure_dirs
from pipeline.cache import should_run
from pipeline.profiling import profile_stage


_GTA_LINK_DIR = osp.join(_REPO_ROOT, "gta-link")


# ---------------------------------------------------------------------------
# Per-frame post-detection math
# ---------------------------------------------------------------------------
def det_and_crops_from_output(output, frame, width, height):
    """Turn ONE frame's raw detector output into ``(det, crops)``.

    Applies the per-frame post-detection math — edge-removal
    ``det[:, 0:4] < 1``, clamp boxes to the frame, drop zero-area boxes, and crop
    the surviving player patches.  This mirrors the sequence demo.py runs inside
    ``imageflow_demo`` between detection and the ReID call.

    Parameters
    ----------
    output:
        One image's detection output as a ``(N, >=5)`` array of YOLOX-shaped rows
        ``(x1,y1,x2,y2,score,...)`` (from ``ultralytics_result_to_yolox_output``),
        or ``None`` when the detector produced nothing for this frame.
    frame:
        The original BGR frame (``H x W x 3`` uint8) the boxes index into.
    width, height:
        The frame's width/height (constant within a single video).

    Returns
    -------
    (det, crops) or (None, None)
        ``det`` is the cleaned detection array ``(x1,y1,x2,y2,score,...)`` and
        ``crops`` the matching list of cropped patches (one per ``det`` row).
        Returns ``(None, None)`` for the ``output is None`` empty case, which
        mirrors demo.py's ``outputs[0] is None`` branch (emit nothing / skip
        the tracker).
    """
    if output is None:
        return None, None

    if hasattr(output, "detach"):
        output = output.detach()
    if hasattr(output, "cpu"):
        output = output.cpu()
    if hasattr(output, "numpy"):
        output = output.numpy()
    det = np.asarray(output, dtype=np.float32).copy()
    rows_to_remove = np.any(det[:, 0:4] < 1, axis=1)  # remove edge detection
    det = det[~rows_to_remove]

    # --- clamp boxes to frame, drop zero-area, crop (verbatim from demo.py) ---
    x1 = np.clip(det[:, 0].astype(int), 0, width)
    y1 = np.clip(det[:, 1].astype(int), 0, height)
    x2 = np.clip(det[:, 2].astype(int), 0, width)
    y2 = np.clip(det[:, 3].astype(int), 0, height)
    keep = (x2 > x1) & (y2 > y1)
    det = det[keep]
    x1, y1, x2, y2 = x1[keep], y1[keep], x2[keep], y2[keep]
    crops = [frame[yt:yb, xl:xr] for xl, yt, xr, yb in zip(x1, y1, x2, y2)]
    return det, crops


# ---------------------------------------------------------------------------
# Detector: YOLOv11 (Ultralytics)
# ---------------------------------------------------------------------------
def _quiet_ultralytics_half_deprecation():
    """Let Ultralytics' ``half`` deprecation WARNING through ONCE, then drop repeats.

    Ultralytics logs "'half' is deprecated ... use 'quantize'" on every
    ``predict`` call that passes ``half=`` — once per batch here, which floods the
    log.  Half precision (``--fp16``) is intentional, so we keep passing it and
    instead install a one-shot filter on the ``ultralytics`` logger: the notice
    appears a single time, the repeats are suppressed.
    """
    import logging

    class _HalfDeprecationOnce(logging.Filter):
        seen = False

        def filter(self, record):
            msg = record.getMessage().lower()
            if "half" in msg and "deprecated" in msg:
                if _HalfDeprecationOnce.seen:
                    return False
                _HalfDeprecationOnce.seen = True
            return True

    ul_logger = logging.getLogger("ultralytics")
    if not any(isinstance(f, _HalfDeprecationOnce) for f in ul_logger.filters):
        ul_logger.addFilter(_HalfDeprecationOnce())


class YOLOv11Detector:
    """Ultralytics YOLO adapter that emits YOLOX-shaped detection rows."""

    def __init__(self, ckpt_path, args, device_str):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "YOLOv11 detector requires the 'ultralytics' package. "
                "Install it with: pip install ultralytics"
            ) from exc

        self.model = YOLO(ckpt_path)
        _quiet_ultralytics_half_deprecation()
        self.device = 0 if device_str == "gpu" else "cpu"
        self.half = bool(args.fp16 and device_str == "gpu")
        self.conf = args.conf
        self.iou = args.nms
        self.imgsz = args.tsize

        if args.fuse:
            fuse = getattr(self.model, "fuse", None)
            if callable(fuse):
                logger.info("Fusing YOLOv11 model...")
                fuse()
            else:
                logger.info("--fuse ignored: this Ultralytics model has no fuse() method")

    def _predict(self, source):
        kwargs = {
            "source": source,
            "device": self.device,
            "half": self.half,
            "verbose": False,
        }
        if self.conf is not None:
            kwargs["conf"] = self.conf
        if self.iou is not None:
            kwargs["iou"] = self.iou
        if self.imgsz is not None:
            kwargs["imgsz"] = self.imgsz
        return self.model.predict(**kwargs)

    def infer_one(self, frame):
        results = self._predict(frame)
        if not results:
            return None
        return ultralytics_result_to_yolox_output(results[0])

    def infer_many(self, frames):
        """Batched detection: one YOLOX-shaped output (or ``None``) per frame.

        This is the batched analogue of ``infer_one``.  Ultralytics' ``predict``
        accepts a LIST source and runs the whole list in one (or a few internal)
        forward pass(es) — replacing ``len(frames)`` separate ``predict`` launches
        with a single call and amortizing the per-call Python/CUDA overhead that
        dominates Stage 1's wall time.

        The SAME ``conf``/``iou``/``imgsz``/``half``/``device`` kwargs as
        ``infer_one`` are used (via the shared ``_predict``), so per-frame
        post-processing is bit-for-bit the single-frame path — only the *grouping*
        of the forward pass changes, never the per-frame detection math.

        Ordering guarantee
        -------------------
        Ultralytics returns its ``Results`` list in input order, so element ``k``
        of the returned list corresponds to ``frames[k]``.  We always return a
        list of exactly ``len(frames)`` entries (``None`` where a frame produced
        no usable output), so callers can zip results back to frames by index.

        Parameters
        ----------
        frames:
            A list of BGR frames (each ``H x W x 3`` uint8), in ascending frame
            order.

        Returns
        -------
        list
            ``len(frames)`` entries; entry ``k`` is the YOLOX-shaped output for
            ``frames[k]`` (from ``ultralytics_result_to_yolox_output``) or
            ``None`` when that frame yielded nothing.
        """
        if not frames:
            return []
        results = self._predict(list(frames))
        # Defensive: if Ultralytics returned nothing at all, treat every frame as
        # empty rather than letting a zip silently drop frames.
        if not results:
            return [None] * len(frames)
        return [ultralytics_result_to_yolox_output(r) for r in results]


# ---------------------------------------------------------------------------
# Producer: pure per-frame perception (no cross-frame state)
# ---------------------------------------------------------------------------
def perceive(frame, detector, extractor, width, height):
    """Detection + edge-removal + clamp/crop + ReID for one frame.

    Returns
    -------
    (det, embs) or (None, None)
        ``det`` is the cleaned/cropped detection array (x1,y1,x2,y2,score,...),
        ``embs`` the matching ReID embeddings.  Returns ``(None, None)`` when the
        detector produces no output for the frame (mirrors demo.py's
        ``outputs[0] is None`` branch, which emits nothing).
    """
    output = detector.infer_one(frame)
    det, crops = det_and_crops_from_output(output, frame, width, height)
    if det is None:
        return None, None
    if not crops:
        return det, np.empty((0, 0), dtype=np.float32)

    # --- ReID appearance embedding ---
    embs = extractor(crops)
    embs = embs.cpu().detach().numpy()
    return det, embs


def perceive_batch(frames, detector, extractor, width, height):
    """Batched producer: ``perceive`` applied to a window of frames at once.

    This is the batched analogue of ``perceive`` and reproduces its per-frame
    contract EXACTLY — the ONLY difference is that detection and ReID are each
    run once for the whole window instead of once per frame:

      * Detection is batched via ``detector.infer_many(frames)`` (one Ultralytics
        ``predict`` over the list instead of N predicts).
      * ReID is batched by concatenating EVERY frame's crops into a single list,
        calling ``extractor(all_crops)`` ONCE, then splitting the embeddings back
        out per frame.

    Per-frame post-detection math (``det_and_crops_from_output``) and the
    per-frame result shapes are untouched, so for any given per-frame detector
    output + crops this returns the identical ``(det, embs)`` that looping
    ``perceive`` would — see the CPU equivalence test in
    ``pipeline/tests/test_stage1_perceive_batch.py``.

    Ordering guarantee (the correctness crux)
    ------------------------------------------
    Crops are concatenated in **frame order**, and **within each frame in
    det-row order** (``det_and_crops_from_output`` returns ``crops`` aligned 1:1
    with ``det`` rows).  We record each frame's crop COUNT, run the extractor on
    the flat list, then slice the returned embeddings back using the running
    offsets — so embedding row ``j`` of frame ``k`` maps to ``det`` row ``j`` of
    frame ``k``.  Because concatenation and the split use the same offsets, the
    mapping is exact and never crosses a frame boundary.

    Parameters
    ----------
    frames:
        A list of BGR frames (each ``H x W x 3`` uint8), in ascending frame
        order.
    detector, extractor, width, height:
        Same objects/values as ``perceive``.

    Returns
    -------
    list of (det, embs)
        One tuple per input frame, in input order.  Each tuple matches
        ``perceive`` exactly:
          * ``det is None``         -> ``(None, None)``       (detector empty)
          * ``len(crops) == 0``     -> ``(det, np.empty((0, 0), dtype=np.float32))``
          * otherwise               -> ``(det, embs_slice)``  (one row per det row)
    """
    if not frames:
        return []

    # 1) Batched detection — one predict over the whole window, in frame order.
    outputs = detector.infer_many(frames)

    # 2) Per-frame post-detection math + crop collection.  We keep three parallel
    #    lists indexed by frame position so we can rebuild results after the
    #    single batched extractor call:
    #      - per_frame_det:   the cleaned det array, or None
    #      - per_frame_count: number of crops for that frame (0 when det is None
    #                         or det has no surviving rows)
    #      - all_crops:       the flat, frame-then-row-ordered crop list
    per_frame_det = []
    per_frame_count = []
    all_crops = []
    for k, frame in enumerate(frames):
        det, crops = det_and_crops_from_output(outputs[k], frame, width, height)
        if det is None:
            per_frame_det.append(None)
            per_frame_count.append(0)
            continue
        per_frame_det.append(det)
        if crops:
            per_frame_count.append(len(crops))
            all_crops.extend(crops)  # frame order preserved; row order within frame
        else:
            per_frame_count.append(0)

    # 3) Single batched ReID call over every crop in the window.  Skip entirely
    #    when the whole window produced no crops (no empty-tensor extractor call).
    all_embs = None
    if all_crops:
        embs = extractor(all_crops)
        all_embs = embs.cpu().detach().numpy()

    # 4) Split the flat embeddings back per frame using running offsets, and
    #    rebuild each frame's result to match ``perceive`` byte-for-byte.
    results = []
    offset = 0
    for det, count in zip(per_frame_det, per_frame_count):
        if det is None:
            # Detector produced nothing for this frame -> mirror perceive's
            # (None, None) / demo.py's "outputs[0] is None" branch.
            results.append((None, None))
            continue
        if count == 0:
            # det present but no surviving crops -> empty embeddings, same shape
            # and dtype perceive returns for the 0-crop case.
            results.append((det, np.empty((0, 0), dtype=np.float32)))
            continue
        # Slice this frame's contiguous block out of the batched embeddings.
        embs_slice = all_embs[offset:offset + count]
        offset += count
        results.append((det, embs_slice))

    return results


# ---------------------------------------------------------------------------
# Consumer: strict-frame-order tracking + assembly
# ---------------------------------------------------------------------------
def track_consume(frame_id, det, embs, tracker, assembler, min_box_area):
    """Tracking consumer for one frame's perception output (strict frame order).

    Calls ``tracker.update``, applies the ``min_box_area`` filter, and appends
    each surviving target to BOTH the MOT results list and its Tracklet (one
    feature per row) via the assembler.  This mirrors demo.py's inner loop.

    Must be called in strictly increasing ``frame_id`` order because the tracker
    carries cross-frame state.
    """
    if det is None:
        return
    online_targets = tracker.update(det, embs)
    for t in online_targets:
        tlwh = t.last_tlwh
        tid = t.track_id
        if tlwh[2] * tlwh[3] > min_box_area:
            # curr_feat is always present on output tracks (current-frame match)
            # and is already L2-normalized.  class_id is the latest matched
            # detection's canonical class (-1 when unknown).
            assembler.add(frame_id, tid, tlwh, t.score, t.curr_feat, t.class_id)


# ---------------------------------------------------------------------------
# Stage driver
# ---------------------------------------------------------------------------
def run_stage1(detector, extractor, args, paths):
    """Run the sequential perceive -> track -> assemble loop over the video.

    Writes ``tracks.txt`` and ``tracklets.pkl`` and returns IO counts used for
    profiling.
    """
    Tracklet = load_tracklet_class(_GTA_LINK_DIR)
    assembler = TrackletAssembler(Tracklet)

    cap = cv2.VideoCapture(args.path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = Deep_EIoU(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    emb_dim = 0

    # Window size for the batched producer.  Detection + ReID run once per window
    # (instead of once per frame), then the consumer replays the window one frame
    # at a time in strict ascending order — the tracker is sequential and cannot
    # be batched.  Default 16 keeps a high-res batch within T4 VRAM.
    batch_size = max(1, int(getattr(args, "batch_size", 16)))

    while True:
        # --- read up to batch_size frames (handles the final partial window) ---
        # We buffer one window of frames, then perceive them all at once.  A short
        # read (fewer than batch_size frames) is the last window before EOF; an
        # empty read means the video is exhausted and we stop.
        window_frames = []
        for _ in range(batch_size):
            ret_val, frame = cap.read()
            if not ret_val:
                break
            window_frames.append(frame)
        if not window_frames:
            break

        timer.tic()
        # Producer — batched perception over the whole window.  No cross-frame
        # state; results come back in frame order, one (det, embs) per frame.
        window = perceive_batch(window_frames, detector, extractor, width, height)
        timer.toc()

        # True frames/sec for the just-perceived batch: timer.diff covers the
        # whole window, so divide by the frame count (NOT 1/avg, which would be
        # batches/sec). Logged after toc so the first line shows a real rate.
        logger.info(
            "Processing frames {}..{} ({:.2f} fps)".format(
                frame_id,
                frame_id + len(window_frames) - 1,
                len(window_frames) / max(1e-5, timer.diff),
            )
        )

        # Consumer — strict frame order; carries the tracker state.  frame_id is
        # advanced exactly once per frame so track_consume always sees strictly
        # ascending ids, identical to the old per-frame loop.
        for det, embs in window:
            if embs is not None and embs.size:
                emb_dim = embs.shape[1]
            track_consume(frame_id, det, embs, tracker, assembler, args.min_box_area)
            frame_id += 1

    cap.release()

    # --- write tracks.txt (byte-equivalent to demo.py's results writelines) ---
    with open(paths.tracks_txt, "w") as f:
        f.writelines(assembler.results)
    logger.info("save results to {}".format(paths.tracks_txt))

    # --- write tracklets.pkl ({id: Tracklet}, features = curr_feat) ---
    with open(paths.tracklets_pkl, "wb") as f:
        pickle.dump(assembler.tracklets, f)
    logger.info("save tracklets to {}".format(paths.tracklets_pkl))

    return {
        "n_frames": frame_id,
        "n_output_rows": assembler.n_rows,
        "n_unique_tracks": assembler.n_unique_tracks,
        "emb_dim": int(emb_dim),
    }


DEFAULT_YOLOV11_CKPT = "checkpoints/yolov11l.pt"


def build_detector(args, device_str):
    """Build the YOLOv11 detector adapter."""
    ckpt_file = args.detector_ckpt or DEFAULT_YOLOV11_CKPT
    logger.info("loading YOLOv11 detector checkpoint from {}".format(ckpt_file))
    return YOLOv11Detector(ckpt_file, args, device_str)


def make_parser():
    """Stage-1 CLI: demo.py's parser (shared tracker/ReID args) + contract args.

    Reuses every tracker/reid argument and default from ``demo.make_parser()``
    and adds the Stage-1 contract args.  The detector is always YOLOv11; pass
    ``--detector-ckpt`` to use a custom YOLOv11 checkpoint.
    """
    parser = demo_make_parser()
    parser.add_argument(
        "--detector-ckpt",
        default=None,
        type=str,
        help="YOLOv11 detector checkpoint path (default: checkpoints/yolov11l.pt).",
    )
    parser.add_argument(
        "--video", required=True, type=str, help="path to the input video (Stage-1 input)"
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        type=str,
        help="base directory for artifacts (default: ./artifacts)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="force recompute even if cached outputs are up-to-date",
    )
    parser.add_argument(
        "--batch-size",
        default=16,
        type=int,
        help=(
            "number of frames to perceive (detect + ReID) per batch before "
            "replaying the tracker frame-by-frame (default: 16). Tracking stays "
            "sequential; only detection/ReID are batched."
        ),
    )
    return parser


def main(args):
    # demo.py routes its input through args.path; keep that internal so the
    # imported loop logic behaves identically.
    args.path = args.video

    device_str = args.device
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    paths = get_artifact_paths(args.video, base_dir=args.artifacts_dir)
    ensure_dirs(paths)

    # --- caching: skip if both outputs are present + newer than the video ---
    if not should_run(paths.tracklets_pkl, [args.video], force=args.force):
        if paths.tracks_txt.exists() and paths.tracklets_pkl.exists():
            logger.info(
                "Stage 1 cached (outputs newer than video); skipping. "
                "Use --force to recompute. Outputs: {} , {}".format(
                    paths.tracks_txt, paths.tracklets_pkl
                )
            )
            return
        # tracklets.pkl fresh but tracks.txt missing -> fall through and run.

    detector = build_detector(args, device_str)

    # Derive the ReID device from --device instead of hardcoding 'cuda'.
    extractor = FeatureExtractor(
        model_name="osnet_x1_0",
        model_path="checkpoints/sports_model.pth.tar-60",
        device="cuda" if device_str == "gpu" else "cpu",
    )

    io_counts = {
        "n_frames": 0,
        "n_output_rows": 0,
        "n_unique_tracks": 0,
        "emb_dim": 0,
        "detector": "yolov11",
        "detector_ckpt": args.detector_ckpt or DEFAULT_YOLOV11_CKPT,
    }
    with profile_stage("01_track", paths, extra=io_counts):
        counts = run_stage1(detector, extractor, args, paths)
        io_counts.update(counts)

    logger.info("Stage 1 done: {}".format(io_counts))


if __name__ == "__main__":
    main(make_parser().parse_args())

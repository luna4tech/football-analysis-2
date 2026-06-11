"""
Stage 1 (sequential) entrypoint: DeepEIoU detection + ReID + online tracking,
emitting the Stage-1 artifact contract:

    <artifacts>/<video_stem>/01_track/tracks.txt      MOT, 0-based frames
    <artifacts>/<video_stem>/01_track/tracklets.pkl   {id: Tracklet}, features=curr_feat
    <artifacts>/<video_stem>/profiles/01_track.json    profiling

This reuses ``demo.py``'s components (``Predictor``, ``preproc_to_tensor``,
``make_parser``) by import and replicates only the inner per-frame logic, kept
behavior-identical to ``demo.py`` so ``tracks.txt`` is byte-equivalent.

Run with CWD = ``Deep-EIoU/Deep-EIoU`` (same as demo.py)::

    cd Deep-EIoU/Deep-EIoU
    python tools/stage1_track.py --video /path/to/clip.mp4 -c checkpoints/best_ckpt.pth.tar

The torch/cv2/yolox-dependent perception + tracking loop only runs on a GPU box
(verified in Colab).  The pure-Python Tracklet assembly lives in
``tools/stage1_assembly.py`` and is unit-tested separately on CPU.

Producer/consumer split (Task 3 will parallelize the *producer* only):
  * ``perceive(frame, predictor, extractor, width, height) -> (det, embs)``
    is the per-frame producer: detection forward + rescale + edge-removal +
    clamp/crop + ReID embedding.  No cross-frame state.
  * ``track_consume(...)`` is the strict-frame-order tracking consumer: it calls
    ``tracker.update(det, embs)``, applies the ``min_box_area`` filter, and feeds
    surviving targets into the ``TrackletAssembler``.  This stays untouched in
    Task 3; only how ``(frame_id, det, embs)`` are *produced* changes.
"""

import argparse
import os
import os.path as osp
import pickle
import sys
import time

import numpy as np
import cv2
import torch
from loguru import logger

# --- import path setup ------------------------------------------------------
# CWD is Deep-EIoU/Deep-EIoU (demo.py does sys.path.append('.')); mirror that so
# `tracker`, `yolox`, `reid`, and sibling `tools` modules import.
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

from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info
from yolox.tracking_utils.timer import Timer

from tracker.Deep_EIoU import Deep_EIoU
from reid.torchreid.utils import FeatureExtractor

# Reuse demo.py's components verbatim — do NOT reimplement them.
from demo import Predictor, make_parser as demo_make_parser  # noqa: F401  (Predictor used)

from stage1_assembly import TrackletAssembler, load_tracklet_class

from pipeline.artifacts import get_artifact_paths, ensure_dirs
from pipeline.cache import should_run
from pipeline.profiling import profile_stage


_GTA_LINK_DIR = osp.join(_REPO_ROOT, "gta-link")


# ---------------------------------------------------------------------------
# Producer: pure per-frame perception (no cross-frame state)
# ---------------------------------------------------------------------------
def perceive(frame, predictor, extractor, width, height):
    """Detection + rescale + edge-removal + clamp/crop + ReID for one frame.

    This is the per-frame *producer*.  It is the exact sequence demo.py runs
    inside ``imageflow_demo`` between ``predictor.inference`` and
    ``tracker.update`` — replicated verbatim for behavior parity.

    Returns
    -------
    (det, embs) or (None, None)
        ``det`` is the cleaned/cropped detection array (x1,y1,x2,y2,score,...),
        ``embs`` the matching ReID embeddings.  Returns ``(None, None)`` when the
        detector produces no output for the frame (mirrors demo.py's
        ``outputs[0] is None`` branch, which emits nothing).
    """
    timer = Timer()
    outputs, img_info = predictor.inference(frame, timer)
    if outputs[0] is None:
        return None, None

    # --- rescale boxes + drop edge detections (verbatim from demo.py) ---
    det = outputs[0].cpu().detach().numpy()
    scale = min(1440 / width, 800 / height)
    det /= scale
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
    cropped_imgs = [frame[yt:yb, xl:xr] for xl, yt, xr, yb in zip(x1, y1, x2, y2)]

    # --- ReID appearance embedding ---
    embs = extractor(cropped_imgs)
    embs = embs.cpu().detach().numpy()
    return det, embs


# ---------------------------------------------------------------------------
# Consumer: strict-frame-order tracking + assembly
# ---------------------------------------------------------------------------
def track_consume(frame_id, det, embs, tracker, assembler, min_box_area):
    """Tracking consumer for one frame's perception output (strict frame order).

    Calls ``tracker.update``, applies the ``min_box_area`` filter, and appends
    each surviving target to BOTH the MOT results list and its Tracklet (one
    feature per row) via the assembler.  This mirrors demo.py's inner loop.

    Must be called in strictly increasing ``frame_id`` order because the tracker
    carries cross-frame state.  Task 3 keeps this function unchanged and only
    changes how ``(frame_id, det, embs)`` are produced/ordered upstream.
    """
    if det is None:
        return
    online_targets = tracker.update(det, embs)
    for t in online_targets:
        tlwh = t.last_tlwh
        tid = t.track_id
        if tlwh[2] * tlwh[3] > min_box_area:
            # curr_feat is always present on output tracks (current-frame match)
            # and is already L2-normalized.
            assembler.add(frame_id, tid, tlwh, t.score, t.curr_feat)


# ---------------------------------------------------------------------------
# Stage driver
# ---------------------------------------------------------------------------
def run_stage1(predictor, extractor, args, paths):
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

    while True:
        if frame_id % 30 == 0:
            logger.info(
                "Processing frame {} ({:.2f} fps)".format(
                    frame_id, 1.0 / max(1e-5, timer.average_time)
                )
            )
        ret_val, frame = cap.read()
        if not ret_val:
            break

        timer.tic()
        # Producer (parallelized in Task 3) — strictly per-frame, no state.
        det, embs = perceive(frame, predictor, extractor, width, height)
        if embs is not None and embs.size:
            emb_dim = embs.shape[1]
        # Consumer — strict frame order; carries the tracker state.
        track_consume(frame_id, det, embs, tracker, assembler, args.min_box_area)
        timer.toc()

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


def build_model(exp, args):
    """Replicate demo.py's model-loading setup (verbatim behavior)."""
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = "checkpoints/best_ckpt.pth.tar"
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        output_dir = osp.join(exp.output_dir, args.experiment_name)
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(trt_file), (
            "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        )
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    return model, trt_file, decoder


def make_parser():
    """Stage-1 CLI: demo.py's parser + contract args.

    Reuses every detector/tracker/reid argument and default from
    ``demo.make_parser()`` and adds the Stage-1 contract args.  Does NOT add
    ``--parallel``/``--batch-size`` (those belong to Task 3).
    """
    parser = demo_make_parser()
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
    return parser


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    # demo.py routes its input through args.path; keep that internal so the
    # imported Predictor/loop logic behaves identically.
    args.path = args.video

    if args.trt:
        args.device = "gpu"
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

    model, trt_file, decoder = build_model(exp, args)
    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16, args.profile)

    # Derive the ReID device from --device instead of hardcoding 'cuda'.
    extractor = FeatureExtractor(
        model_name="osnet_x1_0",
        model_path="checkpoints/sports_model.pth.tar-60",
        device="cuda" if device_str == "gpu" else "cpu",
    )

    io_counts = {"n_frames": 0, "n_output_rows": 0, "n_unique_tracks": 0, "emb_dim": 0}
    with profile_stage("01_track", paths, extra=io_counts):
        counts = run_stage1(predictor, extractor, args, paths)
        io_counts.update(counts)

    logger.info("Stage 1 done: {}".format(io_counts))


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)

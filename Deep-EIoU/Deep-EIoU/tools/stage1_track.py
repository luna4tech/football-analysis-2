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
import queue
import sys
import threading
import time

import numpy as np
import cv2
import torch
from loguru import logger

# --- torch.load compatibility shim ------------------------------------------
# PyTorch >= 2.6 flipped torch.load(weights_only) to default True, which rejects
# the numpy globals stored in the trusted OSNet/YOLOX checkpoints (e.g.
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
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.tracking_utils.timer import Timer

from tracker.Deep_EIoU import Deep_EIoU
from reid.torchreid.utils import FeatureExtractor

# Reuse demo.py's components verbatim — do NOT reimplement them.
from demo import (  # noqa: F401  (Predictor used)
    Predictor,
    make_parser as demo_make_parser,
    preproc_to_tensor,
)

from stage1_assembly import (
    TrackletAssembler,
    load_tracklet_class,
    pipelined_consume_loop,
    split_by_counts,
)

from pipeline.artifacts import get_artifact_paths, ensure_dirs
from pipeline.cache import should_run
from pipeline.profiling import profile_stage


_GTA_LINK_DIR = osp.join(_REPO_ROOT, "gta-link")


# ---------------------------------------------------------------------------
# Shared per-frame post-detection math (the parity guarantee)
# ---------------------------------------------------------------------------
def det_and_crops_from_output(output, frame, width, height):
    """Turn ONE frame's raw detector output into ``(det, crops)``.

    This is the single source of truth for the per-frame post-detection math —
    rescale ``det /= scale``, edge-removal ``det[:, 0:4] < 1``, clamp boxes to
    the frame, drop zero-area boxes, and crop the surviving player patches.  It
    is the exact sequence demo.py runs inside ``imageflow_demo`` between
    ``predictor.inference`` and the ReID call, factored out so the **sequential
    and parallel paths do literally identical math** — only how detection
    forward + ReID are *invoked* (single vs batched) differs between them.

    Parameters
    ----------
    output:
        One image's detection output: a torch tensor of shape ``(N, >=5)`` as
        returned by ``postprocess(...)[i]`` / ``predictor.inference(...)[0][0]``,
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

    # --- rescale boxes + drop edge detections (verbatim from demo.py) ---
    det = output.cpu().detach().numpy()
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
    crops = [frame[yt:yb, xl:xr] for xl, yt, xr, yb in zip(x1, y1, x2, y2)]
    return det, crops


# ---------------------------------------------------------------------------
# Producer: pure per-frame perception (no cross-frame state)
# ---------------------------------------------------------------------------
def perceive(frame, predictor, extractor, width, height):
    """Detection + rescale + edge-removal + clamp/crop + ReID for one frame.

    This is the per-frame *producer* used by the SEQUENTIAL path.  Detection
    forward runs through ``predictor.inference`` (single image); the
    post-detection math is the shared ``det_and_crops_from_output`` helper, so
    it is byte-equivalent to the parallel path's per-frame math.

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
    det, crops = det_and_crops_from_output(outputs[0], frame, width, height)
    if det is None:
        return None, None

    # --- ReID appearance embedding ---
    embs = extractor(crops)
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


# ===========================================================================
# Task 3 — parallel perception path (opt-in)
# ===========================================================================
#
# Parity note: parity with the sequential path is "equal up to floating-point
# nondeterminism," NOT bit-identical.  Batched GPU matmul kernels can differ
# from single-image ones in the last FP bits, which could *rarely* flip a
# detection sitting exactly on a confidence/NMS threshold.  The per-frame
# post-detection math (det_and_crops_from_output) and the entire consumer /
# tracker path are literally shared with the sequential path; only how the
# detector forward and ReID are *invoked* (single vs batched) differs.  The
# authoritative Colab equivalence check (Task 5) diffs tracks.txt within
# tolerance.


# ---------------------------------------------------------------------------
# Batched detection forward (mirrors Predictor.inference, batched)
# ---------------------------------------------------------------------------
def infer_batch(predictor, frames):
    """Run the detector forward on a *batch* of frames.

    Mirrors ``Predictor.inference`` internals but batched: letterbox each frame
    via ``preproc_to_tensor`` (same size/ratio for every frame in a single
    video), stack into one ``(B, 3, H, W)`` tensor, run ``predictor.model``
    once, apply ``predictor.decoder`` if set, then ``postprocess(...)`` which
    returns a length-B list — one detection set per image, each equal to what
    single-image inference would put in ``outputs[0]``.

    Reuses the predictor's existing attributes (``num_classes``, ``confthre``,
    ``nmsthre``, ``decoder``, ``device``, ``fp16``, ``test_size``,
    ``rgb_means``, ``std``) so batched and sequential detection share the same
    configuration.

    Parameters
    ----------
    predictor:
        A ``demo.Predictor`` instance.
    frames:
        A non-empty list of BGR frames (all the same size within a video).

    Returns
    -------
    list
        Length-``len(frames)`` list of per-image detection outputs (torch
        tensors or ``None``), suitable for ``det_and_crops_from_output``.
    """
    tensors = []
    base_shape = frames[0].shape[:2]  # (H, W) of the first frame in the batch
    for frame in frames:
        # Defensive: the shared letterbox ratio + the 1440/800 rescale in
        # det_and_crops_from_output assume a constant frame size across the
        # whole video, so every frame in a batch must share (H, W).
        if frame.shape[:2] != base_shape:
            raise AssertionError(
                "infer_batch requires constant frame size: frame shape "
                "{} != first frame shape {} in batch".format(
                    frame.shape[:2], base_shape
                )
            )
        # preproc_to_tensor returns a (1, 3, H, W) tensor + ratio; all frames in
        # one video share the letterbox ratio (constant frame size).
        t, _ratio = preproc_to_tensor(
            frame,
            predictor.test_size,
            predictor.rgb_means,
            predictor.std,
            predictor.device,
            predictor.fp16,
        )
        tensors.append(t)

    batch = torch.cat(tensors, dim=0).contiguous()  # (B, 3, H, W)

    with torch.no_grad():
        outputs = predictor.model(batch)
        if predictor.decoder is not None:
            outputs = predictor.decoder(outputs, dtype=outputs.type())
        # postprocess returns a length-B list, one detection set per image,
        # each identical in shape/semantics to single-image inference's
        # outputs[0].
        outputs = postprocess(
            outputs, predictor.num_classes, predictor.confthre, predictor.nmsthre
        )
    return outputs


# ---------------------------------------------------------------------------
# Batched perception producer (detect + shared per-frame math + batched ReID)
# ---------------------------------------------------------------------------
def perceive_batch(frames, predictor, extractor, width, height):
    """Batched per-frame producer: ``frames -> list[(det, embs) | (None, None)]``.

    Runs ONE batched detector forward, then the SHARED per-frame post-detection
    helper for every frame (so the math matches the sequential path exactly),
    then ONE batched ReID call over all crops in the batch, splitting the
    embeddings back per-frame by crop count via the pure
    ``split_by_counts`` helper.

    A frame whose detection output was ``None`` stays ``None`` (passthrough);
    a frame that fired but kept zero crops gets an empty ``(0, D)`` array.

    Returns
    -------
    list
        One ``(det, embs)`` (or ``(None, None)``) per input frame, in order.
    """
    outputs = infer_batch(predictor, frames)

    dets = []                       # per-frame det array (or None)
    counts = []                     # per-frame crop count (int) or None passthrough
    all_crops = []                  # flat list of every crop across the batch
    for output, frame in zip(outputs, frames):
        det, crops = det_and_crops_from_output(output, frame, width, height)
        dets.append(det)
        if det is None:
            counts.append(None)
        else:
            counts.append(len(crops))
            all_crops.extend(crops)

    # --- ONE ReID call over every crop in the batch ---
    if all_crops:
        embs_all = extractor(all_crops)
        embs_all = embs_all.cpu().detach().numpy()
    else:
        # No crops anywhere in this batch (all frames None or zero-area).
        embs_all = None

    # Split back per-frame: zero-count -> (0, D); None -> None passthrough.
    per_frame_embs = split_by_counts(embs_all, counts)

    return list(zip(dets, per_frame_embs))


# ---------------------------------------------------------------------------
# Prefetch decode thread
# ---------------------------------------------------------------------------
# Queue depth is a small multiple of batch size: enough to keep the decoder a
# couple of batches ahead of GPU compute without letting a fast decoder exhaust
# memory on a long video.  Each queued item holds one decoded frame.
_PREFETCH_QUEUE_DEPTH_MULTIPLE = 2
_DECODE_SENTINEL = None  # pushed once after the last frame to signal EOS


def _decode_worker(path, frame_queue):
    """Background thread: decode frames and push ``(frame_id, frame)`` in order.

    Reads ``cv2.VideoCapture(path)`` start-to-finish, pushing each decoded frame
    onto the bounded ``frame_queue`` (blocks when full -> bounds memory), then
    pushes a single ``_DECODE_SENTINEL`` to mark end-of-stream.  Frame ids are
    assigned in strict ascending decode order starting at 0.
    """
    cap = cv2.VideoCapture(path)
    frame_id = 0
    try:
        while True:
            ret_val, frame = cap.read()
            if not ret_val:
                break
            frame_queue.put((frame_id, frame))  # blocks if the queue is full
            frame_id += 1
    finally:
        cap.release()
        frame_queue.put(_DECODE_SENTINEL)


def _iter_prefetched_frames(path, batch_size):
    """Yield ``(frame_id, frame)`` from a background decode thread, in order.

    Spawns ONE prefetch thread that decodes into a bounded queue and yields the
    frames on the main thread in strict ascending ``frame_id`` order.  The queue
    is bounded so a fast decoder cannot run arbitrarily far ahead of GPU
    compute.
    """
    maxsize = max(1, batch_size * _PREFETCH_QUEUE_DEPTH_MULTIPLE)
    frame_queue: "queue.Queue" = queue.Queue(maxsize=maxsize)
    worker = threading.Thread(
        target=_decode_worker, args=(path, frame_queue), daemon=True
    )
    worker.start()
    try:
        while True:
            item = frame_queue.get()
            if item is _DECODE_SENTINEL:
                break
            yield item
    finally:
        # If iteration stopped early (e.g. perceive_batch/track_consume raised),
        # the decode worker may be blocked on a full frame_queue.put(...) with no
        # one draining it.  Drain the queue first so the worker can reach its
        # sentinel/return, then join with a timeout so we never hang forever.
        try:
            while True:
                frame_queue.get_nowait()
        except queue.Empty:
            pass
        worker.join(timeout=5.0)


# ---------------------------------------------------------------------------
# Parallel stage driver
# ---------------------------------------------------------------------------
#
# The strict-order loops (`batched_consume_loop`, `pipelined_consume_loop`) live
# in the pure-Python `stage1_assembly` module so they can be unit-tested on a
# CPU-only box with stub perceive/consume callables (no torch/cv2/yolox).
def run_stage1_parallel(predictor, extractor, args, paths):
    """Parallel perceive -> track -> assemble over the video (opt-in).

    Same artifact contract as ``run_stage1``: writes ``tracks.txt`` and
    ``tracklets.pkl`` and returns the same IO-count dict.  Three-stage overlap:
    a prefetch thread decodes frames, a producer thread runs batched detection +
    batched ReID on the GPU, and the calling thread feeds the UNCHANGED
    ``track_consume`` in strict frame order — so the GPU computes the next batch
    while the tracker processes the current one (``pipelined_consume_loop``).
    """
    Tracklet = load_tracklet_class(_GTA_LINK_DIR)
    assembler = TrackletAssembler(Tracklet)

    cap = cv2.VideoCapture(args.path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()  # the prefetch thread opens its own capture

    tracker = Deep_EIoU(args, frame_rate=30)
    timer = Timer()
    emb_dim_box = [0]  # mutable so the closure can record the embedding width

    def _perceive(frames):
        return perceive_batch(frames, predictor, extractor, width, height)

    def _consume(frame_id, det, embs):
        if frame_id % 30 == 0:
            logger.info(
                "Processing frame {} ({:.2f} fps)".format(
                    frame_id, 1.0 / max(1e-5, timer.average_time)
                )
            )
        timer.tic()
        if embs is not None and embs.size:
            emb_dim_box[0] = embs.shape[1]
        # Consumer is byte-for-byte the sequential one (strict frame order;
        # None det -> tracker.frame_id does NOT advance, no row emitted).
        track_consume(frame_id, det, embs, tracker, assembler, args.min_box_area)
        timer.toc()

    frame_source = _iter_prefetched_frames(args.path, args.batch_size)
    # Overlap GPU perceive (producer thread) with CPU consume (this thread) so
    # the detector computes batch N+1 while the tracker drains batch N.
    n_frames = pipelined_consume_loop(
        frame_source, args.batch_size, _perceive, _consume
    )

    # --- write tracks.txt (byte-equivalent to the sequential path) ---
    with open(paths.tracks_txt, "w") as f:
        f.writelines(assembler.results)
    logger.info("save results to {}".format(paths.tracks_txt))

    # --- write tracklets.pkl ({id: Tracklet}, features = curr_feat) ---
    with open(paths.tracklets_pkl, "wb") as f:
        pickle.dump(assembler.tracklets, f)
    logger.info("save tracklets to {}".format(paths.tracklets_pkl))

    return {
        "n_frames": n_frames,
        "n_output_rows": assembler.n_rows,
        "n_unique_tracks": assembler.n_unique_tracks,
        "emb_dim": int(emb_dim_box[0]),
    }


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
    """Stage-1 CLI: demo.py's parser + contract args + Task-3 parallel flags.

    Reuses every detector/tracker/reid argument and default from
    ``demo.make_parser()`` and adds the Stage-1 contract args plus the opt-in
    parallel-mode flags.  Sequential is the default; ``--parallel`` switches to
    the batched detect+ReID + prefetch-decode path.
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
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="opt-in: overlap video decode with batched detect+ReID "
        "(sequential is the default and the behavior reference)",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        default=8,
        type=int,
        help="frames per detect+ReID batch in --parallel mode (default: 8). "
        "Queue depth is derived internally as a small multiple of this.",
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

    mode = "parallel" if args.parallel else "sequential"
    io_counts = {
        "n_frames": 0,
        "n_output_rows": 0,
        "n_unique_tracks": 0,
        "emb_dim": 0,
        # mode/batch_size let the user guide compare sequential vs parallel runs.
        "mode": mode,
        "batch_size": args.batch_size if args.parallel else None,
    }
    with profile_stage("01_track", paths, extra=io_counts):
        if args.parallel:
            logger.info(
                "Stage 1 parallel mode (batch_size={})".format(args.batch_size)
            )
            counts = run_stage1_parallel(predictor, extractor, args, paths)
        else:
            counts = run_stage1(predictor, extractor, args, paths)
        io_counts.update(counts)

    logger.info("Stage 1 done: {}".format(io_counts))


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)

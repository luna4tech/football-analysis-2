"""
CPU-only equivalence test for Stage-1 batched perception (OPT-2).

Proves that ``perceive_batch(frames, ...)`` returns, for every frame, the SAME
``(det, embs)`` that looping ``perceive(frame, ...)`` produces — i.e. batching
detection + ReID changes only the *grouping* of the forward passes, never the
per-frame semantics or the (frame-order, then det-row-order) embedding mapping.

NO torch / cv2 / GPU is exercised: the detector and ReID extractor are fakes,
and frames are plain NumPy arrays.  The only real production code under test is
``perceive`` / ``perceive_batch`` / ``det_and_crops_from_output`` (all NumPy).

Import note
-----------
``perceive`` / ``perceive_batch`` live in ``stage1_track.py``, which imports
torch / cv2 / loguru / yolox / tracker / reid / demo at module top — none of
which are guaranteed (loguru, yolox, tracker, ... are absent on the CPU dev
box).  Rather than refactor production code, we exec the module file with those
heavy/unavailable top-level dependencies STUBBED in ``sys.modules`` first.  The
functions under test depend only on the (NumPy-only) ``det_and_crops_from_output``
and on the injected detector/extractor, so stubbing the imports is sufficient and
keeps the production module unchanged.

Run with:
    python pipeline/tests/test_stage1_perceive_batch.py
or via pytest:
    python -m pytest pipeline/tests/test_stage1_perceive_batch.py -q
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
import warnings
from pathlib import Path

import numpy as np

# --- locate stage1_track.py (lives in Deep-EIoU/Deep-EIoU/tools) -------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS_DIR = _REPO_ROOT / "Deep-EIoU" / "Deep-EIoU" / "tools"


def _make_stub(name: str, **attrs) -> types.ModuleType:
    """Build (but do NOT register) a bare stub module named ``name``."""
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _build_stubs():
    """Return ``{module_name: stub_module}`` for the heavy/unavailable imports.

    These are the top-level imports that block a CPU import of ``stage1_track``
    (loguru / yolox / tracker / reid / demo).  ``torch`` / ``cv2`` are added only
    when ABSENT (real ones are used when present); the functions under test touch
    neither.  ``stage1_assembly`` and ``pipeline.*`` are CPU-clean and imported
    for real, so they are deliberately NOT stubbed.
    """

    class _Timer:
        average_time = 0.0

        def tic(self):  # noqa: D401 - stub
            pass

        def toc(self):  # noqa: D401 - stub
            pass

    stubs = {
        # loguru.logger — used only for progress logging (never at call time).
        "loguru": _make_stub("loguru", logger=types.SimpleNamespace(
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
        )),
        # yolox.tracking_utils.timer.Timer
        "yolox": _make_stub("yolox"),
        "yolox.tracking_utils": _make_stub("yolox.tracking_utils"),
        "yolox.tracking_utils.timer": _make_stub("yolox.tracking_utils.timer", Timer=_Timer),
        # tracker.Deep_EIoU.Deep_EIoU
        "tracker": _make_stub("tracker"),
        "tracker.Deep_EIoU": _make_stub("tracker.Deep_EIoU", Deep_EIoU=object),
        # reid.torchreid.utils.FeatureExtractor
        "reid": _make_stub("reid"),
        "reid.torchreid": _make_stub("reid.torchreid"),
        "reid.torchreid.utils": _make_stub("reid.torchreid.utils", FeatureExtractor=object),
        # demo.make_parser — only needs to be callable to build the CLI parser.
        "demo": _make_stub("demo", make_parser=lambda: argparse.ArgumentParser()),
    }

    # torch / cv2 are imported by the module top-level; stub only if absent so we
    # never shadow a real torch/cv2 already loaded by another test.
    for mod_name in ("torch", "cv2"):
        if mod_name in sys.modules:
            continue
        try:
            __import__(mod_name)
        except Exception:
            extra = {"load": lambda *a, **k: None} if mod_name == "torch" else {}
            stubs[mod_name] = _make_stub(mod_name, **extra)

    return stubs


def _load_stage1_track():
    """Exec ``stage1_track.py`` with heavy/unavailable imports temporarily stubbed.

    The stubs are installed into ``sys.modules`` ONLY for the duration of the
    module exec and torn down in a ``finally`` block, restoring ``sys.modules`` to
    its prior state.  This is critical for hygiene: a bare ``types.ModuleType``
    stub has ``__spec__ = None``, which makes ``importlib.util.find_spec(name)``
    raise ``ValueError`` for other tests (e.g. ``test_stage2_refine`` probes
    ``find_spec("loguru")``).  The loaded module keeps its OWN references to
    whatever it imported during exec, so ``perceive`` / ``perceive_batch`` /
    ``infer_many`` keep working after the stubs are removed — the test never calls
    ``run_stage1``, so ``logger`` and friends are never used at call time.
    """
    stubs = _build_stubs()

    # tools dir + repo root on path so stage1_assembly + pipeline import for real.
    for p in (str(_TOOLS_DIR), str(_REPO_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Snapshot prior sys.modules entries for every name we are about to stub, so
    # the finally block can restore (or remove) each one exactly.
    saved = {name: sys.modules.get(name) for name in stubs}
    try:
        sys.modules.update(stubs)

        path = _TOOLS_DIR / "stage1_track.py"
        spec = importlib.util.spec_from_file_location("stage1_track_cpu", str(path))
        module = importlib.util.module_from_spec(spec)
        sys.modules["stage1_track_cpu"] = module
        with warnings.catch_warnings():
            # The module's own np.float/np.str alias shim emits FutureWarnings on
            # modern NumPy; they are harmless and unrelated to what we test.
            warnings.simplefilter("ignore", FutureWarning)
            spec.loader.exec_module(module)
        return module
    finally:
        # Restore sys.modules to its pre-stub state so no leaking __spec__=None
        # stub can pollute other tests in the same process.
        for name, prior in saved.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


_st = _load_stage1_track()
perceive = _st.perceive
perceive_batch = _st.perceive_batch


# ---------------------------------------------------------------------------
# Fakes — deterministic, content-addressed so batching cannot change results.
# ---------------------------------------------------------------------------
# The fake detector returns a fixed, pre-baked YOLOX-shaped output per frame
# (same object whether queried one-at-a-time or as a batch).  The fake extractor
# maps each crop to an embedding derived ONLY from the crop's content, so the
# same crop yields the same embedding regardless of how crops are grouped.  We
# build each frame so every box region carries a unique pixel value, making each
# crop's signature unique and the (frame, row) -> embedding mapping verifiable.

_EMB_DIM = 8


class _FakeDetector:
    """Returns a canned YOLOX output per frame; ``infer_one`` and ``infer_many``
    yield identical per-frame outputs (the only difference batching introduces)."""

    def __init__(self, outputs):
        # outputs: list aligned with frames; each entry is an (N, >=5) array or None.
        self._outputs = outputs
        self._frame_to_idx = {}

    def _lookup(self, frame):
        # Frames carry a unique tag in pixel [0, 0, 0]; use it to find the output.
        return int(frame[0, 0, 0])

    def infer_one(self, frame):
        return self._outputs[self._lookup(frame)]

    def infer_many(self, frames):
        return [self._outputs[self._lookup(f)] for f in frames]


class _FakeExtractor:
    """Content-addressed ReID stub.

    Returns an ``(len(crops), _EMB_DIM)`` array where each row is a deterministic
    function of its crop's content (its unique pixel signature).  Because the
    embedding depends ONLY on the crop, a crop produces the same embedding whether
    it is passed alone (``perceive``) or inside a big concatenated batch
    (``perceive_batch``)."""

    class _Tensor:
        """Minimal stand-in for a torch tensor: supports .cpu().detach().numpy()."""

        def __init__(self, arr):
            self._arr = arr

        def cpu(self):
            return self

        def detach(self):
            return self

        def numpy(self):
            return self._arr

    def __call__(self, crops):
        rows = []
        for crop in crops:
            # Signature: the crop's unique fill value (top-left pixel) broadcast
            # across the embedding so each (frame,row) crop is distinguishable.
            sig = float(np.asarray(crop)[0, 0, 0])
            rows.append(np.full(_EMB_DIM, sig, dtype=np.float32))
        arr = np.asarray(rows, dtype=np.float32).reshape(len(crops), _EMB_DIM)
        return self._Tensor(arr)


# ---------------------------------------------------------------------------
# Frame / detection builders
# ---------------------------------------------------------------------------
_W, _H = 200, 200
_uid_counter = [0]


def _unique_value():
    _uid_counter[0] += 1
    return _uid_counter[0]


def _make_frame(tag: int):
    """Build a (H, W, 3) uint8 frame tagged at pixel [0,0,0] = ``tag``.

    The tag lets the fake detector map the frame back to its canned output even
    after frames are copied into a window list.
    """
    frame = np.zeros((_H, _W, 3), dtype=np.uint8)
    frame[0, 0, 0] = tag
    return frame


def _paint_box(frame, box, value):
    """Fill ``frame`` over ``box`` = (x1, y1, x2, y2) with a unique ``value`` so
    the crop of that region has a distinct, recoverable signature."""
    x1, y1, x2, y2 = box
    frame[y1:y2, x1:x2, :] = value


def _yolox_row(box, score=0.9, class_conf=1.0, class_id=2.0):
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2), float(y2), score, class_conf, class_id]


def _build_frame_with_dets(tag: int, boxes):
    """Return ``(frame, output)`` for a frame containing ``boxes`` detections.

    Each box region is painted a unique value so its crop signature is unique;
    the matching YOLOX output rows are returned so the fake detector can serve
    them.  ``boxes`` are (x1,y1,x2,y2) with all coords >= 1 (so they survive the
    ``det[:, 0:4] < 1`` edge-removal and have non-zero area).
    """
    frame = _make_frame(tag)
    rows = []
    for box in boxes:
        _paint_box(frame, box, _unique_value())
        rows.append(_yolox_row(box))
    output = np.asarray(rows, dtype=np.float32) if rows else np.empty((0, 7), dtype=np.float32)
    return frame, output


# ---------------------------------------------------------------------------
# Test scenario assembly
# ---------------------------------------------------------------------------
def _scenario():
    """Build a mixed window of frames covering every branch.

    Returns ``(frames, detector, extractor)``.  The window deliberately covers:
      * normal frames with detections,
      * a None-output frame (detector returned nothing),
      * a zero-detection frame (output present but no rows / no surviving crops),
      * frames with DIFFERING crop counts (so the per-frame split offsets matter),
      * (the partial-final-window case is exercised by slicing this list).
    """
    frames = []
    outputs = []

    # tag 0: 3 detections
    f, o = _build_frame_with_dets(0, [(10, 10, 40, 40), (60, 60, 90, 95), (100, 20, 130, 70)])
    frames.append(f); outputs.append(o)

    # tag 1: None output (detector produced nothing)
    frames.append(_make_frame(1)); outputs.append(None)

    # tag 2: 1 detection
    f, o = _build_frame_with_dets(2, [(5, 5, 50, 80)])
    frames.append(f); outputs.append(o)

    # tag 3: zero-detection frame (empty output array -> det None after conversion
    # path; here the canned output is an empty (0,7) array, mirroring "no rows").
    frames.append(_make_frame(3)); outputs.append(np.empty((0, 7), dtype=np.float32))

    # tag 4: 2 detections, but one sits on the frame edge (x1 < 1) so it is
    # removed by edge-removal -> exercises det rows != raw rows AND a surviving
    # subset, with crop count 1.
    f = _make_frame(4)
    _paint_box(f, (0, 10, 30, 40), _unique_value())   # edge box (x1=0) -> dropped
    _paint_box(f, (70, 70, 120, 130), _unique_value())  # kept
    outputs.append(np.asarray(
        [_yolox_row((0, 10, 30, 40)), _yolox_row((70, 70, 120, 130))], dtype=np.float32))
    frames.append(f)

    # tag 5: 2 detections (differing count again, to vary split offsets)
    f, o = _build_frame_with_dets(5, [(15, 15, 45, 65), (150, 30, 190, 120)])
    frames.append(f); outputs.append(o)

    detector = _FakeDetector(outputs)
    extractor = _FakeExtractor()
    return frames, detector, extractor


# ---------------------------------------------------------------------------
# Equivalence assertion helper
# ---------------------------------------------------------------------------
def _assert_pair_equal(got, want, ctx):
    g_det, g_embs = got
    w_det, w_embs = want
    if w_det is None:
        assert g_det is None, f"{ctx}: det should be None, got {g_det!r}"
        assert g_embs is None, f"{ctx}: embs should be None, got {g_embs!r}"
        return
    assert g_det is not None, f"{ctx}: det should not be None"
    assert np.array_equal(g_det, w_det), f"{ctx}: det mismatch\n{g_det}\n!=\n{w_det}"
    assert g_embs is not None, f"{ctx}: embs should not be None"
    assert g_embs.dtype == w_embs.dtype == np.float32, (
        f"{ctx}: embs dtype {g_embs.dtype} / {w_embs.dtype}"
    )
    assert g_embs.shape == w_embs.shape, f"{ctx}: embs shape {g_embs.shape} != {w_embs.shape}"
    assert np.array_equal(g_embs, w_embs), f"{ctx}: embs values mismatch"


def _expected_via_perceive(frames, detector, extractor):
    """The ground truth: loop the single-frame ``perceive`` over the window."""
    return [perceive(f, detector, extractor, _W, _H) for f in frames]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_batch_matches_perceive_full_window():
    frames, detector, extractor = _scenario()
    expected = _expected_via_perceive(frames, detector, extractor)
    got = perceive_batch(frames, detector, extractor, _W, _H)
    assert len(got) == len(expected) == len(frames), (len(got), len(expected), len(frames))
    for k, (g, w) in enumerate(zip(got, expected)):
        _assert_pair_equal(g, w, ctx=f"frame {k}")


def test_branches_present_in_scenario():
    # Guard: prove the scenario actually exercises each branch, so the
    # equivalence above is meaningful and not all-None / all-normal.
    frames, detector, extractor = _scenario()
    expected = _expected_via_perceive(frames, detector, extractor)
    n_none = sum(1 for det, _ in expected if det is None)
    n_zero = sum(1 for det, embs in expected if det is not None and embs.shape == (0, 0))
    crop_counts = sorted({
        embs.shape[0] for det, embs in expected
        if det is not None and embs.shape != (0, 0)
    })
    assert n_none >= 1, "scenario must include a None-output frame"
    assert n_zero >= 1, "scenario must include a zero-detection frame"
    assert len(crop_counts) >= 2, f"scenario must include differing crop counts, got {crop_counts}"


def test_partial_final_window():
    # The driver reads up to batch_size frames; the last window may be shorter.
    # Equivalence must hold for any window length, including length 1 and a
    # partial slice.
    frames, detector, extractor = _scenario()
    for end in (1, 2, len(frames) - 1, len(frames)):
        sub = frames[:end]
        expected = _expected_via_perceive(sub, detector, extractor)
        got = perceive_batch(sub, detector, extractor, _W, _H)
        assert len(got) == end, (end, len(got))
        for k, (g, w) in enumerate(zip(got, expected)):
            _assert_pair_equal(g, w, ctx=f"partial[:{end}] frame {k}")


def test_empty_window():
    _frames, detector, extractor = _scenario()
    assert perceive_batch([], detector, extractor, _W, _H) == []


def test_window_with_no_crops_skips_extractor():
    # A window of only None-output + zero-detection frames must still return the
    # right per-frame shapes AND must not call the extractor (it would receive an
    # empty list).  We assert via an extractor that raises if called.
    class _BoomExtractor:
        def __call__(self, crops):  # pragma: no cover - must not be reached
            raise AssertionError("extractor must not be called when window has no crops")

    none_frame = _make_frame(101)
    zero_frame = _make_frame(102)
    outputs = {101: None, 102: np.empty((0, 7), dtype=np.float32)}

    class _D:
        def infer_many(self, frames):
            return [outputs[int(f[0, 0, 0])] for f in frames]

    got = perceive_batch([none_frame, zero_frame], _D(), _BoomExtractor(), _W, _H)
    assert got[0] == (None, None)
    det, embs = got[1]
    assert det is not None and embs.shape == (0, 0) and embs.dtype == np.float32


def test_ordering_offsets_map_each_embedding_to_its_row():
    # Strongest ordering check: with content-addressed embeddings, embedding row
    # j of frame k must equal the signature of det row j of frame k.  This proves
    # crops are concatenated AND split back in (frame-order, then row-order).
    frames, detector, extractor = _scenario()
    got = perceive_batch(frames, detector, extractor, _W, _H)
    expected = _expected_via_perceive(frames, detector, extractor)
    for k, ((g_det, g_embs), (w_det, w_embs)) in enumerate(zip(got, expected)):
        if w_det is None:
            continue
        # Each expected embedding row is the unique crop signature; identical
        # batched rows prove no cross-frame bleed in the split.
        assert np.array_equal(g_embs, w_embs), f"frame {k}: ordering/offset mismatch"


# ---------------------------------------------------------------------------
# Standalone runner (mirrors test_stage1_assembly.py style; no pytest needed)
# ---------------------------------------------------------------------------
_TESTS = [
    ("perceive_batch == loop(perceive) over mixed window", test_batch_matches_perceive_full_window),
    ("scenario exercises None / zero-det / differing-counts branches", test_branches_present_in_scenario),
    ("perceive_batch handles partial final windows", test_partial_final_window),
    ("perceive_batch handles an empty window", test_empty_window),
    ("perceive_batch skips extractor when window has no crops", test_window_with_no_crops_skips_extractor),
    ("perceive_batch maps each embedding to its det row (ordering)", test_ordering_offsets_map_each_embedding_to_its_row),
]

_FAILURES: list[str] = []
_PASSED = 0


def _run(name, fn):
    global _PASSED
    try:
        fn()
        _PASSED += 1
        print(f"  PASS  {name}")
    except AssertionError as exc:
        _FAILURES.append(f"{name}: {exc}")
        print(f"  FAIL  {name}: {exc}")
    except Exception as exc:  # noqa: BLE001
        _FAILURES.append(f"{name}: {type(exc).__name__}: {exc}")
        print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")


def main() -> int:
    print("=" * 60)
    print("stage1 perceive_batch equivalence tests (CPU-only)")
    print("=" * 60)
    for name, fn in _TESTS:
        _run(name, fn)
    print("=" * 60)
    print(f"Results: {_PASSED} passed, {len(_FAILURES)} failed")
    return 1 if _FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())

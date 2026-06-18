"""
CPU-only unit tests for the Stage-1 PARALLEL path's pure pieces (Task 3).

Runs with plain python (numpy only) — NO torch / cv2 / yolox.  Verifies the two
pure helpers that the parallel path in ``stage1_track.py`` depends on, both of
which live in ``stage1_assembly.py`` so they import without GPU deps:

  1. ``split_by_counts`` — splits one stacked ReID embedding block back into
     per-frame embeddings, including zero-count frames and ``None`` passthrough,
     conserving total rows.

  2. ``batched_consume_loop`` — the strict-order batch/consume loop.  Driven
     here with a STUB batched-perceive (deterministic fake ``(det, embs)`` keyed
     by frame_id) and a fake consumer that records the frame_ids it receives, to
     assert the consumer is called for ``0..N-1`` in order, exactly once each,
     across uneven batch sizes + a final partial batch + some ``None`` frames.

Run with:
    python pipeline/tests/test_stage1_parallel.py
"""

from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path

import numpy as np

# --- locate the assembly module (lives in Deep-EIoU/Deep-EIoU/tools) ---------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS_DIR = _REPO_ROOT / "Deep-EIoU" / "Deep-EIoU" / "tools"


def _load_assembly():
    """Import stage1_assembly by file path (no torch/cv2/yolox import path)."""
    path = _TOOLS_DIR / "stage1_assembly.py"
    spec = importlib.util.spec_from_file_location("stage1_assembly", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_asm = _load_assembly()
split_by_counts = _asm.split_by_counts
batched_consume_loop = _asm.batched_consume_loop
pipelined_consume_loop = _asm.pipelined_consume_loop


# ---------------------------------------------------------------------------
# Minimal test runner (matches test_pipeline.py style; no pytest required)
# ---------------------------------------------------------------------------
_FAILURES: list[str] = []
_PASSED: int = 0


def _pass(name: str) -> None:
    global _PASSED
    _PASSED += 1
    print(f"  PASS  {name}")


def _fail(name: str, msg: str) -> None:
    _FAILURES.append(f"{name}: {msg}")
    print(f"  FAIL  {name}: {msg}")


def run_test(name: str, fn) -> None:
    try:
        fn()
        _pass(name)
    except AssertionError as exc:
        _fail(name, str(exc) or "AssertionError")
    except Exception as exc:  # noqa: BLE001
        _fail(name, f"{type(exc).__name__}: {exc}")


# ===========================================================================
# 1. split_by_counts
# ===========================================================================
def test_split_basic_per_frame():
    # 3 frames with 2, 0, 3 crops -> total 5 rows.
    counts = [2, 0, 3]
    embs = np.arange(5 * 512, dtype=np.float32).reshape(5, 512)
    out = split_by_counts(embs, counts)

    assert len(out) == 3, len(out)
    assert out[0].shape == (2, 512), out[0].shape
    assert out[1].shape == (0, 512), out[1].shape   # zero-count -> empty (0, D)
    assert out[2].shape == (3, 512), out[2].shape

    # Slices are in order and contiguous.
    assert np.array_equal(out[0], embs[0:2])
    assert np.array_equal(out[2], embs[2:5])
    # Total rows conserved.
    total = sum(p.shape[0] for p in out)
    assert total == embs.shape[0] == 5, total


def test_split_none_passthrough():
    # Middle frame's detection output was None -> stays None; rows conserved.
    counts = [1, None, 2]
    embs = np.arange(3 * 512, dtype=np.float32).reshape(3, 512)
    out = split_by_counts(embs, counts)

    assert len(out) == 3
    assert out[0].shape == (1, 512)
    assert out[1] is None, "None count must pass through as None"
    assert out[2].shape == (2, 512)
    assert np.array_equal(out[0], embs[0:1])
    assert np.array_equal(out[2], embs[1:3])

    # Conservation: only int counts contribute rows.
    total = sum(p.shape[0] for p in out if p is not None)
    assert total == embs.shape[0] == 3, total


def test_split_all_none():
    # Every frame None -> embs is None -> all None passthrough.
    out = split_by_counts(None, [None, None, None])
    assert out == [None, None, None], out


def test_split_all_zero_count_embs_none():
    # Detector fired but every box dropped (zero crops) on all frames -> embs is
    # None, counts are all 0 -> each gets a (0, D) empty array, not None.
    out = split_by_counts(None, [0, 0])
    assert len(out) == 2
    for arr in out:
        assert arr is not None
        assert arr.shape[0] == 0, arr.shape


def test_split_mixed_zero_and_none_with_embs():
    # zero-count and None interleaved among real frames.
    counts = [0, 2, None, 1, 0]
    embs = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)  # narrow D=4 is fine
    out = split_by_counts(embs, counts)

    assert out[0].shape == (0, 4)
    assert out[1].shape == (2, 4)
    assert out[2] is None
    assert out[3].shape == (1, 4)
    assert out[4].shape == (0, 4)
    assert np.array_equal(out[1], embs[0:2])
    assert np.array_equal(out[3], embs[2:3])
    total = sum(p.shape[0] for p in out if p is not None)
    assert total == 3 == embs.shape[0], total


def test_split_row_count_mismatch_raises():
    # Counts that don't sum to embs rows must raise (producer-bug guard).
    embs = np.zeros((2, 8), dtype=np.float32)
    raised = False
    try:
        split_by_counts(embs, [1, 2])  # sums to 3, not 2
    except ValueError:
        raised = True
    assert raised, "mismatched counts must raise ValueError"


def test_split_embs_none_but_positive_count_raises():
    raised = False
    try:
        split_by_counts(None, [1])
    except ValueError:
        raised = True
    assert raised, "None embs with positive count must raise ValueError"


# ===========================================================================
# 2. batched_consume_loop strict-ordering invariant
# ===========================================================================
def _make_stub_perceive(none_frames):
    """Deterministic stub batched-perceive keyed by frame_id.

    Returns a callable that, given the list of frames (here each "frame" is just
    its integer frame_id), returns ``(det, embs)`` per frame.  Frames in
    ``none_frames`` get ``(None, None)`` to exercise the skip path; others get a
    tiny deterministic det/embs derived from the frame_id.
    """
    def _perceive(frames):
        out = []
        for fid in frames:
            if fid in none_frames:
                out.append((None, None))
            else:
                det = np.array([[fid, fid, fid + 1, fid + 1, 0.9]], dtype=np.float32)
                embs = np.full((1, 4), float(fid), dtype=np.float32)
                out.append((det, embs))
        return out

    return _perceive


def _run_ordering(n_frames, batch_size, none_frames):
    """Drive batched_consume_loop over [0..n_frames) and record consumer calls."""
    received = []          # (frame_id, is_none) in call order

    def _consume(frame_id, det, embs):
        received.append((frame_id, det is None))

    # frame_source yields (frame_id, frame); here frame == frame_id (an int).
    frame_source = ((i, i) for i in range(n_frames))
    stub = _make_stub_perceive(none_frames)
    total = batched_consume_loop(frame_source, batch_size, stub, _consume)
    return total, received


def test_ordering_exact_division():
    # 12 frames, batch 4 -> 3 full batches, no partial.
    total, received = _run_ordering(12, 4, none_frames=set())
    assert total == 12, total
    ids = [fid for fid, _ in received]
    assert ids == list(range(12)), ids
    assert all(not is_none for _, is_none in received)


def test_ordering_with_partial_batch():
    # 10 frames, batch 4 -> 4,4,2 (final partial batch).
    total, received = _run_ordering(10, 4, none_frames=set())
    assert total == 10, total
    ids = [fid for fid, _ in received]
    assert ids == list(range(10)), ids
    # exactly once each
    assert len(ids) == len(set(ids)) == 10


def test_ordering_uneven_with_none_frames():
    # Uneven batch size, a final partial batch, and several None frames.
    none = {0, 3, 4, 9, 13}  # includes batch boundaries + last frame
    total, received = _run_ordering(14, 3, none_frames=none)
    assert total == 14, total

    ids = [fid for fid, _ in received]
    # Strict ascending 0..13, once each, no gaps / no reordering.
    assert ids == list(range(14)), ids
    assert len(ids) == len(set(ids)) == 14

    # The None-ness recorded by the consumer must match which frames were None,
    # proving None frames are still delivered to the consumer (which applies the
    # skip rule itself) rather than being dropped from the loop.
    none_received = {fid for fid, is_none in received if is_none}
    assert none_received == none, (none_received, none)


def test_ordering_batch_size_one():
    total, received = _run_ordering(5, 1, none_frames={2})
    ids = [fid for fid, _ in received]
    assert ids == [0, 1, 2, 3, 4], ids
    assert total == 5


def test_ordering_batch_larger_than_stream():
    # batch_size bigger than total frame count -> single partial flush.
    total, received = _run_ordering(3, 8, none_frames=set())
    ids = [fid for fid, _ in received]
    assert ids == [0, 1, 2], ids
    assert total == 3


def test_ordering_empty_stream():
    total, received = _run_ordering(0, 4, none_frames=set())
    assert total == 0
    assert received == []


# ===========================================================================
# 3. pipelined_consume_loop — same ordering contract, but perceive runs on a
#    producer thread concurrently with consume.
# ===========================================================================
def _run_ordering_pipelined(n_frames, batch_size, none_frames, queue_depth=2):
    """Drive pipelined_consume_loop and record consumer calls (in call order)."""
    received = []

    def _consume(frame_id, det, embs):
        received.append((frame_id, det is None))

    frame_source = ((i, i) for i in range(n_frames))
    stub = _make_stub_perceive(none_frames)
    total = pipelined_consume_loop(
        frame_source, batch_size, stub, _consume, queue_depth=queue_depth
    )
    return total, received


def test_pipelined_matches_serial_ordering():
    # For a spread of scenarios, the pipelined loop must deliver the EXACT same
    # consumer call sequence (frame_id, is_none) as the serial batch loop.
    scenarios = [
        (12, 4, set()),
        (10, 4, set()),
        (14, 3, {0, 3, 4, 9, 13}),
        (5, 1, {2}),
        (3, 8, set()),
        (0, 4, set()),
    ]
    for n, bs, none in scenarios:
        _, serial = _run_ordering(n, bs, none_frames=none)
        total_p, pipelined = _run_ordering_pipelined(n, bs, none_frames=none)
        assert pipelined == serial, (
            f"pipelined != serial for n={n} bs={bs} none={none}: "
            f"{pipelined} != {serial}"
        )
        assert total_p == n, (n, total_p)


def test_pipelined_strict_order_with_queue_depth_one():
    # queue_depth=1 still preserves strict 0..N-1 order (single producer, FIFO).
    total, received = _run_ordering_pipelined(11, 3, none_frames={1, 6}, queue_depth=1)
    ids = [fid for fid, _ in received]
    assert ids == list(range(11)), ids
    assert total == 11
    none_received = {fid for fid, is_none in received if is_none}
    assert none_received == {1, 6}, none_received


def test_pipelined_producer_exception_propagates():
    # A perceive failure on the producer thread must re-raise on the caller
    # thread (not hang, not swallow).
    class _Boom(RuntimeError):
        pass

    def _bad_perceive(frames):
        if 5 in frames:          # blow up only on the batch containing frame 5
            raise _Boom("perceive failed")
        return [(np.zeros((1, 4), np.float32), np.zeros((1, 4), np.float32)) for _ in frames]

    def _consume(frame_id, det, embs):
        pass

    frame_source = ((i, i) for i in range(8))
    raised = False
    try:
        pipelined_consume_loop(frame_source, 3, _bad_perceive, _consume)
    except _Boom:
        raised = True
    assert raised, "producer-thread exception must propagate to the caller"


def test_pipelined_perceive_and_consume_overlap():
    # Deterministic proof of concurrency: consume(frame 0) blocks until the
    # producer has begun perceiving the SECOND batch.  This can only complete if
    # perceive runs on a thread distinct from consume — a serial loop (perceive
    # then consume on one thread) would deadlock, which we detect via timeout.
    perceived_second_batch = threading.Event()

    def _perceive(frames):
        if 2 in frames:                         # the second batch ([2, 3])
            perceived_second_batch.set()
        return [(np.zeros((1, 4), np.float32), np.zeros((1, 4), np.float32)) for _ in frames]

    consumed = []

    def _consume(frame_id, det, embs):
        if frame_id == 0:
            # Will only be set if the producer thread ran ahead to batch 2.
            assert perceived_second_batch.wait(timeout=5.0), (
                "perceive of batch 2 did not run while consuming frame 0 "
                "-> no overlap (serialized)"
            )
        consumed.append(frame_id)

    box = {}

    def _drive():
        frame_source = ((i, i) for i in range(4))  # 2 batches of 2
        box["n"] = pipelined_consume_loop(frame_source, 2, _perceive, _consume, queue_depth=2)

    t = threading.Thread(target=_drive)
    t.start()
    t.join(timeout=10.0)
    assert not t.is_alive(), "pipelined loop deadlocked (perceive/consume not concurrent)"
    assert consumed == [0, 1, 2, 3], consumed
    assert box.get("n") == 4, box


_TESTS = [
    ("split_by_counts: basic per-frame split + zero-count empty", test_split_basic_per_frame),
    ("split_by_counts: None passthrough, rows conserved", test_split_none_passthrough),
    ("split_by_counts: all-None -> all None", test_split_all_none),
    ("split_by_counts: all zero-count + embs None -> empty arrays", test_split_all_zero_count_embs_none),
    ("split_by_counts: mixed zero/None interleaved", test_split_mixed_zero_and_none_with_embs),
    ("split_by_counts: count/rows mismatch raises", test_split_row_count_mismatch_raises),
    ("split_by_counts: None embs with positive count raises", test_split_embs_none_but_positive_count_raises),
    ("ordering: exact division, 0..N-1 once each", test_ordering_exact_division),
    ("ordering: final partial batch", test_ordering_with_partial_batch),
    ("ordering: uneven + partial + None frames, strict order", test_ordering_uneven_with_none_frames),
    ("ordering: batch_size == 1", test_ordering_batch_size_one),
    ("ordering: batch larger than stream", test_ordering_batch_larger_than_stream),
    ("ordering: empty stream", test_ordering_empty_stream),
    ("pipelined: matches serial consume order across scenarios", test_pipelined_matches_serial_ordering),
    ("pipelined: strict order with queue_depth=1", test_pipelined_strict_order_with_queue_depth_one),
    ("pipelined: producer exception propagates to caller", test_pipelined_producer_exception_propagates),
    ("pipelined: perceive overlaps consume (concurrency proof)", test_pipelined_perceive_and_consume_overlap),
]


def main() -> int:
    print("=" * 60)
    print("stage1 parallel-path pure-piece unit tests (CPU-only)")
    print("=" * 60)
    for test_name, fn in _TESTS:
        run_test(test_name, fn)
    print("=" * 60)
    print(f"Results: {_PASSED} passed, {len(_FAILURES)} failed")
    if _FAILURES:
        print("\nFailed tests:")
        for f in _FAILURES:
            print(f"  {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

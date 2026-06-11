"""
CPU-only unit test for the Stage-1 Tracklet assembly path.

Runs with plain python (numpy only) — NO torch / cv2 / yolox.  Verifies the
pure-Python assembly that ``stage1_track.py`` relies on:

  * per-track times / scores / bboxes / features are aligned and correct,
  * MOT result lines match demo.py's formatting and are row-aligned with
    features,
  * the ``{id: Tracklet}`` dict pickles and unpickles, and the pickle bytes
    reference module ``"Tracklet"`` (so it loads in the Stage-2 process).

Run with:
    python pipeline/tests/test_stage1_assembly.py
"""

from __future__ import annotations

import importlib.util
import pickle
import pickletools
import sys
from pathlib import Path

import numpy as np

# --- locate the assembly module (lives in Deep-EIoU/Deep-EIoU/tools) ---------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS_DIR = _REPO_ROOT / "Deep-EIoU" / "Deep-EIoU" / "tools"
_GTA_LINK_DIR = _REPO_ROOT / "gta-link"


def _load_assembly():
    """Import stage1_assembly by file path (no torch/cv2/yolox import path)."""
    path = _TOOLS_DIR / "stage1_assembly.py"
    spec = importlib.util.spec_from_file_location("stage1_assembly", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_asm = _load_assembly()
TrackletAssembler = _asm.TrackletAssembler
load_tracklet_class = _asm.load_tracklet_class
format_mot_line = _asm.format_mot_line


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


# ---------------------------------------------------------------------------
# Synthetic records: (frame_id, track_id, tlwh, score, feat)
# Two tracks interleaved across three frames, one min_box_area-style survivor
# already filtered out by the caller (so the assembler only ever sees keepers).
# ---------------------------------------------------------------------------
def _synthetic_records():
    rng = np.random.default_rng(0)

    def feat():
        v = rng.standard_normal(512).astype(np.float32)
        return v / np.linalg.norm(v)  # mimic curr_feat (already L2-normalized)

    # (frame_id, track_id, [l, t, w, h], score, feat)
    return [
        (0, 1, [10.0, 20.0, 30.0, 40.0], 0.91, feat()),
        (0, 2, [100.0, 110.0, 25.0, 55.0], 0.85, feat()),
        (1, 1, [12.5, 22.5, 31.0, 41.0], 0.88, feat()),
        (2, 1, [15.0, 25.0, 32.0, 42.0], 0.80, feat()),
        (2, 2, [105.0, 115.0, 26.0, 56.0], 0.77, feat()),
    ]


def _build():
    Tracklet = load_tracklet_class(_GTA_LINK_DIR)
    asm = TrackletAssembler(Tracklet)
    records = _synthetic_records()
    for frame_id, tid, tlwh, score, ft in records:
        asm.add(frame_id, tid, tlwh, score, ft)
    return asm, records, Tracklet


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_module_name_is_Tracklet():
    Tracklet = load_tracklet_class(_GTA_LINK_DIR)
    assert Tracklet.__module__ == "Tracklet", Tracklet.__module__
    assert "Tracklet" in sys.modules


def test_per_track_alignment():
    asm, records, _ = _build()
    tracklets = asm.tracklets

    # Expected per-track grouping in insertion order.
    expected: dict = {}
    for frame_id, tid, tlwh, score, ft in records:
        expected.setdefault(tid, {"times": [], "scores": [], "bboxes": [], "feats": []})
        expected[tid]["times"].append(frame_id)
        expected[tid]["scores"].append(score)
        expected[tid]["bboxes"].append([float(x) for x in tlwh])
        expected[tid]["feats"].append(ft.astype(np.float32))

    assert set(tracklets.keys()) == set(expected.keys())
    for tid, exp in expected.items():
        tr = tracklets[tid]
        assert tr.times == exp["times"], f"times tid={tid}: {tr.times} != {exp['times']}"
        assert tr.scores == exp["scores"], f"scores tid={tid}"
        assert tr.bboxes == exp["bboxes"], f"bboxes tid={tid}: {tr.bboxes}"
        # one feature per detection, aligned with times
        assert len(tr.features) == len(exp["times"]), f"feature count tid={tid}"
        for got, want in zip(tr.features, exp["feats"]):
            assert got.dtype == np.float32, f"feat dtype tid={tid}: {got.dtype}"
            assert np.array_equal(got, want), f"feat mismatch tid={tid}"


def test_features_not_renormalized():
    # Feed a feature that is NOT unit norm; assembler must store it as-is
    # (float32), not renormalize it.
    Tracklet = load_tracklet_class(_GTA_LINK_DIR)
    asm = TrackletAssembler(Tracklet)
    raw = np.array([3.0, 4.0] + [0.0] * 510, dtype=np.float64)  # norm 5.0
    asm.add(0, 7, [1.0, 2.0, 3.0, 4.0], 0.9, raw)
    stored = asm.tracklets[7].features[0]
    assert stored.dtype == np.float32
    assert np.isclose(np.linalg.norm(stored), 5.0), np.linalg.norm(stored)


def test_mot_lines_match_demo_format():
    asm, records, _ = _build()
    # One MOT line per record, in order, row-aligned with features.
    assert asm.n_rows == len(records)
    for line, (frame_id, tid, tlwh, score, _ft) in zip(asm.results, records):
        expected = (
            f"{frame_id},{tid},"
            f"{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},"
            f"{score:.2f},-1,-1,-1\n"
        )
        assert line == expected, f"{line!r} != {expected!r}"

    # Total feature count across tracklets == number of MOT rows (one per row).
    total_feats = sum(len(tr.features) for tr in asm.tracklets.values())
    assert total_feats == asm.n_rows, f"{total_feats} != {asm.n_rows}"


def test_format_mot_line_exact():
    line = format_mot_line(5, 3, [1.005, 2.0, 3.5, 4.499], 0.876)
    # :.2f rounds: 1.005 -> 1.00 (banker-ish via format), 4.499 -> 4.50, 0.876 -> 0.88
    assert line == "5,3,1.00,2.00,3.50,4.50,0.88,-1,-1,-1\n", repr(line)


def test_pickle_roundtrip_and_module_ref():
    asm, records, Tracklet = _build()
    blob = pickle.dumps(asm.tracklets)

    # The pickle must reference module "Tracklet" so Stage-2's `import Tracklet`
    # resolves the class.  Check the opcode stream (robust) and the raw bytes.
    ops = list(pickletools.genops(blob))
    stack_globals = [
        arg for op, arg, _pos in ops
        if op.name in ("STACK_GLOBAL",) and arg is not None
    ]
    # In protocol >=4 the module/name are pushed as SHORT_BINUNICODE strings then
    # STACK_GLOBAL; assert the module string "Tracklet" appears.
    short_strs = [arg for op, arg, _pos in ops if op.name.endswith("BINUNICODE")]
    assert "Tracklet" in short_strs, f"module 'Tracklet' not in pickle strings: {short_strs}"

    # Round-trip: unpickle (Tracklet is registered in sys.modules already).
    restored = pickle.loads(blob)
    assert set(restored.keys()) == set(asm.tracklets.keys())
    for tid, tr in restored.items():
        assert type(tr).__module__ == "Tracklet"
        assert tr.times == asm.tracklets[tid].times
        assert tr.scores == asm.tracklets[tid].scores
        assert tr.bboxes == asm.tracklets[tid].bboxes
        for got, want in zip(tr.features, asm.tracklets[tid].features):
            assert np.array_equal(got, want)


def test_n_unique_tracks():
    asm, _records, _ = _build()
    assert asm.n_unique_tracks == 2, asm.n_unique_tracks


_TESTS = [
    ("assembly: module name is 'Tracklet'", test_module_name_is_Tracklet),
    ("assembly: per-track times/scores/bboxes/features aligned", test_per_track_alignment),
    ("assembly: features stored float32, not renormalized", test_features_not_renormalized),
    ("assembly: MOT lines match demo.py format", test_mot_lines_match_demo_format),
    ("assembly: format_mot_line exact", test_format_mot_line_exact),
    ("assembly: pickle round-trips + references module 'Tracklet'", test_pickle_roundtrip_and_module_ref),
    ("assembly: n_unique_tracks", test_n_unique_tracks),
]


def main() -> int:
    print("=" * 60)
    print("stage1 assembly unit tests (CPU-only)")
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

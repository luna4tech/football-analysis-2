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
ultralytics_result_to_yolox_output = _asm.ultralytics_result_to_yolox_output


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
    # class_id defaults to -1 (unknown), so the line is byte-identical to legacy.
    assert line == "5,3,1.00,2.00,3.50,4.50,0.88,-1,-1,-1\n", repr(line)


def test_format_mot_line_class_in_col8():
    # Canonical class id lands in MOT column 8; cols 1-7 unchanged, cols 9-10 -1.
    line = format_mot_line(5, 3, [1.005, 2.0, 3.5, 4.499], 0.876, class_id=1)
    assert line == "5,3,1.00,2.00,3.50,4.50,0.88,1,-1,-1\n", repr(line)
    cols = line.strip().split(",")
    assert cols[7] == "1", cols          # column 8 (0-based index 7) = class_id
    assert cols[:7] == ["5", "3", "1.00", "2.00", "3.50", "4.50", "0.88"], cols
    assert cols[8:] == ["-1", "-1"], cols


def test_assembler_class_ids_aligned():
    # Two tracks; per-frame canonical classes carried into Tracklet.class_ids and
    # into MOT column 8, aligned 1:1 with times/scores/bboxes/features.
    Tracklet = load_tracklet_class(_GTA_LINK_DIR)
    asm = TrackletAssembler(Tracklet)
    rng = np.random.default_rng(1)

    def feat():
        v = rng.standard_normal(512).astype(np.float32)
        return v / np.linalg.norm(v)

    # (frame_id, track_id, tlwh, score, feat, class_id)
    # Canonical scheme (class_map): goalkeeper=1, player=2, referee=3.
    records = [
        (0, 1, [10.0, 20.0, 30.0, 40.0], 0.9, feat(), 2),   # player
        (0, 2, [50.0, 60.0, 20.0, 50.0], 0.8, feat(), 3),   # referee
        (1, 1, [11.0, 21.0, 30.0, 40.0], 0.85, feat(), 2),  # player (same)
        (2, 1, [12.0, 22.0, 30.0, 40.0], 0.7, feat(), 1),   # latest matched -> gk
    ]
    for fid, tid, tlwh, score, ft, cid in records:
        asm.add(fid, tid, tlwh, score, ft, cid)

    t1 = asm.tracklets[1]
    t2 = asm.tracklets[2]
    assert t1.class_ids == [2, 2, 1], t1.class_ids
    assert t2.class_ids == [3], t2.class_ids
    # class_ids aligned 1:1 with the other parallel arrays.
    for tr in (t1, t2):
        n = len(tr.times)
        assert len(tr.class_ids) == n, (tr.track_id, len(tr.class_ids), n)
        assert len(tr.scores) == n and len(tr.bboxes) == n and len(tr.features) == n

    # MOT column 8 carries the canonical class for each row, in order.
    col8 = [line.strip().split(",")[7] for line in asm.results]
    assert col8 == ["2", "3", "2", "1"], col8


def test_assembler_default_class_is_unknown():
    # add() without a class_id -> -1 everywhere (legacy-safe).
    Tracklet = load_tracklet_class(_GTA_LINK_DIR)
    asm = TrackletAssembler(Tracklet)
    asm.add(0, 1, [1.0, 2.0, 3.0, 4.0], 0.9, np.zeros(512, dtype=np.float32))
    asm.add(1, 1, [1.0, 2.0, 3.0, 4.0], 0.9, np.zeros(512, dtype=np.float32))
    assert asm.tracklets[1].class_ids == [-1, -1], asm.tracklets[1].class_ids
    assert all(line.strip().split(",")[7] == "-1" for line in asm.results)


def test_tracklet_class_ids_constructed_when_absent():
    # Older code paths build Tracklets without class_ids; the field must still be
    # constructed and length-aligned with the frames that WERE provided.
    Tracklet = load_tracklet_class(_GTA_LINK_DIR)
    tr = Tracklet(7, [0, 1, 2], [0.9, 0.8, 0.7], [[1, 2, 3, 4]] * 3)
    assert hasattr(tr, "class_ids"), "class_ids must exist on every Tracklet"
    assert tr.class_ids == [-1, -1, -1], tr.class_ids
    # append_det without a class keeps it aligned + defaults to -1.
    tr.append_det(3, 0.6, [5, 6, 7, 8])
    assert tr.class_ids == [-1, -1, -1, -1], tr.class_ids
    # append_det WITH a class records it.
    tr.append_det(4, 0.5, [9, 10, 11, 12], 1)
    assert tr.class_ids == [-1, -1, -1, -1, 1], tr.class_ids
    # empty Tracklet -> empty class_ids.
    assert Tracklet().class_ids == []


def test_tracklet_legacy_pickle_backfills_class_ids():
    # Simulate a pickle created BEFORE class_ids existed: such instances bypass
    # __init__ on load, so their state dict has no class_ids.  __setstate__ must
    # backfill a -1-filled list aligned with times so append_det / extract work.
    Tracklet = load_tracklet_class(_GTA_LINK_DIR)

    legacy = Tracklet.__new__(Tracklet)  # bypass __init__, like pickle.load does
    legacy_state = {
        "track_id": 9,
        "parent_id": 9,
        "scores": [0.9, 0.8, 0.7],
        "times": [0, 1, 2],
        "bboxes": [[1, 2, 3, 4]] * 3,
        "features": [],
        # NOTE: deliberately no "class_ids" key (legacy on-disk shape).
    }
    legacy.__setstate__(legacy_state)

    assert hasattr(legacy, "class_ids"), "class_ids must be backfilled on restore"
    assert legacy.class_ids == [-1, -1, -1], legacy.class_ids
    assert len(legacy.class_ids) == len(legacy.times)

    # A subsequent append_det keeps every parallel array aligned.
    legacy.append_det(3, 0.6, [5, 6, 7, 8], 1)
    n = len(legacy.times)
    assert legacy.class_ids == [-1, -1, -1, 1], legacy.class_ids
    assert len(legacy.scores) == n and len(legacy.bboxes) == n and len(legacy.class_ids) == n


def test_tracklet_real_legacy_pickle_roundtrip():
    # End-to-end: pickle a Tracklet, strip class_ids from the serialized state,
    # and confirm unpickling backfills an aligned class_ids (no AttributeError).
    Tracklet = load_tracklet_class(_GTA_LINK_DIR)
    tr = Tracklet(5, [0, 1], [0.9, 0.8], [[1, 2, 3, 4], [5, 6, 7, 8]], class_ids=[2, 3])

    state = tr.__dict__.copy()
    state.pop("class_ids")  # mimic a pre-class_ids on-disk Tracklet
    stripped = Tracklet.__new__(Tracklet)
    stripped.__dict__.update(state)
    blob = pickle.dumps(stripped)

    restored = pickle.loads(blob)
    assert restored.class_ids == [-1, -1], restored.class_ids
    assert len(restored.class_ids) == len(restored.times)
    restored.append_det(2, 0.7, [9, 10, 11, 12])
    assert restored.class_ids == [-1, -1, -1], restored.class_ids


def test_tracklet_extract_preserves_class_ids():
    # extract() must carry the per-frame class slice onto the sub-track.
    Tracklet = load_tracklet_class(_GTA_LINK_DIR)
    feats = [np.ones(4, dtype=np.float32) * i for i in range(4)]
    tr = Tracklet(7, [0, 1, 2, 3], [0.9, 0.8, 0.7, 0.6],
                  [[1, 2, 3, 4]] * 4, feats=feats, class_ids=[2, 2, 1, 3])
    sub = tr.extract(1, 2)
    assert sub.times == [1, 2], sub.times
    assert sub.class_ids == [2, 1], sub.class_ids
    assert len(sub.class_ids) == len(sub.times)


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


class _FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls


class _FakeResult:
    def __init__(self, boxes, names=None):
        self.boxes = boxes
        self.names = names


# A typical Ultralytics names mapping for the football detector: raw indices in
# arbitrary order so the test proves we remap by NAME, not by raw index.
_FOOTBALL_NAMES = {0: "Referee", 1: "Player", 2: "GoalKeeper", 3: "ball"}


def test_ultralytics_conversion_rows():
    # Raw classes 1 (Player) and 2 (GoalKeeper) -> canonical 2 and 1 by name
    # (class_map scheme: goalkeeper=1, player=2, referee=3).
    result = _FakeResult(
        _FakeBoxes(
            np.array([[1, 2, 3, 4], [10, 20, 30, 40]], dtype=np.float32),
            np.array([0.9, 0.75], dtype=np.float32),
            np.array([1, 2], dtype=np.float32),
        ),
        names=_FOOTBALL_NAMES,
    )
    out = ultralytics_result_to_yolox_output(result)
    expected = np.array(
        [
            [1, 2, 3, 4, 0.9, 1.0, 2],   # Player -> 2
            [10, 20, 30, 40, 0.75, 1.0, 1],  # GoalKeeper -> 1
        ],
        dtype=np.float32,
    )
    assert out.dtype == np.float32
    assert out.shape == (2, 7), out.shape
    assert np.allclose(out, expected), out


def test_ultralytics_name_remap_canonical_and_unknown():
    # referee -> 3 (case-insensitive), ball -> -1 (not a tracked category),
    # and an unknown raw index -> -1.
    result = _FakeResult(
        _FakeBoxes(
            np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32),
            np.array([0.9, 0.8, 0.7], dtype=np.float32),
            np.array([0, 3, 99], dtype=np.float32),  # Referee, ball, out-of-range
        ),
        names=_FOOTBALL_NAMES,
    )
    out = ultralytics_result_to_yolox_output(result)
    assert list(out[:, 6]) == [3.0, -1.0, -1.0], out[:, 6]


def test_ultralytics_no_names_all_unknown():
    # No names mapping on the result -> every class falls back to -1.
    result = _FakeResult(
        _FakeBoxes(
            np.array([[1, 2, 3, 4]], dtype=np.float32),
            np.array([0.9], dtype=np.float32),
            np.array([1], dtype=np.float32),
        ),
        names=None,
    )
    out = ultralytics_result_to_yolox_output(result)
    assert out[0, 6] == -1.0, out


def test_ultralytics_conversion_empty():
    result = _FakeResult(
        _FakeBoxes(
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    )
    assert ultralytics_result_to_yolox_output(result) is None


_TESTS = [
    ("assembly: module name is 'Tracklet'", test_module_name_is_Tracklet),
    ("assembly: per-track times/scores/bboxes/features aligned", test_per_track_alignment),
    ("assembly: features stored float32, not renormalized", test_features_not_renormalized),
    ("assembly: MOT lines match demo.py format", test_mot_lines_match_demo_format),
    ("assembly: format_mot_line exact", test_format_mot_line_exact),
    ("assembly: format_mot_line class in col 8", test_format_mot_line_class_in_col8),
    ("assembly: class_ids carried + aligned 1:1", test_assembler_class_ids_aligned),
    ("assembly: default class is -1 (legacy-safe)", test_assembler_default_class_is_unknown),
    ("Tracklet: class_ids constructed when absent", test_tracklet_class_ids_constructed_when_absent),
    ("Tracklet: legacy pickle backfills class_ids (__setstate__)", test_tracklet_legacy_pickle_backfills_class_ids),
    ("Tracklet: real legacy pickle round-trip backfills", test_tracklet_real_legacy_pickle_roundtrip),
    ("Tracklet: extract preserves class_ids slice", test_tracklet_extract_preserves_class_ids),
    ("assembly: pickle round-trips + references module 'Tracklet'", test_pickle_roundtrip_and_module_ref),
    ("assembly: n_unique_tracks", test_n_unique_tracks),
    ("ultralytics: convert boxes/conf/classes to YOLOX rows", test_ultralytics_conversion_rows),
    ("ultralytics: name->canonical remap + unknown -> -1", test_ultralytics_name_remap_canonical_and_unknown),
    ("ultralytics: no names -> all -1", test_ultralytics_no_names_all_unknown),
    ("ultralytics: empty boxes -> None", test_ultralytics_conversion_empty),
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

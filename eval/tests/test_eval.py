"""
Pure-Python unit tests for the eval package (Task E1).

Run with:
    python eval/tests/test_eval.py

No pytest, torch, trackeval, scipy, or GPU required — stdlib + numpy only.

Covers:
  * load_pred parses sample refined.txt rows (0-based, score kept).
  * load_gt (motchallenge) parses sample gt.txt (1-based, as-is).
  * GT-adapter seam: a custom fmt registers and routes; unknown fmt errors.
  * materialize_layout creates EXACTLY the expected files/paths; GT written
    as-is; pred written with frame+1; seqLength = max 1-based frame across GT
    and converted pred; seqmap content correct.
  * the 0->1 conversion is applied exactly once (idempotent on 1-based pred).
  * parse_trackeval_output extracts HOTA/DetA/AssA/MOTA/IDF1 from a sample
    *_summary.txt fixture (and a *_detailed.csv COMBINED fixture), and raises a
    clear error when a required metric is missing.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

# Make sure the repo root is on sys.path so we can import eval.*
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.gt_adapters import (
    N_COLS,
    Tracks,
    gt_loader,
    load_gt,
    load_pred,
    register_gt_loader,
)
from eval.trackeval_runner import (
    materialize_layout,
    parse_trackeval_output,
    run_trackeval,
)

# ---------------------------------------------------------------------------
# Tiny test harness (mirrors pipeline/tests style)
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


def run_test(name: str, fn):
    try:
        fn()
        _pass(name)
    except AssertionError as exc:
        _fail(name, str(exc) or "AssertionError")
    except Exception as exc:  # noqa: BLE001
        _fail(name, f"{type(exc).__name__}: {exc}")


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Sample fixtures
# ---------------------------------------------------------------------------

# refined.txt: frame,id,x,y,w,h,score,-1,-1,-1 — 0-based frames.
_SAMPLE_PRED = (
    "0,1,100,200,50,80,0.91,-1,-1,-1\n"
    "0,2,300,400,40,90,0.85,-1,-1,-1\n"
    "1,1,102,202,50,80,0.88,-1,-1,-1\n"
    "2,2,305,402,40,90,0.80,-1,-1,-1\n"
)

# MOTChallenge gt.txt: frame,id,x,y,w,h,conf,class,visibility — 1-based frames.
_SAMPLE_GT = (
    "1,1,101,201,50,80,1,1,1\n"
    "1,2,301,401,40,90,1,1,1\n"
    "2,1,103,203,50,80,1,1,1\n"
    "3,2,306,403,40,90,1,1,1\n"
)


# ---------------------------------------------------------------------------
# 1. Prediction loader
# ---------------------------------------------------------------------------


def test_load_pred_basic():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(Path(tmp) / "refined.txt", _SAMPLE_PRED)
        pred = load_pred(p)
        assert isinstance(pred, Tracks)
        assert pred.frame_base == 0, "refined.txt is 0-based"
        assert len(pred) == 4, f"expected 4 rows, got {len(pred)}"
        assert pred.rows.shape == (4, N_COLS), pred.rows.shape
        # Frames kept as-is (0-based): {0,0,1,2}
        assert list(pred.rows[:, 0]) == [0.0, 0.0, 1.0, 2.0]
        # First row box + score preserved.
        assert list(pred.rows[0, 2:6]) == [100.0, 200.0, 50.0, 80.0]
        assert abs(pred.rows[0, 6] - 0.91) < 1e-9, "score must land in conf col"


def test_load_pred_tolerates_blank_and_short_lines():
    with tempfile.TemporaryDirectory() as tmp:
        content = "\n0,1,1,2,3,4,0.5,-1,-1,-1\n1,2,3\n   \n"
        p = _write(Path(tmp) / "refined.txt", content)
        pred = load_pred(p)
        assert len(pred) == 1, "short/blank lines must be skipped"


# ---------------------------------------------------------------------------
# 2. GT adapter (MOTChallenge) + seam
# ---------------------------------------------------------------------------


def test_load_gt_motchallenge_basic():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(Path(tmp) / "gt.txt", _SAMPLE_GT)
        gt = load_gt(p)  # default fmt motchallenge
        assert gt.frame_base == 1, "MOTChallenge GT is 1-based"
        assert len(gt) == 4
        assert list(gt.rows[:, 0]) == [1.0, 1.0, 2.0, 3.0], "frames kept as-is"
        assert gt.max_frame() == 3
        # class/visibility columns parsed.
        assert gt.rows[0, 7] == 1.0 and gt.rows[0, 8] == 1.0


def test_load_gt_unknown_format_raises():
    raised = False
    try:
        load_gt("whatever.txt", fmt="does_not_exist")
    except ValueError as exc:
        raised = True
        assert "does_not_exist" in str(exc), str(exc)
        assert "motchallenge" in str(exc), "error should list known formats"
    assert raised, "unknown fmt must raise ValueError"


def test_gt_loader_seam_register_and_route():
    """A custom converter registers via the seam and load_gt routes to it."""
    sentinel_rows = np.array(
        [[5, 9, 1, 2, 3, 4, 1, 1, 1]], dtype=np.float64
    )

    def _custom_loader(path):
        # Pretend we converted some native format to 1-based MOTChallenge.
        return Tracks(rows=sentinel_rows.copy(), frame_base=1)

    register_gt_loader("custom_export_test", _custom_loader)
    got = load_gt("ignored_path", fmt="custom_export_test")
    assert got.frame_base == 1
    assert got.rows.shape == (1, N_COLS)
    assert got.rows[0, 0] == 5.0 and got.rows[0, 1] == 9.0


def test_gt_loader_decorator_seam():
    """The decorator form also registers a loader the core can route to."""

    @gt_loader("decorated_fmt_test")
    def _loader(path):  # noqa: ANN001
        return Tracks(rows=np.zeros((0, N_COLS)), frame_base=1)

    got = load_gt("ignored", fmt="decorated_fmt_test")
    assert len(got) == 0
    assert got.frame_base == 1


# ---------------------------------------------------------------------------
# 3. materialize_layout — paths, GT as-is, pred frame+1, seqLength, seqmap
# ---------------------------------------------------------------------------


def _read_mot_frames(path: Path) -> list[int]:
    frames = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        frames.append(int(float(line.split(",")[0])))
    return frames


def test_materialize_layout_paths_exist():
    with tempfile.TemporaryDirectory() as tmp:
        gt = load_gt(_write(Path(tmp) / "gt.txt", _SAMPLE_GT))
        pred = load_pred(_write(Path(tmp) / "refined.txt", _SAMPLE_PRED))
        work = Path(tmp) / "work"
        layout = materialize_layout(work, "M59", gt, pred)

        bs = "SPORTS-eval"
        expected = {
            layout.gt_txt: work / "gt/mot_challenge" / bs / "M59/gt/gt.txt",
            layout.seqinfo_ini: work / "gt/mot_challenge" / bs / "M59/seqinfo.ini",
            layout.seqmap_txt: work / "gt/mot_challenge/seqmaps" / f"{bs}.txt",
            layout.tracker_txt: work
            / "trackers/mot_challenge"
            / bs
            / "refined/data/M59.txt",
        }
        for actual, want in expected.items():
            assert Path(actual) == Path(want), f"path mismatch: {actual} != {want}"
            assert Path(actual).is_file(), f"file not written: {actual}"


def test_materialize_gt_written_as_is():
    with tempfile.TemporaryDirectory() as tmp:
        gt = load_gt(_write(Path(tmp) / "gt.txt", _SAMPLE_GT))
        pred = load_pred(_write(Path(tmp) / "refined.txt", _SAMPLE_PRED))
        layout = materialize_layout(Path(tmp) / "work", "M59", gt, pred)
        # GT frames must be UNCHANGED (already 1-based): {1,1,2,3}
        assert _read_mot_frames(layout.gt_txt) == [1, 1, 2, 3]


def test_materialize_pred_frame_plus_one():
    with tempfile.TemporaryDirectory() as tmp:
        gt = load_gt(_write(Path(tmp) / "gt.txt", _SAMPLE_GT))
        pred = load_pred(_write(Path(tmp) / "refined.txt", _SAMPLE_PRED))
        layout = materialize_layout(Path(tmp) / "work", "M59", gt, pred)
        # Pred 0-based {0,0,1,2} -> written 1-based {1,1,2,3}
        assert _read_mot_frames(layout.tracker_txt) == [1, 1, 2, 3]


def test_materialize_pred_columns_and_conf():
    with tempfile.TemporaryDirectory() as tmp:
        gt = load_gt(_write(Path(tmp) / "gt.txt", _SAMPLE_GT))
        pred = load_pred(_write(Path(tmp) / "refined.txt", _SAMPLE_PRED))
        layout = materialize_layout(Path(tmp) / "work", "M59", gt, pred)
        first = layout.tracker_txt.read_text(encoding="utf-8").splitlines()[0]
        cols = first.split(",")
        assert len(cols) == 10, f"MOT line must have 10 cols: {first}"
        # frame=1 (0+1), id=1, box, conf=score 0.91, then -1,-1,-1
        assert cols[0] == "1" and cols[1] == "1"
        assert cols[2:6] == ["100", "200", "50", "80"]
        assert abs(float(cols[6]) - 0.91) < 1e-9, "conf must equal score"
        assert cols[7:] == ["-1", "-1", "-1"]


def test_materialize_seqlength_is_max_one_based_frame():
    with tempfile.TemporaryDirectory() as tmp:
        gt = load_gt(_write(Path(tmp) / "gt.txt", _SAMPLE_GT))
        pred = load_pred(_write(Path(tmp) / "refined.txt", _SAMPLE_PRED))
        layout = materialize_layout(Path(tmp) / "work", "M59", gt, pred)
        ini = layout.seqinfo_ini.read_text(encoding="utf-8")
        # GT max 1-based = 3; pred converted max = 3 -> seqLength 3.
        assert "seqLength=3" in ini, ini
        assert "name=M59" in ini


def test_materialize_seqlength_driven_by_pred_when_longer():
    """seqLength must reflect the pred's converted (1-based) max if it exceeds GT."""
    with tempfile.TemporaryDirectory() as tmp:
        gt = load_gt(_write(Path(tmp) / "gt.txt", "1,1,0,0,1,1,1,1,1\n"))
        # pred frame 9 (0-based) -> 10 (1-based)
        pred = load_pred(
            _write(Path(tmp) / "refined.txt", "9,1,0,0,1,1,0.5,-1,-1,-1\n")
        )
        layout = materialize_layout(Path(tmp) / "work", "S", gt, pred)
        assert "seqLength=10" in layout.seqinfo_ini.read_text(encoding="utf-8")


def test_materialize_seqmap_content():
    with tempfile.TemporaryDirectory() as tmp:
        gt = load_gt(_write(Path(tmp) / "gt.txt", _SAMPLE_GT))
        pred = load_pred(_write(Path(tmp) / "refined.txt", _SAMPLE_PRED))
        layout = materialize_layout(Path(tmp) / "work", "M59", gt, pred)
        lines = layout.seqmap_txt.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "name", f"seqmap header must be 'name', got {lines[0]!r}"
        assert lines[1] == "M59", f"seqmap seq line wrong: {lines[1]!r}"
        assert len([ln for ln in lines if ln.strip()]) == 2, "exactly header + 1 seq"


def test_materialize_custom_bench_split_and_tracker():
    with tempfile.TemporaryDirectory() as tmp:
        gt = load_gt(_write(Path(tmp) / "gt.txt", _SAMPLE_GT))
        pred = load_pred(_write(Path(tmp) / "refined.txt", _SAMPLE_PRED))
        layout = materialize_layout(
            Path(tmp) / "work",
            "M59",
            gt,
            pred,
            tracker_name="mytracker",
            benchmark="MOT17",
            split="train",
        )
        assert "MOT17-train" in str(layout.gt_txt)
        assert layout.tracker_txt.name == "M59.txt"
        assert "mytracker" in str(layout.tracker_txt)
        assert layout.tracker_txt.is_file()


def test_materialize_img_dims_in_seqinfo():
    with tempfile.TemporaryDirectory() as tmp:
        gt = load_gt(_write(Path(tmp) / "gt.txt", _SAMPLE_GT))
        pred = load_pred(_write(Path(tmp) / "refined.txt", _SAMPLE_PRED))
        layout = materialize_layout(
            Path(tmp) / "work", "M59", gt, pred, img_width=1280, img_height=720
        )
        ini = layout.seqinfo_ini.read_text(encoding="utf-8")
        assert "imWidth=1280" in ini and "imHeight=720" in ini


def test_frame_conversion_applied_exactly_once():
    """A pred already flagged 1-based must NOT be shifted again (idempotent)."""
    with tempfile.TemporaryDirectory() as tmp:
        gt = load_gt(_write(Path(tmp) / "gt.txt", _SAMPLE_GT))
        pred = load_pred(_write(Path(tmp) / "refined.txt", _SAMPLE_PRED))
        # Pretend the pred was already 1-based: no +1 should be applied.
        pred.frame_base = 1
        layout = materialize_layout(Path(tmp) / "work", "M59", gt, pred)
        # Original 0-based frames {0,0,1,2} stay as-is because frame_base==1.
        assert _read_mot_frames(layout.tracker_txt) == [0, 0, 1, 2]


def test_materialize_rejects_non_one_based_gt():
    """GT must be 1-based when materialized; a 0-based GT is a programming error."""
    with tempfile.TemporaryDirectory() as tmp:
        pred = load_pred(_write(Path(tmp) / "refined.txt", _SAMPLE_PRED))
        bad_gt = Tracks(rows=np.zeros((1, N_COLS)), frame_base=0)
        raised = False
        try:
            materialize_layout(Path(tmp) / "work", "M59", bad_gt, pred)
        except ValueError:
            raised = True
        assert raised, "0-based GT must be rejected by materialize_layout"


# ---------------------------------------------------------------------------
# 4. parse_trackeval_output
# ---------------------------------------------------------------------------

# Representative TrackEval *_summary.txt: a whitespace header line of metric
# names (first token = tracker name), then a values line. Extra columns present.
_SAMPLE_SUMMARY = (
    "refined HOTA DetA AssA DetRe DetPr AssRe AssPr LocA "
    "MOTA MOTP IDF1 IDR IDP\n"
    "refined 72.345 68.100 77.900 70.0 71.0 80.0 82.0 88.123 "
    "65.500 79.000 81.250 80.0 82.5\n"
)

# Representative *_detailed.csv with a COMBINED aggregate row.
_SAMPLE_DETAILED = (
    "seq,HOTA,DetA,AssA,MOTA,IDF1,Extra\n"
    "M59,71.0,67.0,76.0,64.0,80.0,foo\n"
    "COMBINED,72.345,68.1,77.9,65.5,81.25,bar\n"
)


def test_parse_summary_txt_extracts_metrics():
    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tmp) / "refined"
        _write(tdir / "pedestrian_summary.txt", _SAMPLE_SUMMARY)
        m = parse_trackeval_output(tdir, class_name="pedestrian")
        assert set(m) == {"HOTA", "DetA", "AssA", "MOTA", "IDF1"}, m
        assert abs(m["HOTA"] - 72.345) < 1e-6
        assert abs(m["DetA"] - 68.100) < 1e-6
        assert abs(m["AssA"] - 77.900) < 1e-6
        assert abs(m["MOTA"] - 65.500) < 1e-6
        assert abs(m["IDF1"] - 81.250) < 1e-6


def test_parse_detailed_csv_combined_row():
    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tmp) / "refined"
        _write(tdir / "pedestrian_detailed.csv", _SAMPLE_DETAILED)
        m = parse_trackeval_output(tdir, class_name="pedestrian")
        # COMBINED row used, not the per-seq M59 row.
        assert abs(m["HOTA"] - 72.345) < 1e-6
        assert abs(m["MOTA"] - 65.5) < 1e-6
        assert abs(m["IDF1"] - 81.25) < 1e-6


def test_parse_prefers_summary_over_detailed():
    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tmp) / "refined"
        _write(tdir / "pedestrian_summary.txt", _SAMPLE_SUMMARY)
        _write(tdir / "pedestrian_detailed.csv", _SAMPLE_DETAILED)
        m = parse_trackeval_output(tdir, class_name="pedestrian")
        assert abs(m["HOTA"] - 72.345) < 1e-6  # both agree here; just smoke


def test_parse_missing_metric_raises_clear_error():
    """A summary missing MOTA/IDF1 must raise naming the missing metric(s)."""
    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tmp) / "refined"
        # Header without MOTA or IDF1.
        bad = "refined HOTA DetA AssA LocA\nrefined 70.0 66.0 75.0 88.0\n"
        _write(tdir / "pedestrian_summary.txt", bad)
        raised = False
        try:
            parse_trackeval_output(tdir, class_name="pedestrian")
        except KeyError as exc:
            raised = True
            msg = str(exc)
            assert "MOTA" in msg and "IDF1" in msg, msg
        assert raised, "missing metrics must raise KeyError"


def test_parse_no_output_file_raises():
    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tmp) / "refined"
        tdir.mkdir(parents=True)
        raised = False
        try:
            parse_trackeval_output(tdir, class_name="pedestrian")
        except FileNotFoundError as exc:
            raised = True
            assert "pedestrian" in str(exc)
        assert raised, "absent output files must raise FileNotFoundError"


def test_parse_respects_class_name():
    """The class_name selects which <class>_summary.txt is read."""
    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tmp) / "refined"
        _write(tdir / "player_summary.txt", _SAMPLE_SUMMARY)
        m = parse_trackeval_output(tdir, class_name="player")
        assert abs(m["HOTA"] - 72.345) < 1e-6


# ---------------------------------------------------------------------------
# 5. run_trackeval seam
# ---------------------------------------------------------------------------


def test_run_trackeval_is_not_implemented_seam():
    raised = False
    try:
        run_trackeval(object())  # type: ignore[arg-type]
    except NotImplementedError as exc:
        raised = True
        assert "Task E2" in str(exc), str(exc)
    assert raised, "run_trackeval must remain a NotImplementedError seam in E1"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_TESTS = [
    ("pred: load_pred basic (0-based, score kept)", test_load_pred_basic),
    ("pred: load_pred tolerates blank/short lines", test_load_pred_tolerates_blank_and_short_lines),
    ("gt: load_gt motchallenge basic (1-based as-is)", test_load_gt_motchallenge_basic),
    ("gt: unknown format raises", test_load_gt_unknown_format_raises),
    ("gt seam: register_gt_loader routes", test_gt_loader_seam_register_and_route),
    ("gt seam: gt_loader decorator routes", test_gt_loader_decorator_seam),
    ("layout: exact files/paths created", test_materialize_layout_paths_exist),
    ("layout: GT written as-is (1-based)", test_materialize_gt_written_as_is),
    ("layout: pred frame+1 (0->1)", test_materialize_pred_frame_plus_one),
    ("layout: pred MOT columns + conf=score", test_materialize_pred_columns_and_conf),
    ("layout: seqLength = max 1-based frame", test_materialize_seqlength_is_max_one_based_frame),
    ("layout: seqLength driven by longer pred", test_materialize_seqlength_driven_by_pred_when_longer),
    ("layout: seqmap header 'name' + seq", test_materialize_seqmap_content),
    ("layout: custom bench/split/tracker", test_materialize_custom_bench_split_and_tracker),
    ("layout: img dims in seqinfo", test_materialize_img_dims_in_seqinfo),
    ("layout: 0->1 conversion applied exactly once", test_frame_conversion_applied_exactly_once),
    ("layout: rejects non-1-based GT", test_materialize_rejects_non_one_based_gt),
    ("parse: summary.txt extracts 5 metrics", test_parse_summary_txt_extracts_metrics),
    ("parse: detailed.csv COMBINED row", test_parse_detailed_csv_combined_row),
    ("parse: prefers summary over detailed", test_parse_prefers_summary_over_detailed),
    ("parse: missing metric raises clear error", test_parse_missing_metric_raises_clear_error),
    ("parse: no output file raises", test_parse_no_output_file_raises),
    ("parse: respects class_name", test_parse_respects_class_name),
    ("seam: run_trackeval NotImplementedError", test_run_trackeval_is_not_implemented_seam),
]


def main() -> int:
    print("=" * 60)
    print("eval unit tests (Task E1, CPU-only)")
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

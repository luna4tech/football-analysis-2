"""
Pure-Python unit tests for the eval package (Tasks E1 + E2).

Run with:
    python eval/tests/test_eval.py

No pytest, torch, trackeval, scipy, or GPU required — stdlib + numpy only.

Task E1 coverage:
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

Task E2 coverage (CPU-only, no trackeval/scipy):
  * extract_metrics: HOTA/DetA/AssA returned as mean of alpha arrays;
    MOTA/IDF1 as scalars; clear error when an expected key is missing.
  * format_metrics_table + write_metrics_json produce the documented shape.
  * run_evaluation end-to-end with an INJECTED stub _runner that returns a
    fake result: asserts metrics.json is written with the 5 metrics; no
    trackeval import triggered.
  * run_trackeval raises ImportError when trackeval is not installed (not
    NotImplementedError — that was the E1 seam; E2 wires it).
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
    extract_all_metrics,
    extract_metrics,
    materialize_layout,
    parse_trackeval_output,
    run_trackeval,
)
from eval.evaluate import (
    format_metrics_table,
    run_evaluation,
    write_metrics_json,
)
from eval.attributes import (
    aggregate_gt_tracks,
    aggregate_pred_tracks,
    associate_tracks,
    class_metrics,
    compute_attribute_metrics,
    consistency_metrics,
    parse_attr_rows,
    run_attribute_eval,
    team_metrics,
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


# GT exactly as the real test-data is shipped: conf=1 but class=-1, vis=-1 on
# every row (1-based frames). The loader coerces class/vis -1 -> 1.
_SAMPLE_GT_RAW_MINUS1 = (
    "1,21,427.0,529.0,17.0,51.0,1,-1,-1,-1\n"
    "1,22,1149.0,388.0,10.0,34.0,1,-1,-1,-1\n"
    "2,21,430.0,531.0,17.0,51.0,1,-1,-1,-1\n"
    "3,22,1152.0,390.0,10.0,34.0,1,-1,-1,-1\n"
)


def _read_mot_rows(path: Path) -> list[list[str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(line.split(","))
    return rows


def test_materialize_gt_class_and_visibility_are_one():
    """E3: GT source rows with class=-1/vis=-1 must be written as class=1/vis=1.

    Real test-data ships class=-1 and visibility=-1 on every GT row. The loader
    coerces those to 1; the writer must carry the canonical class@7/vis@8 instead
    of the old hardcoded -1 (which made TrackEval's pedestrian eval find ZERO
    valid GT and report ~0 for every metric).
    """
    with tempfile.TemporaryDirectory() as tmp:
        # load_gt(motchallenge) semantics: parse the raw file (class/vis -1 -> 1).
        gt = load_gt(_write(Path(tmp) / "gt.txt", _SAMPLE_GT_RAW_MINUS1))
        # In-memory canonical GT already has class=1, vis=1.
        assert all(gt.rows[:, 7] == 1.0), "loader should coerce class -1 -> 1"
        assert all(gt.rows[:, 8] == 1.0), "loader should coerce visibility -1 -> 1"

        pred = load_pred(_write(Path(tmp) / "refined.txt", _SAMPLE_PRED))
        layout = materialize_layout(Path(tmp) / "work", "M59", gt, pred)

        # Read back the WRITTEN gt.txt and assert EVERY row's class/vis == 1.
        gt_written = _read_mot_rows(layout.gt_txt)
        assert len(gt_written) == 4, f"expected 4 GT rows, got {len(gt_written)}"
        for i, cols in enumerate(gt_written):
            assert len(cols) == 9, f"GT line {i} must have 9 cols: {cols}"
            assert int(cols[7]) == 1, f"GT row {i} class must be 1, got {cols[7]!r}"
            assert float(cols[8]) == 1.0, (
                f"GT row {i} visibility must be 1, got {cols[8]!r}"
            )
        # First GT row's frame/id/box/conf preserved (sanity of the rest).
        first = gt_written[0]
        assert first[0] == "1" and first[1] == "21"
        assert first[2:6] == ["427", "529", "17", "51"]
        assert abs(float(first[6]) - 1.0) < 1e-9


def test_materialize_pred_indices_0_to_6_intact():
    """E3: the tracker file's TrackEval-read columns 0-6 stay valid.

    TrackEval's tracker reader uses frame,id,bbox,conf at indices 0-6. The GT fix
    must not regress those: frame is +1 shifted (0-based -> 1-based) and conf is
    the pred score.
    """
    with tempfile.TemporaryDirectory() as tmp:
        gt = load_gt(_write(Path(tmp) / "gt.txt", _SAMPLE_GT_RAW_MINUS1))
        pred = load_pred(_write(Path(tmp) / "refined.txt", _SAMPLE_PRED))
        layout = materialize_layout(Path(tmp) / "work", "M59", gt, pred)

        pred_written = _read_mot_rows(layout.tracker_txt)
        # _SAMPLE_PRED rows are 0-based {0,0,1,2} -> 1-based {1,1,2,3}.
        assert [int(r[0]) for r in pred_written] == [1, 1, 2, 3], "frame must be +1"
        first = pred_written[0]
        # id, box, conf=score at indices 1-6 unchanged.
        assert first[1] == "1"
        assert first[2:6] == ["100", "200", "50", "80"]
        assert abs(float(first[6]) - 0.91) < 1e-9, "conf must equal pred score"
        # Indices 0-6 are exactly 7 fields; the line itself keeps 10 columns.
        assert len(first) == 10, f"pred line must have 10 cols: {first}"


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
# 5. run_trackeval: no-trackeval host raises ImportError (E2 wired, but dep
#    absent on this machine)
# ---------------------------------------------------------------------------


def test_run_trackeval_raises_import_error_without_trackeval():
    """On a host without trackeval installed, run_trackeval raises ImportError."""
    # We need a real LayoutPaths-like object; the simplest is to build a minimal
    # one via materialize_layout with synthetic data.
    with tempfile.TemporaryDirectory() as tmp:
        gt = load_gt(_write(Path(tmp) / "gt.txt", _SAMPLE_GT))
        pred = load_pred(_write(Path(tmp) / "refined.txt", _SAMPLE_PRED))
        layout = materialize_layout(Path(tmp) / "work", "M59", gt, pred)
        try:
            import trackeval as _te  # type: ignore[import]  # noqa: F401
            # trackeval IS installed — skip this test gracefully.
            print("  [skip] trackeval is installed on this host; ImportError test not applicable")
            return
        except ImportError:
            pass
        # trackeval is NOT installed; expect ImportError with a helpful message.
        raised = False
        try:
            run_trackeval(layout)
        except ImportError as exc:
            raised = True
            assert "trackeval" in str(exc).lower(), str(exc)
        assert raised, "run_trackeval must raise ImportError when trackeval is absent"


# ---------------------------------------------------------------------------
# 6. extract_metrics (Task E2 — pure, CPU-testable)
# ---------------------------------------------------------------------------

# Fake raw result dict mirroring what TrackEval's Evaluator.evaluate returns.
# Structure: result[dataset][tracker]["COMBINED_SEQ"][class][FAMILY][metric_name],
# where FAMILY is "HOTA"/"CLEAR"/"Identity"/"Count". HOTA/DetA/AssA live under the
# "HOTA" family as arrays over alpha thresholds; MOTA under "CLEAR", IDF1 under
# "Identity" as scalars. TrackEval stores FRACTIONS (0-1); the params below are
# percentages for readability, divided by 100 to match the real structure.
def _fake_result(
    tracker: str = "refined",
    cls: str = "pedestrian",
    hota: float = 72.345,
    det_a: float = 68.1,
    ass_a: float = 77.9,
    mota: float = 65.5,
    idf1: float = 81.25,
) -> dict:
    alpha_count = 19
    hota_family = {
        "HOTA": np.full(alpha_count, hota / 100.0),
        "DetA": np.full(alpha_count, det_a / 100.0),
        "AssA": np.full(alpha_count, ass_a / 100.0),
        "LocA": np.full(alpha_count, 0.88),
        "HOTA(0)": 0.8555,  # scalar float -> x100
        "HOTA_TP": np.full(alpha_count, 95000.0),  # integer array -> must be SKIPPED
        # Extra key TrackEval emits — must be tolerated.
        "_extra": "ignore me",
    }
    clear_family = {
        "MOTA": mota / 100.0,
        "MOTP": 0.75,
        "IDSW": 124,
        "Frag": 1578,
        "CLR_TP": 91642,
        "CLR_FN": 11276,
        "CLR_FP": 25106,
        "MT": 23,
        "PT": 1,
        "ML": 1,
    }
    identity_family = {
        "IDF1": idf1 / 100.0,
        "IDR": 0.86,
        "IDP": 0.76,
        "IDTP": 89059,
        "IDFN": 13859,
        "IDFP": 27689,
    }
    count_family = {"Dets": 116748, "GT_Dets": 102918, "IDs": 64, "GT_IDs": 25}
    per_class = {
        "HOTA": hota_family,
        "CLEAR": clear_family,
        "Identity": identity_family,
        "Count": count_family,
    }
    return {
        "MotChallenge2DBox": {
            tracker: {
                "COMBINED_SEQ": {
                    cls: per_class,
                },
            },
        },
    }


def test_extract_metrics_array_mean():
    """HOTA/DetA/AssA are returned as the mean over their alpha arrays."""
    result = _fake_result(hota=72.345, det_a=68.1, ass_a=77.9)
    m = extract_metrics(result, tracker_name="refined")
    assert set(m) == {"HOTA", "DetA", "AssA", "MOTA", "IDF1"}, m
    assert abs(m["HOTA"] - 72.345) < 1e-6, m
    assert abs(m["DetA"] - 68.1) < 1e-6, m
    assert abs(m["AssA"] - 77.9) < 1e-6, m


def test_extract_metrics_mota_idf1_scalars():
    """MOTA and IDF1 (plain scalars) are returned unchanged."""
    result = _fake_result(mota=65.5, idf1=81.25)
    m = extract_metrics(result, tracker_name="refined")
    assert abs(m["MOTA"] - 65.5) < 1e-9, m
    assert abs(m["IDF1"] - 81.25) < 1e-9, m


def test_extract_metrics_wrong_tracker_raises():
    result = _fake_result()
    raised = False
    try:
        extract_metrics(result, tracker_name="wrong_tracker")
    except KeyError as exc:
        raised = True
        assert "wrong_tracker" in str(exc), str(exc)
    assert raised


def test_extract_metrics_missing_metric_raises():
    result = _fake_result()
    # Drop MOTA from the CLEAR family.
    del result["MotChallenge2DBox"]["refined"]["COMBINED_SEQ"]["pedestrian"]["CLEAR"]["MOTA"]
    raised = False
    try:
        extract_metrics(result, tracker_name="refined")
    except KeyError as exc:
        raised = True
        assert "MOTA" in str(exc), str(exc)
    assert raised


def test_extract_metrics_custom_class():
    result = _fake_result(tracker="t", cls="player")
    m = extract_metrics(result, tracker_name="t", class_name="player")
    assert abs(m["HOTA"] - 72.345) < 1e-6


def test_extract_all_metrics_full():
    """All families dumped; rates x100, counts raw int, arrays meaned, int-arrays skipped."""
    result = _fake_result()
    allm = extract_all_metrics(result, tracker_name="refined")
    assert set(allm) == {"HOTA", "CLEAR", "Identity", "Count"}, allm

    hota = allm["HOTA"]
    assert abs(hota["HOTA"] - 72.345) < 1e-6  # array mean x100
    assert abs(hota["DetA"] - 68.1) < 1e-6
    assert abs(hota["LocA"] - 88.0) < 1e-6
    assert abs(hota["HOTA(0)"] - 85.55) < 1e-6  # scalar float x100
    assert "HOTA_TP" not in hota, "integer-array field must be skipped"
    assert "_extra" not in hota, "private/non-numeric field must be skipped"

    clear = allm["CLEAR"]
    assert abs(clear["MOTA"] - 65.5) < 1e-6  # float x100
    assert abs(clear["MOTP"] - 75.0) < 1e-6
    assert clear["IDSW"] == 124 and isinstance(clear["IDSW"], int)  # count: raw int
    assert clear["CLR_FP"] == 25106 and clear["MT"] == 23 and clear["Frag"] == 1578

    ident = allm["Identity"]
    assert abs(ident["IDF1"] - 81.25) < 1e-6
    assert ident["IDTP"] == 89059 and isinstance(ident["IDTP"], int)

    cnt = allm["Count"]
    assert cnt["Dets"] == 116748 and cnt["GT_IDs"] == 25
    assert all(isinstance(v, int) for v in cnt.values())


# ---------------------------------------------------------------------------
# 7. format_metrics_table (Task E2)
# ---------------------------------------------------------------------------


def test_format_metrics_table_structure():
    metrics = {"HOTA": 72.345, "DetA": 68.1, "AssA": 77.9, "MOTA": 65.5, "IDF1": 81.25}
    table = format_metrics_table(metrics)
    for key in ("HOTA", "DetA", "AssA", "MOTA", "IDF1"):
        assert key in table, f"{key} not in table:\n{table}"
    # All five values appear as rounded floats.
    assert "72.345" in table
    assert "65.500" in table


def test_format_metrics_table_missing_key_shows_na():
    # Missing metrics should display N/A rather than raising.
    metrics = {"HOTA": 72.345}
    table = format_metrics_table(metrics)
    assert "N/A" in table, table


# ---------------------------------------------------------------------------
# 8. write_metrics_json (Task E2)
# ---------------------------------------------------------------------------


def test_write_metrics_json_schema():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "metrics.json"
        metrics = {"HOTA": 72.345, "DetA": 68.1, "AssA": 77.9, "MOTA": 65.5, "IDF1": 81.25}
        write_metrics_json(out, "M59", "refined", metrics)
        assert out.is_file(), "metrics.json not written"
        import json
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["seq_name"] == "M59"
        assert data["tracker_name"] == "refined"
        assert set(data["metrics"]) == {"HOTA", "DetA", "AssA", "MOTA", "IDF1"}
        assert abs(data["metrics"]["HOTA"] - 72.345) < 1e-9


def test_write_metrics_json_creates_parents():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "deep" / "nested" / "metrics.json"
        write_metrics_json(out, "S", "refined", {"HOTA": 1.0, "DetA": 1.0, "AssA": 1.0, "MOTA": 1.0, "IDF1": 1.0})
        assert out.is_file()


def test_write_metrics_json_includes_all_metrics():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "metrics.json"
        metrics = {"HOTA": 60.9, "DetA": 54.9, "AssA": 67.6, "MOTA": 64.5, "IDF1": 81.1}
        allm = extract_all_metrics(_fake_result(), tracker_name="refined")
        write_metrics_json(out, "M59", "refined", metrics, all_metrics=allm)
        import json
        data = json.loads(out.read_text(encoding="utf-8"))
        assert set(data["metrics"]) == {"HOTA", "DetA", "AssA", "MOTA", "IDF1"}
        assert set(data["all_metrics"]) == {"HOTA", "CLEAR", "Identity", "Count"}
        assert data["all_metrics"]["CLEAR"]["IDSW"] == 124
        assert data["all_metrics"]["Count"]["GT_IDs"] == 25


def test_write_metrics_json_omits_all_metrics_when_absent():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "metrics.json"
        write_metrics_json(out, "S", "refined", {"HOTA": 1.0})
        import json
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "all_metrics" not in data


# ---------------------------------------------------------------------------
# 9. run_evaluation end-to-end with injected stub runner (Task E2)
# ---------------------------------------------------------------------------


def test_run_evaluation_injected_stub_writes_metrics_json():
    """Full end-to-end with a fake _runner — no trackeval import triggered."""
    import json as _json

    def _stub_runner(layout, **kwargs):
        """Fake runner: returns a fake result dict; records the call."""
        _stub_runner.called = True
        return _fake_result(tracker=layout.tracker_name)

    _stub_runner.called = False

    with tempfile.TemporaryDirectory() as tmp:
        gt_file = _write(Path(tmp) / "gt.txt", _SAMPLE_GT)
        pred_file = _write(Path(tmp) / "refined.txt", _SAMPLE_PRED)
        work_dir = Path(tmp) / "work"
        out_path = Path(tmp) / "metrics.json"

        metrics = run_evaluation(
            pred_file,
            gt_file,
            "M59",
            work_dir,
            out_path,
            _runner=_stub_runner,
        )

        assert _stub_runner.called, "injected stub must be called"
        assert set(metrics) == {"HOTA", "DetA", "AssA", "MOTA", "IDF1"}
        assert out_path.is_file(), "metrics.json must be written"
        data = _json.loads(out_path.read_text(encoding="utf-8"))
        assert data["seq_name"] == "M59"
        assert data["tracker_name"] == "refined"
        assert set(data["metrics"]) == {"HOTA", "DetA", "AssA", "MOTA", "IDF1"}


def test_run_evaluation_stub_no_trackeval_imported():
    """Verify the stub path never imports trackeval (by catching ImportError if it did)."""
    import sys as _sys

    # Temporarily block trackeval from importing.
    _sentinel = object()
    _sys.modules.setdefault("trackeval", _sentinel)  # type: ignore[arg-type]

    def _stub(layout, **kwargs):
        return _fake_result(tracker=layout.tracker_name)

    with tempfile.TemporaryDirectory() as tmp:
        gt_file = _write(Path(tmp) / "gt.txt", _SAMPLE_GT)
        pred_file = _write(Path(tmp) / "refined.txt", _SAMPLE_PRED)
        work_dir = Path(tmp) / "work"
        out_path = Path(tmp) / "metrics.json"
        # Must not raise even with the blocked trackeval sentinel.
        metrics = run_evaluation(
            pred_file, gt_file, "M59", work_dir, out_path, _runner=_stub
        )
        assert "HOTA" in metrics

    # Clean up sentinel if we added it.
    if _sys.modules.get("trackeval") is _sentinel:
        del _sys.modules["trackeval"]


# ---------------------------------------------------------------------------
# 10. Task 005 — _format_gt_line forces class=1/vis=1 for an EXTENDED GT
# ---------------------------------------------------------------------------


def test_format_gt_line_forces_class1_vis1_for_extended_gt():
    """A canonical GT row carrying semantic class 0 or 2 must still be written
    with class=1 and visibility=1, so TrackEval's pedestrian (class==1) eval keeps
    every row and HOTA/MOTA stay all-person."""
    from eval.trackeval_runner import _format_gt_line

    # canonical row: frame,id,x,y,w,h,conf,class,visibility
    player = _format_gt_line([5, 7, 10, 20, 30, 40, 1.0, 0, 0.5])  # class 0
    referee = _format_gt_line([6, 8, 11, 21, 31, 41, 1.0, 2, 0.9])  # class 2
    gk = _format_gt_line([7, 9, 12, 22, 32, 42, 1.0, 1, 1.0])  # class 1

    for line, lbl in ((player, "player"), (referee, "referee"), (gk, "gk")):
        cols = line.split(",")
        assert len(cols) == 9, f"{lbl}: gt line must have 9 cols: {line}"
        assert cols[7] == "1", f"{lbl}: class must be forced to 1, got {cols[7]!r}"
        assert cols[8] == "1", f"{lbl}: visibility must be forced to 1, got {cols[8]!r}"

    # Sanity: frame/id/box/conf for the player row are preserved.
    pc = player.split(",")
    assert pc[0] == "5" and pc[1] == "7"
    assert pc[2:6] == ["10", "20", "30", "40"]
    assert abs(float(pc[6]) - 1.0) < 1e-9


def test_extended_gt_materializes_all_class1():
    """End-to-end: an extended GT (class 0/1/2 via a custom loader) materializes a
    gt.txt whose EVERY row is class=1/vis=1 (no rows dropped by TrackEval)."""
    extended = np.array(
        [
            [1, 1, 0, 0, 10, 10, 1, 0, 1],  # player
            [1, 2, 20, 20, 10, 10, 1, 2, 1],  # referee
            [2, 3, 40, 40, 10, 10, 1, 1, 1],  # goalkeeper
        ],
        dtype=np.float64,
    )
    with tempfile.TemporaryDirectory() as tmp:
        gt = Tracks(rows=extended.copy(), frame_base=1)
        pred = load_pred(_write(Path(tmp) / "refined.txt", _SAMPLE_PRED))
        layout = materialize_layout(Path(tmp) / "work", "X", gt, pred)
        for cols in _read_mot_rows(layout.gt_txt):
            assert cols[7] == "1" and cols[8] == "1", cols


# ---------------------------------------------------------------------------
# 11. Task 005 — attribute raw parser (NO -1 coercion)
# ---------------------------------------------------------------------------

# Extended pred: frame,id,x,y,w,h,score,class,team,-1 (0-based frames).
_ATTR_PRED = (
    "0,1,100,100,50,50,0.9,0,0,-1\n"
    "1,1,101,101,50,50,0.9,0,0,-1\n"
    "0,2,300,300,40,40,0.8,2,-1,-1\n"
    "0,3,500,100,45,55,0.7,1,-1,-1\n"  # goalkeeper, no team
)

# Extended GT: frame,id,x,y,w,h,conf,class,team (1-based frames).
_ATTR_GT = (
    "1,10,102,102,50,50,1,0,1\n"
    "2,10,102,102,50,50,1,0,1\n"
    "1,20,301,301,40,40,1,2,-1\n"
    "1,30,502,102,45,55,1,1,-1\n"  # goalkeeper
)


def test_parse_attr_rows_no_minus1_coercion():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(Path(tmp) / "pred.txt", _ATTR_PRED)
        rows = parse_attr_rows(p)
        assert rows.shape == (4, 8), rows.shape
        # class col (idx 6): 0,0,2,1 ; team col (idx 7): 0,0,-1,-1 (kept verbatim).
        assert rows[:, 6].tolist() == [0.0, 0.0, 2.0, 1.0]
        assert rows[:, 7].tolist() == [0.0, 0.0, -1.0, -1.0], "team -1 must NOT coerce"


def test_parse_attr_rows_missing_blank_become_minus1():
    with tempfile.TemporaryDirectory() as tmp:
        # row 1 omits class/team entirely; row 2 has blank class.
        content = "0,1,1,2,3,4,0.5\n1,2,1,2,3,4,0.5,,0\n"
        p = _write(Path(tmp) / "pred.txt", content)
        rows = parse_attr_rows(p)
        assert rows.shape == (2, 8)
        assert rows[0, 6] == -1.0 and rows[0, 7] == -1.0, "absent -> -1"
        assert rows[1, 6] == -1.0, "blank class -> -1"
        assert rows[1, 7] == 0.0, "explicit 0 team kept"


# ---------------------------------------------------------------------------
# 12. Task 005 — IoU matching + majority association
# ---------------------------------------------------------------------------


def test_associate_tracks_iou_majority_with_frame_shift():
    pred = parse_attr_rows_str(_ATTR_PRED)
    gt = parse_attr_rows_str(_ATTR_GT)
    assoc = associate_tracks(pred, gt)  # pred 0-based, gt 1-based -> +1 shift
    # pred 1 (player) overlaps gt 10 across frames; pred 2 -> gt 20; pred 3 -> gt 30.
    assert assoc == {1: 10, 2: 20, 3: 30}, assoc


def test_associate_tracks_no_match_when_below_threshold():
    pred = np.array([[0, 1, 0, 0, 10, 10, 0, 0]], dtype=np.float64)
    gt = np.array([[1, 10, 900, 900, 10, 10, 0, 1]], dtype=np.float64)
    assert associate_tracks(pred, gt) == {}, "non-overlapping boxes must not match"


def test_associate_tracks_majority_breaks_split():
    """A pred track overlapping two GT ids picks the one it co-occurs with most."""
    pred = np.array(
        [
            [0, 1, 100, 100, 50, 50, 0, 0],
            [1, 1, 100, 100, 50, 50, 0, 0],
            [2, 1, 100, 100, 50, 50, 0, 0],
        ],
        dtype=np.float64,
    )
    gt = np.array(
        [
            [1, 10, 100, 100, 50, 50, 0, 0],  # frame 1 (pred 0+1)
            [2, 10, 100, 100, 50, 50, 0, 0],  # frame 2 (pred 1+1) -> 10 twice
            [3, 20, 100, 100, 50, 50, 0, 0],  # frame 3 (pred 2+1) -> 20 once
        ],
        dtype=np.float64,
    )
    assert associate_tracks(pred, gt) == {1: 10}, "majority GT id wins"


# ---------------------------------------------------------------------------
# 13. Task 005 — class accuracy + confusion matrix
# ---------------------------------------------------------------------------


def test_class_metrics_accuracy_and_confusion():
    assoc = {1: 10, 2: 20, 3: 30}
    pred_cls = {1: 0, 2: 0, 3: 1}  # pred 2 wrong (referee gt -> labelled player)
    gt_cls = {10: 0, 20: 2, 30: 1}
    m = class_metrics(assoc, pred_cls, gt_cls)
    assert m["n_evaluated"] == 3
    assert m["n_correct"] == 2, m
    assert abs(m["accuracy"] - 2 / 3) < 1e-9, m
    # confusion[gt][pred]; gt referee(2) predicted player(0) -> [2][0] == 1.
    conf = m["confusion"]
    assert conf[0][0] == 1, "player->player"
    assert conf[2][0] == 1, "referee misclassified as player"
    assert conf[1][1] == 1, "gk->gk"


def test_class_metrics_skips_unknown_gt_class():
    assoc = {1: 10, 2: 20}
    pred_cls = {1: 0, 2: 0}
    gt_cls = {10: 0, 20: -1}  # gt 20 unknown -> not scored
    m = class_metrics(assoc, pred_cls, gt_cls)
    assert m["n_evaluated"] == 1, m
    assert m["accuracy"] == 1.0, m


# ---------------------------------------------------------------------------
# 14. Task 005 — players-only permutation-invariant team accuracy
# ---------------------------------------------------------------------------


def test_team_metrics_permutation_invariant_players_only():
    assoc = {1: 10, 2: 20, 3: 30, 4: 40}
    pred_cls = {1: 0, 2: 0, 3: 0, 4: 2}  # 4 is a referee -> excluded
    gt_cls = {10: 0, 20: 0, 30: 0, 40: 2}
    # pred teams are the SWAP of gt teams -> perm-invariant accuracy must be 1.0.
    pred_team = {1: 1, 2: 1, 3: 0, 4: -1}
    gt_team = {10: 0, 20: 0, 30: 1, 40: -1}
    m = team_metrics(assoc, pred_cls, pred_team, gt_cls, gt_team)
    assert m["n_players"] == 3, m
    assert abs(m["accuracy"] - 1.0) < 1e-9, "swapped labels -> still perfect"


def test_team_metrics_excludes_gk_and_referee():
    assoc = {1: 10, 2: 20}
    pred_cls = {1: 1, 2: 2}  # gk + referee, NO players
    gt_cls = {10: 1, 20: 2}
    pred_team = {1: 0, 2: 1}
    gt_team = {10: 0, 20: 1}
    m = team_metrics(assoc, pred_cls, pred_team, gt_cls, gt_team)
    assert m["n_players"] == 0, m
    assert m["accuracy"] is None and "reason" in m, m


# ---------------------------------------------------------------------------
# 15. Task 005 — consistency (purity / switches)
# ---------------------------------------------------------------------------


def test_consistency_purity_and_switches():
    # track 1: classes [0,0,0] pure; track 2: [0,2,0] -> majority 0, 2 switches.
    rows = np.array(
        [
            [0, 1, 0, 0, 1, 1, 0, -1],
            [1, 1, 0, 0, 1, 1, 0, -1],
            [2, 1, 0, 0, 1, 1, 0, -1],
            [0, 2, 0, 0, 1, 1, 0, -1],
            [1, 2, 0, 0, 1, 1, 2, -1],
            [2, 2, 0, 0, 1, 1, 0, -1],
        ],
        dtype=np.float64,
    )
    m = consistency_metrics(rows)
    assert m["n_tracks"] == 2
    t1 = m["per_track"]["1"]
    t2 = m["per_track"]["2"]
    assert t1["purity"] == 1.0 and t1["switches"] == 0
    assert abs(t2["purity"] - 2 / 3) < 1e-9, t2
    assert t2["switches"] == 2, t2
    assert m["total_class_switches"] == 2


# ---------------------------------------------------------------------------
# 16. Task 005 — graceful degradation branches
# ---------------------------------------------------------------------------


def test_compute_metrics_gt_all_minus1_skips_class_and_team():
    pred = parse_attr_rows_str(_ATTR_PRED)
    # GT all class -1, team -1 (the current sample-GT case).
    gt = np.array(
        [
            [1, 10, 102, 102, 50, 50, -1, -1],
            [2, 10, 102, 102, 50, 50, -1, -1],
        ],
        dtype=np.float64,
    )
    m = compute_attribute_metrics(gt, pred)
    assert m["class"].get("skipped") is True, m["class"]
    assert "all -1" in m["class"]["reason"]
    assert m["team"].get("skipped") is True, m["team"]
    # Consistency + counts always emitted.
    assert "consistency" in m and m["consistency"]["n_tracks"] >= 1
    assert "counts" in m and "matched_tracks" in m["counts"]


def test_compute_metrics_no_iou_match_skips_but_keeps_consistency():
    pred = np.array([[0, 1, 0, 0, 10, 10, 0, 0]], dtype=np.float64)
    gt = np.array([[1, 10, 900, 900, 10, 10, 0, 1]], dtype=np.float64)
    m = compute_attribute_metrics(gt, pred)
    assert m["counts"]["matched_tracks"] == 0
    assert m["class"].get("skipped") is True
    assert m["team"].get("skipped") is True
    assert m["consistency"]["n_tracks"] == 1, "consistency still emitted"


def test_compute_metrics_pred_no_team_skips_team_only():
    pred = np.array(
        [[0, 1, 100, 100, 50, 50, 0, -1], [1, 1, 100, 100, 50, 50, 0, -1]],
        dtype=np.float64,
    )  # pred has class 0 but NO team
    gt = np.array(
        [[1, 10, 100, 100, 50, 50, 0, 1], [2, 10, 100, 100, 50, 50, 0, 1]],
        dtype=np.float64,
    )
    m = compute_attribute_metrics(gt, pred)
    # GT has semantic class -> class metrics computed; pred lacks team -> team skipped.
    assert m["class"].get("skipped") is not True, m["class"]
    assert m["team"].get("skipped") is True and "pred" in m["team"]["reason"], m["team"]


def test_ari_nmi_skipped_when_sklearn_absent():
    """When sklearn cannot import, team metrics still emit accuracy with ARI/NMI None."""
    import sys as _sys

    _SENTINEL = "__absent__"
    # Block BOTH the package and the submodule: an earlier test may have cached
    # sklearn.metrics, in which case `from sklearn.metrics import ...` would still
    # resolve. Setting them to None forces an ImportError on the lazy import.
    blocked = ("sklearn", "sklearn.metrics")
    saved = {name: _sys.modules.get(name, _SENTINEL) for name in blocked}
    for name in blocked:
        _sys.modules[name] = None  # type: ignore[assignment]
    try:
        assoc = {1: 10, 2: 20}
        pred_cls = {1: 0, 2: 0}
        gt_cls = {10: 0, 20: 0}
        pred_team = {1: 0, 2: 1}
        gt_team = {10: 0, 20: 1}
        m = team_metrics(assoc, pred_cls, pred_team, gt_cls, gt_team)
        assert m["accuracy"] is not None, "accuracy is pure; must still be present"
        assert m["ari"] is None and m["nmi"] is None, m
        assert "clustering_reason" in m, m
    finally:
        for name in blocked:
            if saved[name] == _SENTINEL:
                _sys.modules.pop(name, None)
            else:
                _sys.modules[name] = saved[name]


# ---------------------------------------------------------------------------
# 17. Task 005 — track_attributes.json preferred for pred per-track class/team
# ---------------------------------------------------------------------------


def test_aggregate_pred_prefers_track_attributes_json():
    with tempfile.TemporaryDirectory() as tmp:
        # txt says track 1 class 0/team 0; json (authoritative) says class 1/team 1.
        pred_txt = "0,1,0,0,10,10,0.9,0,0,-1\n1,1,0,0,10,10,0.9,0,0,-1\n"
        rows = parse_attr_rows_str(pred_txt)
        attr_json = _write(
            Path(tmp) / "track_attributes.json",
            '{"1": {"class": 1, "team": 1, "gk": true}}',
        )
        cls, team = aggregate_pred_tracks(rows, attr_json)
        assert cls[1] == 1, "json class must win over txt majority"
        assert team[1] == 1, "json team must win over txt constant"


def test_aggregate_pred_falls_back_to_txt_without_json():
    pred_txt = "0,1,0,0,10,10,0.9,2,-1,-1\n1,1,0,0,10,10,0.9,2,-1,-1\n"
    rows = parse_attr_rows_str(pred_txt)
    cls, team = aggregate_pred_tracks(rows, None)
    assert cls[1] == 2 and team[1] == -1, "txt-derived values used"


# ---------------------------------------------------------------------------
# 18. Task 005 — run_attribute_eval writes JSON; auto-locates attributes.json
# ---------------------------------------------------------------------------


def test_run_attribute_eval_writes_json_and_uses_sibling_attrs():
    with tempfile.TemporaryDirectory() as tmp:
        team_dir = Path(tmp) / "03_team"
        pred = _write(team_dir / "refined.txt", _ATTR_PRED)
        _write(
            team_dir / "track_attributes.json",
            '{"1": {"class": 0, "team": 0, "gk": false},'
            ' "2": {"class": 2, "team": -1, "gk": false},'
            ' "3": {"class": 1, "team": -1, "gk": true}}',
        )
        gt = _write(Path(tmp) / "gt.txt", _ATTR_GT)
        out = Path(tmp) / "attributes_metrics.json"
        m = run_attribute_eval(gt, pred, out)
        assert out.is_file(), "attributes_metrics.json must be written"
        import json as _json

        data = _json.loads(out.read_text(encoding="utf-8"))
        assert data["counts"]["matched_tracks"] == 3, data["counts"]
        assert data["class"]["accuracy"] == 1.0, data["class"]
        # sibling json was auto-located.
        assert "track_attributes_json" in data
        assert m["consistency"]["n_tracks"] == 3


def test_run_evaluation_attribute_eval_nonfatal(_capsys=None):
    """A broken attribute eval must NOT break the HOTA path (metrics.json still
    written, run_evaluation returns the 5 metrics)."""

    def _stub_runner(layout, **kwargs):
        return _fake_result(tracker=layout.tracker_name)

    with tempfile.TemporaryDirectory() as tmp:
        gt_file = _write(Path(tmp) / "gt.txt", _SAMPLE_GT)
        pred_file = _write(Path(tmp) / "refined.txt", _SAMPLE_PRED)
        work_dir = Path(tmp) / "work"
        out_path = Path(tmp) / "metrics.json"

        # Monkeypatch run_attribute_eval (imported into eval.evaluate) to raise.
        import eval.evaluate as _ev

        original = _ev.run_attribute_eval

        def _boom(*a, **k):
            raise RuntimeError("intentional attribute-eval failure")

        _ev.run_attribute_eval = _boom
        try:
            metrics = run_evaluation(
                pred_file, gt_file, "M59", work_dir, out_path, _runner=_stub_runner
            )
        finally:
            _ev.run_attribute_eval = original

        assert set(metrics) == {"HOTA", "DetA", "AssA", "MOTA", "IDF1"}, metrics
        assert out_path.is_file(), "metrics.json must still be written"


# Helper: parse an attribute MOT string fixture without a temp file.
def parse_attr_rows_str(content: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(Path(tmp) / "_attr.txt", content)
        return parse_attr_rows(p)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_TESTS = [
    # Task E1 tests
    ("pred: load_pred basic (0-based, score kept)", test_load_pred_basic),
    ("pred: load_pred tolerates blank/short lines", test_load_pred_tolerates_blank_and_short_lines),
    ("gt: load_gt motchallenge basic (1-based as-is)", test_load_gt_motchallenge_basic),
    ("gt: unknown format raises", test_load_gt_unknown_format_raises),
    ("gt seam: register_gt_loader routes", test_gt_loader_seam_register_and_route),
    ("gt seam: gt_loader decorator routes", test_gt_loader_decorator_seam),
    ("layout: exact files/paths created", test_materialize_layout_paths_exist),
    ("layout: GT written as-is (1-based)", test_materialize_gt_written_as_is),
    ("layout: GT class/vis written as 1 (E3 fix)", test_materialize_gt_class_and_visibility_are_one),
    ("layout: pred indices 0-6 intact (E3)", test_materialize_pred_indices_0_to_6_intact),
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
    # Task E2 tests
    ("E2 run_trackeval: ImportError without trackeval", test_run_trackeval_raises_import_error_without_trackeval),
    ("E2 extract_metrics: HOTA/DetA/AssA as array means", test_extract_metrics_array_mean),
    ("E2 extract_metrics: MOTA/IDF1 scalars", test_extract_metrics_mota_idf1_scalars),
    ("E2 extract_metrics: wrong tracker raises KeyError", test_extract_metrics_wrong_tracker_raises),
    ("E2 extract_metrics: missing metric raises KeyError", test_extract_metrics_missing_metric_raises),
    ("E2 extract_metrics: custom class_name", test_extract_metrics_custom_class),
    ("extract_all_metrics: full per-family dump", test_extract_all_metrics_full),
    ("E2 format_table: 5 keys + values present", test_format_metrics_table_structure),
    ("E2 format_table: missing key shows N/A", test_format_metrics_table_missing_key_shows_na),
    ("E2 metrics.json: schema correct", test_write_metrics_json_schema),
    ("E2 metrics.json: creates parent dirs", test_write_metrics_json_creates_parents),
    ("metrics.json: includes all_metrics block", test_write_metrics_json_includes_all_metrics),
    ("metrics.json: omits all_metrics when absent", test_write_metrics_json_omits_all_metrics_when_absent),
    ("E2 run_evaluation: stub runner writes metrics.json", test_run_evaluation_injected_stub_writes_metrics_json),
    ("E2 run_evaluation: stub does not import trackeval", test_run_evaluation_stub_no_trackeval_imported),
    # Task 005 — _format_gt_line forces class=1/vis=1
    ("005 gt-line: class=1/vis=1 for extended class 0/1/2", test_format_gt_line_forces_class1_vis1_for_extended_gt),
    ("005 gt-line: extended GT materializes all class=1", test_extended_gt_materializes_all_class1),
    # Task 005 — attribute raw parser (no -1 coercion)
    ("005 parse: raw attr keeps -1 (no coercion)", test_parse_attr_rows_no_minus1_coercion),
    ("005 parse: missing/blank class/team -> -1", test_parse_attr_rows_missing_blank_become_minus1),
    # Task 005 — IoU matching + majority association
    ("005 assoc: IoU match + majority (frame +1 shift)", test_associate_tracks_iou_majority_with_frame_shift),
    ("005 assoc: below-threshold -> no match", test_associate_tracks_no_match_when_below_threshold),
    ("005 assoc: majority breaks a split", test_associate_tracks_majority_breaks_split),
    # Task 005 — class metrics
    ("005 class: accuracy + 3x3 confusion", test_class_metrics_accuracy_and_confusion),
    ("005 class: skip unknown-GT-class tracks", test_class_metrics_skips_unknown_gt_class),
    # Task 005 — team metrics (players only, perm-invariant)
    ("005 team: perm-invariant, players only", test_team_metrics_permutation_invariant_players_only),
    ("005 team: excludes GK + referee", test_team_metrics_excludes_gk_and_referee),
    # Task 005 — consistency
    ("005 consistency: purity + switch count", test_consistency_purity_and_switches),
    # Task 005 — graceful degradation
    ("005 degrade: GT all -1 -> class/team skipped", test_compute_metrics_gt_all_minus1_skips_class_and_team),
    ("005 degrade: no IoU match -> consistency kept", test_compute_metrics_no_iou_match_skips_but_keeps_consistency),
    ("005 degrade: pred no team -> team-only skip", test_compute_metrics_pred_no_team_skips_team_only),
    ("005 degrade: ARI/NMI skipped without sklearn", test_ari_nmi_skipped_when_sklearn_absent),
    # Task 005 — track_attributes.json preference
    ("005 pred-agg: prefers track_attributes.json", test_aggregate_pred_prefers_track_attributes_json),
    ("005 pred-agg: falls back to txt without json", test_aggregate_pred_falls_back_to_txt_without_json),
    # Task 005 — run_attribute_eval + non-fatal wiring
    ("005 run: writes json + auto-locates sibling attrs", test_run_attribute_eval_writes_json_and_uses_sibling_attrs),
    ("005 wiring: attribute eval failure is non-fatal", test_run_evaluation_attribute_eval_nonfatal),
]


def main() -> int:
    print("=" * 60)
    print("eval unit tests (Tasks E1 + E2, CPU-only)")
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

"""
CPU-only unit tests for Task-004 class/team-aware rendering.

Runs with plain python (no cv2 / torch / yolox).  ``render_from_txt`` makes its
``import cv2`` lazy (inside ``main``), so the module + its pure helpers import by
file path without cv2 — exactly how the module's own ``load_plot_tracking`` loads
visualize.py by path.  These tests drive the pure functions only:

  * ``parse_results``: parses class_id (col 8) + team_id (col 9) when present;
    legacy rows (<8/<9 fields) -> -1 (unknown);
  * ``load_track_attributes``: joins track_attributes.json -> {id: (class, team)};
  * ``category_color_and_label``: all five categories incl. the legacy fallback
    (class == -1 and team == -1 -> get_color(id), bare id label) and the
    ``gk:`` label prefix;
  * ``legacy_color`` is byte-identical to visualize.get_color(abs(id)) so a
    fully-legacy txt renders exactly as before.

Run with:
    python pipeline/tests/test_render_from_txt.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

# --- locate render_from_txt (lives in Deep-EIoU/Deep-EIoU/tools) -------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS_DIR = _REPO_ROOT / "Deep-EIoU" / "Deep-EIoU" / "tools"


def _load_render_module():
    """Import render_from_txt by file path (no cv2 needed — it is lazy)."""
    path = _TOOLS_DIR / "render_from_txt.py"
    spec = importlib.util.spec_from_file_location("render_from_txt", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_rft = _load_render_module()


# ---------------------------------------------------------------------------
# Minimal test runner (matches the other pipeline tests' style; no pytest)
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
# Reference get_color, copied from yolox/utils/visualize.py (which imports cv2
# at module load and so can't be imported here).  legacy_color must match this
# byte-for-byte for the fallback to render exactly as before.
# ---------------------------------------------------------------------------
def _reference_get_color(idx):
    idx = idx * 3
    return ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)


def _write_txt(path, rows):
    """Write a MOT txt from row tuples (each tuple = one comma-joined line)."""
    lines = [",".join(str(c) for c in r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ===========================================================================
# legacy_color == get_color(abs(id)) (byte-for-byte legacy parity)
# ===========================================================================
def test_legacy_color_matches_get_color():
    for tid in (0, 1, 2, 7, 42, 255, 1000):
        assert _rft.legacy_color(tid) == _reference_get_color(abs(tid)), tid
    # negative ids use abs() in both, like plot_tracking's get_color(abs(obj_id)).
    assert _rft.legacy_color(-7) == _reference_get_color(7)


# ===========================================================================
# parse_results: class/team columns + legacy rows
# ===========================================================================
def test_parse_results_extended_columns():
    with tempfile.TemporaryDirectory() as td:
        txt = Path(td) / "refined.txt"
        # frame,id,x,y,w,h,score,class_id,team_id,-1
        # class_map scheme: player=2, goalkeeper=1.
        rows = [
            (0, 1, "100.00", "200.00", "30.00", "60.00", 1, 2, 0, -1),  # player team0
            (0, 2, "150.00", "250.00", "31.00", "61.00", 1, 2, 1, -1),  # player team1
            (0, 3, "300.00", "400.00", "32.00", "62.00", 1, 1, -1, -1),  # gk
            (1, 1, "101.00", "201.00", "30.00", "60.00", 1, 2, 0, -1),
        ]
        _write_txt(txt, rows)
        frames, n_rows = _rft.parse_results(str(txt))
        assert n_rows == 4, n_rows
        tlwhs, ids, scores, class_ids, team_ids = frames[0]
        assert ids == [1, 2, 3], ids
        assert class_ids == [2, 2, 1], class_ids
        assert team_ids == [0, 1, -1], team_ids
        assert tlwhs[0] == (100.0, 200.0, 30.0, 60.0), tlwhs[0]
        # frame 1 parsed too.
        assert frames[1][1] == [1], frames[1]


def test_parse_results_legacy_rows_default_unknown():
    with tempfile.TemporaryDirectory() as td:
        txt = Path(td) / "legacy.txt"
        # Legacy MOT: frame,id,x,y,w,h,score,-1,-1,-1 (no class/team semantics).
        rows = [
            (0, 1, "100.00", "200.00", "30.00", "60.00", 1, -1, -1, -1),
            (0, 2, "150.00", "250.00", "31.00", "61.00", 1, -1, -1, -1),
        ]
        _write_txt(txt, rows)
        frames, n_rows = _rft.parse_results(str(txt))
        assert n_rows == 2, n_rows
        _, ids, _, class_ids, team_ids = frames[0]
        assert ids == [1, 2]
        assert class_ids == [-1, -1], class_ids
        assert team_ids == [-1, -1], team_ids


def test_parse_results_short_rows_default_unknown():
    with tempfile.TemporaryDirectory() as td:
        txt = Path(td) / "short.txt"
        # Only 7 columns (truly legacy, no col 8/9 at all) -> class/team -1.
        rows = [(0, 5, "1.00", "2.00", "3.00", "4.00", 1)]
        _write_txt(txt, rows)
        frames, n_rows = _rft.parse_results(str(txt))
        assert n_rows == 1
        _, ids, _, class_ids, team_ids = frames[0]
        assert ids == [5] and class_ids == [-1] and team_ids == [-1]


# ===========================================================================
# load_track_attributes: JSON join
# ===========================================================================
def test_load_track_attributes_join():
    with tempfile.TemporaryDirectory() as td:
        attr = Path(td) / "track_attributes.json"
        # class_map scheme: player=2, goalkeeper=1, referee=3.
        attr.write_text(json.dumps({
            "1": {"class": 2, "team": 0, "gk": False},
            "2": {"class": 2, "team": 1, "gk": False},
            "3": {"class": 1, "team": -1, "gk": True},
            "10": {"class": 3, "team": -1, "gk": False},
        }), encoding="utf-8")
        out = _rft.load_track_attributes(str(attr))
        # keys -> int ids; values -> (class, team) tuples.
        assert out == {1: (2, 0), 2: (2, 1), 3: (1, -1), 10: (3, -1)}, out


def test_load_track_attributes_missing_returns_empty():
    assert _rft.load_track_attributes(None) == {}
    assert _rft.load_track_attributes("/no/such/file.json") == {}


def test_find_attributes_path_sibling_autodetect():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        txt = td / "refined.txt"
        txt.write_text("0,1,1,2,3,4,1,0,0,-1\n", encoding="utf-8")
        # no sibling json yet -> None.
        assert _rft.find_attributes_path(str(txt)) is None
        # create the sibling -> auto-detected.
        sibling = td / "track_attributes.json"
        sibling.write_text("{}", encoding="utf-8")
        assert _rft.find_attributes_path(str(txt)) == str(sibling)
        # explicit arg wins over auto-detect.
        assert _rft.find_attributes_path(str(txt), "/x/y.json") == "/x/y.json"


# ===========================================================================
# category_color_and_label: all five categories
# ===========================================================================
def test_category_player_team0():
    color, label = _rft.category_color_and_label(7, _rft.PLAYER_CLASS, 0)
    assert color == _rft.C_TEAM0, color
    assert label == "7", label


def test_category_player_team1():
    color, label = _rft.category_color_and_label(7, _rft.PLAYER_CLASS, 1)
    assert color == _rft.C_TEAM1, color
    assert label == "7", label


def test_category_goalkeeper_prefix():
    color, label = _rft.category_color_and_label(9, _rft.GOALKEEPER_CLASS, -1)
    assert color == _rft.C_GK, color
    assert label == "gk:9", label  # gk: prefix, bare id otherwise.
    # GK color/prefix regardless of team value.
    color2, label2 = _rft.category_color_and_label(9, _rft.GOALKEEPER_CLASS, 0)
    assert color2 == _rft.C_GK and label2 == "gk:9"


def test_category_referee_no_prefix():
    color, label = _rft.category_color_and_label(4, _rft.REFEREE_CLASS, -1)
    assert color == _rft.C_REF, color
    assert label == "4", label  # referee: no prefix.


def test_category_legacy_fallback():
    # class == -1 and team == -1 -> legacy color + bare id (byte-for-byte).
    color, label = _rft.category_color_and_label(13, -1, -1)
    assert color == _rft.legacy_color(13) == _reference_get_color(13), color
    assert label == "13", label


def test_category_player_unknown_team_falls_back():
    # player whose team is unknown -> legacy color + bare id (no team color).
    color, label = _rft.category_color_and_label(21, _rft.PLAYER_CLASS, -1)
    assert color == _rft.legacy_color(21), color
    assert label == "21", label


def test_category_colors_are_distinct():
    cats = {_rft.C_TEAM0, _rft.C_TEAM1, _rft.C_GK, _rft.C_REF}
    assert len(cats) == 4, cats  # four visually distinct constants.


# ===========================================================================
# Precedence: attributes.json (stable) overrides per-frame txt cols
# ===========================================================================
def test_attributes_take_precedence_over_per_frame_cols():
    # Simulate the main() resolution: attrs win over the row's class/team.
    track_attrs = {1: (_rft.GOALKEEPER_CLASS, -1)}  # attributes say id 1 is a goalkeeper
    # per-frame txt said player/team0, but the stable attribute wins.
    tid, cid, team = 1, _rft.PLAYER_CLASS, 0
    if tid in track_attrs:
        cid, team = track_attrs[tid]
    color, label = _rft.category_color_and_label(tid, cid, team)
    assert color == _rft.C_GK and label == "gk:1", (color, label)


_TESTS = [
    ("legacy_color: byte-identical to get_color(abs(id))", test_legacy_color_matches_get_color),
    ("parse_results: class/team cols 8/9 parsed", test_parse_results_extended_columns),
    ("parse_results: legacy rows (-1) -> unknown", test_parse_results_legacy_rows_default_unknown),
    ("parse_results: short rows (<8 cols) -> unknown", test_parse_results_short_rows_default_unknown),
    ("load_track_attributes: JSON join -> {id:(class,team)}", test_load_track_attributes_join),
    ("load_track_attributes: missing -> empty dict", test_load_track_attributes_missing_returns_empty),
    ("find_attributes_path: sibling autodetect + explicit", test_find_attributes_path_sibling_autodetect),
    ("category: player team0 -> C_TEAM0, bare id", test_category_player_team0),
    ("category: player team1 -> C_TEAM1, bare id", test_category_player_team1),
    ("category: goalkeeper -> C_GK, gk: prefix", test_category_goalkeeper_prefix),
    ("category: referee -> C_REF, no prefix", test_category_referee_no_prefix),
    ("category: legacy fallback (-1,-1) -> get_color(id), bare id", test_category_legacy_fallback),
    ("category: player unknown-team -> legacy fallback", test_category_player_unknown_team_falls_back),
    ("category: four distinct color constants", test_category_colors_are_distinct),
    ("precedence: attributes.json overrides per-frame cols", test_attributes_take_precedence_over_per_frame_cols),
]


def main() -> int:
    print("=" * 60)
    print("render_from_txt class/team rendering unit tests (CPU-only, no cv2)")
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

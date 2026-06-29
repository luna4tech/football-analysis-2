"""
CPU-only unit tests for Stage-3 team assignment.

Runs with plain python (numpy only) — NO sklearn / torch.  ``team_assignment``
imports CPU-light because the only heavy dep (sklearn KMeans) is lazy-imported
inside ``_kmeans_cluster``; these tests NEVER call the real clusterer.  Instead
they inject a deterministic FAKE cluster function (mirroring stage2_refine's
``RefineDeps`` injection) and drive the surrounding logic:

  * confidence-weighted class mode (+ unweighted fallback);
  * player-only gating (GK / referee never enter clustering, team = -1);
  * deterministic team naming (larger cluster -> 0, tie-break lowest min id);
  * < 2 player tracks -> all team -1 but both outputs still written;
  * ``03_team/refined.txt`` cols 1-8 byte-identical to the input, only col 9
    rewritten by the row's track team;
  * ``track_attributes.json`` schema + one entry per id.

Fixtures use the real (light) ``gta-link/Tracklet.py`` class directly.

Run with:
    python pipeline/tests/test_team_assignment.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

# Repo root + gta-link on sys.path so `import pipeline.*` and `import Tracklet`
# both resolve (team_assignment also does this itself; done here for the fixture
# Tracklet import below).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_GTA_LINK_DIR = _REPO_ROOT / "gta-link"
for _p in (_REPO_ROOT, _GTA_LINK_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pipeline import team_assignment as ta
import Tracklet  # noqa: E402, F401 — registers sys.modules["Tracklet"] (light)


def _tracklet_cls():
    """The Tracklet class CURRENTLY registered as the top-level ``Tracklet``.

    Resolved lazily (not captured at import) because other tests
    (``test_stage1_assembly``) re-load ``gta-link/Tracklet.py`` by file path and
    overwrite ``sys.modules["Tracklet"]``.  Pickling resolves the class by its
    module name, so a fixture built from a stale class object would fail with
    "not the same object as Tracklet.Tracklet" once the suite has swapped the
    module.  Fetching it from ``sys.modules`` at build time keeps dump/load on
    the one class object pickle resolves to, regardless of test ordering.
    """
    return sys.modules["Tracklet"].Tracklet


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
# Fixtures
# ---------------------------------------------------------------------------
def _mk_track(tid, n, class_ids, feat, scores=None):
    """Build a real Tracklet with ``n`` frames sharing one feature vector.

    ``class_ids`` may be a single id (broadcast over n frames) or a list.
    ``feat`` is the per-frame feature vector (same for every frame so the mean
    embedding equals it after L2-norm).
    """
    frames = list(range(n))
    if not isinstance(class_ids, list):
        class_ids = [class_ids] * n
    if scores is None:
        scores = [1.0] * n
    feats = [np.asarray(feat, dtype=np.float64) for _ in range(n)]
    bboxes = [[float(f), float(f), 10.0, 20.0] for f in frames]
    return _tracklet_cls()(tid, frames, scores, bboxes, feats=feats, class_ids=class_ids)


# A fake cluster function: assigns labels by the SIGN of the first embedding
# component (deterministic, no sklearn).  >= 0 -> label 0, < 0 -> label 1.
def _fake_cluster_by_sign(embeddings, k, **kwargs):
    return [0 if np.asarray(e).ravel()[0] >= 0 else 1 for e in embeddings]


_POS = [1.0, 0.0, 0.0, 0.0]   # cluster A direction (first comp >= 0)
_NEG = [-1.0, 0.0, 0.0, 0.0]  # cluster B direction (first comp < 0)


# ===========================================================================
# Confidence-weighted class aggregation
# ===========================================================================
def test_aggregate_class_confidence_weighted():
    # Two frames player (low conf) vs one frame goalkeeper (high conf): weighted
    # mode is the goalkeeper even though player has more frames.
    P, G = ta.PLAYER_CLASS, ta.GOALKEEPER_CLASS  # 2, 1
    assert ta.aggregate_class([P, P, G], [0.1, 0.1, 0.9]) == G


def test_aggregate_class_unweighted_fallback():
    P, G = ta.PLAYER_CLASS, ta.GOALKEEPER_CLASS  # 2, 1
    # scores absent -> unweighted count: player wins (2 vs 1).
    assert ta.aggregate_class([P, P, G], None) == P
    # scores too SHORT -> unweighted fallback (never silently down-weight).
    assert ta.aggregate_class([P, P, G], [0.9]) == P


def test_aggregate_class_empty_is_unknown():
    assert ta.aggregate_class([], None) == -1


def test_aggregate_class_tie_breaks_lowest_id():
    # Equal weight on goalkeeper (1) and referee (3) -> lowest id (1) wins
    # deterministically.
    G, R = ta.GOALKEEPER_CLASS, ta.REFEREE_CLASS  # 1, 3
    assert ta.aggregate_class([G, R], [0.5, 0.5]) == G


# ===========================================================================
# Player-only gating + GK/referee -> team -1
# ===========================================================================
def test_gk_and_referee_get_team_minus_one():
    tracklets = {
        1: _mk_track(1, 5, ta.PLAYER_CLASS, _POS),
        2: _mk_track(2, 5, ta.PLAYER_CLASS, _NEG),
        3: _mk_track(3, 5, ta.GOALKEEPER_CLASS, _POS),
        4: _mk_track(4, 5, ta.REFEREE_CLASS, _NEG),
    }
    attrs = ta.assign_teams(tracklets, cluster_fn=_fake_cluster_by_sign)
    # GK + referee never clustered -> team -1; gk flag set only for the GK.
    assert attrs[3]["team"] == -1 and attrs[3]["gk"] is True, attrs[3]
    assert attrs[4]["team"] == -1 and attrs[4]["gk"] is False, attrs[4]
    # players got real teams (0/1).
    assert attrs[1]["team"] in (0, 1) and attrs[2]["team"] in (0, 1), attrs


def test_only_players_enter_clustering():
    # 1 player + 1 GK + 1 ref => only ONE player => < 2 players => all team -1.
    tracklets = {
        1: _mk_track(1, 5, ta.PLAYER_CLASS, _POS),
        2: _mk_track(2, 5, ta.GOALKEEPER_CLASS, _NEG),
        3: _mk_track(3, 5, ta.REFEREE_CLASS, _POS),
    }
    attrs = ta.assign_teams(tracklets, cluster_fn=_fake_cluster_by_sign)
    assert all(a["team"] == -1 for a in attrs.values()), attrs


# ===========================================================================
# Deterministic team naming (larger cluster -> 0; tie-break lowest min id)
# ===========================================================================
def test_larger_cluster_becomes_team_zero():
    # 3 players in the +cluster, 1 in the -cluster -> larger cluster (3) = team 0.
    tracklets = {
        1: _mk_track(1, 5, ta.PLAYER_CLASS, _POS),
        2: _mk_track(2, 5, ta.PLAYER_CLASS, _POS),
        3: _mk_track(3, 5, ta.PLAYER_CLASS, _POS),
        4: _mk_track(4, 5, ta.PLAYER_CLASS, _NEG),
    }
    attrs = ta.assign_teams(tracklets, cluster_fn=_fake_cluster_by_sign)
    assert attrs[1]["team"] == 0 and attrs[2]["team"] == 0 and attrs[3]["team"] == 0, attrs
    assert attrs[4]["team"] == 1, attrs


def test_team_naming_tie_breaks_on_lowest_min_id():
    # Equal-size clusters {2,3} (+) and {1,4} (-): the cluster with the lowest
    # min track id (1, the -cluster) wins team 0.
    tracklets = {
        1: _mk_track(1, 5, ta.PLAYER_CLASS, _NEG),
        2: _mk_track(2, 5, ta.PLAYER_CLASS, _POS),
        3: _mk_track(3, 5, ta.PLAYER_CLASS, _POS),
        4: _mk_track(4, 5, ta.PLAYER_CLASS, _NEG),
    }
    attrs = ta.assign_teams(tracklets, cluster_fn=_fake_cluster_by_sign)
    # {1,4} has min id 1 < {2,3}'s min id 2 -> {1,4} = team 0.
    assert attrs[1]["team"] == 0 and attrs[4]["team"] == 0, attrs
    assert attrs[2]["team"] == 1 and attrs[3]["team"] == 1, attrs


def test_name_teams_helper_directly():
    # raw labels: id7->A, id3->A, id9->B  => cluster A={7,3} (size2),
    # B={9} (size1).  Larger -> 0.
    team_of = ta._name_teams([7, 3, 9], [0, 0, 1])
    assert team_of == {7: 0, 3: 0, 9: 1}, team_of


# ===========================================================================
# < 2 player tracks -> all -1 but both outputs still written
# ===========================================================================
def test_fewer_than_two_players_all_minus_one():
    tracklets = {
        1: _mk_track(1, 5, ta.PLAYER_CLASS, _POS),  # the only player
        2: _mk_track(2, 5, ta.GOALKEEPER_CLASS, _NEG),
    }
    attrs = ta.assign_teams(tracklets, cluster_fn=_fake_cluster_by_sign)
    assert all(a["team"] == -1 for a in attrs.values()), attrs


# ===========================================================================
# refined.txt rewrite: cols 1-8 identical, ONLY col 9 changes
# ===========================================================================
def _write_input_refined(path, rows):
    """Write a MOT refined.txt (10 cols/row) from row tuples."""
    lines = []
    for r in rows:
        lines.append(",".join(str(c) for c in r))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_team_txt_only_col9_changes():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        refined_txt = td / "refined.txt"
        team_txt = td / "team_refined.txt"
        # frame,id,x,y,w,h,score,class,team(-1 placeholder),-1
        # class_map scheme: player=2, goalkeeper=1.
        rows = [
            (0, 1, "100.00", "200.00", "30.00", "60.00", 1, 2, -1, -1),
            (0, 2, "150.00", "250.00", "31.00", "61.00", 1, 2, -1, -1),
            (1, 1, "101.00", "201.00", "30.00", "60.00", 1, 2, -1, -1),
            (1, 3, "300.00", "400.00", "32.00", "62.00", 1, 1, -1, -1),  # a GK
        ]
        _write_input_refined(refined_txt, rows)

        attributes = {
            1: {"class": 2, "team": 0, "gk": False},
            2: {"class": 2, "team": 1, "gk": False},
            3: {"class": 1, "team": -1, "gk": True},
        }
        n = ta.write_team_txt(str(refined_txt), str(team_txt), attributes)
        assert n == 4, n

        in_lines = refined_txt.read_text(encoding="utf-8").splitlines()
        out_lines = team_txt.read_text(encoding="utf-8").splitlines()
        assert len(in_lines) == len(out_lines) == 4

        for in_line, out_line in zip(in_lines, out_lines):
            ic = in_line.split(",")
            oc = out_line.split(",")
            tid = int(ic[1])
            # cols 1-8 (idx 0..7) and col 10 (idx 9) byte-identical.
            assert ic[:8] == oc[:8], (in_line, out_line)
            assert ic[9] == oc[9], (in_line, out_line)
            # col 9 (idx 8) == this track's team id.
            assert oc[8] == str(attributes[tid]["team"]), (out_line, attributes[tid])

        # Specifically: rows for id 1 -> team 0, id 2 -> 1, GK id 3 -> -1.
        assert out_lines[0].split(",")[8] == "0"
        assert out_lines[1].split(",")[8] == "1"
        assert out_lines[3].split(",")[8] == "-1"


# ===========================================================================
# track_attributes.json schema + one entry per id
# ===========================================================================
def test_track_attributes_json_schema():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "track_attributes.json"
        # class_map scheme: player=2, goalkeeper=1, referee=3.
        attributes = {
            1: {"class": 2, "team": 0, "gk": False},
            2: {"class": 1, "team": -1, "gk": True},
            10: {"class": 3, "team": -1, "gk": False},
        }
        ta.write_track_attributes(str(path), attributes)
        data = json.loads(path.read_text(encoding="utf-8"))
        # one entry per id; keys are STRINGS.
        assert set(data.keys()) == {"1", "2", "10"}, data.keys()
        for key, entry in data.items():
            assert set(entry.keys()) == {"class", "team", "gk"}, entry
            assert isinstance(entry["class"], int), entry
            assert isinstance(entry["team"], int), entry
            assert isinstance(entry["gk"], bool), entry
        assert data["2"]["gk"] is True and data["2"]["team"] == -1, data["2"]


# ===========================================================================
# End-to-end run_team_assignment with the fake cluster fn (no sklearn)
# ===========================================================================
def test_run_team_assignment_end_to_end():
    import pickle

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        refined_pkl = td / "refined_tracklets.pkl"
        refined_txt = td / "refined.txt"
        team_txt = td / "03_refined.txt"
        attrs_json = td / "track_attributes.json"

        tracklets = {
            1: _mk_track(1, 3, ta.PLAYER_CLASS, _POS),
            2: _mk_track(2, 3, ta.PLAYER_CLASS, _POS),
            3: _mk_track(3, 3, ta.PLAYER_CLASS, _NEG),
            4: _mk_track(4, 3, ta.GOALKEEPER_CLASS, _POS),
        }
        with open(refined_pkl, "wb") as f:
            pickle.dump(tracklets, f)

        # Build a refined.txt whose ids == pkl ids (one row per track frame).
        rows = []
        for tid, trk in tracklets.items():
            for f in trk.times:
                rows.append((f, tid, "1.00", "2.00", "3.00", "4.00", 1, trk.class_ids[0], -1, -1))
        _write_input_refined(refined_txt, rows)

        counts = ta.run_team_assignment(
            str(refined_pkl), str(refined_txt), str(team_txt), str(attrs_json),
            cluster_fn=_fake_cluster_by_sign,
        )

        # both outputs written.
        assert team_txt.is_file() and attrs_json.is_file()
        # counts sanity: 3 players, 1 GK.
        assert counts["n_tracks"] == 4 and counts["n_players"] == 3, counts
        # larger +cluster {1,2} -> team 0; {3} -> team 1; GK -> -1.
        data = json.loads(attrs_json.read_text(encoding="utf-8"))
        assert data["1"]["team"] == 0 and data["2"]["team"] == 0, data
        assert data["3"]["team"] == 1, data
        assert data["4"]["team"] == -1 and data["4"]["gk"] is True, data

        # team txt: every row's col 9 matches its track's team.
        for line in team_txt.read_text(encoding="utf-8").splitlines():
            cols = line.split(",")
            tid = int(cols[1])
            assert cols[8] == str(data[str(tid)]["team"]), line


# ===========================================================================
# Optional: real sklearn k-means clusters separable embeddings (skips w/o dep)
# ===========================================================================
def _sklearn_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("sklearn") is not None


def test_optional_real_kmeans_separates():
    if not _sklearn_available():
        print("    SKIP (sklearn unavailable: real k-means cannot run)")
        return
    # Two well-separated unit directions -> k-means must split them.
    embeddings = [_POS, _POS, _NEG, _NEG]
    labels = ta._kmeans_cluster(embeddings, 2)
    assert len(labels) == 4, labels
    # the two _POS share a label, the two _NEG share the other label.
    assert labels[0] == labels[1] and labels[2] == labels[3], labels
    assert labels[0] != labels[2], labels


_TESTS = [
    ("aggregate_class: confidence-weighted mode", test_aggregate_class_confidence_weighted),
    ("aggregate_class: unweighted fallback (absent/short scores)", test_aggregate_class_unweighted_fallback),
    ("aggregate_class: empty -> unknown -1", test_aggregate_class_empty_is_unknown),
    ("aggregate_class: tie-breaks on lowest class id", test_aggregate_class_tie_breaks_lowest_id),
    ("gating: GK/referee -> team -1", test_gk_and_referee_get_team_minus_one),
    ("gating: only players cluster (1 player -> all -1)", test_only_players_enter_clustering),
    ("naming: larger cluster -> team 0", test_larger_cluster_becomes_team_zero),
    ("naming: tie-break lowest min track id", test_team_naming_tie_breaks_on_lowest_min_id),
    ("naming: _name_teams helper", test_name_teams_helper_directly),
    ("gating: < 2 players -> all team -1", test_fewer_than_two_players_all_minus_one),
    ("refined.txt: only col 9 changes (cols 1-8 identical)", test_team_txt_only_col9_changes),
    ("track_attributes.json: schema + one entry per id", test_track_attributes_json_schema),
    ("run_team_assignment: end-to-end with fake cluster fn", test_run_team_assignment_end_to_end),
    ("optional: real sklearn k-means separates (skips w/o dep)", test_optional_real_kmeans_separates),
]


def main() -> int:
    print("=" * 60)
    print("stage3 team assignment unit tests (CPU-only, fake cluster fn)")
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

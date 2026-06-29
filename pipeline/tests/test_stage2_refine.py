"""
CPU-only unit tests for Stage-2's refine orchestration.

Runs with plain python (numpy only) — NO torch / sklearn / scipy / matplotlib /
seaborn / loguru.  The refine ALGORITHM functions (split / spatial) need those
heavy deps, so this test does NOT import ``refine_tracklets``; instead it drives
``stage2_refine.run_refine`` with INJECTED stub callables (a ``RefineDeps``
bundle) plus the real, numpy-only ``_fast_connect`` connect step, and asserts:

  * the full flow runs split THEN connect (fast) THEN save;
  * spatial constraints are computed from the ORIGINAL tracklets + factor;
  * split params are forwarded by name;
  * the save target is the contract ``refined_txt`` path;
  * ``_fast_connect`` merges/keeps tracklets correctly (feature distance,
    temporal overlap, spatial gate).

``stage2_refine`` imports cleanly here because it only imports ``pipeline.*``
(numpy/stdlib) at module level; ``refine_tracklets`` is imported lazily inside
``_build_deps`` / ``main`` (never reached by these tests, except the optional
end-to-end one which SKIPS gracefully when the heavy deps are missing).

Run with:
    python pipeline/tests/test_stage2_refine.py
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

# --- locate the stage2_refine module (lives in gta-link/) -------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_GTA_LINK_DIR = _REPO_ROOT / "gta-link"


def _load_stage2():
    """Import stage2_refine by file path.

    With CWD untouched, the module's top-level imports are only ``pipeline.*``
    (it inserts the repo root onto sys.path itself) and stdlib/numpy — so this
    succeeds on a CPU-only box.  ``refine_tracklets`` is imported lazily inside
    the module and is NOT triggered here.
    """
    path = _GTA_LINK_DIR / "stage2_refine.py"
    spec = importlib.util.spec_from_file_location("stage2_refine", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_s2 = _load_stage2()
run_refine = _s2.run_refine
RefineDeps = _s2.RefineDeps

# Stage 3 + the real Tracklet class for the OPT-1 slim-pkl equivalence tests.
# stage2_refine put the repo root on sys.path at import; add gta-link so the
# real `Tracklet` (used for a faithful pickle round-trip) resolves as the
# top-level module name the pickle references.
if str(_GTA_LINK_DIR) not in sys.path:
    sys.path.insert(0, str(_GTA_LINK_DIR))
from pipeline import team_assignment as _ta  # noqa: E402
import Tracklet  # noqa: E402, F401 — registers sys.modules["Tracklet"] (light)


def _Tracklet(*args, **kwargs):
    """Construct a Tracklet from the class CURRENTLY registered as ``Tracklet``.

    Other tests (e.g. ``test_stage1_assembly``) re-load ``gta-link/Tracklet.py``
    by file path and overwrite ``sys.modules["Tracklet"]`` with a fresh module
    object.  Pickling resolves the class by its module name, so a fixture built
    from a stale class object would fail with "not the same object as
    Tracklet.Tracklet" once the suite has swapped the module.  Resolving the
    class lazily from ``sys.modules`` at build time keeps dump/load using the
    one class object pickle will resolve to, regardless of test ordering.
    """
    return sys.modules["Tracklet"].Tracklet(*args, **kwargs)


# ---------------------------------------------------------------------------
# Minimal test runner (no pytest)
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
# Minimal Tracklet stand-in for _fast_connect (numpy-only)
# ---------------------------------------------------------------------------
class _FakeTrack:
    def __init__(self, track_id, times, feats, bboxes=None, class_ids=None):
        self.track_id = track_id
        self.parent_id = track_id
        self.times = list(times)
        self.features = [np.asarray(f, dtype=np.float64) for f in feats]
        self.bboxes = (
            list(bboxes) if bboxes is not None
            else [[0.0, 0.0, 1.0, 1.0] for _ in times]
        )
        self.scores = [1.0 for _ in times]
        # per-frame class ids, aligned 1:1 with times (default unknown -1).
        self.class_ids = list(class_ids) if class_ids is not None else [-1 for _ in times]


_V0 = [1.0, 0.0, 0.0, 0.0]   # one unit direction
_V1 = [0.0, 1.0, 0.0, 0.0]   # an orthogonal direction (cosine distance 1.0)


def _params(**overrides):
    p = {
        "min_len": 100,
        "eps": 0.6,
        "min_samples": 10,
        "max_k": 3,
        "spatial_factor": 1.0,
        "merge_dist_thres": 0.4,
    }
    p.update(overrides)
    return p


# ---------------------------------------------------------------------------
# Recorder stubs for the injected refine callables
# ---------------------------------------------------------------------------
class _Recorder:
    """Records the orchestration steps + their inputs.

    ``split_tracklets`` returns a caller-supplied dict of real ``_FakeTrack``
    objects so the real, numpy-only ``_fast_connect`` connect step can run over
    them.  ``check_spatial_constraints`` is the gate ``_fast_connect`` calls.
    """

    def __init__(self, split_output, spatial_ok=True):
        self.calls = []
        self.spatial_arg = None
        self.split_input = None
        self.split_params = None
        self.split_output = split_output
        self.spatial_ok = spatial_ok
        self.saved_path = None
        self.saved_obj = None

    def get_spatial_constraints(self, tid2track, factor):
        self.calls.append("spatial")
        self.spatial_arg = (tid2track, factor)
        return (1e9, 1e9)

    def split_tracklets(self, tmp, eps=None, max_k=None, min_samples=None, len_thres=None):
        self.calls.append("split")
        self.split_input = tmp
        self.split_params = dict(eps=eps, max_k=max_k, min_samples=min_samples, len_thres=len_thres)
        return self.split_output

    def check_spatial_constraints(self, trk1, trk2, max_x_range, max_y_range):
        self.calls.append("spatial_check")
        return self.spatial_ok

    def save_results(self, out_path, tracklets):
        self.calls.append("save")
        self.saved_path = out_path
        self.saved_obj = tracklets


def _deps_from(rec: _Recorder) -> "RefineDeps":
    return RefineDeps(
        get_spatial_constraints=rec.get_spatial_constraints,
        split_tracklets=rec.split_tracklets,
        check_spatial_constraints=rec.check_spatial_constraints,
        save_results=rec.save_results,
    )


_REFINED_TXT = "/contract/path/02_refine/refined.txt"


# ===========================================================================
# Orchestration tests (split -> fast connect -> save)
# ===========================================================================
def test_full_flow_split_then_connect_then_save():
    # split returns two identical non-overlapping tracklets -> connect merges to 1.
    split_out = {
        1: _FakeTrack(1, range(0, 5), [_V0] * 5),
        2: _FakeTrack(2, range(10, 15), [_V0] * 5),
    }
    tmp = {1: object(), 2: object(), 3: object()}  # original (only len/identity used)
    rec = _Recorder(split_output=split_out, spatial_ok=True)
    counts = run_refine(tmp, _REFINED_TXT, _params(), _deps_from(rec), seq_name="clip")

    # Order: spatial first, split second, save last; the connect step invokes the
    # spatial gate in between.
    assert rec.calls[0] == "spatial", rec.calls
    assert rec.calls[1] == "split", rec.calls
    assert rec.calls[-1] == "save", rec.calls
    assert "spatial_check" in rec.calls, rec.calls
    # split feeds connect: the ORIGINAL tmp is what split saw.
    assert rec.split_input is tmp, "split must run on the original tracklets"
    # save receives the MERGED result (one tracklet).
    assert rec.saved_path == _REFINED_TXT, rec.saved_path
    assert len(rec.saved_obj) == 1, rec.saved_obj
    assert counts["n_tracklets_in"] == 3, counts
    assert counts["n_tracklets_after_split"] == 2, counts
    assert counts["n_tracklets_out"] == 1, counts


def test_spatial_uses_original_and_factor():
    split_out = {1: _FakeTrack(1, range(0, 5), [_V0] * 5)}
    tmp = {1: object(), 2: object()}
    rec = _Recorder(split_output=split_out)
    run_refine(tmp, _REFINED_TXT, _params(spatial_factor=2.5), _deps_from(rec), seq_name="clip")
    assert rec.spatial_arg[0] is tmp, "spatial uses ORIGINAL tracklets"
    assert rec.spatial_arg[1] == 2.5, rec.spatial_arg


def test_split_params_forwarded():
    split_out = {1: _FakeTrack(1, range(0, 5), [_V0] * 5)}
    rec = _Recorder(split_output=split_out)
    p = _params(eps=0.42, max_k=7, min_samples=4, min_len=33)
    run_refine({1: object()}, _REFINED_TXT, p, _deps_from(rec), seq_name="clip")
    assert rec.split_params == dict(eps=0.42, max_k=7, min_samples=4, len_thres=33), rec.split_params


def test_save_target_is_contract_path():
    split_out = {1: _FakeTrack(1, range(0, 5), [_V0] * 5)}
    rec = _Recorder(split_output=split_out)
    run_refine({1: object()}, _REFINED_TXT, _params(), _deps_from(rec), seq_name="clip")
    assert rec.saved_path == _REFINED_TXT, rec.saved_path


# ===========================================================================
# Fast connect/merge (_fast_connect) — CPU-only, numpy + stubbed spatial gate
# ===========================================================================
def _fast_deps(spatial_ok=True):
    """Deps bundle for direct _fast_connect calls: only check_spatial_constraints
    is used; the other callables are simple no-ops."""
    return RefineDeps(
        get_spatial_constraints=lambda t, f: (1e9, 1e9),
        split_tracklets=lambda *a, **k: None,
        check_spatial_constraints=lambda a, b, mx, my: spatial_ok,
        save_results=lambda p, o: None,
    )


def test_fast_merges_identical_nonoverlapping():
    # Two non-overlapping tracklets with identical feats -> distance ~0 -> merge.
    trks = {
        1: _FakeTrack(1, range(0, 5), [_V0] * 5, class_ids=[0] * 5),
        2: _FakeTrack(2, range(10, 15), [_V0] * 5, class_ids=[1] * 5),
    }
    out = _s2._fast_connect(trks, _fast_deps(spatial_ok=True),
                            max_x_range=1e9, max_y_range=1e9, merge_dist_thres=0.4)
    assert len(out) == 1, "identical non-overlapping tracklets must merge to one"
    survivor = next(iter(out.values()))
    # merged track keeps t1's id (smaller index) and concatenates times/bboxes.
    assert sorted(survivor.times) == [0, 1, 2, 3, 4, 10, 11, 12, 13, 14], survivor.times
    assert len(survivor.bboxes) == 10, len(survivor.bboxes)
    # merge must concatenate features/scores/class_ids too, keeping ALL five
    # per-frame arrays aligned (Task 002 invariant).
    n = len(survivor.times)
    assert (
        len(survivor.bboxes) == n
        and len(survivor.features) == n
        and len(survivor.scores) == n
        and len(survivor.class_ids) == n
    ), (n, len(survivor.bboxes), len(survivor.features),
        len(survivor.scores), len(survivor.class_ids))
    # class_ids are concatenated in merge order (track1 then track2).
    assert survivor.class_ids == [0] * 5 + [1] * 5, survivor.class_ids


def test_fast_keeps_orthogonal_apart():
    # Orthogonal feats -> distance ~1.0 >= thres -> never merge.
    trks = {
        1: _FakeTrack(1, range(0, 5), [_V0] * 5),
        2: _FakeTrack(2, range(10, 15), [_V1] * 5),
    }
    out = _s2._fast_connect(trks, _fast_deps(spatial_ok=True),
                            max_x_range=1e9, max_y_range=1e9, merge_dist_thres=0.4)
    assert len(out) == 2, "orthogonal tracklets must not merge"


def test_fast_overlap_blocks_merge():
    # Identical feats but SHARED frames (2,3,4) -> distance forced to 1 -> no merge.
    trks = {
        1: _FakeTrack(1, range(0, 5), [_V0] * 5),
        2: _FakeTrack(2, range(2, 7), [_V0] * 5),
    }
    out = _s2._fast_connect(trks, _fast_deps(spatial_ok=True),
                            max_x_range=1e9, max_y_range=1e9, merge_dist_thres=0.4)
    assert len(out) == 2, "temporally overlapping tracklets must not merge"


def test_fast_spatial_gate_blocks_merge():
    # Mergeable by distance, but the spatial gate vetoes -> stays apart.
    trks = {
        1: _FakeTrack(1, range(0, 5), [_V0] * 5),
        2: _FakeTrack(2, range(10, 15), [_V0] * 5),
    }
    out = _s2._fast_connect(trks, _fast_deps(spatial_ok=False),
                            max_x_range=1e9, max_y_range=1e9, merge_dist_thres=0.4)
    assert len(out) == 2, "spatial-gate veto must prevent the merge"


def test_fast_chain_merges_only_similar():
    # 1 & 2 identical (merge); 3 orthogonal (stays). Result: {merged(1,2), 3}.
    trks = {
        1: _FakeTrack(1, range(0, 5), [_V0] * 5),
        2: _FakeTrack(2, range(10, 15), [_V0] * 5),
        3: _FakeTrack(3, range(20, 25), [_V1] * 5),
    }
    out = _s2._fast_connect(trks, _fast_deps(spatial_ok=True),
                            max_x_range=1e9, max_y_range=1e9, merge_dist_thres=0.4)
    assert len(out) == 2, out.keys()
    lens = sorted(len(t.times) for t in out.values())
    assert lens == [5, 10], lens   # the orthogonal one (5) + the merged pair (10)


# ===========================================================================
# Refined-tracklets pkl export (_renumber_and_export_pkl) — CPU-only
# ===========================================================================
def test_export_pkl_keys_match_save_results_renumbering():
    # Arbitrary (non-1..n) keys -> exported keys must be i+1 over sorted(keys),
    # which is EXACTLY the id renumbering save_results writes to refined.txt.
    out = {
        7: _FakeTrack(7, range(0, 3), [_V0] * 3),
        2: _FakeTrack(2, range(10, 13), [_V0] * 3),
        5: _FakeTrack(5, range(20, 23), [_V0] * 3),
    }
    with tempfile.TemporaryDirectory() as td:
        pkl_path = str(Path(td) / "refined_tracklets.pkl")
        _s2._renumber_and_export_pkl(out, pkl_path)
        import pickle as _pickle
        with open(pkl_path, "rb") as f:
            loaded = _pickle.load(f)

    # sorted(keys) = [2, 5, 7] -> new ids 1, 2, 3 (same as save_results' i+1).
    assert sorted(loaded.keys()) == [1, 2, 3], loaded.keys()
    # dict key and Tracklet.track_id agree (both equal the refined.txt id).
    for new_id, track in loaded.items():
        assert track.track_id == new_id, (new_id, track.track_id)


def test_export_pkl_track_id_set_on_objects():
    out = {3: _FakeTrack(3, range(0, 2), [_V0] * 2), 1: _FakeTrack(1, range(5, 7), [_V0] * 2)}
    with tempfile.TemporaryDirectory() as td:
        pkl_path = str(Path(td) / "refined_tracklets.pkl")
        _s2._renumber_and_export_pkl(out, pkl_path)
    # original tid 1 -> new id 1, original tid 3 -> new id 2 (sorted + i+1).
    assert out[1].track_id == 1, out[1].track_id
    assert out[3].track_id == 2, out[3].track_id


def test_export_pkl_asserts_array_alignment():
    bad = _FakeTrack(1, range(0, 5), [_V0] * 5)
    bad.class_ids = [0, 0]  # deliberately misaligned -> export must reject it
    raised = False
    try:
        with tempfile.TemporaryDirectory() as td:
            _s2._renumber_and_export_pkl({1: bad}, str(Path(td) / "x.pkl"))
    except AssertionError:
        raised = True
    assert raised, "misaligned per-frame arrays must trip the invariant assertion"


def test_run_refine_writes_pkl_with_matching_ids():
    # End-to-end through run_refine: the exported pkl keys must equal the set of
    # ids save_results would write to refined.txt (i+1 over sorted merged keys).
    split_out = {
        1: _FakeTrack(1, range(0, 5), [_V0] * 5, class_ids=[0] * 5),
        2: _FakeTrack(2, range(10, 15), [_V1] * 5, class_ids=[2] * 5),  # orthogonal: no merge
    }
    rec = _Recorder(split_output=split_out, spatial_ok=True)
    with tempfile.TemporaryDirectory() as td:
        pkl_path = str(Path(td) / "refined_tracklets.pkl")
        # keep_features=True so this asserts the FULL-array export invariant (the
        # slim default clears times/bboxes/features by design — covered by the
        # OPT-1 slim_pkl tests below).
        run_refine(
            {1: object(), 2: object()}, _REFINED_TXT, _params(),
            _deps_from(rec), seq_name="clip", refined_pkl=pkl_path,
            keep_features=True,
        )
        import pickle as _pickle
        with open(pkl_path, "rb") as f:
            loaded = _pickle.load(f)
    # two un-merged tracks -> ids {1, 2}; every track keeps all 5 arrays aligned.
    assert sorted(loaded.keys()) == [1, 2], loaded.keys()
    for new_id, track in loaded.items():
        n = len(track.times)
        assert (
            len(track.bboxes) == n
            and len(track.features) == n
            and len(track.scores) == n
            and len(track.class_ids) == n
        ), (new_id, n)


def test_run_refine_no_pkl_when_path_omitted():
    # Backward-compat: omitting refined_pkl must not attempt any pkl export.
    split_out = {1: _FakeTrack(1, range(0, 5), [_V0] * 5)}
    rec = _Recorder(split_output=split_out)
    counts = run_refine({1: object()}, _REFINED_TXT, _params(), _deps_from(rec), seq_name="clip")
    assert counts["n_tracklets_out"] == 1, counts  # ran fine without a pkl path


# ===========================================================================
# OPT-1: slim refined_tracklets.pkl — Stage 3 output is bit-identical whether
# it consumes the FULL pkl or the SLIMMED-then-exported pkl (CPU-only).
# ===========================================================================
# Distinct class ids from the shared map: player=2, goalkeeper=1, referee=3.
_PLAYER = _ta.PLAYER_CLASS
_GK = _ta.GOALKEEPER_CLASS
_REF = _ta.REFEREE_CLASS

# Two well-separated unit directions so the fake cluster fn (sign of comp 0)
# cleanly splits the players into two teams.
_POS = [1.0, 0.0, 0.0, 0.0]
_NEG = [-1.0, 0.0, 0.0, 0.0]


def _fake_cluster_by_sign(embeddings, k, **kwargs):
    """Deterministic CPU clusterer: label by the sign of the first component."""
    return [0 if np.asarray(e).ravel()[0] >= 0 else 1 for e in embeddings]


def _mk_real_track(tid, n, class_id, feat):
    """Build a real ``Tracklet`` with ``n`` frames (varied per-frame features).

    Per-frame features are jittered around ``feat`` so the stored mean is a
    genuine average over differing vectors — exercising the order-independence
    that makes the precomputed mean equal to Stage 3's on-the-fly mean.
    """
    rng = np.random.default_rng(tid)
    base = np.asarray(feat, dtype=np.float64)
    frames = list(range(n))
    scores = [1.0] * n
    bboxes = [[float(f), float(f), 10.0, 20.0] for f in frames]
    feats = [base + 0.05 * rng.standard_normal(base.size) for _ in frames]
    class_ids = [class_id] * n
    return _Tracklet(tid, frames, scores, bboxes, feats=feats, class_ids=class_ids)


def _fixture_tracklets():
    """A small ``{id: Tracklet}``: 3 players (2 +cluster, 1 -cluster), 1 GK, 1 ref."""
    return {
        1: _mk_real_track(1, 6, _PLAYER, _POS),
        2: _mk_real_track(2, 7, _PLAYER, _POS),
        3: _mk_real_track(3, 5, _PLAYER, _NEG),
        4: _mk_real_track(4, 4, _GK, _POS),
        5: _mk_real_track(5, 8, _REF, _NEG),
    }


def _export_and_load(out, keep_features):
    """Export ``out`` via Stage 2's pkl writer, then unpickle it back."""
    import pickle as _pickle

    with tempfile.TemporaryDirectory() as td:
        pkl_path = str(Path(td) / "refined_tracklets.pkl")
        _s2._renumber_and_export_pkl(out, pkl_path, keep_features=keep_features)
        with open(pkl_path, "rb") as f:
            return _pickle.load(f)


def test_slim_pkl_stage3_identical_to_full():
    # Stage 3 (assign_teams) over the FULL export must equal Stage 3 over the
    # SLIMMED-then-exported pkl: the precomputed mean_emb == on-the-fly mean.
    full = _export_and_load(_fixture_tracklets(), keep_features=True)
    slim = _export_and_load(_fixture_tracklets(), keep_features=False)

    attrs_full = _ta.assign_teams(full, cluster_fn=_fake_cluster_by_sign)
    attrs_slim = _ta.assign_teams(slim, cluster_fn=_fake_cluster_by_sign)
    assert attrs_full == attrs_slim, (attrs_full, attrs_slim)


def test_slim_pkl_track_attributes_json_identical_to_full():
    # The WRITTEN track_attributes.json must be byte-identical for full vs slim.
    import json

    full = _export_and_load(_fixture_tracklets(), keep_features=True)
    slim = _export_and_load(_fixture_tracklets(), keep_features=False)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        af, as_ = td / "full.json", td / "slim.json"
        _ta.write_track_attributes(
            str(af), _ta.assign_teams(full, cluster_fn=_fake_cluster_by_sign)
        )
        _ta.write_track_attributes(
            str(as_), _ta.assign_teams(slim, cluster_fn=_fake_cluster_by_sign)
        )
        assert af.read_text(encoding="utf-8") == as_.read_text(encoding="utf-8")
        # And it is the expected schema/content (3 players teamed, GK/ref -1).
        data = json.loads(as_.read_text(encoding="utf-8"))
        assert data["1"]["team"] == 0 and data["2"]["team"] == 0, data
        assert data["3"]["team"] == 1, data
        assert data["4"]["team"] == -1 and data["4"]["gk"] is True, data
        assert data["5"]["team"] == -1 and data["5"]["gk"] is False, data


def test_slim_pkl_drops_arrays_keeps_mean_and_class():
    # Acceptance: slim carries mean_emb + class_ids + scores; NO per-detection
    # features / times / bboxes.
    slim = _export_and_load(_fixture_tracklets(), keep_features=False)
    for tid, track in slim.items():
        assert getattr(track, "mean_emb", None) is not None, tid
        assert track.features == [] and track.times == [] and track.bboxes == [], tid
        assert len(track.class_ids) > 0 and len(track.scores) > 0, tid


def test_keep_features_reproduces_full_pkl():
    # --keep-features path: per-detection arrays intact, NO mean_emb attribute.
    full = _export_and_load(_fixture_tracklets(), keep_features=True)
    for tid, track in full.items():
        assert not hasattr(track, "mean_emb"), tid
        n = len(track.times)
        assert (
            len(track.bboxes) == n
            and len(track.features) == n
            and len(track.scores) == n
            and len(track.class_ids) == n
            and n > 0
        ), (tid, n)


def _featureless_fixture():
    """3 players, but track 3 has NO features (all five arrays empty, aligned).

    Built fresh on each call because ``_renumber_and_export_pkl`` mutates the
    Tracklets in place (slim mode clears arrays / sets ``mean_emb``).
    """
    out = {
        1: _mk_real_track(1, 6, _PLAYER, _POS),
        2: _mk_real_track(2, 6, _PLAYER, _NEG),
        3: _mk_real_track(3, 5, _PLAYER, _POS),
    }
    # Strip track 3's features (keep ALL five arrays mutually aligned — here all
    # empty — so the export-time alignment assert still passes on the full arrays).
    out[3].features = []
    out[3].times = []
    out[3].bboxes = []
    out[3].scores = []
    out[3].class_ids = []
    return out


def test_slim_pkl_mean_emb_none_for_featureless_track():
    # Edge case: a player track with NO features -> mean_emb is None -> it is
    # skipped from clustering, exactly as the full-pkl path skips it.
    slim = _export_and_load(_featureless_fixture(), keep_features=False)
    # The featureless track's stored mean is None.
    none_track = next(t for t in slim.values() if not t.class_ids)
    assert getattr(none_track, "mean_emb", "missing") is None

    # The equivalent FULL export (same data) -> Stage 3 output must match: the
    # featureless player is excluded from clustering in BOTH.
    full = _export_and_load(_featureless_fixture(), keep_features=True)
    attrs_full = _ta.assign_teams(full, cluster_fn=_fake_cluster_by_sign)
    attrs_slim = _ta.assign_teams(slim, cluster_fn=_fake_cluster_by_sign)
    assert attrs_full == attrs_slim, (attrs_full, attrs_slim)


# ===========================================================================
# Optional real end-to-end refine (SKIPS gracefully when heavy deps missing)
# ===========================================================================
def _heavy_deps_available() -> bool:
    for mod in ("torch", "sklearn", "scipy", "matplotlib", "seaborn", "loguru"):
        if importlib.util.find_spec(mod) is None:
            return False
    return True


def test_optional_end_to_end_real_refine():
    """Tiny real refine on a synthetic 2-tracklet pkl (skips if deps missing)."""
    if not _heavy_deps_available():
        print("    SKIP (heavy deps unavailable: refine_tracklets cannot import)")
        return

    # Import gta-link's Tracklet + refine_tracklets by adding gta-link to path.
    # (Only reached when all heavy deps are present, e.g. on Colab.)
    if str(_GTA_LINK_DIR) not in sys.path:
        sys.path.insert(0, str(_GTA_LINK_DIR))
    import refine_tracklets  # noqa: F401, WPS433
    from Tracklet import Tracklet  # noqa: WPS433

    rng = np.random.default_rng(0)
    # Two short, temporally non-overlapping tracklets with near-identical feats
    # so the merge step has a candidate.  Lengths < min_len so split is a no-op.
    # Distinct per-frame class ids so we can check they reach MOT col 8.
    def _mk(tid, frames, class_id):
        feats = [rng.standard_normal(512).astype(np.float32) for _ in frames]
        scores = [1.0 for _ in frames]
        bboxes = [[100.0 + f, 100.0 + f, 30.0, 60.0] for f in frames]
        class_ids = [class_id for _ in frames]
        return Tracklet(tid, list(frames), scores, bboxes, feats=feats, class_ids=class_ids)

    tmp = {1: _mk(1, range(0, 10), 0), 2: _mk(2, range(20, 30), 2)}

    deps = _s2._build_deps()
    with tempfile.TemporaryDirectory() as td:
        out_path = str(Path(td) / "refined.txt")
        pkl_path = str(Path(td) / "refined_tracklets.pkl")
        counts = run_refine(
            tmp, out_path, _params(), deps, seq_name="synthetic", refined_pkl=pkl_path
        )
        assert Path(out_path).is_file(), "refined.txt must be written"
        rows = Path(out_path).read_text().strip().splitlines()
        assert len(rows) >= 1, "refined.txt should have at least one MOT row"
        # Each row is the 10-field MOT format with score=1, trailing -1,-1,-1.
        first = rows[0].split(",")
        assert len(first) == 10, first
        assert counts["n_tracklets_in"] == 2, counts

        # --- col 8 carries the per-frame class (0 or 2 here), never the old -1 ---
        txt_ids = set()
        for row in rows:
            cols = row.split(",")
            txt_ids.add(int(cols[1]))
            assert cols[7] in ("0", "2"), "col 8 must be the per-frame class: " + row

        # --- pkl exists, keys == the ids written to refined.txt ---
        import pickle as _pickle
        with open(pkl_path, "rb") as f:
            refined_tracklets = _pickle.load(f)
        assert set(refined_tracklets.keys()) == txt_ids, (
            sorted(refined_tracklets.keys()), sorted(txt_ids)
        )
        # every exported track keeps all five parallel arrays aligned.
        for new_id, track in refined_tracklets.items():
            assert track.track_id == new_id, (new_id, track.track_id)
            n = len(track.times)
            assert (
                len(track.bboxes) == n
                and len(track.features) == n
                and len(track.scores) == n
                and len(track.class_ids) == n
            ), (new_id, n)
    print("    real refine wrote {} rows".format(len(rows)))


def test_optional_split_preserves_class_ids():
    """Real split_tracklets must keep per-frame class_ids aligned (skips w/o deps).

    Builds ONE long tracklet whose features form two well-separated clusters so
    DBSCAN splits it; the two halves carry distinct class ids.  After the split,
    each emitted sub-tracklet must have class_ids aligned with its frames and
    drawn from the correct half (no class loss on split).
    """
    if not _heavy_deps_available():
        print("    SKIP (heavy deps unavailable: refine_tracklets cannot import)")
        return

    if str(_GTA_LINK_DIR) not in sys.path:
        sys.path.insert(0, str(_GTA_LINK_DIR))
    import refine_tracklets  # noqa: WPS433
    from Tracklet import Tracklet  # noqa: WPS433

    rng = np.random.default_rng(1)
    # Two tight, far-apart feature clusters -> a clear id-switch to split on.
    half = 80
    centerA = np.zeros(512, dtype=np.float32); centerA[0] = 50.0
    centerB = np.zeros(512, dtype=np.float32); centerB[1] = 50.0
    feats = (
        [centerA + 0.01 * rng.standard_normal(512).astype(np.float32) for _ in range(half)]
        + [centerB + 0.01 * rng.standard_normal(512).astype(np.float32) for _ in range(half)]
    )
    frames = list(range(2 * half))
    scores = [1.0] * (2 * half)
    bboxes = [[float(f), float(f), 30.0, 60.0] for f in frames]
    class_ids = [0] * half + [2] * half  # class differs per cluster
    trk = Tracklet(1, frames, scores, bboxes, feats=feats, class_ids=class_ids)

    out = refine_tracklets.split_tracklets(
        {1: trk}, eps=0.6, max_k=3, min_samples=10, len_thres=50
    )
    assert len(out) >= 2, "the synthetic id-switch tracklet should split"
    for sub in out.values():
        n = len(sub.times)
        # class_ids exist, aligned 1:1 with the sub-tracklet's frames.
        assert len(sub.class_ids) == n, (n, len(sub.class_ids))
        assert len(sub.scores) == n and len(sub.bboxes) == n and len(sub.features) == n
        # within a sub-tracklet the class must be the one tied to its frames
        # (frame f < half -> class 0, else class 2): no cross-contamination.
        for f, c in zip(sub.times, sub.class_ids):
            assert c == (0 if f < half else 2), (f, c)


_TESTS = [
    ("run_refine: split -> connect -> save full flow", test_full_flow_split_then_connect_then_save),
    ("run_refine: spatial from original + factor", test_spatial_uses_original_and_factor),
    ("run_refine: split params forwarded by name", test_split_params_forwarded),
    ("run_refine: save target is the contract refined_txt", test_save_target_is_contract_path),
    ("fast_connect: identical non-overlapping tracklets merge", test_fast_merges_identical_nonoverlapping),
    ("fast_connect: orthogonal tracklets stay apart", test_fast_keeps_orthogonal_apart),
    ("fast_connect: temporal overlap blocks merge", test_fast_overlap_blocks_merge),
    ("fast_connect: spatial-gate veto blocks merge", test_fast_spatial_gate_blocks_merge),
    ("fast_connect: chain merges only the similar pair", test_fast_chain_merges_only_similar),
    ("export_pkl: keys match save_results i+1 renumbering", test_export_pkl_keys_match_save_results_renumbering),
    ("export_pkl: track_id set on exported objects", test_export_pkl_track_id_set_on_objects),
    ("export_pkl: misaligned arrays trip the invariant assertion", test_export_pkl_asserts_array_alignment),
    ("run_refine: writes pkl with ids matching refined.txt", test_run_refine_writes_pkl_with_matching_ids),
    ("run_refine: no pkl export when path omitted", test_run_refine_no_pkl_when_path_omitted),
    ("slim_pkl: Stage 3 assign_teams identical full vs slim", test_slim_pkl_stage3_identical_to_full),
    ("slim_pkl: track_attributes.json identical full vs slim", test_slim_pkl_track_attributes_json_identical_to_full),
    ("slim_pkl: drops features/times/bboxes, keeps mean_emb+class+scores", test_slim_pkl_drops_arrays_keeps_mean_and_class),
    ("slim_pkl: --keep-features reproduces full pkl (no mean_emb)", test_keep_features_reproduces_full_pkl),
    ("slim_pkl: mean_emb None for featureless track (edge)", test_slim_pkl_mean_emb_none_for_featureless_track),
    ("run_refine: optional real end-to-end refine (skips w/o deps)", test_optional_end_to_end_real_refine),
    ("split: optional real split preserves class_ids (skips w/o deps)", test_optional_split_preserves_class_ids),
]


def main() -> int:
    print("=" * 60)
    print("stage2 refine orchestration unit tests (CPU-only, stubbed)")
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

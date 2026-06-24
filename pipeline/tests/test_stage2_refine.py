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
    def __init__(self, track_id, times, feats, bboxes=None):
        self.track_id = track_id
        self.parent_id = track_id
        self.times = list(times)
        self.features = [np.asarray(f, dtype=np.float64) for f in feats]
        self.bboxes = (
            list(bboxes) if bboxes is not None
            else [[0.0, 0.0, 1.0, 1.0] for _ in times]
        )
        self.scores = [1.0 for _ in times]


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
        1: _FakeTrack(1, range(0, 5), [_V0] * 5),
        2: _FakeTrack(2, range(10, 15), [_V0] * 5),
    }
    out = _s2._fast_connect(trks, _fast_deps(spatial_ok=True),
                            max_x_range=1e9, max_y_range=1e9, merge_dist_thres=0.4)
    assert len(out) == 1, "identical non-overlapping tracklets must merge to one"
    survivor = next(iter(out.values()))
    # merged track keeps t1's id (smaller index) and concatenates times/bboxes.
    assert sorted(survivor.times) == [0, 1, 2, 3, 4, 10, 11, 12, 13, 14], survivor.times
    assert len(survivor.bboxes) == 10, len(survivor.bboxes)


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
    def _mk(tid, frames):
        feats = [rng.standard_normal(512).astype(np.float32) for _ in frames]
        scores = [1.0 for _ in frames]
        bboxes = [[100.0 + f, 100.0 + f, 30.0, 60.0] for f in frames]
        return Tracklet(tid, list(frames), scores, bboxes, feats=feats)

    tmp = {1: _mk(1, range(0, 10)), 2: _mk(2, range(20, 30))}

    deps = _s2._build_deps()
    with tempfile.TemporaryDirectory() as td:
        out_path = str(Path(td) / "refined.txt")
        counts = run_refine(tmp, out_path, _params(), deps, seq_name="synthetic")
        assert Path(out_path).is_file(), "refined.txt must be written"
        rows = Path(out_path).read_text().strip().splitlines()
        assert len(rows) >= 1, "refined.txt should have at least one MOT row"
        # Each row is the 10-field MOT format with score=1, trailing -1,-1,-1.
        first = rows[0].split(",")
        assert len(first) == 10, first
        assert counts["n_tracklets_in"] == 2, counts
    print("    real refine wrote {} rows".format(len(rows)))


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
    ("run_refine: optional real end-to-end refine (skips w/o deps)", test_optional_end_to_end_real_refine),
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

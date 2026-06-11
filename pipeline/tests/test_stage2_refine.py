"""
CPU-only unit tests for Stage-2's step-decision orchestration (Task 4).

Runs with plain python (numpy only) — NO torch / sklearn / scipy / matplotlib /
seaborn / loguru.  The refine ALGORITHM functions need those heavy deps, so this
test does NOT import ``refine_tracklets``; instead it drives
``stage2_refine.run_refine`` with INJECTED stub callables (a ``RefineDeps``
bundle) and asserts the step-decision logic:

  * ``--use_split`` only      -> split called, merge NOT called; save gets the
                                 split result.
  * ``--use_connect`` only    -> split NOT called, merge called.
  * both                      -> split THEN merge (split feeds merge).
  * neither                   -> raises ValueError.
  * save target is the contract ``refined_txt`` path.

``stage2_refine`` imports cleanly here because it only imports ``pipeline.*``
(numpy/stdlib) at module level; ``refine_tracklets`` is imported lazily inside
``_build_deps`` / ``main`` (never reached by these tests).

Optionally, if torch+sklearn+scipy+matplotlib+seaborn+loguru all happen to be
importable, a tiny real end-to-end refine on a synthetic 2-tracklet pkl runs and
asserts a non-empty refined.txt is written.  It SKIPS gracefully (does not fail)
when any dep is missing — which is the case on this host (loguru/matplotlib/
seaborn absent).

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
# Minimal test runner (matches test_stage1_parallel.py style; no pytest)
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
# Recording stubs for the injected refine callables
# ---------------------------------------------------------------------------
class _Recorder:
    """Records which refine steps were called, in order, with their inputs.

    Each stub returns a sentinel dict so we can prove which object flowed into
    save_results (the split dict vs the merged dict).
    """

    def __init__(self):
        self.calls = []                 # ordered list of step names
        self.spatial_arg = None
        self.split_input = None
        self.split_output = {"SPLIT": 1}
        self.dist_input = None
        self.merge_input = None         # tracklets passed to merge
        self.merge_kwargs = None
        self.merge_output = {"MERGED": 1}
        self.saved_path = None
        self.saved_obj = None

    # get_spatial_constraints(tid2track, factor) -> (max_x, max_y)
    def get_spatial_constraints(self, tid2track, factor):
        self.calls.append("spatial")
        self.spatial_arg = (tid2track, factor)
        return (111.0, 222.0)

    # split_tracklets(tmp, eps, max_k, min_samples, len_thres) -> dict
    def split_tracklets(self, tmp, eps=None, max_k=None, min_samples=None, len_thres=None):
        self.calls.append("split")
        self.split_input = tmp
        self.split_params = dict(eps=eps, max_k=max_k, min_samples=min_samples, len_thres=len_thres)
        return self.split_output

    # get_distance_matrix(tracklets) -> ndarray
    def get_distance_matrix(self, tracklets):
        self.calls.append("dist")
        self.dist_input = tracklets
        return np.zeros((2, 2), dtype=np.float32)

    # merge_tracklets(tracklets, seq2Dist, Dist, seq_name, max_x_range, max_y_range, merge_dist_thres)
    def merge_tracklets(self, tracklets, seq2Dist, Dist, seq_name=None,
                        max_x_range=None, max_y_range=None, merge_dist_thres=None):
        self.calls.append("merge")
        self.merge_input = tracklets
        self.merge_kwargs = dict(
            seq2Dist=seq2Dist, seq_name=seq_name, max_x_range=max_x_range,
            max_y_range=max_y_range, merge_dist_thres=merge_dist_thres,
        )
        return self.merge_output

    # save_results(out_path, tracklets) -> None
    def save_results(self, out_path, tracklets):
        self.calls.append("save")
        self.saved_path = out_path
        self.saved_obj = tracklets


def _deps_from(rec: _Recorder) -> "RefineDeps":
    return RefineDeps(
        get_spatial_constraints=rec.get_spatial_constraints,
        split_tracklets=rec.split_tracklets,
        get_distance_matrix=rec.get_distance_matrix,
        merge_tracklets=rec.merge_tracklets,
        save_results=rec.save_results,
    )


def _params(use_split, use_connect):
    return {
        "use_split": use_split,
        "use_connect": use_connect,
        "min_len": 100,
        "eps": 0.6,
        "min_samples": 10,
        "max_k": 3,
        "spatial_factor": 1.0,
        "merge_dist_thres": 0.4,
    }


_TMP_TRACKLETS = {1: object(), 2: object(), 3: object()}  # opaque {tid: Tracklet}
_REFINED_TXT = "/contract/path/02_refine/refined.txt"


# ===========================================================================
# Step-decision tests
# ===========================================================================
def test_split_only_no_merge():
    rec = _Recorder()
    counts = run_refine(_TMP_TRACKLETS, _REFINED_TXT, _params(True, False),
                        _deps_from(rec), seq_name="clip")

    assert "split" in rec.calls, rec.calls
    assert "merge" not in rec.calls, "merge must NOT run for --use_split only"
    assert "dist" not in rec.calls, "distance matrix must NOT run without merge"
    # save gets the SPLIT result (deviation: not merged).
    assert rec.saved_obj is rec.split_output, "save must receive the split result"
    assert rec.saved_path == _REFINED_TXT, rec.saved_path
    # counts reflect split sizes (stub split output has len 1).
    assert counts["n_tracklets_in"] == 3, counts
    assert counts["n_tracklets_after_split"] == len(rec.split_output), counts
    assert counts["n_tracklets_out"] == len(rec.split_output), counts


def test_connect_only_no_split():
    rec = _Recorder()
    counts = run_refine(_TMP_TRACKLETS, _REFINED_TXT, _params(False, True),
                        _deps_from(rec), seq_name="clip")

    assert "split" not in rec.calls, "split must NOT run for --use_connect only"
    assert "dist" in rec.calls and "merge" in rec.calls, rec.calls
    # When not splitting, the merge input is the ORIGINAL tmp tracklets.
    assert rec.merge_input is _TMP_TRACKLETS, "merge must run on the original tracklets"
    assert rec.dist_input is _TMP_TRACKLETS
    assert rec.saved_obj is rec.merge_output, "save must receive the merged result"
    assert rec.saved_path == _REFINED_TXT
    # after_split == in (no split happened).
    assert counts["n_tracklets_after_split"] == 3, counts
    assert counts["n_tracklets_out"] == len(rec.merge_output), counts


def test_both_split_then_merge():
    rec = _Recorder()
    run_refine(_TMP_TRACKLETS, _REFINED_TXT, _params(True, True),
               _deps_from(rec), seq_name="clip")

    # Order: spatial, split, dist, merge, save.
    assert rec.calls == ["spatial", "split", "dist", "merge", "save"], rec.calls
    # split feeds merge: merge runs on the split OUTPUT, not the original.
    assert rec.split_input is _TMP_TRACKLETS
    assert rec.merge_input is rec.split_output, "split output must feed merge"
    assert rec.dist_input is rec.split_output
    assert rec.saved_obj is rec.merge_output


def test_neither_raises():
    rec = _Recorder()
    raised = False
    try:
        run_refine(_TMP_TRACKLETS, _REFINED_TXT, _params(False, False),
                   _deps_from(rec), seq_name="clip")
    except ValueError:
        raised = True
    assert raised, "neither --use_split nor --use_connect must raise ValueError"
    # No algorithm step should have run before the raise.
    assert rec.calls == [], rec.calls


def test_save_target_is_contract_path():
    # Explicitly assert across all enabled-flag combos that save target == refined_txt.
    for us, uc in [(True, False), (False, True), (True, True)]:
        rec = _Recorder()
        run_refine(_TMP_TRACKLETS, _REFINED_TXT, _params(us, uc),
                   _deps_from(rec), seq_name="clip")
        assert rec.saved_path == _REFINED_TXT, (us, uc, rec.saved_path)


def test_spatial_uses_original_and_factor():
    # spatial constraints computed from original tmp with the spatial_factor,
    # and the (max_x, max_y) are threaded into merge.
    rec = _Recorder()
    p = _params(True, True)
    p["spatial_factor"] = 2.5
    run_refine(_TMP_TRACKLETS, _REFINED_TXT, p, _deps_from(rec), seq_name="clip")
    assert rec.spatial_arg[0] is _TMP_TRACKLETS, "spatial uses ORIGINAL tracklets"
    assert rec.spatial_arg[1] == 2.5, rec.spatial_arg
    # merge receives the spatial ranges + seq_name + threshold.
    assert rec.merge_kwargs["max_x_range"] == 111.0
    assert rec.merge_kwargs["max_y_range"] == 222.0
    assert rec.merge_kwargs["seq_name"] == "clip"
    assert rec.merge_kwargs["merge_dist_thres"] == 0.4
    assert rec.merge_kwargs["seq2Dist"] == {}, "merge gets an empty seq2Dist debug dict"


def test_split_params_forwarded():
    rec = _Recorder()
    p = _params(True, False)
    p.update(eps=0.42, max_k=7, min_samples=4, min_len=33)
    run_refine(_TMP_TRACKLETS, _REFINED_TXT, p, _deps_from(rec), seq_name="clip")
    assert rec.split_params == dict(eps=0.42, max_k=7, min_samples=4, len_thres=33), rec.split_params


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
    import refine_tracklets  # noqa: WPS433
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
        counts = run_refine(tmp, out_path, _params(True, True), deps, seq_name="synthetic")
        assert Path(out_path).is_file(), "refined.txt must be written"
        rows = Path(out_path).read_text().strip().splitlines()
        assert len(rows) >= 1, "refined.txt should have at least one MOT row"
        # Each row is the 10-field MOT format with score=1, trailing -1,-1,-1.
        first = rows[0].split(",")
        assert len(first) == 10, first
        assert counts["n_tracklets_in"] == 2, counts
    print("    real refine wrote {} rows".format(len(rows)))


_TESTS = [
    ("run_refine: --use_split only -> split, no merge, save split", test_split_only_no_merge),
    ("run_refine: --use_connect only -> merge, no split", test_connect_only_no_split),
    ("run_refine: both -> split THEN merge (split feeds merge)", test_both_split_then_merge),
    ("run_refine: neither -> raises ValueError", test_neither_raises),
    ("run_refine: save target is the contract refined_txt", test_save_target_is_contract_path),
    ("run_refine: spatial from original+factor, threaded into merge", test_spatial_uses_original_and_factor),
    ("run_refine: split params forwarded by name", test_split_params_forwarded),
    ("run_refine: optional real end-to-end refine (skips w/o deps)", test_optional_end_to_end_real_refine),
]


def main() -> int:
    print("=" * 60)
    print("stage2 refine step-decision unit tests (CPU-only, stubbed)")
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

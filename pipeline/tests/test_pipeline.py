"""
Pure-Python unit tests for the pipeline package.

Run with:
    python pipeline/tests/test_pipeline.py

No pytest, torch, psutil, or GPU required.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

# Make sure the repo root is on sys.path so we can import pipeline
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.artifacts import ArtifactPaths, ensure_dirs, get_artifact_paths
from pipeline.cache import is_stale, should_run
from pipeline.profiling import profile_stage, write_summary

# ---------------------------------------------------------------------------
# Helpers
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


# ---------------------------------------------------------------------------
# 1. Artifact path layout
# ---------------------------------------------------------------------------


def test_artifact_paths_layout():
    paths = get_artifact_paths("/data/videos/match.mp4", base_dir="/tmp/art")
    root = Path("/tmp/art/match")
    assert paths.root == root, f"root mismatch: {paths.root}"
    assert paths.track_dir == root / "01_track"
    assert paths.tracks_txt == root / "01_track" / "tracks.txt"
    assert paths.tracklets_pkl == root / "01_track" / "tracklets.pkl"
    assert paths.refine_dir == root / "02_refine"
    assert paths.refined_txt == root / "02_refine" / "refined.txt"
    assert paths.profiles_dir == root / "profiles"
    assert paths.profile_json("01_track") == root / "profiles" / "01_track.json"


def test_artifact_paths_default_base_dir():
    paths = get_artifact_paths("game.mp4")
    assert paths.root == Path("artifacts") / "game"


def test_artifact_paths_no_extension():
    paths = get_artifact_paths("game", base_dir="/out")
    assert paths.root == Path("/out/game")


def test_ensure_dirs_creates_tree():
    with tempfile.TemporaryDirectory() as tmp:
        paths = get_artifact_paths("vid.mp4", base_dir=tmp)
        ensure_dirs(paths)
        assert paths.track_dir.is_dir(), "track_dir not created"
        assert paths.refine_dir.is_dir(), "refine_dir not created"
        assert paths.profiles_dir.is_dir(), "profiles_dir not created"


def test_ensure_dirs_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        paths = get_artifact_paths("vid.mp4", base_dir=tmp)
        ensure_dirs(paths)
        ensure_dirs(paths)  # must not raise


# ---------------------------------------------------------------------------
# 2. is_stale / should_run
# ---------------------------------------------------------------------------


def _write(path: Path, content: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_is_stale_missing_output():
    with tempfile.TemporaryDirectory() as tmp:
        inp = _write(Path(tmp) / "inp.txt")
        out = Path(tmp) / "out.txt"  # does not exist
        assert is_stale(out, [inp]) is True, "missing output must be stale"


def test_is_stale_missing_input():
    with tempfile.TemporaryDirectory() as tmp:
        out = _write(Path(tmp) / "out.txt")
        inp = Path(tmp) / "inp.txt"  # does not exist
        assert is_stale(out, [inp]) is True, "missing input must be stale"


def test_is_stale_up_to_date():
    with tempfile.TemporaryDirectory() as tmp:
        inp = _write(Path(tmp) / "inp.txt")
        # Ensure output is strictly newer than input
        time.sleep(0.01)
        out = _write(Path(tmp) / "out.txt")
        assert is_stale(out, [inp]) is False, "newer output should not be stale"


def test_is_stale_input_newer_than_output():
    with tempfile.TemporaryDirectory() as tmp:
        out = _write(Path(tmp) / "out.txt")
        time.sleep(0.01)
        inp = _write(Path(tmp) / "inp.txt")  # newer than out
        assert is_stale(out, [inp]) is True, "newer input must make output stale"


def test_is_stale_equal_mtime_not_stale():
    """An input with mtime EQUAL to the output is up-to-date (strict > comparison)."""
    with tempfile.TemporaryDirectory() as tmp:
        inp = _write(Path(tmp) / "inp.txt")
        out = _write(Path(tmp) / "out.txt")
        # Force both files to the exact same mtime (and atime) so getmtime ties.
        ts = 1_600_000_000  # fixed timestamp, identical for both files
        os.utime(inp, (ts, ts))
        os.utime(out, (ts, ts))
        assert os.path.getmtime(inp) == os.path.getmtime(out), "precondition: mtimes must be equal"
        assert is_stale(out, [inp]) is False, "equal mtime must NOT be stale"


def test_is_stale_no_inputs():
    with tempfile.TemporaryDirectory() as tmp:
        out = _write(Path(tmp) / "out.txt")
        assert is_stale(out, []) is False, "no inputs + existing output = not stale"


def test_should_run_force():
    with tempfile.TemporaryDirectory() as tmp:
        inp = _write(Path(tmp) / "inp.txt")
        time.sleep(0.01)
        out = _write(Path(tmp) / "out.txt")
        # Would normally be False (not stale), but force=True overrides
        assert should_run(out, [inp], force=True) is True


def test_should_run_not_forced_stale():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.txt"  # missing
        assert should_run(out, [], force=False) is True


def test_should_run_not_forced_fresh():
    with tempfile.TemporaryDirectory() as tmp:
        inp = _write(Path(tmp) / "inp.txt")
        time.sleep(0.01)
        out = _write(Path(tmp) / "out.txt")
        assert should_run(out, [inp], force=False) is False


# ---------------------------------------------------------------------------
# 3. profile_stage — JSON output, CPU-only
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = {
    "schema_version",
    "stage",
    "wall_time_s",
    "gpu_peak_mb",
    "cpu_rss_start_mb",
    "cpu_rss_end_mb",
    "cpu_peak_mb",
    "io",
}


def _make_artifacts(tmp: str, stem: str = "vid") -> ArtifactPaths:
    paths = get_artifact_paths(f"{stem}.mp4", base_dir=tmp)
    ensure_dirs(paths)
    return paths


def test_profile_stage_writes_json():
    with tempfile.TemporaryDirectory() as tmp:
        artifacts = _make_artifacts(tmp)
        with profile_stage("test_stage", artifacts, extra={"n_frames": 10}):
            pass  # trivial body
        profile_path = artifacts.profile_json("test_stage")
        assert profile_path.exists(), "profile JSON not written"
        with open(profile_path, encoding="utf-8") as fh:
            data = json.load(fh)
        missing = _REQUIRED_KEYS - set(data.keys())
        assert not missing, f"missing keys: {missing}"


def test_profile_stage_schema_version():
    with tempfile.TemporaryDirectory() as tmp:
        artifacts = _make_artifacts(tmp)
        with profile_stage("s1", artifacts):
            pass
        with open(artifacts.profile_json("s1"), encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["schema_version"] == 1


def test_profile_stage_wall_time_positive():
    with tempfile.TemporaryDirectory() as tmp:
        artifacts = _make_artifacts(tmp)
        with profile_stage("s2", artifacts):
            time.sleep(0.02)
        with open(artifacts.profile_json("s2"), encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["wall_time_s"] >= 0.01, "wall time too small"


def test_profile_stage_io_dict():
    with tempfile.TemporaryDirectory() as tmp:
        artifacts = _make_artifacts(tmp)
        extra = {"n_frames": 300, "emb_shape": [25, 512]}
        with profile_stage("s3", artifacts, extra=extra):
            pass
        with open(artifacts.profile_json("s3"), encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["io"] == extra, f"io mismatch: {data['io']}"


def test_profile_stage_gpu_null_when_no_cuda():
    """gpu_peak_mb should be null if CUDA unavailable (expected on this machine)."""
    try:
        import torch  # noqa: F401

        if torch.cuda.is_available():
            # Can't test the null branch on a CUDA machine — skip gracefully
            return
    except ImportError:
        pass  # torch absent — null is guaranteed

    with tempfile.TemporaryDirectory() as tmp:
        artifacts = _make_artifacts(tmp)
        with profile_stage("s4", artifacts):
            pass
        with open(artifacts.profile_json("s4"), encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["gpu_peak_mb"] is None, "expected null gpu_peak_mb without CUDA"


def test_profile_stage_writes_on_exception():
    """Profile JSON must be written even if the stage body raises."""
    with tempfile.TemporaryDirectory() as tmp:
        artifacts = _make_artifacts(tmp)
        try:
            with profile_stage("s5", artifacts):
                raise RuntimeError("deliberate error")
        except RuntimeError:
            pass
        assert artifacts.profile_json("s5").exists(), "profile not written on exception"


def test_profile_stage_creates_profiles_dir():
    """profiles_dir is created even if ensure_dirs was not called."""
    with tempfile.TemporaryDirectory() as tmp:
        # Build paths but do NOT call ensure_dirs
        artifacts = get_artifact_paths("vid.mp4", base_dir=tmp)
        with profile_stage("s6", artifacts):
            pass
        assert artifacts.profile_json("s6").exists()


# ---------------------------------------------------------------------------
# 4. write_summary
# ---------------------------------------------------------------------------

_FAKE_PROFILE_1: dict = {
    "schema_version": 1,
    "stage": "01_track",
    "wall_time_s": 12.5,
    "gpu_peak_mb": 1024.0,
    "cpu_rss_start_mb": 200.0,
    "cpu_rss_end_mb": 250.0,
    "cpu_peak_mb": None,
    "io": {"n_frames": 300},
}

_FAKE_PROFILE_2: dict = {
    "schema_version": 1,
    "stage": "02_refine",
    "wall_time_s": 3.2,
    "gpu_peak_mb": None,
    "cpu_rss_start_mb": 250.0,
    "cpu_rss_end_mb": 260.0,
    "cpu_peak_mb": None,
    "io": {},
}


def _write_fake_profiles(profiles_dir: Path) -> None:
    profiles_dir.mkdir(parents=True, exist_ok=True)
    for p in (_FAKE_PROFILE_1, _FAKE_PROFILE_2):
        path = profiles_dir / f"{p['stage']}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(p, fh)


def test_write_summary_creates_files():
    with tempfile.TemporaryDirectory() as tmp:
        pdir = Path(tmp) / "profiles"
        _write_fake_profiles(pdir)
        write_summary(pdir)
        assert (pdir / "summary.json").exists(), "summary.json not created"
        assert (pdir / "summary.md").exists(), "summary.md not created"


def test_write_summary_json_structure():
    with tempfile.TemporaryDirectory() as tmp:
        pdir = Path(tmp) / "profiles"
        _write_fake_profiles(pdir)
        write_summary(pdir)
        with open(pdir / "summary.json", encoding="utf-8") as fh:
            data = json.load(fh)
        assert "profiles" in data, "missing 'profiles' key"
        assert "totals" in data, "missing 'totals' key"
        assert len(data["profiles"]) == 2, f"expected 2 profiles, got {len(data['profiles'])}"


def test_write_summary_totals_wall_time():
    with tempfile.TemporaryDirectory() as tmp:
        pdir = Path(tmp) / "profiles"
        _write_fake_profiles(pdir)
        write_summary(pdir)
        with open(pdir / "summary.json", encoding="utf-8") as fh:
            data = json.load(fh)
        expected_total = _FAKE_PROFILE_1["wall_time_s"] + _FAKE_PROFILE_2["wall_time_s"]
        assert abs(data["totals"]["wall_time_s"] - expected_total) < 1e-9


def test_write_summary_totals_gpu_peak():
    with tempfile.TemporaryDirectory() as tmp:
        pdir = Path(tmp) / "profiles"
        _write_fake_profiles(pdir)
        write_summary(pdir)
        with open(pdir / "summary.json", encoding="utf-8") as fh:
            data = json.load(fh)
        # max of 1024.0 and None → 1024.0
        assert data["totals"]["gpu_peak_mb"] == 1024.0


def test_write_summary_md_contains_stage_names():
    with tempfile.TemporaryDirectory() as tmp:
        pdir = Path(tmp) / "profiles"
        _write_fake_profiles(pdir)
        write_summary(pdir)
        md = (pdir / "summary.md").read_text(encoding="utf-8")
        assert "01_track" in md, "stage name missing from summary.md"
        assert "02_refine" in md, "stage name missing from summary.md"


def test_write_summary_excludes_itself():
    """Re-running write_summary must not include the old summary.json as a profile."""
    with tempfile.TemporaryDirectory() as tmp:
        pdir = Path(tmp) / "profiles"
        _write_fake_profiles(pdir)
        write_summary(pdir)
        write_summary(pdir)  # second run
        with open(pdir / "summary.json", encoding="utf-8") as fh:
            data = json.load(fh)
        assert len(data["profiles"]) == 2, "summary.json must not include itself"


def test_write_summary_empty_dir():
    """write_summary on an empty directory should not raise."""
    with tempfile.TemporaryDirectory() as tmp:
        pdir = Path(tmp) / "profiles"
        pdir.mkdir()
        write_summary(pdir)
        with open(pdir / "summary.json", encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["profiles"] == []
        assert data["totals"]["wall_time_s"] == 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_TESTS = [
    ("artifacts: path layout correct", test_artifact_paths_layout),
    ("artifacts: default base_dir", test_artifact_paths_default_base_dir),
    ("artifacts: no extension", test_artifact_paths_no_extension),
    ("artifacts: ensure_dirs creates tree", test_ensure_dirs_creates_tree),
    ("artifacts: ensure_dirs idempotent", test_ensure_dirs_idempotent),
    ("cache: is_stale missing output", test_is_stale_missing_output),
    ("cache: is_stale missing input", test_is_stale_missing_input),
    ("cache: is_stale up-to-date", test_is_stale_up_to_date),
    ("cache: is_stale input newer", test_is_stale_input_newer_than_output),
    ("cache: is_stale equal mtime not stale", test_is_stale_equal_mtime_not_stale),
    ("cache: is_stale no inputs", test_is_stale_no_inputs),
    ("cache: should_run force=True", test_should_run_force),
    ("cache: should_run stale", test_should_run_not_forced_stale),
    ("cache: should_run fresh", test_should_run_not_forced_fresh),
    ("profiling: writes JSON", test_profile_stage_writes_json),
    ("profiling: schema_version=1", test_profile_stage_schema_version),
    ("profiling: wall_time positive", test_profile_stage_wall_time_positive),
    ("profiling: io dict preserved", test_profile_stage_io_dict),
    ("profiling: gpu_peak null without CUDA", test_profile_stage_gpu_null_when_no_cuda),
    ("profiling: writes JSON on exception", test_profile_stage_writes_on_exception),
    ("profiling: creates profiles_dir", test_profile_stage_creates_profiles_dir),
    ("summary: creates files", test_write_summary_creates_files),
    ("summary: JSON structure", test_write_summary_json_structure),
    ("summary: totals wall_time", test_write_summary_totals_wall_time),
    ("summary: totals gpu_peak", test_write_summary_totals_gpu_peak),
    ("summary: md contains stage names", test_write_summary_md_contains_stage_names),
    ("summary: excludes itself on re-run", test_write_summary_excludes_itself),
    ("summary: empty directory", test_write_summary_empty_dir),
]


def main() -> int:
    print("=" * 60)
    print("pipeline unit tests")
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

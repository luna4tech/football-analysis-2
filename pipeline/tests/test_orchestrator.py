"""
CPU-only unit tests for the orchestrator + compare_tracks (Task 5).

Runs with plain python (numpy only) — NO torch / cv2 / GPU.  The orchestrator is
exercised with an INJECTED fake runner that records the commands it would launch
(it never starts a subprocess), and ``compare_tracks`` is driven on temp MOT
files.

Run with:
    python pipeline/tests/test_orchestrator.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Repo root on sys.path so `import pipeline` works.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline import orchestrator
from pipeline.compare_tracks import compare_tracks
from pipeline.orchestrator import (
    CommandResult,
    RunOptions,
    StageError,
    build_stage1_command,
    build_stage2_command,
    run,
)

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
# Fake runner: records (cmd, cwd) and returns a scripted return code per call.
# ---------------------------------------------------------------------------
class FakeRunner:
    def __init__(self, return_codes=None):
        # return_codes[i] is the rc for the i-th call; default all 0.
        self._return_codes = list(return_codes) if return_codes is not None else []
        self.calls = []  # list of (cmd, cwd)

    def __call__(self, cmd, cwd):
        self.calls.append((list(cmd), str(cwd)))
        idx = len(self.calls) - 1
        rc = self._return_codes[idx] if idx < len(self._return_codes) else 0
        return CommandResult(rc)


def _value_after(cmd, flag):
    """Return the single value following ``flag`` in an argv list."""
    i = cmd.index(flag)
    return cmd[i + 1]


def _opt_str(value):
    return str(value)


# A throwaway video path under a temp artifacts dir keeps every test hermetic.
def _make_env(tmp):
    video = Path(tmp) / "clip.mp4"
    video.write_bytes(b"\x00")  # must exist so .resolve() is stable
    artifacts = Path(tmp) / "artifacts"
    return str(video), str(artifacts)


# ===========================================================================
# Stage-1 command wiring
# ===========================================================================
def test_stage1_sequential_command():
    opts = RunOptions(device="gpu")
    cmd = build_stage1_command("/abs/clip.mp4", "/abs/artifacts", opts)
    assert cmd[0] == sys.executable, cmd
    assert cmd[1] == "tools/stage1_track.py", cmd
    assert _value_after(cmd, "--video") == "/abs/clip.mp4", cmd
    assert _value_after(cmd, "--artifacts-dir") == "/abs/artifacts", cmd
    assert _value_after(cmd, "--device") == "gpu", cmd
    # sequential: no parallel/batch/force.
    assert "--parallel" not in cmd, cmd
    assert "--batch-size" not in cmd, cmd
    assert "--force" not in cmd, cmd


def test_stage1_parallel_forwards_batch():
    opts = RunOptions(device="cpu", parallel=True, batch_size=16)
    cmd = build_stage1_command("/abs/clip.mp4", "/abs/artifacts", opts)
    assert "--parallel" in cmd, cmd
    assert _value_after(cmd, "--batch-size") == "16", cmd
    assert _value_after(cmd, "--device") == "cpu", cmd


def test_stage1_force_adds_force():
    opts = RunOptions(force_stage1=True)
    cmd = build_stage1_command("/abs/clip.mp4", "/abs/artifacts", opts)
    assert "--force" in cmd, cmd


# ===========================================================================
# Stage-2 command wiring
# ===========================================================================
def test_stage2_refine_params_forwarded():
    opts = RunOptions(
        use_split=True,
        use_connect=True,
        eps=0.42,
        min_samples=7,
        max_k=5,
        min_len=33,
        spatial_factor=2.5,
        merge_dist_thres=0.33,
    )
    cmd = build_stage2_command("/abs/clip.mp4", "/abs/artifacts", opts)
    assert cmd[0] == sys.executable, cmd
    assert cmd[1] == "stage2_refine.py", cmd
    assert _value_after(cmd, "--video") == "/abs/clip.mp4", cmd
    assert _value_after(cmd, "--artifacts-dir") == "/abs/artifacts", cmd
    assert "--use_split" in cmd, cmd
    assert "--use_connect" in cmd, cmd
    assert _value_after(cmd, "--eps") == "0.42", cmd
    assert _value_after(cmd, "--min_samples") == "7", cmd
    assert _value_after(cmd, "--max_k") == "5", cmd
    assert _value_after(cmd, "--min_len") == "33", cmd
    assert _value_after(cmd, "--spatial_factor") == "2.5", cmd
    assert _value_after(cmd, "--merge_dist_thres") == "0.33", cmd
    assert "--force" not in cmd, cmd


def test_stage2_no_split_no_connect_flags_omitted():
    # --no-split -> use_split False -> flag absent (still has connect).
    cmd_a = build_stage2_command("/v", "/a", RunOptions(use_split=False, use_connect=True))
    assert "--use_split" not in cmd_a, cmd_a
    assert "--use_connect" in cmd_a, cmd_a

    cmd_b = build_stage2_command("/v", "/a", RunOptions(use_split=True, use_connect=False))
    assert "--use_split" in cmd_b, cmd_b
    assert "--use_connect" not in cmd_b, cmd_b


# ===========================================================================
# Orchestrated run with the fake runner
# ===========================================================================
def test_run_two_stages_cwd_and_abs_paths():
    with tempfile.TemporaryDirectory() as tmp:
        video, artifacts = _make_env(tmp)
        runner = FakeRunner([0, 0])
        run(video, RunOptions(artifacts_dir=artifacts), runner=runner)

        assert len(runner.calls) == 2, "both stages must run on success"
        (cmd1, cwd1), (cmd2, cwd2) = runner.calls

        # Stage 1 in Deep-EIoU/Deep-EIoU; Stage 2 in gta-link.
        assert cwd1 == str(orchestrator.STAGE1_CWD), cwd1
        assert cwd2 == str(orchestrator.STAGE2_CWD), cwd2
        assert cmd1[1] == "tools/stage1_track.py", cmd1
        assert cmd2[1] == "stage2_refine.py", cmd2

        # SAME absolute video + artifacts dir to BOTH stages.
        v1 = _value_after(cmd1, "--video")
        v2 = _value_after(cmd2, "--video")
        a1 = _value_after(cmd1, "--artifacts-dir")
        a2 = _value_after(cmd2, "--artifacts-dir")
        assert v1 == v2, (v1, v2)
        assert a1 == a2, (a1, a2)
        assert Path(v1).is_absolute(), v1
        assert Path(a1).is_absolute(), a1
        assert v1 == str(Path(video).resolve()), v1
        assert a1 == str(Path(artifacts).resolve()), a1


def test_run_full_pipeline_defaults_both_components():
    with tempfile.TemporaryDirectory() as tmp:
        video, artifacts = _make_env(tmp)
        runner = FakeRunner([0, 0])
        run(video, RunOptions(artifacts_dir=artifacts), runner=runner)
        cmd2 = runner.calls[1][0]
        assert "--use_split" in cmd2, cmd2
        assert "--use_connect" in cmd2, cmd2


def test_force_all_forces_both_stages():
    with tempfile.TemporaryDirectory() as tmp:
        video, artifacts = _make_env(tmp)
        runner = FakeRunner([0, 0])
        opts = RunOptions(artifacts_dir=artifacts, force_stage1=True, force_stage2=True)
        run(video, opts, runner=runner)
        cmd1, cmd2 = runner.calls[0][0], runner.calls[1][0]
        assert "--force" in cmd1, "Stage 1 must be forced"
        assert "--force" in cmd2, "Stage 2 must be forced"


def test_force_stage2_only():
    with tempfile.TemporaryDirectory() as tmp:
        video, artifacts = _make_env(tmp)
        runner = FakeRunner([0, 0])
        opts = RunOptions(artifacts_dir=artifacts, force_stage1=False, force_stage2=True)
        run(video, opts, runner=runner)
        cmd1, cmd2 = runner.calls[0][0], runner.calls[1][0]
        assert "--force" not in cmd1, "Stage 1 must NOT be forced"
        assert "--force" in cmd2, "Stage 2 must be forced"


def test_stage1_failure_aborts_before_stage2():
    with tempfile.TemporaryDirectory() as tmp:
        video, artifacts = _make_env(tmp)
        runner = FakeRunner([1, 0])  # Stage 1 returns non-zero
        raised = False
        try:
            run(video, RunOptions(artifacts_dir=artifacts), runner=runner)
        except StageError as exc:
            raised = True
            assert "Stage 1" in str(exc), str(exc)
        assert raised, "non-zero Stage 1 must raise StageError"
        assert len(runner.calls) == 1, "Stage 2 must NOT run after Stage 1 fails"


def test_write_summary_called_on_success():
    # Use fake profile JSONs so we can assert summary aggregation across stages.
    import json

    with tempfile.TemporaryDirectory() as tmp:
        video, artifacts = _make_env(tmp)

        # The fake runner writes per-stage profile JSONs exactly where the real
        # stages would (paths.profiles_dir), so write_summary aggregates them.
        from pipeline.artifacts import get_artifact_paths

        paths = get_artifact_paths(str(Path(video).resolve()), base_dir=str(Path(artifacts).resolve()))

        def _writing_runner(cmd, cwd):
            paths.profiles_dir.mkdir(parents=True, exist_ok=True)
            if cmd[1].endswith("stage1_track.py"):
                rec = {"stage": "01_track", "wall_time_s": 12.0, "gpu_peak_mb": 100.0,
                       "cpu_rss_start_mb": None, "cpu_rss_end_mb": None, "cpu_peak_mb": None, "io": {}}
                (paths.profiles_dir / "01_track.json").write_text(json.dumps(rec))
            else:
                rec = {"stage": "02_refine", "wall_time_s": 3.0, "gpu_peak_mb": 250.0,
                       "cpu_rss_start_mb": None, "cpu_rss_end_mb": None, "cpu_peak_mb": None, "io": {}}
                (paths.profiles_dir / "02_refine.json").write_text(json.dumps(rec))
            return CommandResult(0)

        result = run(video, RunOptions(artifacts_dir=artifacts), runner=_writing_runner)

        assert result.summary_md.is_file(), "summary.md must be written on success"
        summary_json = paths.profiles_dir / "summary.json"
        assert summary_json.is_file(), "summary.json must be written"
        data = json.loads(summary_json.read_text())
        # Aggregation: total wall = 12 + 3 = 15; gpu peak = max(100, 250) = 250.
        assert abs(data["totals"]["wall_time_s"] - 15.0) < 1e-9, data["totals"]
        assert data["totals"]["gpu_peak_mb"] == 250.0, data["totals"]
        assert len(data["profiles"]) == 2, data["profiles"]


# ===========================================================================
# compare_tracks
# ===========================================================================
def _write_mot(path, rows):
    """rows: list of (frame, id, x, y, w, h)."""
    lines = []
    for (fr, tid, x, y, w, h) in rows:
        lines.append(f"{fr},{tid},{x},{y},{w},{h},1.0,-1,-1,-1")
    Path(path).write_text("\n".join(lines) + "\n")


_ROWS = [
    (0, 1, 100.0, 100.0, 30.0, 60.0),
    (0, 2, 200.0, 150.0, 28.0, 58.0),
    (1, 1, 101.0, 101.0, 30.0, 60.0),
]


def test_compare_identical_equivalent():
    with tempfile.TemporaryDirectory() as tmp:
        a = Path(tmp) / "a.txt"
        b = Path(tmp) / "b.txt"
        _write_mot(a, _ROWS)
        _write_mot(b, _ROWS)
        r = compare_tracks(a, b, tol=1.0)
        assert r.equivalent, r.report()
        assert r.keys_match, r.report()
        assert r.max_coord_diff == 0.0, r.max_coord_diff


def test_compare_within_tol_jitter_equivalent():
    with tempfile.TemporaryDirectory() as tmp:
        a = Path(tmp) / "a.txt"
        b = Path(tmp) / "b.txt"
        _write_mot(a, _ROWS)
        # jitter every coord by <= 0.5px (under the 1.0 tol)
        jittered = [(fr, tid, x + 0.5, y - 0.5, w + 0.4, h - 0.3) for (fr, tid, x, y, w, h) in _ROWS]
        _write_mot(b, jittered)
        r = compare_tracks(a, b, tol=1.0)
        assert r.equivalent, r.report()
        assert 0.0 < r.max_coord_diff <= 1.0, r.max_coord_diff


def test_compare_out_of_tol_not_equivalent():
    with tempfile.TemporaryDirectory() as tmp:
        a = Path(tmp) / "a.txt"
        b = Path(tmp) / "b.txt"
        _write_mot(a, _ROWS)
        # shift one box by 5px -> exceeds tol
        big = list(_ROWS)
        big[0] = (0, 1, 105.0, 100.0, 30.0, 60.0)
        _write_mot(b, big)
        r = compare_tracks(a, b, tol=1.0)
        assert not r.equivalent, r.report()
        assert r.keys_match, "keys still match; failure is the coord diff"
        assert r.max_coord_diff == 5.0, r.max_coord_diff


def test_compare_key_mismatch_not_equivalent():
    with tempfile.TemporaryDirectory() as tmp:
        a = Path(tmp) / "a.txt"
        b = Path(tmp) / "b.txt"
        _write_mot(a, _ROWS)
        # drop the last row in B -> (1,1) only in A
        _write_mot(b, _ROWS[:-1])
        r = compare_tracks(a, b, tol=1.0)
        assert not r.equivalent, r.report()
        assert not r.keys_match, r.report()
        assert (1, 1) in r.only_in_a, r.only_in_a
        assert r.only_in_b == [], r.only_in_b


_TESTS = [
    ("stage1 sequential command shape", test_stage1_sequential_command),
    ("stage1 --parallel forwards --batch-size", test_stage1_parallel_forwards_batch),
    ("stage1 force adds --force", test_stage1_force_adds_force),
    ("stage2 refine params forwarded", test_stage2_refine_params_forwarded),
    ("stage2 no-split/no-connect flags omitted", test_stage2_no_split_no_connect_flags_omitted),
    ("run: two stages, correct cwd + abs paths to both", test_run_two_stages_cwd_and_abs_paths),
    ("run: full pipeline defaults both components", test_run_full_pipeline_defaults_both_components),
    ("force-all forces both stages", test_force_all_forces_both_stages),
    ("force-stage2 forces only stage 2", test_force_stage2_only),
    ("stage1 failure aborts before stage2", test_stage1_failure_aborts_before_stage2),
    ("write_summary called + aggregates on success", test_write_summary_called_on_success),
    ("compare: identical -> equivalent", test_compare_identical_equivalent),
    ("compare: within-tol jitter -> equivalent", test_compare_within_tol_jitter_equivalent),
    ("compare: out-of-tol -> not equivalent", test_compare_out_of_tol_not_equivalent),
    ("compare: key mismatch -> not equivalent", test_compare_key_mismatch_not_equivalent),
]


def main() -> int:
    print("=" * 60)
    print("orchestrator + compare_tracks unit tests (CPU-only, fake runner)")
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

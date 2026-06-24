"""
CPU-only unit tests for the orchestrator.

Runs with plain python (numpy only) — NO torch / cv2 / GPU.  The orchestrator is
exercised with an INJECTED fake runner that records the commands it would launch
(it never starts a subprocess).

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
from pipeline.orchestrator import (
    CommandResult,
    RunOptions,
    StageError,
    build_stage1_command,
    build_stage2_command,
    build_stage3_command,
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
def test_stage1_command():
    opts = RunOptions(device="gpu")
    cmd = build_stage1_command("/abs/clip.mp4", "/abs/artifacts", opts)
    assert cmd[0] == sys.executable, cmd
    assert cmd[1] == "tools/stage1_track.py", cmd
    assert _value_after(cmd, "--video") == "/abs/clip.mp4", cmd
    assert _value_after(cmd, "--artifacts-dir") == "/abs/artifacts", cmd
    assert _value_after(cmd, "--device") == "gpu", cmd
    assert _value_after(cmd, "--detector-ckpt") == "checkpoints/yolov11l.pt", cmd
    # default: not forced.
    assert "--force" not in cmd, cmd


def test_stage1_device_forwarded():
    cmd = build_stage1_command("/abs/clip.mp4", "/abs/artifacts", RunOptions(device="cpu"))
    assert _value_after(cmd, "--device") == "cpu", cmd


def test_stage1_force_adds_force():
    opts = RunOptions(force_stage1=True)
    cmd = build_stage1_command("/abs/clip.mp4", "/abs/artifacts", opts)
    assert "--force" in cmd, cmd


def test_stage1_fp16_and_fuse_off_by_default():
    cmd = build_stage1_command("/abs/clip.mp4", "/abs/artifacts", RunOptions())
    assert "--fp16" not in cmd, cmd
    assert "--fuse" not in cmd, cmd


def test_stage1_fp16_and_fuse_forwarded():
    opts = RunOptions(fp16=True, fuse=True)
    cmd = build_stage1_command("/abs/clip.mp4", "/abs/artifacts", opts)
    assert "--fp16" in cmd, cmd
    assert "--fuse" in cmd, cmd


# ===========================================================================
# Stage-2 command wiring
# ===========================================================================
def test_stage2_refine_params_forwarded():
    opts = RunOptions(
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
    assert _value_after(cmd, "--eps") == "0.42", cmd
    assert _value_after(cmd, "--min_samples") == "7", cmd
    assert _value_after(cmd, "--max_k") == "5", cmd
    assert _value_after(cmd, "--min_len") == "33", cmd
    assert _value_after(cmd, "--spatial_factor") == "2.5", cmd
    assert _value_after(cmd, "--merge_dist_thres") == "0.33", cmd
    assert "--force" not in cmd, cmd


# ===========================================================================
# Stage-3 command wiring
# ===========================================================================
def test_stage3_command():
    opts = RunOptions()
    cmd = build_stage3_command("/abs/clip.mp4", "/abs/artifacts", opts)
    assert cmd[0] == sys.executable, cmd
    assert cmd[1] == "pipeline/team_assignment.py", cmd
    assert _value_after(cmd, "--video") == "/abs/clip.mp4", cmd
    assert _value_after(cmd, "--artifacts-dir") == "/abs/artifacts", cmd
    # default: not forced.
    assert "--force" not in cmd, cmd


def test_stage3_force_adds_force():
    opts = RunOptions(force_stage3=True)
    cmd = build_stage3_command("/abs/clip.mp4", "/abs/artifacts", opts)
    assert "--force" in cmd, cmd


# ===========================================================================
# Orchestrated run with the fake runner
# ===========================================================================
def test_run_three_stages_cwd_and_abs_paths():
    with tempfile.TemporaryDirectory() as tmp:
        video, artifacts = _make_env(tmp)
        runner = FakeRunner([0, 0, 0])
        run(video, RunOptions(artifacts_dir=artifacts), runner=runner)

        assert len(runner.calls) == 3, "all three stages must run on success"
        (cmd1, cwd1), (cmd2, cwd2), (cmd3, cwd3) = runner.calls

        # Stage 1 in Deep-EIoU/Deep-EIoU; Stage 2 in gta-link; Stage 3 in repo root.
        assert cwd1 == str(orchestrator.STAGE1_CWD), cwd1
        assert cwd2 == str(orchestrator.STAGE2_CWD), cwd2
        assert cwd3 == str(orchestrator.STAGE3_CWD), cwd3
        assert cmd1[1] == "tools/stage1_track.py", cmd1
        assert cmd2[1] == "stage2_refine.py", cmd2
        assert cmd3[1] == "pipeline/team_assignment.py", cmd3

        # SAME absolute video + artifacts dir to ALL stages.
        vs = [_value_after(c, "--video") for c in (cmd1, cmd2, cmd3)]
        as_ = [_value_after(c, "--artifacts-dir") for c in (cmd1, cmd2, cmd3)]
        assert vs[0] == vs[1] == vs[2], vs
        assert as_[0] == as_[1] == as_[2], as_
        assert Path(vs[0]).is_absolute(), vs[0]
        assert Path(as_[0]).is_absolute(), as_[0]
        assert vs[0] == str(Path(video).resolve()), vs[0]
        assert as_[0] == str(Path(artifacts).resolve()), as_[0]


def test_run_stage2_gets_refine_params():
    with tempfile.TemporaryDirectory() as tmp:
        video, artifacts = _make_env(tmp)
        runner = FakeRunner([0, 0])
        run(video, RunOptions(artifacts_dir=artifacts), runner=runner)
        cmd2 = runner.calls[1][0]
        # Stage 2 always runs the full split + connect refine; the refine params
        # are forwarded.
        assert _value_after(cmd2, "--eps") == "0.6", cmd2
        assert _value_after(cmd2, "--merge_dist_thres") == "0.4", cmd2


def test_force_all_forces_all_stages():
    with tempfile.TemporaryDirectory() as tmp:
        video, artifacts = _make_env(tmp)
        runner = FakeRunner([0, 0, 0])
        opts = RunOptions(
            artifacts_dir=artifacts,
            force_stage1=True,
            force_stage2=True,
            force_stage3=True,
        )
        run(video, opts, runner=runner)
        cmd1, cmd2, cmd3 = runner.calls[0][0], runner.calls[1][0], runner.calls[2][0]
        assert "--force" in cmd1, "Stage 1 must be forced"
        assert "--force" in cmd2, "Stage 2 must be forced"
        assert "--force" in cmd3, "Stage 3 must be forced"


def test_force_stage3_only():
    with tempfile.TemporaryDirectory() as tmp:
        video, artifacts = _make_env(tmp)
        runner = FakeRunner([0, 0, 0])
        opts = RunOptions(
            artifacts_dir=artifacts,
            force_stage1=False,
            force_stage2=False,
            force_stage3=True,
        )
        run(video, opts, runner=runner)
        cmd1, cmd2, cmd3 = runner.calls[0][0], runner.calls[1][0], runner.calls[2][0]
        assert "--force" not in cmd1, "Stage 1 must NOT be forced"
        assert "--force" not in cmd2, "Stage 2 must NOT be forced"
        assert "--force" in cmd3, "Stage 3 must be forced"


def test_stage1_failure_aborts_before_stage2():
    with tempfile.TemporaryDirectory() as tmp:
        video, artifacts = _make_env(tmp)
        runner = FakeRunner([1, 0, 0])  # Stage 1 returns non-zero
        raised = False
        try:
            run(video, RunOptions(artifacts_dir=artifacts), runner=runner)
        except StageError as exc:
            raised = True
            assert "Stage 1" in str(exc), str(exc)
        assert raised, "non-zero Stage 1 must raise StageError"
        assert len(runner.calls) == 1, "Stage 2 must NOT run after Stage 1 fails"


def test_stage2_failure_aborts_before_stage3():
    with tempfile.TemporaryDirectory() as tmp:
        video, artifacts = _make_env(tmp)
        runner = FakeRunner([0, 1, 0])  # Stage 2 returns non-zero
        raised = False
        try:
            run(video, RunOptions(artifacts_dir=artifacts), runner=runner)
        except StageError as exc:
            raised = True
            assert "Stage 2" in str(exc), str(exc)
        assert raised, "non-zero Stage 2 must raise StageError"
        assert len(runner.calls) == 2, "Stage 3 must NOT run after Stage 2 fails"


def test_stage3_failure_raises():
    with tempfile.TemporaryDirectory() as tmp:
        video, artifacts = _make_env(tmp)
        runner = FakeRunner([0, 0, 1])  # Stage 3 returns non-zero
        raised = False
        try:
            run(video, RunOptions(artifacts_dir=artifacts), runner=runner)
        except StageError as exc:
            raised = True
            assert "Stage 3" in str(exc), str(exc)
        assert raised, "non-zero Stage 3 must raise StageError"
        assert len(runner.calls) == 3, "all three stages run; Stage 3 failed last"


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
            elif cmd[1].endswith("stage2_refine.py"):
                rec = {"stage": "02_refine", "wall_time_s": 3.0, "gpu_peak_mb": 250.0,
                       "cpu_rss_start_mb": None, "cpu_rss_end_mb": None, "cpu_peak_mb": None, "io": {}}
                (paths.profiles_dir / "02_refine.json").write_text(json.dumps(rec))
            else:
                rec = {"stage": "03_team", "wall_time_s": 1.0, "gpu_peak_mb": None,
                       "cpu_rss_start_mb": None, "cpu_rss_end_mb": None, "cpu_peak_mb": None, "io": {}}
                (paths.profiles_dir / "03_team.json").write_text(json.dumps(rec))
            return CommandResult(0)

        result = run(video, RunOptions(artifacts_dir=artifacts), runner=_writing_runner)

        assert result.summary_md.is_file(), "summary.md must be written on success"
        summary_json = paths.profiles_dir / "summary.json"
        assert summary_json.is_file(), "summary.json must be written"
        data = json.loads(summary_json.read_text())
        # Aggregation: total wall = 12 + 3 + 1 = 16; gpu peak = max(100, 250) = 250.
        assert abs(data["totals"]["wall_time_s"] - 16.0) < 1e-9, data["totals"]
        assert data["totals"]["gpu_peak_mb"] == 250.0, data["totals"]
        assert len(data["profiles"]) == 3, data["profiles"]


_TESTS = [
    ("stage1 command shape", test_stage1_command),
    ("stage1 --device forwarded", test_stage1_device_forwarded),
    ("stage1 force adds --force", test_stage1_force_adds_force),
    ("stage1 --fp16/--fuse off by default", test_stage1_fp16_and_fuse_off_by_default),
    ("stage1 --fp16/--fuse forwarded", test_stage1_fp16_and_fuse_forwarded),
    ("stage2 refine params forwarded", test_stage2_refine_params_forwarded),
    ("stage3 command shape", test_stage3_command),
    ("stage3 force adds --force", test_stage3_force_adds_force),
    ("run: three stages, correct cwd + abs paths to all", test_run_three_stages_cwd_and_abs_paths),
    ("run: stage 2 gets refine params", test_run_stage2_gets_refine_params),
    ("force-all forces all stages", test_force_all_forces_all_stages),
    ("force-stage3 forces only stage 3", test_force_stage3_only),
    ("stage1 failure aborts before stage2", test_stage1_failure_aborts_before_stage2),
    ("stage2 failure aborts before stage3", test_stage2_failure_aborts_before_stage3),
    ("stage3 failure raises", test_stage3_failure_raises),
    ("write_summary called + aggregates on success", test_write_summary_called_on_success),
]


def main() -> int:
    print("=" * 60)
    print("orchestrator unit tests (CPU-only, fake runner)")
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

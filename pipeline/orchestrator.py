"""
pipeline.orchestrator — chain Stage 1 then Stage 2 as subprocesses.

Why subprocesses (not in-process)?
----------------------------------
Deep-EIoU and GtaLink each bundle their OWN copy of ``torchreid`` under
incompatible top-level module names; importing both into one interpreter would
clash.  The locked design (PLAN.md "Option A") keeps each stage intact in its own
directory and runs it as its OWN process in its OWN working directory:

  * Stage 1 (``tools/stage1_track.py``)   CWD = ``<repo>/Deep-EIoU/Deep-EIoU``
  * Stage 2 (``stage2_refine.py``)        CWD = ``<repo>/gta-link``

The orchestrator itself runs from the repo root and only touches ``pipeline.*``
helpers, so this module is import-light (no torch / cv2) and runs on a CPU box.

Absolute paths across the CWD boundary
--------------------------------------
Because each stage runs in a DIFFERENT working directory, a relative ``--video``
or ``--artifacts-dir`` would resolve to two different places.  The orchestrator
therefore resolves ONE absolute video path and ONE absolute artifacts base up
front and passes the SAME absolute values to BOTH stages — it never relies on a
stage's per-CWD relative default.

Caching caveat (documented + surfaced in --help)
------------------------------------------------
Each stage self-caches on input-file MTIME (``pipeline.cache``).  Changing a
PARAMETER (a tracking threshold, ``eps``, ``merge_dist_thres``, …) does NOT
change any input file's mtime, so it does NOT bust the cache.  Re-tuning params
therefore requires the matching ``--force-*`` flag.

Public API
----------
RunOptions               — typed bundle of all run() knobs.
SubprocessRunner         — default runner (thin wrapper over subprocess.run).
build_stage1_command(...)/build_stage2_command(...) — pure command builders.
run(video, opts, runner=None) -> RunResult
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from pipeline.artifacts import ensure_dirs, get_artifact_paths
from pipeline.profiling import write_summary

# --- repo layout (resolved relative to this file, never the CWD) ------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE1_CWD = _REPO_ROOT / "Deep-EIoU" / "Deep-EIoU"
STAGE2_CWD = _REPO_ROOT / "gta-link"
STAGE1_SCRIPT = "tools/stage1_track.py"   # relative to STAGE1_CWD
STAGE2_SCRIPT = "stage2_refine.py"        # relative to STAGE2_CWD


# ===========================================================================
# Options
# ===========================================================================
class RunOptions:
    """All knobs for an orchestrated run.

    Stage 1
    -------
    device : str
        ``"gpu"`` (default) or ``"cpu"`` — forwarded to Stage 1's ``--device``.
    parallel : bool
        Opt-in batched detect+ReID + prefetch decode (forwarded as ``--parallel``).
    batch_size : int
        Frames per batch in parallel mode (forwarded as ``--batch-size`` only
        when ``parallel`` is set).
    fp16 : bool
        Half-precision detector inference (forwarded as ``--fp16``).  This is the
        main GPU throughput lever — batching does NOT speed up an already
        compute-saturated detector, but FP16 typically gives ~1.5-2x.
    fuse : bool
        Fuse detector conv+BN layers (forwarded as ``--fuse``); small free gain.

    Stage 2 refine params (forwarded with the same names Stage 2 declares)
    ---------------------------------------------------------------------
    use_split, use_connect : bool
        Default to the FULL pipeline (BOTH True).  Disable one via
        ``--no-split`` / ``--no-connect`` on the CLI; at least one must remain.
    eps, min_samples, max_k, min_len, spatial_factor, merge_dist_thres
        Refine algorithm params (same defaults as Stage 2's parser).
    fast_merge : bool
        Use Stage 2's exact batched connect/merge (``--fast_merge``) instead of
        the original per-pair GPU path. Same output (up to float rounding),
        orders of magnitude faster on the connect step. Only affects
        ``--use_connect``.

    Force
    -----
    force_stage1, force_stage2 : bool
        Add ``--force`` to the corresponding stage.  ``--force-all`` sets both.

    Misc
    ----
    artifacts_dir : str | None
        Base dir for artifacts; resolved to an absolute path before use.
        ``None`` -> ``<repo>/artifacts``.
    """

    def __init__(
        self,
        *,
        device: str = "gpu",
        parallel: bool = False,
        batch_size: int = 8,
        fp16: bool = False,
        fuse: bool = False,
        use_split: bool = True,
        use_connect: bool = True,
        eps: float = 0.6,
        min_samples: int = 10,
        max_k: int = 3,
        min_len: int = 100,
        spatial_factor: float = 1.0,
        merge_dist_thres: float = 0.4,
        fast_merge: bool = False,
        force_stage1: bool = False,
        force_stage2: bool = False,
        artifacts_dir: "Optional[str]" = None,
    ) -> None:
        self.device = device
        self.parallel = parallel
        self.batch_size = batch_size
        self.fp16 = fp16
        self.fuse = fuse
        self.use_split = use_split
        self.use_connect = use_connect
        self.eps = eps
        self.min_samples = min_samples
        self.max_k = max_k
        self.min_len = min_len
        self.spatial_factor = spatial_factor
        self.merge_dist_thres = merge_dist_thres
        self.fast_merge = fast_merge
        self.force_stage1 = force_stage1
        self.force_stage2 = force_stage2
        self.artifacts_dir = artifacts_dir


# ===========================================================================
# Runner (injectable)
# ===========================================================================
class CommandResult:
    """Minimal result returned by a runner: just the process return code."""

    def __init__(self, returncode: int) -> None:
        self.returncode = returncode


class SubprocessRunner:
    """Default runner: run a command live (inherit stdout/stderr, no capture).

    Stages stream their progress straight to the parent's terminal so the user
    sees live output.  Tests inject a fake runner instead so they can assert the
    built commands without launching anything.
    """

    def __call__(self, cmd: Sequence[str], cwd: "str | Path") -> CommandResult:
        # No capture: child inherits the parent's stdout/stderr (live stream).
        completed = subprocess.run(list(cmd), cwd=str(cwd))
        return CommandResult(completed.returncode)


# ===========================================================================
# Pure command builders (no side effects; unit-tested directly)
# ===========================================================================
def build_stage1_command(video_abs: str, artifacts_abs: str, opts: RunOptions) -> List[str]:
    """Build the Stage-1 argv (to run with ``cwd=STAGE1_CWD``).

    Always passes the absolute ``--video`` / ``--artifacts-dir``.  Forwards
    ``--device`` always; ``--parallel --batch-size N`` only when parallel;
    ``--fp16`` / ``--fuse`` only when enabled; and ``--force`` only when Stage 1
    is forced.
    """
    cmd: List[str] = [
        sys.executable,
        STAGE1_SCRIPT,
        "--video", video_abs,
        "--artifacts-dir", artifacts_abs,
        "--device", opts.device,
    ]
    if opts.parallel:
        cmd += ["--parallel", "--batch-size", str(opts.batch_size)]
    if opts.fp16:
        cmd += ["--fp16"]
    if opts.fuse:
        cmd += ["--fuse"]
    if opts.force_stage1:
        cmd += ["--force"]
    return cmd


def build_stage2_command(video_abs: str, artifacts_abs: str, opts: RunOptions) -> List[str]:
    """Build the Stage-2 argv (to run with ``cwd=STAGE2_CWD``).

    Always passes the absolute ``--video`` / ``--artifacts-dir``.  Forwards the
    refine params; ``--use_split`` / ``--use_connect`` are store_true flags so
    they are added only when enabled.  ``--force`` only when Stage 2 is forced.
    """
    cmd: List[str] = [
        sys.executable,
        STAGE2_SCRIPT,
        "--video", video_abs,
        "--artifacts-dir", artifacts_abs,
    ]
    if opts.use_split:
        cmd += ["--use_split"]
    if opts.use_connect:
        cmd += ["--use_connect"]
    cmd += [
        "--eps", str(opts.eps),
        "--min_samples", str(opts.min_samples),
        "--max_k", str(opts.max_k),
        "--min_len", str(opts.min_len),
        "--spatial_factor", str(opts.spatial_factor),
        "--merge_dist_thres", str(opts.merge_dist_thres),
    ]
    if opts.fast_merge:
        cmd += ["--fast_merge"]
    if opts.force_stage2:
        cmd += ["--force"]
    return cmd


# ===========================================================================
# Result
# ===========================================================================
class RunResult:
    """Outcome of a successful orchestrated run."""

    def __init__(
        self,
        *,
        paths: Any,
        stage1_seconds: float,
        stage2_seconds: float,
        summary_md: Path,
    ) -> None:
        self.paths = paths
        self.stage1_seconds = stage1_seconds
        self.stage2_seconds = stage2_seconds
        self.summary_md = summary_md


class StageError(RuntimeError):
    """Raised when a stage subprocess returns a non-zero exit code."""


# ===========================================================================
# Orchestrator
# ===========================================================================
def run(
    video: "str | Path",
    opts: RunOptions,
    runner: "Optional[SubprocessRunner]" = None,
) -> RunResult:
    """Run Stage 1 then Stage 2 as subprocesses; aggregate the profile summary.

    Parameters
    ----------
    video:
        Path to the input video (relative or absolute; resolved to absolute).
    opts:
        A :class:`RunOptions` bundle.
    runner:
        Callable ``runner(cmd, cwd) -> object with .returncode``.  Defaults to
        :class:`SubprocessRunner` (live, no capture).  INJECTABLE so tests can
        assert the built commands without launching anything.

    Behavior
    --------
    * Resolves ONE absolute video path and ONE absolute artifacts base; passes
      the SAME absolute ``--video`` / ``--artifacts-dir`` to BOTH stages.
    * ``ensure_dirs`` so the artifact tree (incl. ``profiles/``) exists.
    * Runs Stage 1 (cwd=STAGE1_CWD).  If it returns non-zero, ABORTS before
      Stage 2 and raises :class:`StageError` naming Stage 1.
    * Runs Stage 2 (cwd=STAGE2_CWD).  Non-zero -> :class:`StageError` naming
      Stage 2.
    * On success calls ``write_summary(paths.profiles_dir)`` and prints per-stage
      wall times + the ``summary.md`` path.

    Returns
    -------
    RunResult
    """
    if runner is None:
        runner = SubprocessRunner()

    # --- resolve absolute paths up front (the CWD-boundary guarantee) -------
    video_abs = str(Path(video).resolve())
    if opts.artifacts_dir is None:
        artifacts_base = _REPO_ROOT / "artifacts"
    else:
        artifacts_base = Path(opts.artifacts_dir).resolve()
    artifacts_abs = str(artifacts_base)

    # paths so we know profiles_dir; uses the SAME absolute base the stages get.
    paths = get_artifact_paths(video_abs, base_dir=artifacts_abs)
    ensure_dirs(paths)

    # --- Stage 1 ------------------------------------------------------------
    stage1_cmd = build_stage1_command(video_abs, artifacts_abs, opts)
    print(f"[pipeline] Stage 1 -> {' '.join(stage1_cmd)}  (cwd={STAGE1_CWD})")
    t0 = time.perf_counter()
    res1 = runner(stage1_cmd, STAGE1_CWD)
    stage1_seconds = time.perf_counter() - t0
    if res1.returncode != 0:
        raise StageError(
            f"Stage 1 (tracking) failed with exit code {res1.returncode}; "
            f"aborting before Stage 2. Command: {' '.join(stage1_cmd)}"
        )

    # --- Stage 2 ------------------------------------------------------------
    stage2_cmd = build_stage2_command(video_abs, artifacts_abs, opts)
    print(f"[pipeline] Stage 2 -> {' '.join(stage2_cmd)}  (cwd={STAGE2_CWD})")
    t0 = time.perf_counter()
    res2 = runner(stage2_cmd, STAGE2_CWD)
    stage2_seconds = time.perf_counter() - t0
    if res2.returncode != 0:
        raise StageError(
            f"Stage 2 (refine) failed with exit code {res2.returncode}. "
            f"Command: {' '.join(stage2_cmd)}"
        )

    # --- aggregate the profile summary --------------------------------------
    write_summary(paths.profiles_dir)
    summary_md = paths.profiles_dir / "summary.md"

    print("[pipeline] done.")
    print(f"[pipeline]   Stage 1 wall: {stage1_seconds:.2f} s")
    print(f"[pipeline]   Stage 2 wall: {stage2_seconds:.2f} s")
    print(f"[pipeline]   summary:      {summary_md}")

    return RunResult(
        paths=paths,
        stage1_seconds=stage1_seconds,
        stage2_seconds=stage2_seconds,
        summary_md=summary_md,
    )

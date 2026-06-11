"""
pipeline.profiling — per-stage wall-time and memory profiling.

JSON profile schema (written to ``profiles/<stage_name>.json``)
---------------------------------------------------------------
{
  "schema_version": 1,
  "stage":          <str>   stage identifier,
  "wall_time_s":    <float> elapsed wall-clock seconds,
  "gpu_peak_mb":    <float | null>
                    peak GPU memory in MB (torch.cuda.max_memory_allocated),
                    null when torch is unavailable or no CUDA device present,
  "cpu_rss_start_mb": <float | null>
                    process RSS in MB measured just before stage body ran,
                    resolved via the chain psutil -> resource -> null
                    (null when neither psutil nor resource is available),
  "cpu_rss_end_mb": <float | null>
                    process RSS in MB measured just after stage body returned,
  "cpu_peak_mb":    <float | null>
                    peak resident-set size in MB during the stage if the
                    platform exposes it (psutil); otherwise null,
  "io":             <object>
                    arbitrary caller-supplied key→value metadata
                    (e.g. {"n_frames": 300, "emb_shape": [25, 512]}),
}

Public API
----------
profile_stage(name, artifacts, extra=None)  — context manager
write_summary(profiles_dir)                 — aggregate JSON + Markdown table
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

# ---------------------------------------------------------------------------
# Optional-dependency probing (all at import time so they are checked once)
# ---------------------------------------------------------------------------

# ---- torch / CUDA ----------------------------------------------------------
_torch_available = False
try:
    import torch as _torch  # type: ignore[import]

    _torch_available = True
except ImportError:
    _torch = None  # type: ignore[assignment]

# ---- psutil ----------------------------------------------------------------
_psutil_available = False
try:
    import psutil as _psutil  # type: ignore[import]

    _psutil_available = True
except ImportError:
    _psutil = None  # type: ignore[assignment]

# ---- resource (POSIX only) -------------------------------------------------
_resource_available = False
try:
    import resource as _resource  # type: ignore[import]

    _resource_available = True
except ImportError:
    _resource = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MB = 1024.0 * 1024.0


def _rss_mb() -> Optional[float]:
    """Return current process RSS in MB, or None if unavailable."""
    if _psutil_available:
        try:
            proc = _psutil.Process(os.getpid())
            return proc.memory_info().rss / _MB
        except Exception:  # pragma: no cover
            pass
    if _resource_available:
        try:
            # getrusage returns ru_maxrss in kilobytes on Linux, bytes on macOS
            import sys

            usage = _resource.getrusage(_resource.RUSAGE_SELF)  # type: ignore[union-attr]
            ru_maxrss = usage.ru_maxrss
            if sys.platform == "darwin":
                return ru_maxrss / _MB
            else:
                return ru_maxrss / 1024.0  # KB → MB
        except Exception:  # pragma: no cover
            pass
    return None


def _gpu_reset() -> None:
    """Reset CUDA peak-memory stats if CUDA is available; otherwise no-op."""
    if _torch_available and _torch.cuda.is_available():  # type: ignore[union-attr]
        _torch.cuda.reset_peak_memory_stats()  # type: ignore[union-attr]


def _gpu_peak_mb() -> Optional[float]:
    """Return peak GPU memory in MB, or None if unavailable."""
    if _torch_available and _torch.cuda.is_available():  # type: ignore[union-attr]
        try:
            return _torch.cuda.max_memory_allocated() / _MB  # type: ignore[union-attr]
        except Exception:  # pragma: no cover
            pass
    return None


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def profile_stage(
    name: str,
    artifacts: Any,
    extra: Optional[Dict[str, Any]] = None,
) -> Iterator[None]:
    """Context manager that profiles the enclosed stage and writes a JSON file.

    Usage::

        with profile_stage("01_track", artifacts, extra={"n_frames": 300}):
            run_tracking(...)

    Parameters
    ----------
    name:
        Stage identifier, also used as the JSON filename stem.
    artifacts:
        An :class:`~pipeline.artifacts.ArtifactPaths` instance.  The profile
        is written to ``artifacts.profile_json(name)``.
    extra:
        Optional dict of caller-supplied IO metadata (e.g. frame counts,
        embedding shapes).  Must be JSON-serialisable.  Defaults to ``{}``.

    Side-effects
    ------------
    * Writes ``<profiles_dir>/<name>.json`` on normal exit *and* on exception
      (wall_time reflects the time up to the exception).
    * Creates ``profiles_dir`` if it does not exist.
    """
    if extra is None:
        extra = {}

    # Ensure directory exists
    profiles_dir: Path = artifacts.profiles_dir
    profiles_dir.mkdir(parents=True, exist_ok=True)

    profile_path: Path = artifacts.profile_json(name)

    # --- entry measurements -------------------------------------------------
    _gpu_reset()
    rss_start = _rss_mb()
    t_start = time.perf_counter()

    try:
        yield
    finally:
        # --- exit measurements ----------------------------------------------
        wall_time_s = time.perf_counter() - t_start
        rss_end = _rss_mb()
        gpu_peak = _gpu_peak_mb()

        # cpu_peak: not available via standard libs; reserved for future use
        cpu_peak: Optional[float] = None

        record: Dict[str, Any] = {
            "schema_version": 1,
            "stage": name,
            "wall_time_s": wall_time_s,
            "gpu_peak_mb": gpu_peak,
            "cpu_rss_start_mb": rss_start,
            "cpu_rss_end_mb": rss_end,
            "cpu_peak_mb": cpu_peak,
            "io": extra,
        }

        with open(profile_path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

_SUMMARY_EXCLUDE = {"summary.json", "summary.md"}

# Columns written to the Markdown table (in order)
_MD_COLUMNS = [
    ("stage", "Stage"),
    ("wall_time_s", "Wall (s)"),
    ("gpu_peak_mb", "GPU peak (MB)"),
    ("cpu_rss_start_mb", "CPU RSS start (MB)"),
    ("cpu_rss_end_mb", "CPU RSS end (MB)"),
    ("cpu_peak_mb", "CPU peak (MB)"),
]


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_summary(profiles_dir: "str | os.PathLike[str]") -> None:
    """Aggregate all stage JSON profiles into ``summary.json`` and ``summary.md``.

    Reads every ``*.json`` file in *profiles_dir* except ``summary.json``
    itself and writes:

    * ``summary.json`` — list of profile objects plus a ``totals`` entry.
    * ``summary.md``   — Markdown table for human inspection.

    Parameters
    ----------
    profiles_dir:
        Directory containing the per-stage ``.json`` profile files.
    """
    profiles_dir = Path(profiles_dir)
    profile_files = sorted(
        p
        for p in profiles_dir.glob("*.json")
        if p.name not in _SUMMARY_EXCLUDE
    )

    profiles = []
    for pf in profile_files:
        with open(pf, encoding="utf-8") as fh:
            profiles.append(json.load(fh))

    # Totals row
    total_wall = sum(
        p.get("wall_time_s", 0.0) or 0.0 for p in profiles
    )
    gpu_values = [p.get("gpu_peak_mb") for p in profiles if p.get("gpu_peak_mb") is not None]
    total_gpu: Optional[float] = max(gpu_values) if gpu_values else None

    summary: Dict[str, Any] = {
        "schema_version": 1,
        "profiles": profiles,
        "totals": {
            "wall_time_s": total_wall,
            "gpu_peak_mb": total_gpu,
        },
    }

    summary_json_path = profiles_dir / "summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # Markdown table
    headers = [col_label for _, col_label in _MD_COLUMNS]
    sep = ["---" for _ in _MD_COLUMNS]

    rows = []
    for p in profiles:
        row = [_fmt(p.get(key)) for key, _ in _MD_COLUMNS]
        rows.append(row)

    # totals row
    totals_row = [
        "**TOTAL**",
        _fmt(total_wall),
        _fmt(total_gpu),
        "",
        "",
        "",
    ]
    rows.append(totals_row)

    def _md_row(cells: list) -> str:
        return "| " + " | ".join(cells) + " |"

    lines = [
        "# Pipeline Profile Summary",
        "",
        _md_row(headers),
        _md_row(sep),
    ]
    for row in rows:
        lines.append(_md_row(row))
    lines.append("")

    summary_md_path = profiles_dir / "summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

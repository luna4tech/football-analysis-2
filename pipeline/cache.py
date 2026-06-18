"""
pipeline.cache — mtime-based caching helpers.

Design decisions
----------------
* A missing *output* is always considered stale (stage must run).
* A missing *input* is treated as stale — if a required input doesn't exist
  the stage can't safely be skipped anyway, and any downstream stage should
  fail gracefully when it tries to open the file.
* Staleness uses a strict ``>`` comparison: an input whose mtime is *equal* to
  the output's mtime is considered up-to-date (NOT stale). Equal mtimes most
  commonly mean the output was produced from that input in the same operation,
  so re-running would be wasted work.
* Only ``os.path.getmtime`` is used; no hashing, no manifests.

Public API
----------
is_stale(output_path, input_paths) -> bool
should_run(output_path, input_paths, *, force=False) -> bool
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def is_stale(
    output_path: "str | os.PathLike[str]",
    input_paths: "Iterable[str | os.PathLike[str]]",
) -> bool:
    """Return ``True`` if the stage should be (re-)run based on file mtimes.

    A stage is considered stale — and must run — when **any** of the
    following is true:

    * *output_path* does not exist.
    * Any path in *input_paths* does not exist (missing input → cannot safely
      skip; the stage will fail anyway when it tries to open the file).
    * Any input file is **newer** than the output file (mtime comparison).

    Parameters
    ----------
    output_path:
        The file produced by the stage (e.g. ``tracks.txt``).
    input_paths:
        The files the stage reads (e.g. the source video).  May be empty,
        in which case only the existence of *output_path* is checked.

    Returns
    -------
    bool
        ``True``  → stage is stale and must run.
        ``False`` → output is up-to-date; stage may be skipped.
    """
    output_path = Path(output_path)
    if not output_path.exists():
        return True

    output_mtime = os.path.getmtime(output_path)

    for inp in input_paths:
        inp = Path(inp)
        if not inp.exists():
            return True
        if os.path.getmtime(inp) > output_mtime:
            return True

    return False


def should_run(
    output_path: "str | os.PathLike[str]",
    input_paths: "Iterable[str | os.PathLike[str]]",
    *,
    force: bool = False,
) -> bool:
    """Return ``True`` if the stage should execute.

    This is the single entry point all stages use so that caching semantics
    stay consistent.

    Parameters
    ----------
    output_path:
        The primary output file of the stage.
    input_paths:
        Files the stage reads.
    force:
        When ``True``, always return ``True`` regardless of mtimes (i.e.
        the caller is requesting a forced recompute).

    Returns
    -------
    bool
        ``True``  → stage should run (either forced or stale).
        ``False`` → stage may be skipped.
    """
    if force:
        return True
    return is_stale(output_path, input_paths)

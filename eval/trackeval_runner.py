"""
eval.trackeval_runner — TrackEval MOTChallenge layout writer, runner, and
output parser.

This module builds the on-disk directory tree that TrackEval's MOTChallenge
dataset expects, runs TrackEval (HOTA + CLEAR + Identity) over that layout, and
parses the raw result back into a metrics dict.

:func:`run_trackeval` imports ``trackeval`` lazily (inside the function only) so
the rest of this module remains import-light (stdlib + numpy) and CPU-testable
without TrackEval installed.

MOTChallenge layout written by :func:`materialize_layout`
---------------------------------------------------------
For ``benchmark=<BENCH>``, ``split=<SPLIT>``, sequence ``<seq>`` and tracker
``<tracker>`` under ``<work>``::

    <work>/gt/mot_challenge/<BENCH>-<SPLIT>/<seq>/gt/gt.txt
    <work>/gt/mot_challenge/<BENCH>-<SPLIT>/<seq>/seqinfo.ini
    <work>/gt/mot_challenge/seqmaps/<BENCH>-<SPLIT>.txt
    <work>/trackers/mot_challenge/<BENCH>-<SPLIT>/<tracker>/data/<seq>.txt

Frame base (centralized 0->1 conversion)
----------------------------------------
GT is written AS-IS (already 1-based). Predictions are 0-based on disk; the
single +1 shift to 1-based happens here in :func:`_pred_to_motchallenge_rows`
(driven by the prediction's ``frame_base``). This is the ONE place the frame
base is converted, by design — a stray double-shift or missing shift is the #1
silent killer of MOT evaluation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import numpy as np

from .gt_adapters import N_COLS, Tracks

# Default canvas size. NOTE: imWidth/imHeight do NOT affect IoU-based HOTA/MOTA/
# IDF1 (those metrics are scale-invariant over box overlaps); they are written
# only because TrackEval reads seqinfo.ini. Override via the CLI if known.
DEFAULT_IMG_WIDTH = 1920
DEFAULT_IMG_HEIGHT = 1080
DEFAULT_FRAME_RATE = 30


class LayoutPaths:
    """Resolved paths produced by :func:`materialize_layout`."""

    def __init__(
        self,
        work_dir: Path,
        bench_split: str,
        seq_name: str,
        tracker_name: str,
    ) -> None:
        self.work_dir: Path = work_dir
        self.bench_split: str = bench_split
        self.seq_name: str = seq_name
        self.tracker_name: str = tracker_name

        gt_root = work_dir / "gt" / "mot_challenge"
        bs_dir = gt_root / bench_split
        self.seq_dir: Path = bs_dir / seq_name
        self.gt_dir: Path = self.seq_dir / "gt"
        self.gt_txt: Path = self.gt_dir / "gt.txt"
        self.seqinfo_ini: Path = self.seq_dir / "seqinfo.ini"
        self.seqmaps_dir: Path = gt_root / "seqmaps"
        self.seqmap_txt: Path = self.seqmaps_dir / f"{bench_split}.txt"

        trackers_root = work_dir / "trackers" / "mot_challenge"
        self.tracker_dir: Path = trackers_root / bench_split / tracker_name
        self.tracker_data_dir: Path = self.tracker_dir / "data"
        self.tracker_txt: Path = self.tracker_data_dir / f"{seq_name}.txt"

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"LayoutPaths(work_dir={self.work_dir!r}, "
            f"bench_split={self.bench_split!r}, seq={self.seq_name!r}, "
            f"tracker={self.tracker_name!r})"
        )


# ---------------------------------------------------------------------------
# Row formatting
# ---------------------------------------------------------------------------


def _format_gt_line(row: "np.ndarray | list") -> str:
    """Format one canonical GT row to a MOTChallenge ``gt.txt`` CSV line.

    Output columns (the order TrackEval's MotChallenge2DBox reads):
    ``frame,id,x,y,w,h,conf,class,visibility``. ``class`` and ``visibility`` are
    written as the CONSTANT pedestrian values ``1`` and ``1`` — NOT taken from the
    canonical row — so that the materialized ``gt.txt`` stays all-person regardless
    of the semantic ``class_id`` (1/2/3) the extended GT now carries in canonical
    column 7. TrackEval's MOTChallenge pedestrian eval keeps ONLY ``class==1``
    rows; writing the semantic class verbatim would drop every player (2) and
    referee (3) row and silently break HOTA/MOTA. The semantic class/team are
    evaluated separately by :mod:`eval.attributes` straight off the raw txt.

    For the current all-``-1`` sample GT this is identical to the previous
    behaviour (the loader coerced ``-1`` -> ``1``, so it already landed here as
    ``class=1, visibility=1``); for an extended GT it is the fix that keeps HOTA
    all-person.

    ``frame`` and ``id`` are written as integers; the box and ``conf`` keep their
    (float) precision. ``class`` and ``visibility`` are the constant ``1``.
    """
    frame = int(round(float(row[0])))
    tid = int(round(float(row[1])))
    x, y, w, h = (float(v) for v in row[2:6])
    conf = float(row[6])
    return f"{frame},{tid},{x:g},{y:g},{w:g},{h:g},{conf:g},1,1"


def _format_pred_line(row: "np.ndarray | list") -> str:
    """Format one prediction row to a MOTChallenge tracker-file CSV line.

    Output columns: ``frame,id,x,y,w,h,conf,-1,-1,-1``. TrackEval's tracker
    reader uses only indices 0-6 (``frame,id,bbox,conf``=score) and ignores
    class/visibility, so the trailing ``-1,-1,-1`` are left as inert
    placeholders. ``frame`` and ``id`` are written as integers; the box and
    ``conf`` keep their (float) precision.
    """
    frame = int(round(float(row[0])))
    tid = int(round(float(row[1])))
    x, y, w, h = (float(v) for v in row[2:6])
    conf = float(row[6])
    return f"{frame},{tid},{x:g},{y:g},{w:g},{h:g},{conf:g},-1,-1,-1"


def _gt_to_motchallenge_rows(gt: Tracks) -> np.ndarray:
    """Return GT rows for the MOTChallenge ``gt.txt``, frames AS-IS (1-based).

    GT is assumed already 1-based (``frame_base == 1``); we do NOT shift it.
    """
    if gt.frame_base != 1:
        raise ValueError(
            "GT must be 1-based (frame_base==1) before materializing; "
            f"got frame_base={gt.frame_base}. Convert in the GT adapter."
        )
    return gt.rows.copy()


def _pred_to_motchallenge_rows(pred: Tracks) -> np.ndarray:
    """Return prediction rows converted to 1-based frames (the ONLY 0->1 spot).

    The shift applied is ``1 - frame_base`` so a 0-based pred gains ``+1`` and an
    already-1-based pred is left untouched (idempotent / never double-shifted).
    """
    rows = pred.rows.copy()
    shift = 1 - pred.frame_base  # 0-based -> +1 ; 1-based -> +0
    if rows.shape[0]:
        rows[:, 0] = rows[:, 0] + shift
    return rows


# ---------------------------------------------------------------------------
# Layout writer
# ---------------------------------------------------------------------------


def materialize_layout(
    work_dir: "str | os.PathLike[str]",
    seq_name: str,
    gt: Tracks,
    pred: Tracks,
    *,
    tracker_name: str = "refined",
    benchmark: str = "SPORTS",
    split: str = "eval",
    img_width: int = DEFAULT_IMG_WIDTH,
    img_height: int = DEFAULT_IMG_HEIGHT,
    frame_rate: int = DEFAULT_FRAME_RATE,
) -> LayoutPaths:
    """Write the MOTChallenge layout TrackEval expects and return its paths.

    Parameters
    ----------
    work_dir:
        Root under which ``gt/`` and ``trackers/`` trees are created.
    seq_name:
        Sequence name (also the seqmap entry and the tracker data filename).
    gt:
        Canonical GT (must be 1-based; written AS-IS to ``gt/gt.txt``).
    pred:
        Canonical predictions (0-based ``refined.txt``); converted to 1-based
        here and written to ``trackers/.../data/<seq>.txt`` with ``conf=score``.
    tracker_name:
        Tracker subdirectory name (default ``"refined"``).
    benchmark, split:
        Combine into ``<BENCH>-<SPLIT>`` used throughout the layout.
    img_width, img_height, frame_rate:
        Written into ``seqinfo.ini`` (do not affect IoU-based metrics).

    Returns
    -------
    LayoutPaths
        Resolved paths of every file written.
    """
    work_dir = Path(work_dir)
    bench_split = f"{benchmark}-{split}"
    paths = LayoutPaths(work_dir, bench_split, seq_name, tracker_name)

    # Create directories.
    paths.gt_dir.mkdir(parents=True, exist_ok=True)
    paths.seqmaps_dir.mkdir(parents=True, exist_ok=True)
    paths.tracker_data_dir.mkdir(parents=True, exist_ok=True)

    # --- GT gt.txt (AS-IS, 1-based; class=1/vis=1 forced -> all-person) ---
    gt_rows = _gt_to_motchallenge_rows(gt)
    _write_mot_file(paths.gt_txt, gt_rows, _format_gt_line)

    # --- Pred data/<seq>.txt (centralized 0->1 conversion, conf=score) ---
    pred_rows = _pred_to_motchallenge_rows(pred)
    _write_mot_file(paths.tracker_txt, pred_rows, _format_pred_line)

    # --- seqinfo.ini (seqLength = max 1-based frame across GT and conv. pred) ---
    seq_length = _seq_length(gt_rows, pred_rows)
    _write_seqinfo(
        paths.seqinfo_ini,
        seq_name=seq_name,
        seq_length=seq_length,
        img_width=img_width,
        img_height=img_height,
        frame_rate=frame_rate,
    )

    # --- seqmap (<BENCH>-<SPLIT>.txt: header "name" then the single seq) ---
    _write_seqmap(paths.seqmap_txt, seq_name)

    return paths


def _write_mot_file(path: Path, rows: np.ndarray, formatter) -> None:
    """Write *rows* as MOTChallenge CSV lines (one per row) to *path*.

    *formatter* is the per-row line formatter — :func:`_format_gt_line` for the
    GT ``gt.txt`` (constant class=1/visibility=1) or :func:`_format_pred_line` for
    the tracker file (inert ``-1,-1,-1`` trailing columns).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [formatter(r) for r in rows]
    # Trailing newline; empty file (no rows) is still valid.
    text = "\n".join(lines)
    if lines:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _seq_length(gt_rows: np.ndarray, pred_rows: np.ndarray) -> int:
    """Max 1-based frame across GT and converted pred (both 1-based here)."""
    max_frame = 0
    if gt_rows.shape[0]:
        max_frame = max(max_frame, int(gt_rows[:, 0].max()))
    if pred_rows.shape[0]:
        max_frame = max(max_frame, int(pred_rows[:, 0].max()))
    return max_frame


def _write_seqinfo(
    path: Path,
    *,
    seq_name: str,
    seq_length: int,
    img_width: int,
    img_height: int,
    frame_rate: int,
) -> None:
    """Write a MOTChallenge ``seqinfo.ini`` ([Sequence] section)."""
    # imWidth/imHeight are recorded for completeness; IoU-based HOTA/MOTA/IDF1
    # are scale-invariant and do not depend on them.
    content = (
        "[Sequence]\n"
        f"name={seq_name}\n"
        "imDir=img1\n"
        f"frameRate={frame_rate}\n"
        f"seqLength={seq_length}\n"
        f"imWidth={img_width}\n"
        f"imHeight={img_height}\n"
        "imExt=.jpg\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_seqmap(path: Path, seq_name: str) -> None:
    """Write a seqmap: first line ``name``, then the single sequence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"name\n{seq_name}\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# TrackEval output parser
# ---------------------------------------------------------------------------

# Metrics every caller relies on. Keys are the canonical names we return;
# values are TrackEval's column header names (identical here, but kept explicit
# so a header rename in a future TrackEval version is a one-line change).
REQUIRED_METRICS: Dict[str, str] = {
    "HOTA": "HOTA",
    "DetA": "DetA",
    "AssA": "AssA",
    "MOTA": "MOTA",
    "IDF1": "IDF1",
}


def parse_trackeval_output(
    tracker_dir: "str | os.PathLike[str]",
    class_name: str = "pedestrian",
) -> Dict[str, float]:
    """Parse TrackEval's per-tracker text output into a metrics dict.

    Looks for, in order:
      1. ``<tracker_dir>/<class_name>_summary.txt`` — a whitespace-delimited
         header line of metric names followed by a single values line.
      2. ``<tracker_dir>/<class_name>_detailed.csv`` — a header row plus rows
         keyed by sequence; the ``COMBINED`` row is used.

    Returns ``{HOTA, DetA, AssA, MOTA, IDF1}`` as floats. Tolerant of extra
    columns. Raises a clear error if no file is found or a required metric is
    absent.

    Parameters
    ----------
    tracker_dir:
        The tracker directory (``trackers/mot_challenge/<BENCH>-<SPLIT>/<tracker>``).
    class_name:
        Metric class TrackEval reports under (default ``"pedestrian"``; sports
        configs commonly reuse the pedestrian class).
    """
    tracker_dir = Path(tracker_dir)
    summary = tracker_dir / f"{class_name}_summary.txt"
    detailed = tracker_dir / f"{class_name}_detailed.csv"

    if summary.is_file():
        header, values = _read_summary_txt(summary)
        source = summary
    elif detailed.is_file():
        header, values = _read_detailed_csv_combined(detailed)
        source = detailed
    else:
        raise FileNotFoundError(
            "no TrackEval output found for class "
            f"{class_name!r} in {tracker_dir} "
            f"(looked for {summary.name} and {detailed.name})"
        )

    return _extract_metrics(header, values, source)


def _read_summary_txt(path: Path) -> "tuple[list[str], list[str]]":
    """Read a ``*_summary.txt``: a header line then a values line.

    TrackEval prepends the tracker name as the first token of each line; we
    align header and values by length from the right so an extra leading
    label on either line does not misalign the metric columns.
    """
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln]
    if len(lines) < 2:
        raise ValueError(
            f"{path} has too few lines ({len(lines)}); expected a header line "
            "and a values line"
        )
    header = lines[0].split()
    values = lines[1].split()
    return header, values


def _read_detailed_csv_combined(path: Path) -> "tuple[list[str], list[str]]":
    """Read a ``*_detailed.csv`` and return (header, COMBINED-row values).

    The first column is the sequence name; the row whose first cell is
    ``COMBINED`` (case-insensitive) holds the aggregate metrics.
    """
    rows = [
        ln.strip().split(",")
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    if not rows:
        raise ValueError(f"{path} is empty")
    header = [c.strip() for c in rows[0]]
    combined = None
    for r in rows[1:]:
        if r and r[0].strip().upper() == "COMBINED":
            combined = [c.strip() for c in r]
            break
    if combined is None:
        raise ValueError(
            f"{path} has no COMBINED row; rows: "
            f"{[r[0] for r in rows[1:] if r]!r}"
        )
    return header, combined


def _extract_metrics(
    header: list[str], values: list[str], source: Path
) -> Dict[str, float]:
    """Map header->value and pull the required metrics out by name.

    Header and values are aligned from the RIGHT, so a leading tracker-name or
    sequence-name token present on one line but not the other is ignored.
    Tolerant of extra trailing columns. Raises naming any missing metric.
    """
    n = min(len(header), len(values))
    # Align from the right (metric columns are right-justified after any label).
    h_tail = header[-n:]
    v_tail = values[-n:]
    table: Dict[str, str] = {}
    for key, val in zip(h_tail, v_tail):
        table[key] = val

    out: Dict[str, float] = {}
    missing: list[str] = []
    for canonical, col in REQUIRED_METRICS.items():
        if col not in table:
            missing.append(canonical)
            continue
        try:
            out[canonical] = float(table[col])
        except ValueError:
            missing.append(canonical)
    if missing:
        raise KeyError(
            "missing required metric(s) "
            f"{missing} in TrackEval output {source} "
            f"(available columns: {sorted(table)})"
        )
    return out


# ---------------------------------------------------------------------------
# TrackEval run seam (wired in Task E2)
# ---------------------------------------------------------------------------

# Single place where the dataset / benchmark constants are defined so that
# run_trackeval and materialize_layout cannot diverge.  materialize_layout
# accepts these as keyword arguments; run_trackeval reads from the LayoutPaths
# object (which already has bench_split baked in).
_DEFAULT_BENCHMARK = "SPORTS"
_DEFAULT_SPLIT = "eval"
_DEFAULT_CLASS = "pedestrian"


def run_trackeval(
    layout: LayoutPaths,
    *,
    class_name: str = _DEFAULT_CLASS,
    trackeval_path: "str | None" = None,
) -> dict:
    """Run TrackEval over a materialized layout and return the raw result dict.

    This is the thin un-testable seam that imports and invokes ``trackeval``
    (HOTA + CLEAR + Identity) with a sports config (``do_preproc=False``).
    All extraction and reporting live in :func:`extract_metrics` (pure,
    CPU-testable) and :func:`evaluate.main`.

    Parameters
    ----------
    layout:
        Paths produced by :func:`materialize_layout`.  The benchmark and split
        encoded in ``layout.bench_split`` are used automatically.
    class_name:
        Class TrackEval evaluates; default ``"pedestrian"`` (sports configs
        commonly reuse this class label).
    trackeval_path:
        Optional path to a TrackEval git clone to insert on ``sys.path``
        before importing.  Use when TrackEval is not installed as a package.

    Returns
    -------
    dict
        Raw nested result dict from
        ``trackeval.Evaluator(...).evaluate(dataset_list, metrics_list)``.
        Pass it to :func:`extract_metrics` for the headline 5-metric summary.

    Raises
    ------
    ImportError
        If ``trackeval`` cannot be imported (dependency not installed / cloned).
    """
    if trackeval_path is not None:
        import sys as _sys

        tp = str(trackeval_path)
        if tp not in _sys.path:
            _sys.path.insert(0, tp)

    # Lazy import — trackeval is NOT required on this host.
    try:
        import trackeval  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "trackeval not found. Clone TrackEval and pass --trackeval-path, "
            "or install it (pip install trackeval). "
            f"Original error: {exc}"
        ) from exc

    # ---- Evaluator config ----
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["DISPLAY_LESS_PROGRESS"] = True
    eval_config["OUTPUT_DETAILED"] = True  # write *_detailed.csv for fallback
    eval_config["PLOT_CURVES"] = False

    # ---- Dataset config ----
    # Single point where dataset field names live — adjust here if TrackEval
    # renames a field in a future version.
    bench_split_parts = layout.bench_split.split("-", 1)
    benchmark = bench_split_parts[0] if len(bench_split_parts) == 2 else layout.bench_split
    split = bench_split_parts[1] if len(bench_split_parts) == 2 else _DEFAULT_SPLIT

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config["GT_FOLDER"] = str(layout.work_dir / "gt" / "mot_challenge")
    dataset_config["TRACKERS_FOLDER"] = str(
        layout.work_dir / "trackers" / "mot_challenge"
    )
    dataset_config["BENCHMARK"] = benchmark
    dataset_config["SPLIT_TO_EVAL"] = split
    dataset_config["TRACKERS_TO_EVAL"] = [layout.tracker_name]
    dataset_config["CLASSES_TO_EVAL"] = [class_name]
    dataset_config["SEQMAP_FILE"] = str(layout.seqmap_txt)
    # Sports config: disable MOT17 pedestrian-distractor removal.
    dataset_config["DO_PREPROC"] = False
    dataset_config["PRINT_CONFIG"] = False

    # ---- Metrics ----
    metrics_list = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR(),
        trackeval.metrics.Identity(),
    ]

    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    evaluator = trackeval.Evaluator(eval_config)
    result, _ = evaluator.evaluate(dataset_list, metrics_list)
    return result


# ---------------------------------------------------------------------------
# extract_metrics — pure, CPU-testable (Task E2)
# ---------------------------------------------------------------------------

# Field-type classification mirroring TrackEval's metric definitions, used by
# extract_all_metrics to format the FULL dump to match TrackEval's printed table:
# float/rate fields -> percentages (x100), count/integer fields -> raw ints,
# per-alpha float arrays -> mean (x100). Integer-array fields (HOTA_TP/FN/FP) are
# omitted (TrackEval's summary table omits them too). SINGLE place to adjust if a
# TrackEval version renames fields.
_INTEGER_FIELDS = {
    "CLEAR": {"CLR_TP", "CLR_FN", "CLR_FP", "IDSW", "MT", "PT", "ML", "Frag"},
    "Identity": {"IDTP", "IDFN", "IDFP"},
    "Count": {"Dets", "GT_Dets", "IDs", "GT_IDs"},
}
_SKIP_FIELDS = {"HOTA": {"HOTA_TP", "HOTA_FN", "HOTA_FP"}}


def _combined_seq_class(result: dict, tracker_name: str, class_name: str) -> dict:
    """Navigate result -> dataset -> tracker -> COMBINED_SEQ -> class.

    Returns the per-class dict of metric families. Each level raises a labelled
    ``KeyError`` naming exactly where traversal failed (the single Colab 1-line
    fix point if TrackEval renames a key).
    """
    try:
        if len(result) != 1:
            raise KeyError(
                f"expected exactly 1 dataset in result, got {len(result)}: "
                f"{list(result.keys())}"
            )
        dataset_key = next(iter(result))
        by_tracker: dict = result[dataset_key]
    except (AttributeError, TypeError) as exc:
        raise KeyError(f"result has unexpected structure: {exc}") from exc

    if tracker_name not in by_tracker:
        raise KeyError(
            f"tracker {tracker_name!r} not found in result; "
            f"available: {list(by_tracker.keys())}"
        )
    by_seq: dict = by_tracker[tracker_name]

    combined_key = "COMBINED_SEQ"
    if combined_key not in by_seq:
        raise KeyError(
            f"{combined_key!r} not in result[...][{tracker_name!r}]; "
            f"available: {list(by_seq.keys())}"
        )
    by_class: dict = by_seq[combined_key]

    if class_name not in by_class:
        raise KeyError(
            f"class {class_name!r} not in COMBINED_SEQ; "
            f"available: {list(by_class.keys())}"
        )
    return by_class[class_name]


def extract_metrics(
    result: dict,
    *,
    tracker_name: str,
    class_name: str = _DEFAULT_CLASS,
) -> Dict[str, float]:
    """Extract headline metrics from a TrackEval raw result dict.

    The raw result comes from
    ``trackeval.Evaluator(...).evaluate(dataset_list, metrics_list)``
    — a nested dict keyed by dataset name, then tracker, then
    ``"COMBINED_SEQ"``, then class name, then metric FAMILY
    (``"HOTA"`` / ``"CLEAR"`` / ``"Identity"`` / ``"Count"``), each a dict of
    ``metric_name -> value``. So HOTA/DetA/AssA live under the ``"HOTA"`` family,
    MOTA under ``"CLEAR"``, IDF1 under ``"Identity"``.

    GOTCHA — array vs scalar
    ~~~~~~~~~~~~~~~~~~~~~~~~
    TrackEval's HOTA / DetA / AssA are 1-D arrays over IoU alpha thresholds
    (default 19 thresholds from 0.05 to 0.95).  The headline scalar reported
    by the MOTChallenge leaderboard is the **mean over those alphas**
    (``float(np.mean(arr))``).  MOTA (from CLEAR) and IDF1 (from Identity)
    are plain scalars.  This function normalises both shapes and returns floats.

    The result-key path is the SINGLE location to update if a future TrackEval
    version renames a key.

    Parameters
    ----------
    result:
        Raw nested dict from ``Evaluator.evaluate``.
    tracker_name:
        The tracker name used when calling :func:`materialize_layout`.
    class_name:
        Class evaluated; default ``"pedestrian"``.

    Returns
    -------
    dict
        ``{HOTA, DetA, AssA, MOTA, IDF1}`` as floats, expressed as PERCENTAGES
        (0-100) to match TrackEval's printed table and the MOT literature
        (TrackEval stores fractions internally; we multiply by 100).

    Raises
    ------
    KeyError
        If a required key is missing anywhere along the result path.
    """
    metrics_raw = _combined_seq_class(result, tracker_name, class_name)

    # Level 5: metric FAMILY. TrackEval nests per-class results one level deeper,
    # keyed by the metric class that produced them ("HOTA", "CLEAR", "Identity",
    # "Count"); each family is a dict of metric_name -> value.
    def _family(name: str) -> dict:
        if name not in metrics_raw:
            raise KeyError(
                f"metric family {name!r} missing from COMBINED_SEQ[{class_name!r}]; "
                f"available: {sorted(metrics_raw)}"
            )
        fam = metrics_raw[name]
        if not isinstance(fam, dict):
            raise KeyError(
                f"expected metric family {name!r} to be a dict; got {type(fam).__name__}"
            )
        return fam

    def _pct(fam: dict, fam_name: str, key: str) -> float:
        """Metric as a percentage (0-100). HOTA/DetA/AssA are arrays over alpha
        thresholds (mean is the headline); MOTA/IDF1 are scalars. TrackEval stores
        fractions, so we * 100 to match its printed table / the MOT literature."""
        if key not in fam:
            raise KeyError(
                f"metric {key!r} missing from {fam_name!r} family; "
                f"available: {sorted(k for k in fam if not str(k).startswith('_'))}"
            )
        val = fam[key]
        try:
            arr = np.asarray(val, dtype=float)
        except (TypeError, ValueError) as exc:
            raise KeyError(f"cannot convert {fam_name}.{key}={val!r} to float: {exc}") from exc
        scalar = float(arr) if arr.ndim == 0 else float(np.mean(arr))
        return scalar * 100.0

    # SINGLE place mapping our headline metrics to their (family, key).
    hota_fam = _family("HOTA")
    clear_fam = _family("CLEAR")
    id_fam = _family("Identity")
    return {
        "HOTA": _pct(hota_fam, "HOTA", "HOTA"),
        "DetA": _pct(hota_fam, "HOTA", "DetA"),
        "AssA": _pct(hota_fam, "HOTA", "AssA"),
        "MOTA": _pct(clear_fam, "CLEAR", "MOTA"),
        "IDF1": _pct(id_fam, "Identity", "IDF1"),
    }


def extract_all_metrics(
    result: dict,
    *,
    tracker_name: str,
    class_name: str = _DEFAULT_CLASS,
) -> "Dict[str, Dict[str, float]]":
    """Return EVERY metric TrackEval computed, organised by family.

    Unlike :func:`extract_metrics` (the 5 headline values), this dumps all fields
    of all families (``HOTA``, ``CLEAR``, ``Identity``, ``Count``) for writing to
    ``metrics.json``. Formatting matches TrackEval's printed table:

    * float / rate fields  -> percentages (``value * 100``),
    * integer / count fields (see :data:`_INTEGER_FIELDS`) -> raw ints,
    * per-alpha float arrays (HOTA family) -> their mean, then ``* 100``.

    Integer-array fields (``HOTA_TP``/``FN``/``FP``) and private ``_``-keys are
    omitted, mirroring TrackEval's summary. Non-numeric values are skipped.

    Returns ``{family_name: {metric_name: float}}``.
    """
    metrics_raw = _combined_seq_class(result, tracker_name, class_name)
    out: "Dict[str, Dict[str, float]]" = {}
    for family, fields in metrics_raw.items():
        if not isinstance(fields, dict):
            continue
        int_fields = _INTEGER_FIELDS.get(family, set())
        skip = _SKIP_FIELDS.get(family, set())
        fam_out: "Dict[str, float]" = {}
        for key, val in fields.items():
            if str(key).startswith("_") or key in skip:
                continue
            try:
                arr = np.asarray(val, dtype=float)
            except (TypeError, ValueError):
                continue  # non-numeric (e.g. stray strings) — skip
            if arr.ndim >= 1:
                fam_out[key] = float(np.mean(arr)) * 100.0
            elif key in int_fields:
                fam_out[key] = int(round(float(arr)))
            else:
                fam_out[key] = float(arr) * 100.0
        out[family] = fam_out
    return out

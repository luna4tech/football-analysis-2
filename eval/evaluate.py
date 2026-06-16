"""
eval.evaluate — CLI: load GT + predictions, materialize TrackEval layout, run
evaluation, and report metrics.

This is the standalone tracking-evaluation entrypoint, completely separate from
the pipeline: it reads ``refined.txt`` and a GT file from disk post-hoc.

Flow
----
1. Resolve the prediction path (``--pred``, or auto-locate from ``--video`` via
   ``pipeline.artifacts.get_artifact_paths``).
2. ``load_gt`` (GT format seam) -> ``load_pred`` (0-based refined.txt).
3. ``materialize_layout`` writes the MOTChallenge tree TrackEval expects.
4. ``run_trackeval`` (INJECTABLE via the ``_runner`` parameter — default calls the
   real TrackEval; override in tests to avoid the heavy dep) runs evaluation.
5. ``extract_metrics`` pulls HOTA/DetA/AssA/MOTA/IDF1 from the raw result.
6. Print an aligned metrics table; write ``metrics.json``.

Run::

    python -m eval.evaluate --pred path/refined.txt --gt path/gt.txt --seq-name M59
    python -m eval.evaluate --video match.mp4 --gt gt.txt   # auto-locate refined.txt

Import-light: stdlib + numpy only. Does NOT import trackeval at the top level.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, Optional

# Ensure the repo root is importable so ``pipeline.artifacts`` resolves when
# this module is run as a script or via ``-m eval.evaluate`` from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.gt_adapters import GT_LOADERS, load_gt, load_pred
from eval.trackeval_runner import (
    extract_metrics,
    materialize_layout,
    run_trackeval,
)


# ---------------------------------------------------------------------------
# Reporting helpers (pure, CPU-testable)
# ---------------------------------------------------------------------------

_METRIC_KEYS = ("HOTA", "DetA", "AssA", "MOTA", "IDF1")


def format_metrics_table(metrics: Dict[str, float]) -> str:
    """Return an aligned metrics table string.

    Example::

        Metric       Value
        ----------  -------
        HOTA         72.345
        DetA         68.100
        AssA         77.900
        MOTA         65.500
        IDF1         81.250
    """
    lines = [f"{'Metric':<12}{'Value':>8}", "-" * 12 + "  " + "-" * 7]
    for key in _METRIC_KEYS:
        val = metrics.get(key)
        if val is None:
            lines.append(f"{key:<12}{'N/A':>8}")
        else:
            lines.append(f"{key:<12}{val:>8.3f}")
    return "\n".join(lines)


def write_metrics_json(
    out_path: Path,
    seq_name: str,
    tracker_name: str,
    metrics: Dict[str, float],
) -> None:
    """Write ``metrics.json`` with the 5 headline metrics.

    Schema::

        {
          "seq_name": "M59",
          "tracker_name": "refined",
          "metrics": {"HOTA": ..., "DetA": ..., "AssA": ..., "MOTA": ..., "IDF1": ...}
        }
    """
    payload = {
        "seq_name": seq_name,
        "tracker_name": tracker_name,
        "metrics": {k: metrics[k] for k in _METRIC_KEYS if k in metrics},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.evaluate",
        description=(
            "Score the pipeline's refined.txt against ground truth via "
            "TrackEval (HOTA/DetA/AssA/MOTA/IDF1). "
            "Reads refined.txt and a GT file post-hoc; no GPU or pipeline "
            "runtime required."
        ),
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--pred",
        type=str,
        default=None,
        help="path to the pipeline's refined.txt (0-based frames)",
    )
    src.add_argument(
        "--video",
        type=str,
        default=None,
        help=(
            "path to the source video; refined.txt is auto-located via "
            "pipeline.artifacts.get_artifact_paths(video, artifacts_dir)"
        ),
    )

    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=None,
        help="base artifacts dir for --video auto-location (default: artifacts/)",
    )
    parser.add_argument(
        "--gt",
        type=str,
        required=True,
        help="path to the ground-truth file (required)",
    )
    parser.add_argument(
        "--seq-name",
        type=str,
        default=None,
        help="sequence name (default: stem of --video, else of --pred)",
    )
    parser.add_argument(
        "--gt-format",
        type=str,
        default="motchallenge",
        choices=sorted(GT_LOADERS),
        help="ground-truth format (default: motchallenge)",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="dir to materialize the TrackEval layout in (default: a temp dir)",
    )
    parser.add_argument(
        "--tracker-name",
        type=str,
        default="refined",
        help="tracker subdirectory name in the layout (default: refined)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="SPORTS",
        help="MOTChallenge benchmark label (default: SPORTS)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="eval",
        help="MOTChallenge split label (default: eval)",
    )
    parser.add_argument(
        "--img-width",
        type=int,
        default=None,
        help="image width for seqinfo.ini (does not affect IoU-based metrics)",
    )
    parser.add_argument(
        "--img-height",
        type=int,
        default=None,
        help="image height for seqinfo.ini (does not affect IoU-based metrics)",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default="pedestrian",
        help=(
            "class label TrackEval evaluates under (default: pedestrian). "
            "Some sports datasets use 'player' or 'athlete'."
        ),
    )
    parser.add_argument(
        "--trackeval-path",
        type=str,
        default=None,
        help=(
            "path to a TrackEval git clone; inserted on sys.path before "
            "importing trackeval (use when TrackEval is not pip-installed)"
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "path for metrics.json output "
            "(default: same directory as --pred / the auto-located refined.txt)"
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_pred_path(args: argparse.Namespace) -> Path:
    """Return the refined.txt path from --pred or auto-located from --video."""
    if args.pred:
        return Path(args.pred)
    from pipeline.artifacts import get_artifact_paths

    paths = get_artifact_paths(args.video, base_dir=args.artifacts_dir)
    return paths.refined_txt


def _resolve_seq_name(args: argparse.Namespace, pred_path: Path) -> str:
    """Pick the sequence name: --seq-name, else --video stem, else --pred stem."""
    if args.seq_name:
        return args.seq_name
    if args.video:
        return Path(args.video).stem
    return pred_path.stem


def _resolve_work_dir(args: argparse.Namespace) -> Path:
    """Return the work dir, creating a temp one if not provided."""
    if args.work_dir:
        wd = Path(args.work_dir)
        wd.mkdir(parents=True, exist_ok=True)
        return wd
    import tempfile

    return Path(tempfile.mkdtemp(prefix="eval_trackeval_"))


def _resolve_out_path(args: argparse.Namespace, pred_path: Path) -> Path:
    """Resolve the metrics.json output path."""
    if args.out:
        return Path(args.out)
    return pred_path.parent / "metrics.json"


# ---------------------------------------------------------------------------
# Core logic (injectable runner for testing)
# ---------------------------------------------------------------------------


def run_evaluation(
    pred_path: Path,
    gt_path: Path,
    seq_name: str,
    work_dir: Path,
    out_path: Path,
    *,
    gt_format: str = "motchallenge",
    tracker_name: str = "refined",
    benchmark: str = "SPORTS",
    split: str = "eval",
    class_name: str = "pedestrian",
    img_width: Optional[int] = None,
    img_height: Optional[int] = None,
    trackeval_path: Optional[str] = None,
    _runner: Optional[Callable] = None,
) -> Dict[str, float]:
    """Load, materialize, run TrackEval (or a stub), extract metrics, and report.

    The ``_runner`` parameter is injectable so CPU tests can pass a stub without
    triggering the real TrackEval import.  Defaults to :func:`run_trackeval`.

    Parameters
    ----------
    class_name:
        Class label TrackEval evaluates under (default ``"pedestrian"``; some
        sports datasets use ``"player"`` or ``"athlete"``). Forwarded to both
        :func:`run_trackeval` and :func:`extract_metrics`.

    Returns
    -------
    dict
        ``{HOTA, DetA, AssA, MOTA, IDF1}`` as floats.
    """
    if _runner is None:
        _runner = run_trackeval

    print(f"[eval] GT       : {gt_path}  (format={gt_format})")
    print(f"[eval] pred     : {pred_path}")
    print(f"[eval] seq      : {seq_name}")
    print(f"[eval] work-dir : {work_dir}")

    gt = load_gt(gt_path, fmt=gt_format)
    pred = load_pred(pred_path)
    print(
        f"[eval] loaded GT rows={len(gt)} (frame_base={gt.frame_base}), "
        f"pred rows={len(pred)} (frame_base={pred.frame_base})"
    )

    layout_kwargs: dict = {}
    if img_width is not None:
        layout_kwargs["img_width"] = img_width
    if img_height is not None:
        layout_kwargs["img_height"] = img_height

    layout = materialize_layout(
        work_dir,
        seq_name,
        gt,
        pred,
        tracker_name=tracker_name,
        benchmark=benchmark,
        split=split,
        **layout_kwargs,
    )

    print("[eval] materialized TrackEval layout:")
    print(f"         gt.txt   : {layout.gt_txt}")
    print(f"         seqinfo  : {layout.seqinfo_ini}")
    print(f"         seqmap   : {layout.seqmap_txt}")
    print(f"         pred data: {layout.tracker_txt}")

    # ---- Run TrackEval (or injected stub) ----
    runner_kwargs: dict = {"class_name": class_name}
    if trackeval_path is not None:
        runner_kwargs["trackeval_path"] = trackeval_path

    raw_result = _runner(layout, **runner_kwargs)

    # ---- Extract headline metrics ----
    metrics = extract_metrics(raw_result, tracker_name=tracker_name, class_name=class_name)

    # ---- Report ----
    print("\n[eval] Tracking metrics:")
    print(format_metrics_table(metrics))

    write_metrics_json(out_path, seq_name, tracker_name, metrics)
    print(f"\n[eval] metrics.json written to: {out_path}")

    return metrics


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main(argv: "list[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)

    pred_path = _resolve_pred_path(args)
    gt_path = Path(args.gt)
    seq_name = _resolve_seq_name(args, pred_path)
    work_dir = _resolve_work_dir(args)
    out_path = _resolve_out_path(args, pred_path)

    if not pred_path.is_file():
        print(f"ERROR: prediction file not found: {pred_path}", file=sys.stderr)
        return 2
    if not gt_path.is_file():
        print(f"ERROR: ground-truth file not found: {gt_path}", file=sys.stderr)
        return 2

    try:
        run_evaluation(
            pred_path,
            gt_path,
            seq_name,
            work_dir,
            out_path,
            gt_format=args.gt_format,
            tracker_name=args.tracker_name,
            benchmark=args.benchmark,
            split=args.split,
            class_name=args.class_name,
            img_width=args.img_width,
            img_height=args.img_height,
            trackeval_path=args.trackeval_path,
        )
    except ImportError as exc:
        print(
            f"[eval] ERROR: {exc}\n"
            "Install/clone TrackEval and pass --trackeval-path, or run in an "
            "environment where it is pip-installed.",
            file=sys.stderr,
        )
        return 1
    except (KeyError, ValueError) as exc:
        print(
            f"[eval] ERROR: {exc}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

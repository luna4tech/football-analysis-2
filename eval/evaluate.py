"""
eval.evaluate — CLI: load GT + predictions and materialize the TrackEval layout.

This is the standalone tracking-evaluation entrypoint. It is COMPLETELY separate
from the pipeline: it reads ``refined.txt`` and a GT file from disk post-hoc.

Flow
----
1. Resolve the prediction path (``--pred``, or auto-locate from ``--video`` via
   ``pipeline.artifacts.get_artifact_paths``).
2. ``load_gt`` (GT format seam) -> ``load_pred`` (0-based refined.txt).
3. ``materialize_layout`` writes the MOTChallenge tree TrackEval expects.
4. The actual TrackEval run is a single seam (``run_trackeval``) that raises
   ``NotImplementedError("wired in Task E2")`` — Task E2 fills that hole and the
   reporting.

Run::

    python -m eval.evaluate --pred path/refined.txt --gt path/gt.txt --seq-name M59
    python -m eval.evaluate --video match.mp4 --gt gt.txt   # auto-locate refined.txt

Import-light: stdlib + numpy only. Does NOT import trackeval.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the repo root is importable so ``pipeline.artifacts`` resolves when this
# module is run as a script or via ``-m eval.evaluate`` from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.gt_adapters import GT_LOADERS, load_gt, load_pred
from eval.trackeval_runner import materialize_layout, run_trackeval  # noqa: F401


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.evaluate",
        description=(
            "Score the pipeline's refined.txt against ground truth via "
            "TrackEval (HOTA/MOTA/IDF1). Task E1 loads inputs and materializes "
            "the TrackEval layout; the run itself is wired in Task E2."
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
        help="path to the source video; refined.txt is auto-located via "
        "pipeline.artifacts.get_artifact_paths(video, artifacts_dir)",
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
    return parser


def _resolve_pred_path(args: argparse.Namespace) -> Path:
    """Return the refined.txt path from --pred or auto-located from --video."""
    if args.pred:
        return Path(args.pred)
    # --video path: defer the pipeline import until needed.
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


def main(argv: "list[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)

    pred_path = _resolve_pred_path(args)
    gt_path = Path(args.gt)
    seq_name = _resolve_seq_name(args, pred_path)
    work_dir = _resolve_work_dir(args)

    if not pred_path.is_file():
        print(f"ERROR: prediction file not found: {pred_path}", file=sys.stderr)
        return 2
    if not gt_path.is_file():
        print(f"ERROR: ground-truth file not found: {gt_path}", file=sys.stderr)
        return 2

    print(f"[eval] GT      : {gt_path}  (format={args.gt_format})")
    print(f"[eval] pred    : {pred_path}")
    print(f"[eval] seq     : {seq_name}")
    print(f"[eval] work-dir: {work_dir}")

    gt = load_gt(gt_path, fmt=args.gt_format)
    pred = load_pred(pred_path)
    print(f"[eval] loaded GT rows={len(gt)} (frame_base={gt.frame_base}), "
          f"pred rows={len(pred)} (frame_base={pred.frame_base})")

    layout_kwargs = {}
    if args.img_width is not None:
        layout_kwargs["img_width"] = args.img_width
    if args.img_height is not None:
        layout_kwargs["img_height"] = args.img_height

    layout = materialize_layout(
        work_dir,
        seq_name,
        gt,
        pred,
        tracker_name=args.tracker_name,
        benchmark=args.benchmark,
        split=args.split,
        **layout_kwargs,
    )

    print("[eval] materialized TrackEval layout:")
    print(f"         gt.txt   : {layout.gt_txt}")
    print(f"         seqinfo  : {layout.seqinfo_ini}")
    print(f"         seqmap   : {layout.seqmap_txt}")
    print(f"         pred data: {layout.tracker_txt}")

    # ---- single TrackEval-run seam (wired in Task E2) ----
    try:
        metrics = run_trackeval(layout)
        print("[eval] metrics:", metrics)
    except NotImplementedError as exc:
        print(
            f"[eval] TrackEval run not wired yet ({exc}). "
            "Layout is materialized and ready; Task E2 fills the run + report.",
        )
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())

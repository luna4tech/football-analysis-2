"""
pipeline.__main__ — the ``python -m pipeline ...`` CLI.

Subcommands
-----------
run      Chain Stage 1 (tracking) then Stage 2 (refine) as subprocesses over the
         shared artifact contract, with caching/force, and write an aggregated
         profile summary.  Defaults to the FULL pipeline (both --use_split and
         --use_connect enabled); use --no-split / --no-connect to drop one.

compare  Tolerant MOT diff (``compare_tracks``): match rows by (frame, id), then
         report key-set diffs + max/mean abs box-coord diff.  Files are
         "equivalent" iff key sets match and max coord diff <= --tol (default
         1.0).  Used for the parallel==sequential equivalence check.

Caching caveat (re-stated in --help)
------------------------------------
Each stage caches on input-file MTIME, so changing a PARAMETER (a tracking
threshold, eps, merge_dist_thres, …) does NOT bust the cache.  Re-tuning params
requires the matching --force-* flag.

This module is import-light (no torch / cv2) so ``--help`` builds on a CPU box.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

# Allow `python pipeline/__main__.py ...` and `python -m pipeline ...` alike by
# ensuring the repo root is importable.
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.compare_tracks import build_parser as _build_compare_parser
from pipeline.compare_tracks import main as _compare_main
from pipeline.orchestrator import RunOptions, StageError, run

_CACHE_CAVEAT = (
    "CACHE CAVEAT: each stage caches on input-file MTIME, so changing a "
    "PARAMETER (tracking threshold, eps, merge_dist_thres, ...) does NOT bust "
    "the cache. Re-tuning params requires the matching --force-* flag."
)


def _add_run_parser(subparsers: "argparse._SubParsersAction") -> None:
    p = subparsers.add_parser(
        "run",
        help="run the full pipeline (Stage 1 then Stage 2) as subprocesses",
        description=(
            "Chain Stage 1 (DeepEIoU tracking) then Stage 2 (GtaLink refine) as "
            "subprocesses over the shared artifact contract. Defaults to the "
            "FULL pipeline: BOTH --use_split and --use_connect are enabled "
            "unless you pass --no-split / --no-connect (at least one must "
            "remain). " + _CACHE_CAVEAT
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", required=True, help="path to the input video")
    p.add_argument(
        "--artifacts-dir",
        dest="artifacts_dir",
        default=None,
        help="base artifacts dir (default: <repo>/artifacts)",
    )

    # --- Stage 1 ---
    g1 = p.add_argument_group("Stage 1 (tracking)")
    g1.add_argument(
        "--device",
        default="gpu",
        choices=["gpu", "cpu"],
        help="device for Stage 1 detection + ReID",
    )
    g1.add_argument(
        "--detector",
        default="yolov11",
        choices=["yolov11", "yolox"],
        help="Stage 1 detector backend; YOLOv11 uses an Ultralytics checkpoint, "
        "YOLOX uses the original DeepEIoU detector",
    )
    g1.add_argument(
        "--detector-ckpt",
        dest="detector_ckpt",
        default=None,
        help="Stage 1 detector checkpoint. Defaults to checkpoints/yolov11l.pt "
        "for YOLOv11 and checkpoints/best_ckpt.pth.tar for YOLOX.",
    )
    g1.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="opt-in batched detect+ReID + prefetch decode (sequential default)",
    )
    g1.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=8,
        help="frames per batch in --parallel mode",
    )
    g1.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="half-precision detector inference (big speedup on GPU; the main "
        "throughput lever — batching does NOT help a saturated detector)",
    )
    g1.add_argument(
        "--fuse",
        action="store_true",
        default=False,
        help="fuse conv+BN in the detector (small free speedup)",
    )

    # --- Stage 2 ---
    g2 = p.add_argument_group("Stage 2 (refine)")
    g2.add_argument(
        "--no-split",
        dest="use_split",
        action="store_false",
        default=True,
        help="disable the split component (default: enabled)",
    )
    g2.add_argument(
        "--no-connect",
        dest="use_connect",
        action="store_false",
        default=True,
        help="disable the connect/merge component (default: enabled)",
    )
    g2.add_argument("--eps", type=float, default=0.6, help="DBSCAN eps for split")
    g2.add_argument("--min_samples", type=int, default=10, help="DBSCAN min_samples")
    g2.add_argument("--max_k", type=int, default=3, help="max subtracklets per split")
    g2.add_argument("--min_len", type=int, default=100, help="min tracklet len to split")
    g2.add_argument("--spatial_factor", type=float, default=1.0, help="spatial distance factor")
    g2.add_argument(
        "--merge_dist_thres",
        type=float,
        default=0.4,
        help="max cosine distance to merge two tracklets",
    )
    g2.add_argument(
        "--fast-merge",
        dest="fast_merge",
        action="store_true",
        default=False,
        help="exact batched connect/merge (same output, orders of magnitude "
        "faster than the per-pair GPU path); only affects --use_connect",
    )

    # --- Force ---
    gf = p.add_argument_group("force recompute (params alone do NOT bust the cache)")
    gf.add_argument(
        "--force-all",
        dest="force_all",
        action="store_true",
        default=False,
        help="force recompute of BOTH stages",
    )
    gf.add_argument(
        "--force-stage1",
        dest="force_stage1",
        action="store_true",
        default=False,
        help="force recompute of Stage 1 only",
    )
    gf.add_argument(
        "--force-stage2",
        dest="force_stage2",
        action="store_true",
        default=False,
        help="force recompute of Stage 2 only",
    )


def _run_from_args(args: argparse.Namespace) -> int:
    if not args.use_split and not args.use_connect:
        print(
            "error: Stage 2 needs at least one component; do not pass both "
            "--no-split and --no-connect.",
            file=sys.stderr,
        )
        return 2

    detector_ckpt = args.detector_ckpt
    if detector_ckpt is None:
        detector_ckpt = (
            "checkpoints/best_ckpt.pth.tar"
            if args.detector == "yolox"
            else "checkpoints/yolov11l.pt"
        )

    opts = RunOptions(
        device=args.device,
        detector=args.detector,
        detector_ckpt=detector_ckpt,
        parallel=args.parallel,
        batch_size=args.batch_size,
        fp16=args.fp16,
        fuse=args.fuse,
        use_split=args.use_split,
        use_connect=args.use_connect,
        eps=args.eps,
        min_samples=args.min_samples,
        max_k=args.max_k,
        min_len=args.min_len,
        spatial_factor=args.spatial_factor,
        merge_dist_thres=args.merge_dist_thres,
        fast_merge=args.fast_merge,
        force_stage1=args.force_all or args.force_stage1,
        force_stage2=args.force_all or args.force_stage2,
        artifacts_dir=args.artifacts_dir,
    )
    try:
        run(args.video, opts)
    except StageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline",
        description="Two-stage tracking pipeline (DeepEIoU tracking + GtaLink refine).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_run_parser(subparsers)

    # compare subcommand reuses compare_tracks' own parser definition.
    compare_src = _build_compare_parser()
    subparsers.add_parser(
        "compare",
        help="tolerant MOT diff (parallel==sequential equivalence check)",
        parents=[compare_src],
        add_help=False,
        description=compare_src.description,
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return _run_from_args(args)
    if args.command == "compare":
        # Delegate to compare_tracks.main with the parsed flags.
        return _compare_main(["--a", args.a, "--b", args.b, "--tol", str(args.tol)])
    parser.error(f"unknown command: {args.command}")  # pragma: no cover
    return 2  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())

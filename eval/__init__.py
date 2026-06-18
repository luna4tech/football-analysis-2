# eval — standalone tracking-evaluation tool (HOTA / MOTA / IDF1 via TrackEval).
#
# This package is COMPLETELY separate from `pipeline/`: it is never invoked by
# `python -m pipeline run`. It reads the pipeline's on-disk artifacts post-hoc
# (`refined.txt`) plus a ground-truth file and scores them with TrackEval.
#
# Task E1 builds everything EXCEPT the actual TrackEval invocation (the CLI, the
# GT/pred loaders, the 0->1-based frame handling, and the TrackEval on-disk
# layout writer + output parser). The real `trackeval` run is a clearly-marked
# seam (`trackeval_runner.run_trackeval`) wired in Task E2.

# Plan: Standalone tracking evaluation (HOTA / MOTA / IDF1)

## Goal
A standalone evaluation tool that scores the pipeline's `refined.txt` against ground truth using
the standard MOT metrics — **HOTA, MOTA, IDF1** (plus HOTA's DetA/AssA) — via TrackEval. It is
**completely separate from the pipeline**: never invoked by `python -m pipeline run`; it reads the
pipeline's on-disk artifacts post-hoc.

## Key decisions (locked)
- **Metric engine = TrackEval** (the reference impl behind MOTChallenge/SportsMOT/SoccerNet). The
  vendored `motmetrics` can do MOTA/IDF1 but NOT HOTA, so TrackEval is required and covers all three
  consistently.
- **Tracking metrics only.** No detector mAP, no raw-detection dump.
- **Single video / single sequence.** No batch/multi-seq aggregation.
- **Score `refined.txt` only** (the final pipeline output) vs GT. No baseline-vs-refined comparison.
- **GT format = MOTChallenge `gt.txt`** for now (`frame,id,x,y,w,h,conf,class,visibility`). The user's
  own custom-export format is handled LATER by a converter that plugs into a GT-adapter seam.
- **Decoupled from the pipeline**: reads `refined.txt` + a GT `gt.txt` from disk; needs no video
  (seqLength derived from the data). Not wired into the orchestrator.

## Critical correctness details
- **Frame base**: the pipeline emits 0-based frames; MOTChallenge/TrackEval (and the assumed GT) are
  1-based. The pred must be `+1`-shifted when materialized for TrackEval. (Silent killer of MOT eval.)
- **Sports config**: disable MOT17 pedestrian preprocessing/distractor removal (`do_preproc=False`);
  treat GT as a single valid class. SportsMOT/SoccerNet evaluate this way.

## Layout
```
eval/
  evaluate.py          # CLI entrypoint
  gt_adapters.py       # GT loader seam (MOTChallenge now; custom-export converter plugs in later)
  trackeval_runner.py  # build TrackEval MOTChallenge layout, run TrackEval, parse results
  README.md
```

## Tasks
1. **Eval scaffold + MOTChallenge materialization.** CLI (`--pred`/`--gt`/`--seq-name`, optional
   `--video`/`--artifacts-dir` to auto-locate `refined.txt`), `gt_adapters.py` (MOTChallenge loader +
   converter seam), pred loader + 0->1 frame conversion, the TrackEval on-disk layout writer
   (gt/gt.txt, seqinfo.ini, seqmap, trackers/.../data/<seq>.txt), and a result parser for TrackEval's
   output. Fully CPU-testable here (frame conversion, layout correctness, parsing a sample TrackEval
   summary) with no TrackEval/torch dependency.
2. **TrackEval integration + reporting.** Wire the actual TrackEval run (HOTA + CLEAR + Identity,
   sports config), print the metrics table, write `metrics.json`, and `eval/README.md`. The real run
   is deferred to Colab (TrackEval + scipy absent on this host).

## Verification model
- Agents (this machine): CPU tests for the pure pieces (frame conversion, layout writer, GT/pred
  loaders, result parser on a captured sample).
- User (Colab): the real TrackEval run on `refined.txt` + GT, producing the metrics.

## Non-goals (YAGNI)
No mAP/detector eval. No multi-sequence batch. No baseline-vs-refined. Not part of the pipeline /
orchestrator. No handling of the user's custom GT export yet (converter seam only).

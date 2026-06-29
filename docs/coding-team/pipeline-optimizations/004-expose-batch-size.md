# Task 004 — Expose `--batch-size` through the orchestrator

## Context
OPT-2 added `--batch-size` to `Deep-EIoU/Deep-EIoU/tools/stage1_track.py` (perception batch
window, default 16), but it is NOT reachable from `python -m pipeline run` — the orchestrator
always uses the stage default. Thread it through so users can tune it from the main entrypoint.
(`--keep-features` stays stage-only — do NOT expose it.)

## Objective
`python -m pipeline run --batch-size N` forwards `--batch-size N` to Stage 1. Default 16 →
behavior unchanged when the flag is omitted.

## Scope
- `pipeline/orchestrator.py` — `RunOptions` + `build_stage1_command`.
- `pipeline/__main__.py` — Stage 1 CLI arg + `RunOptions` construction in `_run_from_args`.
- `pipeline/tests/test_orchestrator.py` (and any other test asserting the Stage 1 argv).
- `USER_GUIDE.md` — move `--batch-size` from the stage-only table to the orchestrator table.

## Implementation
1. **`RunOptions.__init__`**: add keyword-only `batch_size: int = 16`; set `self.batch_size`.
   Add a one-line mention under the Stage 1 section of the class docstring.
2. **`build_stage1_command`**: append `["--batch-size", str(opts.batch_size)]` — pass it ALWAYS
   (same pattern as `--device` / `--detector-ckpt`, which are always passed; the value equals the
   stage default when unset, so it's harmless and keeps the command explicit).
3. **`pipeline/__main__.py`**: in `_add_run_parser`, add to the Stage 1 group `g1`:
   `--batch-size` (`dest="batch_size"`, `type=int`, `default=16`, help describing the perception
   batch window). In `_run_from_args`, pass `batch_size=args.batch_size` into `RunOptions(...)`.
4. **Docs (`USER_GUIDE.md`)**: move the `--batch-size` row from the "Stage-script-only flags"
   table (§B4) into the orchestrator flag table; leave `--keep-features` in the stage-only table.
   Update the §B9 GPU-OOM troubleshooting line ("run a direct Stage 1 with a smaller
   `--batch-size`") to note it can now be passed to `python -m pipeline run --batch-size` directly.

## Caveat (the main risk)
Adding an always-passed arg changes the Stage 1 argv. **Find and update any existing test that
asserts the exact `build_stage1_command` output** (likely in `test_orchestrator.py`) — an
`assert cmd == [...]` will now include `--batch-size 16`. Add/extend a test asserting the flag is
present with the default and with a configured value.

## Non-goals
- Do NOT expose or touch `--keep-features`.
- Do NOT change `stage1_track.py` (it already has the flag) or any default behavior.

## Acceptance criteria
- `python -m pipeline run --batch-size 8` forwards `--batch-size 8` to Stage 1; omitting it
  forwards `16`.
- `build_stage1_command` includes `--batch-size <value>`; tests cover default + custom.
- Full suite green (`python -m pytest pipeline/tests -q`).
- USER_GUIDE.md lists `--batch-size` as an orchestrator flag.

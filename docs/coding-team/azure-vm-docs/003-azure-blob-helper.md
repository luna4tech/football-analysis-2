# Task 003 — Small Azure Blob helper script (no `az login`)

## Context
The §A4 "fresh GPU VM / Azure" flow currently uses `az storage blob` commands. The owner finds the
`az` auth flow fiddly. Since we use **connection-string** auth (just an env var), a tiny Python
helper using `azure-storage-blob` is simpler — no `az login`. Add it and point the guide at it.

## Objective
A standalone `azure_blob.py` (repo root) that mirrors the four blob ops the workflow needs, using
`AZURE_STORAGE_CONNECTION_STRING`. Then update USER_GUIDE §A4 to use it.

## The script — `azure_blob.py` (repo root)
- Auth: connection string from `os.environ["AZURE_STORAGE_CONNECTION_STRING"]`. If unset, exit with
  a clear message. No `az login`, no SAS, no other auth modes.
- **Lazy import** `azure.storage.blob` *inside* the command handlers (after argparse) with a clear
  "pip install azure-storage-blob" error if missing — so `python azure_blob.py --help` works on a
  box without the package.
- argparse subcommands (mirror `az storage blob download / download-batch / upload-batch`):
  - `download   <container> <blob>   <dest_file>`
  - `upload     <container> <blob>   <src_file>`
  - `download-dir <container> <prefix> <dest_dir>`   — downloads every blob under `<prefix>` into
    `<dest_dir>`, recreating the relative tree. Empty `prefix` ("") = whole container. Skip the
    zero-length "directory placeholder" blob names.
  - `upload-dir   <container> <prefix> <src_dir>`    — uploads every file under `<src_dir>` to
    `<prefix>/<relpath>` (posix-style blob names), `overwrite=True`.
- Keep the transfer functions at module level and **dependency-injectable** so they're testable
  without the azure package: e.g. `download_dir(container_client, prefix, dest_dir)` /
  `upload_dir(container_client, prefix, src_dir)` take a client-like object (with `list_blobs`,
  `download_blob`, `upload_blob`); a thin `main(argv)` builds the real client and dispatches.
- Print a one-line progress per file and a final count. Pure stdlib + (lazy) azure SDK only.

## Test — `pipeline/tests/test_azure_blob.py`
CPU-only, **no azure import / no network** — pass a FAKE container client (a small stub recording
`upload_blob` calls and serving canned `list_blobs`/`download_blob`). Cover:
- `upload-dir`: `src/a/b.txt` + prefix `outputs/clip` → blob name `outputs/clip/a/b.txt`; empty
  prefix → `a/b.txt`.
- `download-dir`: blob `outputs/clip/a/b.txt` + prefix `outputs/clip` → writes `dest/a/b.txt`
  (prefix stripped); placeholder/zero-rel entries skipped.
- single `download`/`upload` call the expected client methods with the right args.
Wire it into the suite the same way the sibling tests are (the repo's test files are
self-contained scripts that pytest also collects).

## Docs — USER_GUIDE §A4
Replace the `az storage blob …` block with the `azure_blob.py` equivalents, keeping the
`export AZURE_STORAGE_CONNECTION_STRING=…` line. Example shape:
```bash
python azure_blob.py download     videos clip.mp4 ./clip.mp4
python azure_blob.py download-dir  checkpoints "" Deep-EIoU/Deep-EIoU/checkpoints
python -m pipeline run --video ./clip.mp4 --artifacts-dir ./out --device gpu --fp16 --fuse
python azure_blob.py upload-dir    outputs <stem> ./out/<stem>
```
Add ONE line noting `az storage blob` / `azcopy` remain alternatives (azcopy for very large
transfers, via SAS). Do not expand into a full alternatives essay.

## Non-goals
- No SAS/AAD/`az login` support; connection string only.
- Don't add `azure-storage-blob` to any requirements file (it's a VM-side convenience; document the
  `pip install azure-storage-blob` need in §A4 or the script error).
- Don't touch pipeline/eval code.

## Acceptance criteria
- `python azure_blob.py --help` works WITHOUT `azure-storage-blob` installed.
- The four subcommands behave as specified; missing env var / missing package give clear errors.
- `python -m pytest pipeline/tests -q` stays green incl. the new test (which imports no azure).
- USER_GUIDE §A4 uses `azure_blob.py`.

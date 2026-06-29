"""
azure_blob.py — a tiny Azure Blob Storage CLI for the §A4 GPU-VM workflow.

Mirrors the four blob ops the workflow needs (single download/upload + recursive
download-dir/upload-dir) using ``azure-storage-blob`` with **connection-string
auth only** (the ``AZURE_STORAGE_CONNECTION_STRING`` env var).  No ``az login``,
no SAS, no AAD — just one env var.

Why this exists
---------------
The §A4 flow used ``az storage blob ...``; the ``az`` auth flow is fiddly.  Since
we already authenticate by connection string, this self-contained helper is
simpler on a fresh VM.  It is a VM-side convenience, NOT a pipeline dependency:
``azure-storage-blob`` is intentionally absent from the requirements files —
``pip install azure-storage-blob`` on the box that needs it.

The ``azure.storage.blob`` import is **lazy** (inside the command handlers, after
argparse), so ``python azure_blob.py --help`` works on a box without the package.

The transfer functions (``download_file`` / ``upload_file`` / ``download_dir`` /
``upload_dir``) are module-level and take a *client-like* object (anything with
``list_blobs`` / ``download_blob`` / ``upload_blob``), so they are unit-testable
without the azure SDK (see ``pipeline/tests/test_azure_blob.py``).

Run with:
    export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;..."
    python azure_blob.py download     videos clip.mp4 ./clip.mp4
    python azure_blob.py upload        outputs clip/refined.txt ./out/clip/refined.txt
    python azure_blob.py download-dir  checkpoints "" Deep-EIoU/Deep-EIoU/checkpoints
    python azure_blob.py upload-dir    outputs clip ./out/clip
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path, PurePosixPath
from typing import List, Optional

_CONN_STR_ENV = "AZURE_STORAGE_CONNECTION_STRING"


# ---------------------------------------------------------------------------
# Module-level, dependency-injectable transfer functions.
#
# Each takes a *container client* — any object exposing ``list_blobs`` /
# ``download_blob`` / ``upload_blob`` like azure's ``ContainerClient`` — so the
# transfer logic is testable with a fake stub and no azure SDK / network.
# ---------------------------------------------------------------------------
def download_file(container_client, blob: str, dest_file: str) -> None:
    """Download a single ``blob`` to ``dest_file`` (parent dirs created)."""
    dest = Path(dest_file)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"download {blob} -> {dest}")
    with open(dest, "wb") as fh:
        container_client.download_blob(blob).readinto(fh)


def upload_file(container_client, blob: str, src_file: str) -> None:
    """Upload a single ``src_file`` to ``blob`` (overwrite=True)."""
    src = Path(src_file)
    print(f"upload {src} -> {blob}")
    with open(src, "rb") as fh:
        container_client.upload_blob(name=blob, data=fh, overwrite=True)


def download_dir(container_client, prefix: str, dest_dir: str) -> int:
    """Download every blob under ``prefix`` into ``dest_dir``.

    Recreates the relative tree under ``dest_dir`` (the ``prefix`` itself is
    stripped from each blob name).  An empty ``prefix`` ("") means the whole
    container.  Zero-length "directory placeholder" blobs (names ending in "/",
    or whose relative path is empty) are skipped.  Returns the file count.
    """
    dest_root = Path(dest_dir)
    # Normalise the prefix to a posix path with no trailing slash for stripping.
    prefix_norm = prefix.strip("/")
    count = 0
    for blob in container_client.list_blobs(name_starts_with=prefix):
        name = _blob_name(blob)
        if name.endswith("/"):
            continue  # directory placeholder
        rel = _strip_prefix(name, prefix_norm)
        if not rel:
            continue  # placeholder / the prefix entry itself
        dest = dest_root / Path(*PurePosixPath(rel).parts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"download {name} -> {dest}")
        with open(dest, "wb") as fh:
            container_client.download_blob(name).readinto(fh)
        count += 1
    print(f"downloaded {count} file(s) from prefix {prefix!r}")
    return count


def upload_dir(container_client, prefix: str, src_dir: str) -> int:
    """Upload every file under ``src_dir`` to ``<prefix>/<relpath>``.

    Blob names are posix-style; ``overwrite=True``.  An empty ``prefix`` uploads
    each file under its bare relative path.  Returns the file count.
    """
    src_root = Path(src_dir)
    prefix_norm = prefix.strip("/")
    count = 0
    for path in sorted(p for p in src_root.rglob("*") if p.is_file()):
        rel = path.relative_to(src_root).as_posix()
        blob = f"{prefix_norm}/{rel}" if prefix_norm else rel
        print(f"upload {path} -> {blob}")
        with open(path, "rb") as fh:
            container_client.upload_blob(name=blob, data=fh, overwrite=True)
        count += 1
    print(f"uploaded {count} file(s) to prefix {prefix!r}")
    return count


def _blob_name(blob) -> str:
    """Blob name from either a string or an azure BlobProperties-like object."""
    if isinstance(blob, str):
        return blob
    return blob.name


def _strip_prefix(name: str, prefix_norm: str) -> str:
    """Strip ``prefix_norm`` (+ a leading slash) from a posix blob ``name``."""
    if not prefix_norm:
        return name.lstrip("/")
    if name == prefix_norm:
        return ""
    if name.startswith(prefix_norm + "/"):
        return name[len(prefix_norm) + 1:]
    return name


# ---------------------------------------------------------------------------
# Lazy azure-SDK client construction (kept out of import path for --help).
# ---------------------------------------------------------------------------
def _container_client(container: str):
    """Build a ContainerClient from the connection-string env var.

    The ``azure.storage.blob`` import is lazy so ``--help`` works without the
    package.  Raises ``SystemExit`` with a clear message if the package is
    missing or the env var is unset.
    """
    conn_str = os.environ.get(_CONN_STR_ENV)
    if not conn_str:
        raise SystemExit(
            f"error: {_CONN_STR_ENV} is not set. Export your Blob connection "
            "string first, e.g.:\n"
            f'  export {_CONN_STR_ENV}="DefaultEndpointsProtocol=https;'
            'AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"'
        )
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        raise SystemExit(
            "error: azure-storage-blob is not installed. Install it with:\n"
            "  pip install azure-storage-blob"
        )
    service = BlobServiceClient.from_connection_string(conn_str)
    return service.get_container_client(container)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python azure_blob.py",
        description=(
            "Tiny Azure Blob Storage helper (connection-string auth only via "
            f"${_CONN_STR_ENV}). Needs `pip install azure-storage-blob` on the "
            "box that runs a transfer."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_dl = subparsers.add_parser("download", help="download a single blob to a file")
    p_dl.add_argument("container", help="container name")
    p_dl.add_argument("blob", help="blob name (path within the container)")
    p_dl.add_argument("dest_file", help="local destination file path")

    p_up = subparsers.add_parser("upload", help="upload a single file to a blob")
    p_up.add_argument("container", help="container name")
    p_up.add_argument("blob", help="blob name (path within the container)")
    p_up.add_argument("src_file", help="local source file path")

    p_dld = subparsers.add_parser(
        "download-dir",
        help="download every blob under a prefix, recreating the tree",
    )
    p_dld.add_argument("container", help="container name")
    p_dld.add_argument("prefix", help='blob name prefix ("" = whole container)')
    p_dld.add_argument("dest_dir", help="local destination directory")

    p_upd = subparsers.add_parser(
        "upload-dir",
        help="upload every file under a directory to <prefix>/<relpath>",
    )
    p_upd.add_argument("container", help="container name")
    p_upd.add_argument("prefix", help='blob name prefix ("" = bare relative paths)')
    p_upd.add_argument("src_dir", help="local source directory")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "download":
        download_file(_container_client(args.container), args.blob, args.dest_file)
    elif args.command == "upload":
        upload_file(_container_client(args.container), args.blob, args.src_file)
    elif args.command == "download-dir":
        download_dir(_container_client(args.container), args.prefix, args.dest_dir)
    elif args.command == "upload-dir":
        upload_dir(_container_client(args.container), args.prefix, args.src_dir)
    else:  # pragma: no cover — argparse(required=True) rejects anything else.
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

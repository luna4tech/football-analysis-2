"""
CPU-only unit tests for azure_blob.py (the §A4 Blob helper).

NO azure import, NO network.  A small FAKE container-client stub records
``upload_blob`` calls and serves canned ``list_blobs`` / ``download_blob``
responses, so the transfer functions are exercised entirely in-process.

Covers:
  * upload_dir: src/a/b.txt + prefix "outputs/clip" -> blob "outputs/clip/a/b.txt";
    empty prefix -> "a/b.txt".
  * download_dir: blob "outputs/clip/a/b.txt" + prefix "outputs/clip" -> writes
    dest/a/b.txt (prefix stripped); placeholder / zero-rel entries skipped.
  * single download / upload call the expected client methods with the right args.

Run with:
    python pipeline/tests/test_azure_blob.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Repo root on sys.path so `import azure_blob` (a repo-root module) works —
# same trick the sibling tests use to reach `pipeline`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import azure_blob  # noqa: E402  (must follow the sys.path insert)


# ---------------------------------------------------------------------------
# Minimal test runner (matches the other pipeline tests' style; no pytest)
# ---------------------------------------------------------------------------
_FAILURES: list[str] = []
_PASSED: int = 0


def _pass(name: str) -> None:
    global _PASSED
    _PASSED += 1
    print(f"  PASS  {name}")


def _fail(name: str, msg: str) -> None:
    _FAILURES.append(f"{name}: {msg}")
    print(f"  FAIL  {name}: {msg}")


def run_test(name: str, fn) -> None:
    try:
        fn()
        _pass(name)
    except AssertionError as exc:
        _fail(name, str(exc) or "AssertionError")
    except Exception as exc:  # noqa: BLE001
        _fail(name, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Fake container client + blob stubs (no azure SDK).
# ---------------------------------------------------------------------------
class _FakeBlobProps:
    """Stand-in for azure BlobProperties — only ``.name`` is used."""

    def __init__(self, name: str):
        self.name = name


class _FakeDownloader:
    """Stand-in for azure StorageStreamDownloader — only ``readinto`` is used."""

    def __init__(self, data: bytes):
        self._data = data

    def readinto(self, fh) -> int:
        return fh.write(self._data)


class FakeContainerClient:
    """Records upload_blob calls; serves canned list_blobs / download_blob.

    ``blobs`` maps blob-name -> bytes (the canned download payloads); their keys
    are also what ``list_blobs`` enumerates (filtered by ``name_starts_with``).
    """

    def __init__(self, blobs=None):
        self.blobs = dict(blobs or {})
        self.uploads = []  # list of (name, data_bytes, overwrite)
        self.downloaded = []  # list of requested blob names

    def list_blobs(self, name_starts_with=""):
        prefix = name_starts_with or ""
        for name in sorted(self.blobs):
            if name.startswith(prefix):
                yield _FakeBlobProps(name)

    def download_blob(self, name):
        self.downloaded.append(name)
        return _FakeDownloader(self.blobs[name])

    def upload_blob(self, name, data, overwrite=False):
        self.uploads.append((name, data.read(), overwrite))


def _write(path: Path, content: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


# ===========================================================================
# upload-dir
# ===========================================================================
def test_upload_dir_with_prefix():
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        _write(src / "a" / "b.txt", b"hello")
        client = FakeContainerClient()
        n = azure_blob.upload_dir(client, "outputs/clip", str(src))
        assert n == 1, n
        names = [u[0] for u in client.uploads]
        assert names == ["outputs/clip/a/b.txt"], names
        # data + overwrite forwarded correctly.
        assert client.uploads[0][1] == b"hello", client.uploads[0]
        assert client.uploads[0][2] is True, client.uploads[0]


def test_upload_dir_empty_prefix_uses_bare_relpath():
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        _write(src / "a" / "b.txt", b"hi")
        client = FakeContainerClient()
        n = azure_blob.upload_dir(client, "", str(src))
        assert n == 1, n
        assert [u[0] for u in client.uploads] == ["a/b.txt"], client.uploads


def test_upload_dir_multiple_files_posix_names():
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        _write(src / "top.txt")
        _write(src / "deep" / "nested" / "x.bin")
        client = FakeContainerClient()
        n = azure_blob.upload_dir(client, "p", str(src))
        assert n == 2, n
        names = sorted(u[0] for u in client.uploads)
        assert names == ["p/deep/nested/x.bin", "p/top.txt"], names


# ===========================================================================
# download-dir
# ===========================================================================
def test_download_dir_strips_prefix():
    with tempfile.TemporaryDirectory() as td:
        dest = Path(td) / "dest"
        client = FakeContainerClient({"outputs/clip/a/b.txt": b"data"})
        n = azure_blob.download_dir(client, "outputs/clip", str(dest))
        assert n == 1, n
        out = dest / "a" / "b.txt"
        assert out.is_file(), "prefix must be stripped -> dest/a/b.txt"
        assert out.read_bytes() == b"data", out.read_bytes()


def test_download_dir_skips_placeholders():
    with tempfile.TemporaryDirectory() as td:
        dest = Path(td) / "dest"
        client = FakeContainerClient({
            "outputs/clip/": b"",            # directory placeholder (trailing /)
            "outputs/clip": b"",             # the prefix entry itself -> zero rel
            "outputs/clip/a/b.txt": b"keep",
        })
        n = azure_blob.download_dir(client, "outputs/clip", str(dest))
        assert n == 1, "only the real file should be downloaded"
        assert (dest / "a" / "b.txt").read_bytes() == b"keep"
        # placeholders must not have been fetched.
        assert client.downloaded == ["outputs/clip/a/b.txt"], client.downloaded


def test_download_dir_empty_prefix_whole_container():
    with tempfile.TemporaryDirectory() as td:
        dest = Path(td) / "dest"
        client = FakeContainerClient({"a/b.txt": b"1", "c.txt": b"2"})
        n = azure_blob.download_dir(client, "", str(dest))
        assert n == 2, n
        assert (dest / "a" / "b.txt").read_bytes() == b"1"
        assert (dest / "c.txt").read_bytes() == b"2"


# ===========================================================================
# single download / upload
# ===========================================================================
def test_download_file_calls_client_and_writes():
    with tempfile.TemporaryDirectory() as td:
        dest = Path(td) / "nested" / "clip.mp4"
        client = FakeContainerClient({"clip.mp4": b"video-bytes"})
        azure_blob.download_file(client, "clip.mp4", str(dest))
        assert client.downloaded == ["clip.mp4"], client.downloaded
        assert dest.read_bytes() == b"video-bytes", dest.read_bytes()


def test_upload_file_calls_client_with_overwrite():
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "clip.mp4"
        src.write_bytes(b"payload")
        client = FakeContainerClient()
        azure_blob.upload_file(client, "videos/clip.mp4", str(src))
        assert len(client.uploads) == 1, client.uploads
        name, data, overwrite = client.uploads[0]
        assert name == "videos/clip.mp4", name
        assert data == b"payload", data
        assert overwrite is True, overwrite


_TESTS = [
    ("upload-dir: prefix -> outputs/clip/a/b.txt", test_upload_dir_with_prefix),
    ("upload-dir: empty prefix -> a/b.txt", test_upload_dir_empty_prefix_uses_bare_relpath),
    ("upload-dir: multiple files, posix blob names", test_upload_dir_multiple_files_posix_names),
    ("download-dir: prefix stripped -> dest/a/b.txt", test_download_dir_strips_prefix),
    ("download-dir: placeholder/zero-rel skipped", test_download_dir_skips_placeholders),
    ("download-dir: empty prefix = whole container", test_download_dir_empty_prefix_whole_container),
    ("download: calls client + writes file", test_download_file_calls_client_and_writes),
    ("upload: calls client with overwrite=True", test_upload_file_calls_client_with_overwrite),
]


def main() -> int:
    print("=" * 60)
    print("azure_blob helper unit tests (CPU-only, no azure import)")
    print("=" * 60)
    for test_name, fn in _TESTS:
        run_test(test_name, fn)
    print("=" * 60)
    print(f"Results: {_PASSED} passed, {len(_FAILURES)} failed")
    if _FAILURES:
        print("\nFailed tests:")
        for f in _FAILURES:
            print(f"  {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

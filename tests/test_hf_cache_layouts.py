"""The Hugging Face cache in both layouts: per-repository blobs (huggingface_hub 0.x) and the cache-wide
shared blob store (huggingface_hub 1.x), where `hub/<repo>/blobs/<etag>` is a symlink to
`hub/blobs/<xx>/<hash>`. Measurement and model removal must count a shared payload once, attribute it
to the repositories that link to it, and never delete one another repository still links to."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

from kura.paths import hf_cached_file

from tests.platform_support import posix_only
from tests.test_reclaim import _cleanup, _items, _run, _workspace, _write


COMMIT = "c0ffee"
REF = len(COMMIT)  # each repository's refs/main, a regular file it holds besides its blobs


def _store(hub: Path) -> Path:
    store = hub / "blobs"
    _write(store / ".huggingface-shared-blobs", "1\n")
    return store


def _payload(hub: Path, digit: str, size: int) -> Path:
    """A shared-store payload, as huggingface_hub 1.x publishes it (`blobs/<xx>/<hash>` plus a `.refs` manifest)."""
    digest = digit * 64
    payload = _write(_store(hub) / digest[:2] / digest, digit.encode() * size)
    _write(payload.with_name(f"{digest}.refs"), "")
    _write(payload.with_name(f"{digest}.lock"), "")
    return payload


def _link_shared(hub: Path, repo: str, filename: str, payload: Path, etag: str, kind: str = "models") -> Path:
    """A repository file in the shared layout: snapshot -> ../../blobs/<etag> -> ../../blobs/<xx>/<hash>."""
    repo_dir = hub / f"{kind}--{repo.replace('/', '--')}"
    blob = repo_dir / "blobs" / etag
    blob.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(os.path.relpath(payload, blob.parent), blob)
    refs = payload.with_name(f"{payload.name}.refs")
    refs.write_text(refs.read_text() + f"{blob.relative_to(hub).as_posix()}\n")
    snapshot = repo_dir / "snapshots" / COMMIT / filename
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(os.path.relpath(blob, snapshot.parent), snapshot)
    _write(repo_dir / "refs" / "main", COMMIT)
    return snapshot


def _classic(hub: Path, repo: str, filename: str, size: int, etag: str = "e" * 40) -> Path:
    """A repository file in the per-repository layout: snapshot -> ../../blobs/<etag>, a regular file."""
    repo_dir = hub / f"models--{repo.replace('/', '--')}"
    blob = _write(repo_dir / "blobs" / etag, b"c" * size)
    snapshot = repo_dir / "snapshots" / COMMIT / filename
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(os.path.relpath(blob, snapshot.parent), snapshot)
    _write(repo_dir / "refs" / "main", COMMIT)
    return snapshot


SYMLINKS = "the Hugging Face cache layouts are symlinks, which native Windows creates only in Developer Mode"


def _repos(root: Path) -> dict[str, dict]:
    return {item["id"]: item for item in _items(root) if item["kind"] == "hf-repository"}


@posix_only(SYMLINKS)
class SharedLayoutMeasurementTests(unittest.TestCase):
    def test_a_repository_in_the_shared_layout_holds_the_bytes_of_its_payloads(self) -> None:
        with _workspace() as root:
            hub = root / "cache" / "huggingface" / "hub"
            _link_shared(hub, "org/shared", "w.safetensors", _payload(hub, "a", 5000), "1" * 64)
            items = _repos(root)
        self.assertNotIn("blobs", {item["id"] for item in items.values()})
        self.assertEqual(items["org/shared"]["size_bytes"], 5000 + REF)

    def test_a_mixed_cache_counts_a_shared_payload_once_and_names_who_shares_it(self) -> None:
        with _workspace() as root:
            hub = root / "cache" / "huggingface" / "hub"
            _classic(hub, "org/classic", "c.safetensors", 700)
            only_b = _payload(hub, "b", 3000)
            common = _payload(hub, "c", 400)
            _link_shared(hub, "org/b", "w.safetensors", only_b, "2" * 64)
            _link_shared(hub, "org/b", "ae.safetensors", common, "3" * 64)
            _link_shared(hub, "org/c", "ae.safetensors", common, "3" * 64)
            items = _repos(root)
        self.assertEqual(items["org/classic"]["size_bytes"], 700 + REF)
        # What deleting it alone frees: its own payload, not the one org/c also links to.
        self.assertEqual(items["org/b"]["size_bytes"], 3000 + REF)
        self.assertEqual(items["org/b"]["shared_bytes"], 400)
        self.assertEqual(items["org/b"]["shared_with"], ["org/c"])
        self.assertEqual(items["org/c"]["size_bytes"], REF)
        self.assertEqual(items["org/c"]["shared_with"], ["org/b"])
        self.assertNotIn("shared_bytes", items["org/classic"])

    def test_a_cached_file_is_found_and_sized_through_the_shared_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory).resolve()
            _link_shared(cache / "hub", "org/shared", "w.safetensors", _payload(cache / "hub", "a", 1234), "1" * 64)
            found = hf_cached_file(cache, "org/shared", "w.safetensors")
            self.assertIsNotNone(found)
            self.assertEqual(found.stat().st_size, 1234)


@posix_only(SYMLINKS)
class SharedLayoutRemovalTests(unittest.TestCase):
    def test_removing_a_model_frees_its_own_payloads_and_keeps_one_another_model_links_to(self) -> None:
        with _workspace() as root:
            hub = root / "cache" / "huggingface" / "hub"
            classic = _classic(hub, "org/classic", "c.safetensors", 700)
            only_b = _payload(hub, "b", 3000)
            common = _payload(hub, "c", 400)
            _link_shared(hub, "org/b", "w.safetensors", only_b, "2" * 64)
            _link_shared(hub, "org/b", "ae.safetensors", common, "3" * 64)
            c_file = _link_shared(hub, "org/c", "ae.safetensors", common, "3" * 64)
            code, out, err = _cleanup(["--models", "org/b"])
            self.assertEqual(code, 0, err)
            self.assertEqual(json.loads(out)["total_bytes"], 3000 + REF)
            code, out, err = _cleanup(["--models", "org/b", "--yes"])
            self.assertEqual(code, 0, err)
            self.assertFalse((hub / "models--org--b").exists())
            self.assertFalse(only_b.exists())
            self.assertFalse(only_b.with_name(f"{only_b.name}.refs").exists())
            # org/c still links to the common payload: it stays, and org/c's file still reads.
            self.assertTrue(common.is_file())
            self.assertEqual(c_file.read_bytes(), b"c" * 400)
            self.assertEqual(classic.read_bytes(), b"c" * 700)
            self.assertTrue((hub / "blobs" / ".huggingface-shared-blobs").is_file())
            self.assertEqual(_repos(root)["org/c"]["size_bytes"], 400 + REF)
            self.assertEqual(_cleanup(["--models", "org/c", "--yes"])[0], 0)
            self.assertFalse(common.exists())
            self.assertTrue(classic.is_file())

    def test_a_payload_no_repository_links_to_is_listed_and_freed_on_request(self) -> None:
        with _workspace() as root:
            hub = root / "cache" / "huggingface" / "hub"
            orphan = _payload(hub, "d", 900)
            used = _payload(hub, "e", 100)
            _link_shared(hub, "org/used", "w.safetensors", used, "4" * 64)
            _run(root, "done", ended="2026-10-01T00:00:00+00:00", repos=("org/used",))
            items = {item["id"]: item for item in _items(root)}
            self.assertEqual(items["hub/blobs"]["kind"], "hf-unreferenced-blobs")
            self.assertEqual(items["hub/blobs"]["size_bytes"], 900)
            self.assertEqual(_cleanup(["--models", "unreferenced", "--yes"])[0], 0)
            self.assertFalse(orphan.exists())
            self.assertTrue(used.is_file())
            self.assertNotIn("hub/blobs", {item["id"] for item in _items(root)})

    def test_a_link_out_of_the_cache_is_neither_counted_nor_followed_when_removing(self) -> None:
        with tempfile.TemporaryDirectory() as elsewhere:
            outside = _write(Path(elsewhere).resolve() / "keep.bin", b"k" * 800)
            with _workspace() as root:
                hub = root / "cache" / "huggingface" / "hub"
                blob = hub / "models--org--odd" / "blobs" / ("5" * 64)
                blob.parent.mkdir(parents=True)
                os.symlink(outside, blob)
                _store(hub)
                self.assertEqual(_repos(root)["org/odd"]["size_bytes"], 0)
                self.assertEqual(_cleanup(["--models", "org/odd", "--yes"])[0], 0)
                self.assertFalse((hub / "models--org--odd").exists())
            self.assertEqual(outside.read_bytes(), b"k" * 800)

    def test_an_unmarked_store_is_never_touched(self) -> None:
        with _workspace() as root:
            hub = root / "cache" / "huggingface" / "hub"
            payload = _payload(hub, "f", 600)
            (hub / "blobs" / ".huggingface-shared-blobs").unlink()
            _link_shared(hub, "org/x", "w.safetensors", payload, "6" * 64)
            self.assertNotIn("hub/blobs", {item["id"] for item in _items(root)})
            self.assertEqual(_cleanup(["--models", "org/x", "--yes"])[0], 0)
            self.assertTrue(payload.is_file())

    def test_a_payload_whose_lock_cannot_be_taken_is_kept(self) -> None:
        with _workspace() as root:
            hub = root / "cache" / "huggingface" / "hub"
            payload = _payload(hub, "a", 500)
            lock = payload.with_name(f"{payload.name}.lock")
            lock.unlink()
            lock.mkdir()  # not a lock file: the sweep cannot hold huggingface_hub's lock, so it keeps the payload
            _link_shared(hub, "org/x", "w.safetensors", payload, "7" * 64)
            code, _, err = _cleanup(["--models", "org/x", "--yes"])
            self.assertEqual(code, 0, err)
            self.assertFalse((hub / "models--org--x").exists())
            self.assertTrue(payload.is_file())

    def test_a_payload_any_hub_entry_links_to_is_kept_whatever_its_repository_type(self) -> None:
        with _workspace() as root:
            hub = root / "cache" / "huggingface" / "hub"
            payload = _payload(hub, "a", 500)
            _link_shared(hub, "org/x", "w.safetensors", payload, "1" * 64)
            _link_shared(hub, "k/x", "w.bin", payload, "1" * 64, kind="kernels")
            self.assertEqual(_repos(root)["org/x"]["size_bytes"], REF)
            self.assertEqual(_cleanup(["--models", "org/x", "--yes"])[0], 0)
            self.assertTrue(payload.is_file())
            self.assertNotIn("hub/blobs", {item["id"] for item in _items(root)})

    @unittest.skipIf(getattr(os, "geteuid", lambda: 0)() == 0, "root reads every directory")
    def test_a_cache_that_cannot_be_read_whole_keeps_every_payload(self) -> None:
        with _workspace() as root:
            hub = root / "cache" / "huggingface" / "hub"
            payload = _payload(hub, "a", 500)
            _link_shared(hub, "org/x", "w.safetensors", payload, "1" * 64)
            _link_shared(hub, "org/hidden", "w.safetensors", payload, "1" * 64)
            hidden = hub / "models--org--hidden" / "blobs"
            hidden.chmod(0)
            try:
                code, out, err = _cleanup(["plan", "--json"])
                self.assertIn("cannot be read", out)
                self.assertEqual(_cleanup(["--models", "org/x", "--yes"])[0], 0)
                self.assertTrue(payload.is_file())
            finally:
                hidden.chmod(0o755)

    def test_the_sweep_waits_for_a_download_holding_the_lock_and_sees_its_link(self) -> None:
        import fcntl

        with _workspace() as root:
            hub = root / "cache" / "huggingface" / "hub"
            payload = _payload(hub, "a", 500)
            _link_shared(hub, "org/x", "w.safetensors", payload, "1" * 64)
            lock = payload.with_name(f"{payload.name}.lock")
            held = threading.Event()

            def download() -> None:
                # As a newer filelock does: link under the lock, then delete the lock file and release it.
                descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT, 0o666)
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                held.set()
                time.sleep(0.5)
                _link_shared(hub, "org/late", "w.safetensors", payload, "1" * 64)
                os.unlink(lock)
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

            thread = threading.Thread(target=download)
            thread.start()
            held.wait()
            code, _, err = _cleanup(["--models", "org/x", "--yes"])
            thread.join()
            self.assertEqual(code, 0, err)
            self.assertFalse((hub / "models--org--x").exists())
            self.assertTrue(payload.is_file())
            self.assertEqual((hub / "models--org--late" / "snapshots" / COMMIT / "w.safetensors").read_bytes(), b"a" * 500)


if __name__ == "__main__":
    unittest.main()

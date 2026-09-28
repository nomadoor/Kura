"""Selected-file RunPod transfer: inventory, proven archive, and stage binding."""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from kura.dataset_transfer import (
    _add_entry,
    _safe_destination,
    build_transfer_inventory,
    estimate_transfer,
    verify_stage_matches_compile,
    write_transfer_archive,
    write_transfer_manifest,
)
from kura.executors.runpod import stage_runpod

sys.path.insert(0, str(Path(__file__).resolve().parent))
from handoff_fixtures import freeze_fixture  # noqa: E402


class DatasetTransferTests(unittest.TestCase):
    @staticmethod
    def _config() -> dict:
        return {"storage_mode": "upload", "gpu_type_ids": ["NVIDIA A40"]}

    def _compiled(self, root: Path, *, resume: bool = False, link_target: bool = False) -> tuple[Path, dict]:
        run = {
            "id": "example",
            "type": "train",
            "backend": {"name": "ai-toolkit", "config": {"model_arch": "sdxl"}},
            "datasets": [{"id": "tiny"}],
        }
        run_dir = root / "runs" / "example"
        resolved = run_dir / "resolved"
        resolved.mkdir(parents=True)
        dataset = root / "datasets" / "tiny"
        dataset.mkdir(parents=True)
        (dataset / "dataset.yaml").write_text("id: tiny\nitems_schema_version: 2\n", encoding="utf-8")
        (dataset / "a.png").write_bytes(b"selected image")
        (dataset / "unselected.bin").write_bytes(b"must never be transferred")
        target = "a.png"
        if link_target:
            (dataset / "storage").mkdir()
            (dataset / "a.png").rename(dataset / "storage" / "a.png")
            (dataset / "link.png").symlink_to("storage/a.png")
            target = "link.png"
        (dataset / "items.jsonl").write_text(json.dumps({
            "id": "a",
            "files": [{"type": "file", "role": "target", "path": target}],
            "caption": {"text": "caption"},
        }) + "\n", encoding="utf-8")
        if resume:
            payload = root / "artifacts" / "training-state" / "state-1" / "payload"
            payload.mkdir(parents=True)
            (payload / "optimizer.bin").write_bytes(b"state")
            manifest = {
                "schema_version": 1,
                "id": "state-1",
                "payload": "artifacts/training-state/state-1/payload",
                "files": [{
                    "path": "optimizer.bin", "size": 5,
                    "sha256": hashlib.sha256(b"state").hexdigest(),
                }],
            }
            raw = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
            (payload.parent / "manifest.json").write_bytes(raw)
            run["continuation"] = {"mode": "resume", "source": {
                "artifact_id": "state-1", "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            }}
        from kura.backends import get_backend
        from kura.dataset_handoff import freeze_dataset_handoff

        adapter = get_backend("ai-toolkit")
        freeze_dataset_handoff(
            run, root, resolved, backend="ai-toolkit",
            project=lambda selection: adapter.project_dataset(run, selection),
        )
        (run_dir / "run.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
        (resolved / "manifest.lock.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
        (run_dir / "status.json").write_text(json.dumps({"state": "compiled"}), encoding="utf-8")
        return run_dir, run

    def test_inventory_selects_only_locked_files_in_explicit_namespaces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, run = self._compiled(root, resume=True)

            inventory = build_transfer_inventory(root, run_dir, run)

            by_namespace: dict[str, list[str]] = {}
            for item in inventory["entries"]:
                self.assertEqual(item["archive_name"], f"{item['namespace']}/{item['destination']}")
                by_namespace.setdefault(item["namespace"], []).append(item["destination"])
            self.assertEqual(by_namespace["source"], ["datasets/tiny/a.png"])
            self.assertIn("runs/example/resolved/dataset-input.lock.json", by_namespace["envelope"])
            self.assertEqual(by_namespace["resume"], [
                "artifacts/training-state/state-1/manifest.json",
                "artifacts/training-state/state-1/payload/optimizer.bin",
            ])
            self.assertEqual(inventory["resume"]["artifact_id"], "state-1")
            names = [item["archive_name"] for item in inventory["entries"]]
            self.assertEqual(names, sorted(names))

    def test_unsafe_destinations_are_rejected(self) -> None:
        for value in ("../x", "/datasets/x", "datasets//x", "datasets\\x", "datasets/./x", "other/x"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "safe path"):
                _safe_destination(value, prefix="datasets")

    def test_archive_is_reproducible_and_proves_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, run = self._compiled(root)
            inventory = build_transfer_inventory(root, run_dir, run)
            (run_dir / "transfer").mkdir()

            first = write_transfer_archive(root, run_dir, inventory, run_dir / "transfer" / "one.tar")
            second = write_transfer_archive(root, run_dir, inventory, run_dir / "transfer" / "two.tar")

            self.assertEqual(first["archive_sha256"], second["archive_sha256"])
            self.assertEqual(
                (run_dir / "transfer" / "one.tar").read_bytes(), (run_dir / "transfer" / "two.tar").read_bytes(),
            )
            self.assertEqual(first["tar_bytes"], estimate_transfer(inventory)["tar_bytes"])
            self.assertEqual(
                first["archive_sha256"],
                hashlib.sha256((run_dir / "transfer" / "one.tar").read_bytes()).hexdigest(),
            )
            source = next(item for item in first["entries"] if item["namespace"] == "source")
            self.assertEqual(source["sha256"], hashlib.sha256(b"selected image").hexdigest())
            with tarfile.open(run_dir / "transfer" / "one.tar") as archive:
                members = archive.getmembers()
            self.assertTrue(all(member.isreg() and member.mtime == 0 and member.uid == 0 for member in members))
            self.assertNotIn("source/datasets/tiny/unselected.bin", [member.name for member in members])

    def test_archive_rejects_content_that_differs_from_the_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, run = self._compiled(root)
            inventory = build_transfer_inventory(root, run_dir, run)
            (run_dir / "transfer").mkdir()
            (root / "datasets" / "tiny" / "a.png").write_bytes(b"SELECTED IMAGE")

            # Stat drift is caught before writing; the hash catches the rest.
            with self.assertRaisesRegex(ValueError, "changed before transfer"):
                write_transfer_archive(root, run_dir, inventory, run_dir / "transfer" / "x.tar")
            with (
                patch("kura.dataset_transfer.inspect_dataset_sources", return_value=[]),
                self.assertRaisesRegex(ValueError, "content differs from the lock"),
            ):
                write_transfer_archive(root, run_dir, inventory, run_dir / "transfer" / "x.tar")
            self.assertEqual(list((run_dir / "transfer").iterdir()), [])

    def test_archive_rejects_a_file_that_changes_while_it_is_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "a.bin"
            source.write_bytes(b"payload")
            real_fstat = os.fstat
            calls = {"count": 0}

            def drifting_fstat(descriptor: int) -> object:
                observed = real_fstat(descriptor)
                calls["count"] += 1
                return SimpleNamespace(
                    st_mode=observed.st_mode,
                    st_size=observed.st_size,
                    st_mtime_ns=observed.st_mtime_ns + (1 if calls["count"] == 2 else 0),
                    st_ctime_ns=observed.st_ctime_ns,
                )

            item = {
                "namespace": "source", "archive_name": "source/datasets/x/a.bin",
                "destination": "datasets/x/a.bin", "root": directory, "relative": "a.bin",
                "size": 7, "sha256": None,
            }
            with tarfile.open(fileobj=io.BytesIO(), mode="w", format=tarfile.PAX_FORMAT) as archive:
                with (
                    patch("kura.dataset_transfer.os.fstat", side_effect=drifting_fstat),
                    self.assertRaisesRegex(ValueError, "changed while it was archived"),
                ):
                    _add_entry(archive, item)

    def test_reading_refuses_a_symlink_at_any_component_below_the_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            outside = Path(directory) / "outside"
            (root / "real").mkdir(parents=True)
            outside.mkdir()
            (root / "real" / "a.bin").write_bytes(b"inside")
            (outside / "a.bin").write_bytes(b"outside")
            (root / "linked").symlink_to(outside, target_is_directory=True)
            (root / "real" / "file-link.bin").symlink_to(root / "real" / "a.bin")
            for relative in ("linked/a.bin", "real/file-link.bin"):
                item = {
                    "namespace": "source", "archive_name": f"source/datasets/x/{relative}",
                    "destination": f"datasets/x/{relative}", "root": str(root), "relative": relative,
                    "size": None, "sha256": None,
                }
                with self.subTest(relative=relative):
                    with tarfile.open(fileobj=io.BytesIO(), mode="w", format=tarfile.PAX_FORMAT) as archive:
                        with self.assertRaisesRegex(ValueError, "cannot read transfer source"):
                            _add_entry(archive, item)

    def test_in_root_dataset_symlink_transfers_the_resolved_bytes_at_its_logical_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, run = self._compiled(root, link_target=True)

            record = stage_runpod(workspace=root, run_dir=run_dir, config=self._config())

            with tarfile.open(run_dir / record["archive"]) as archive:
                names = archive.getnames()
                content = archive.extractfile("source/datasets/tiny/link.png").read()
            self.assertIn("source/datasets/tiny/link.png", names)
            self.assertNotIn("source/datasets/tiny/storage/a.png", names)
            self.assertEqual(content, b"selected image")

    def test_resume_payload_cannot_escape_through_a_parent_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, run = self._compiled(root, resume=True)
            payload = root / "artifacts" / "training-state" / "state-1" / "payload"
            outside = root / "outside-payload"
            payload.rename(outside)
            payload.symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "training-state|symlink|cannot read"):
                inventory = build_transfer_inventory(root, run_dir, run)
                (run_dir / "transfer").mkdir()
                write_transfer_archive(root, run_dir, inventory, run_dir / "transfer" / "x.tar")

    def test_stage_record_is_bound_and_launch_requires_an_exact_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, run = self._compiled(root)

            record = stage_runpod(workspace=root, run_dir=run_dir, config=self._config())

            for key in ("input_sha256", "projection_sha256", "archive_sha256", "tar_bytes", "payload_bytes"):
                self.assertIn(key, record)
            self.assertEqual(record["transfer"], "selected-files")
            self.assertTrue((run_dir / record["archive"]).is_file())
            manifest = json.loads((run_dir / record["manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["archive_sha256"], record["archive_sha256"])
            self.assertEqual(manifest["entries"], record["entries"])
            verify_stage_matches_compile(root, run_dir, run, record)

            (run_dir / "resolved" / "backend-command.lock.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stage it again"):
                verify_stage_matches_compile(root, run_dir, run, record)

    def test_launch_rejects_a_missing_truncated_or_replaced_staged_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, run = self._compiled(root)
            record = stage_runpod(workspace=root, run_dir=run_dir, config=self._config())
            archive = run_dir / record["archive"]
            manifest = run_dir / record["manifest"]
            original = archive.read_bytes()
            original_manifest = manifest.read_text(encoding="utf-8")
            outside = root / "elsewhere.tar"
            outside.write_bytes(original)

            def restore() -> None:
                if archive.is_symlink() or archive.exists():
                    archive.unlink()
                archive.write_bytes(original)
                manifest.write_text(original_manifest, encoding="utf-8")

            tampers = {
                "missing": lambda: archive.unlink(),
                "truncated": lambda: archive.write_bytes(original[:-512]),
                "replaced": lambda: archive.write_bytes(b"x" * len(original)),
                "symlinked": lambda: (archive.unlink(), archive.symlink_to(outside)),
                "manifest": lambda: manifest.write_text("{}", encoding="utf-8"),
            }
            for name, tamper in tampers.items():
                with self.subTest(tamper=name):
                    tamper()
                    with self.assertRaisesRegex(ValueError, "stage it again"):
                        verify_stage_matches_compile(root, run_dir, run, record)
                    restore()
            verify_stage_matches_compile(root, run_dir, run, record)

    def test_launch_rejects_a_record_and_manifest_rewritten_together(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, run = self._compiled(root)
            record = stage_runpod(workspace=root, run_dir=run_dir, config=self._config())
            archive = run_dir / record["archive"]
            original = archive.read_bytes()

            def rewrite_entry(field: str, value: object) -> dict:
                altered = json.loads(json.dumps(record))
                altered["entries"][0][field] = value
                return altered

            def with_payload_bytes(value: int) -> dict:
                altered = json.loads(json.dumps(record))
                altered["payload_bytes"] = value
                return altered

            def with_member_swapped() -> dict:
                # A consistent forgery: rewrite the tar with other content of the
                # same size and update every digest in the record and manifest.
                altered = json.loads(json.dumps(record))
                source = next(item for item in altered["entries"] if item["namespace"] == "source")
                forged_content = b"X" * source["size"]
                buffer = io.BytesIO()
                with tarfile.open(archive) as original_tar, tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as forged:
                    for member in original_tar.getmembers():
                        data = original_tar.extractfile(member).read()
                        if member.name == source["archive_name"]:
                            data = forged_content
                        forged.addfile(member, io.BytesIO(data))
                archive.write_bytes(buffer.getvalue())
                altered["tar_bytes"] = len(buffer.getvalue())
                altered["archive_sha256"] = hashlib.sha256(buffer.getvalue()).hexdigest()
                return altered

            forgeries = {
                "entry size": lambda: rewrite_entry("size", record["entries"][0]["size"] + 1),
                "archive name": lambda: rewrite_entry("archive_name", "envelope/elsewhere"),
                "payload bytes": lambda: with_payload_bytes(record["payload_bytes"] + 1),
                "member content": with_member_swapped,
            }
            for name, forge in forgeries.items():
                with self.subTest(forgery=name):
                    altered = forge()
                    write_transfer_manifest(run_dir / record["manifest"], altered)
                    with self.assertRaisesRegex(ValueError, "stage it again"):
                        verify_stage_matches_compile(root, run_dir, run, altered)
                    archive.write_bytes(original)

    def test_stage_stops_when_local_space_is_insufficient(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, _ = self._compiled(root)
            with (
                patch("kura.executors.runpod.shutil.disk_usage", return_value=SimpleNamespace(free=10)),
                self.assertRaisesRegex(ValueError, "RunPod stage needs"),
            ):
                stage_runpod(workspace=root, run_dir=run_dir, config=self._config())
            self.assertFalse(any((run_dir / "transfer").iterdir()))


    def test_plan_shows_the_four_transfer_sizes_separately(self) -> None:
        from kura.run_commands.plan import format_run_plan

        output = format_run_plan({
            "id": "example", "type": "train",
            "backend": {"name": "ai-toolkit", "config": {}},
            "model": {}, "compute": {}, "datasets": [],
            "dataset_input": {
                "status": "current", "verification": "content-hash-at-compile",
                "selection": [], "changes": [], "runtime_checks": [], "views": [],
                "projection_rules": [],
                "runpod_transfer": {
                    "payload_bytes": 1000, "tar_bytes": 10240,
                    "local_stage_free_bytes": 10240, "remote_peak_bytes": 11240,
                },
            },
        })

        self.assertIn("RunPod selected-file transfer:", output)
        for label in ("payload", "tar", "local_stage_free", "pod_peak"):
            self.assertRegex(output, rf"\n    {label}\s+\S")


if __name__ == "__main__":
    unittest.main()

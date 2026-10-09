"""Selected-file RunPod transfer: inventory, proven archive, and stage binding."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
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
from kura.container_scripts import script_source
from kura.executors.runpod import stage_runpod

sys.path.insert(0, str(Path(__file__).resolve().parent))
from handoff_fixtures import freeze_fixture  # noqa: E402
from tests.platform_support import DATASET_IO, posix_only


class _CompiledRunFixture:
    @staticmethod
    def _config() -> dict:
        return {"storage_mode": "upload", "gpu_type_ids": ["NVIDIA A40"]}

    @classmethod
    def _workspace_config(cls) -> dict:
        return {"runpod": cls._config()}

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


@posix_only(DATASET_IO)
class DatasetTransferTests(_CompiledRunFixture, unittest.TestCase):
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

            record = stage_runpod(workspace=root, run_dir=run_dir, config=self._workspace_config())

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

            record = stage_runpod(workspace=root, run_dir=run_dir, config=self._workspace_config())

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
            record = stage_runpod(workspace=root, run_dir=run_dir, config=self._workspace_config())
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
            record = stage_runpod(workspace=root, run_dir=run_dir, config=self._workspace_config())
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
                # WSL: the Linux disk looks roomy, the Windows drive behind it is full.
                patch("kura.storage.is_wsl", return_value=True),
                patch("kura.storage._findmnt_for", return_value={"available": True, "fstype": "ext4", "source": "/dev/sdd"}),
                patch("kura.storage._auto_wsl_host_drive", return_value="C:"),
                patch("kura.storage._windows_drive_free_bytes", side_effect=lambda drive: 10 if drive == "D:" else 900 * 1024**3),
                patch("kura.storage.shutil.disk_usage", return_value=SimpleNamespace(free=900 * 1024**3, total=1000 * 1024**3)),
                self.assertRaisesRegex(ValueError, "RunPod stage needs about .* on D:"),
            ):
                stage_runpod(workspace=root, run_dir=run_dir, config={**self._workspace_config(), "storage": {"host_drive": "D"}})
            self.assertFalse(any((run_dir / "transfer").iterdir()))


    def test_plan_reads_write_roots_in_the_command_lock_shape(self) -> None:
        from kura.run_commands.plan import _command_write_roots

        lock = {"write_roots": [
            {"role": "model-cache", "path": "/workspace/cache/ai-toolkit/models", "env": "MODELS_PATH"},
            {"role": "broken"}, "not-a-mapping",
        ]}
        self.assertEqual(_command_write_roots(lock), ["/workspace/cache/ai-toolkit/models"])
        self.assertEqual(_command_write_roots({}), [])
        self.assertEqual(_command_write_roots(None), [])

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



    def test_launch_refuses_v2_without_a_verified_selected_file_stage_before_any_api_call(self) -> None:
        from kura.executors.runpod import launch_runpod

        spec = {"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, _ = self._compiled(root)
            cases = {
                "no stage": ({"storage_mode": "upload"}, None, "no staged bundle"),
                "legacy stage": ({"storage_mode": "upload"}, {"storage_mode": "upload", "archive_name": "old.tar.gz"}, "selected-file stage"),
                "container disk": ({"storage_mode": "container_disk"}, None, "storage_mode=upload"),
            }
            for name, (settings, stage, message) in cases.items():
                with self.subTest(case=name):
                    status = {"state": "compiled"}
                    if stage is not None:
                        (run_dir / "realizations").mkdir(exist_ok=True)
                        (run_dir / "realizations" / "stage-old.json").write_text(json.dumps(stage), encoding="utf-8")
                        status["last_stage"] = "realizations/stage-old.json"
                    (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
                    with (
                        patch("kura.executors.runpod._runpod_request") as request,
                        patch("kura.executors.runpod._runpod_graphql") as graphql,
                        self.assertRaisesRegex(ValueError, message),
                    ):
                        launch_runpod(max_lease_sec=3600, 
                            run_dir=run_dir, spec=spec, image="registry/image:tag",
                            config={**self._config(), **settings}, yes=True,
                        )
                    request.assert_not_called()
                    graphql.assert_not_called()


    def test_upload_sends_only_the_launch_pin_and_refuses_a_replaced_stage(self) -> None:
        from kura.dataset_transfer import StagedTransferChanged, pin_transfer_manifest, verify_pinned_transfer

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, run = self._compiled(root)
            record = stage_runpod(workspace=root, run_dir=run_dir, config=self._workspace_config())
            pinned = run_dir / "realizations" / "real-1.transfer-manifest.json"
            digest = pin_transfer_manifest(root, run_dir, run, record, pinned)
            self.assertEqual(hashlib.sha256(pinned.read_bytes()).hexdigest(), digest)

            # A deterministic restage of the same compile still equals the pin.
            again = stage_runpod(workspace=root, run_dir=run_dir, config=self._workspace_config())
            self.assertEqual(verify_pinned_transfer(root, run_dir, run, again, pinned, digest), pinned.read_bytes())

            # A consistent forgery that adds an unselected file is refused.
            forged = json.loads(json.dumps(record))
            forged["entries"].append(dict(forged["entries"][-1], archive_name="source/datasets/tiny/extra.bin", destination="datasets/tiny/extra.bin"))
            write_transfer_manifest(run_dir / record["manifest"], forged)
            with self.assertRaises(StagedTransferChanged):
                verify_pinned_transfer(root, run_dir, run, forged, pinned, digest)

            # Editing the pinned copy itself is refused.
            pinned.write_bytes(pinned.read_bytes() + b" ")
            with self.assertRaisesRegex(StagedTransferChanged, "pinned at launch was modified"):
                verify_pinned_transfer(root, run_dir, run, record, pinned, digest)

    def test_a_pre_upload_refusal_stops_the_unused_pod_without_a_review_hold(self) -> None:
        from kura.dataset_transfer import StagedTransferChanged
        from kura.run_commands.launch import run_remote

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runs" / "example").mkdir(parents=True)
            with (
                patch("kura.run_commands.launch._run_path", return_value=root / "runs" / "example"),
                patch("kura.run_commands.launch.stage_run", return_value=0),
                patch("kura.run_commands.launch.launch_run", return_value=0),
                patch("kura.run_commands.launch._runpod_run_over_ssh", side_effect=StagedTransferChanged("replaced")),
                patch("kura.run_commands.launch.download_with_retries") as download,
                patch("kura.run_commands.launch.stop_runpod", return_value={}) as stop,
                patch("kura.run_commands.launch.time.sleep") as sleep,
                patch("kura.run_commands.launch._notify"),
                patch("sys.stderr", new_callable=io.StringIO),
            ):
                code = run_remote(
                    "example", upload_timeout=1, job_timeout=0, download_attempts=1, download_interval=0,
                )

            self.assertEqual(code, 1)
            stop.assert_called_once()
            self.assertEqual(stop.call_args.args[0].name, "example")
            download.assert_not_called()
            sleep.assert_not_called()


    def test_any_pre_ssh_preparation_failure_is_a_refusal_before_upload(self) -> None:
        from kura.dataset_transfer import TransferRefused, pin_transfer_manifest
        from kura.run_commands.runpod_ssh import _runpod_run_over_ssh

        def prepared(root: Path) -> Path:
            run_dir, run = self._compiled(root)
            record = stage_runpod(workspace=root, run_dir=run_dir, config=self._workspace_config())
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            pinned = "realizations/real-1.transfer-manifest.json"
            digest = pin_transfer_manifest(root, run_dir, run, record, run_dir / pinned)
            (run_dir / "realizations" / "real-1.json").write_text(json.dumps({
                "id": "real-1",
                "request": {"env": {"KURA_WORKSPACE": "/workspace"}},
                "container_cwd": "/opt/tool",
                "backend_command": ["python", "train.py"],
                "transfer": {"stage": status["last_stage"], "pinned_manifest": pinned, "manifest_sha256": digest},
            }), encoding="utf-8")
            status["last_realization"] = "realizations/real-1.json"
            (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
            return run_dir

        def remove_archive(run_dir: Path) -> None:
            stage = json.loads((run_dir / json.loads((run_dir / "status.json").read_text())["last_stage"]).read_text())
            (run_dir / stage["archive"]).unlink()

        def list_realization(run_dir: Path) -> None:
            (run_dir / "realizations" / "real-1.json").write_text("[]", encoding="utf-8")

        cases = {
            "realization is a JSON list": list_realization,
            "missing archive": remove_archive,
            "missing pin": lambda run_dir: (run_dir / "realizations" / "real-1.transfer-manifest.json").unlink(),
            "missing resolved manifest": lambda run_dir: (run_dir / "resolved" / "manifest.lock.yaml").unlink(),
        }
        for name, damage in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                run_dir = prepared(Path(directory))
                damage(run_dir)
                with (
                    patch("kura.run_commands.runpod_ssh._runpod_ssh_details") as ssh,
                    self.assertRaises(TransferRefused),
                ):
                    _runpod_run_over_ssh(run_dir, ssh_timeout_sec=1, job_timeout_sec=0)
                ssh.assert_not_called()


@posix_only(DATASET_IO)
class RunPodInputVerifyTests(_CompiledRunFixture, unittest.TestCase):
    """The Pod-side receive path, exercised through the generated job script."""

    def _pod(self, root: Path) -> tuple[Path, Path, dict]:
        run_dir, run = self._compiled(root / "local")
        record = stage_runpod(workspace=root / "local", run_dir=run_dir, config=self._workspace_config())
        pod = root / "pod"
        remote_dir = pod / ".kura-transfer" / "example"
        remote_dir.mkdir(parents=True)
        (remote_dir / record["archive_name"]).write_bytes((run_dir / record["archive"]).read_bytes())
        manifest_bytes = (run_dir / record["manifest"]).read_bytes()
        (remote_dir / Path(record["manifest"]).name).write_bytes(manifest_bytes)
        # What the controller embeds: the digest of the manifest it verified.
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        return pod, remote_dir, record

    def _run_job(
        self, pod: Path, remote_dir: Path, record: dict, command: str = "touch trainer-started",
        command_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        from kura.run_commands.runpod_ssh import _runpod_remote_job_script

        script = _runpod_remote_job_script(
            workspace=str(pod),
            run_id="example",
            realization_id="real-1",
            remote_secret_path=str(pod / "no-secrets.env"),
            archive_name=record["archive_name"],
            remote_archive=str(remote_dir / record["archive_name"]),
            cwd=str(pod),
            command=command,
            transfer_manifest=str(remote_dir / Path(record["manifest"]).name),
            transfer_manifest_sha256=self.manifest_sha256,
            command_env=command_env,
        )
        return subprocess.run(["sh", "-c", script], text=True, capture_output=True, check=False)

    def _verify_record(self, pod: Path) -> dict:
        return json.loads(
            (pod / "runs" / "example" / "realizations" / "real-1.runpod-input.json").read_text(encoding="utf-8")
        )

    def test_verified_transfer_publishes_inputs_and_views_before_the_trainer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pod, remote_dir, record = self._pod(Path(directory))

            result = self._run_job(pod, remote_dir, record)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((pod / "trainer-started").exists())
            proof = self._verify_record(pod)
            self.assertEqual(proof["status"], "verified")
            self.assertEqual(proof["archive_sha256"], record["archive_sha256"])
            self.assertEqual((pod / "datasets" / "tiny" / "a.png").read_bytes(), b"selected image")
            self.assertFalse((pod / "datasets" / "tiny" / "unselected.bin").exists())
            self.assertTrue((pod / "runs" / "example" / "resolved" / "dataset-input.lock.json").is_file())
            lock = json.loads((pod / "runs" / "example" / "resolved" / "dataset-input.lock.json").read_text(encoding="utf-8"))
            for view in lock["views"]:
                for link in view["links"]:
                    self.assertEqual(os.readlink(pod / link["path"]), link["target"])
            self.assertEqual(proof["view_links"], sum(len(view["links"]) for view in lock["views"]))
            self.assertIn("datasets/tiny/a.png", proof["source_baseline"])
            self.assertFalse((pod / ".kura-transfer").exists() and any((pod / ".kura-transfer").iterdir()))

    def test_the_trainer_sees_the_frozen_command_env_under_kura_owned_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pod, remote_dir, record = self._pod(Path(directory))
            command = (
                "sh -c 'test \"$KURA_MUSUBI_IMAGE_SUFFIXES\" = \"[.png]\" "
                "&& test \"$KURA_WORKSPACE\" = \"" + str(pod) + "\" && touch trainer-started'"
            )

            result = self._run_job(pod, remote_dir, record, command=command, command_env={
                "KURA_MUSUBI_IMAGE_SUFFIXES": "[.png]",
                "KURA_WORKSPACE": "/somewhere-else",
            })

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((pod / "trainer-started").exists())

    def test_any_verification_failure_publishes_nothing_and_never_starts_the_trainer(self) -> None:
        def corrupt_byte(remote_dir: Path, record: dict) -> None:
            archive = remote_dir / record["archive_name"]
            data = bytearray(archive.read_bytes())
            data[1024] ^= 0xFF
            archive.write_bytes(bytes(data))

        def extra_entry(remote_dir: Path, record: dict) -> None:
            manifest_path = remote_dir / Path(record["manifest"]).name
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["entries"].append(dict(manifest["entries"][-1], archive_name="source/datasets/tiny/extra"))
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

        def wrong_input_digest(remote_dir: Path, record: dict) -> None:
            # Even a manifest rebound to its own new digest must match the envelope.
            manifest_path = remote_dir / Path(record["manifest"]).name
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["input_sha256"] = "sha256:" + "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

        def occupied_target(remote_dir: Path, record: dict) -> None:
            (remote_dir.parent.parent / "datasets").mkdir()

        def swapped_manifest(remote_dir: Path, record: dict) -> None:
            manifest_path = remote_dir / Path(record["manifest"]).name
            manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + " ", encoding="utf-8")

        def list_manifest(remote_dir: Path, record: dict) -> None:
            manifest_path = remote_dir / Path(record["manifest"]).name
            manifest_path.write_text("[]", encoding="utf-8")
            self.manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

        cases = {
            "manifest is a JSON list": list_manifest,
            "swapped manifest": swapped_manifest,
            "corrupt byte": corrupt_byte,
            "extra manifest entry": extra_entry,
            "wrong input digest": wrong_input_digest,
            "occupied target": occupied_target,
        }
        for name, tamper in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                pod, remote_dir, record = self._pod(Path(directory))
                tamper(remote_dir, record)

                result = self._run_job(pod, remote_dir, record)

                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((pod / "trainer-started").exists())
                self.assertEqual(self._verify_record(pod)["status"], "failed")
                self.assertFalse((pod / "runs" / "example" / "resolved").exists())
                if name != "occupied target":
                    self.assertFalse((pod / "datasets").exists())
                exits = list((pod / "runs" / "example" / "realizations").glob("remote-exit-*.json"))
                self.assertEqual(len(exits), 1)
                self.assertNotEqual(json.loads(exits[0].read_text(encoding="utf-8"))["exit_code"], 0)


    def _postflight(self, pod: Path) -> dict:
        return json.loads(
            (pod / "runs" / "example" / "realizations" / "real-1.runpod-input-postflight.json").read_text(encoding="utf-8")
        )

    def test_postflight_records_matched_inputs_even_when_the_trainer_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pod, remote_dir, record = self._pod(Path(directory))

            result = self._run_job(pod, remote_dir, record, command="sh -c 'exit 3'")

            self.assertEqual(result.returncode, 3)
            postflight = self._postflight(pod)
            self.assertEqual(postflight["status"], "matched")
            self.assertEqual(postflight["view_link_verification"], "matched")

    def test_postflight_records_drift_without_changing_the_trainer_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pod, remote_dir, record = self._pod(Path(directory))
            lock_view = "runs/example/cache/dataset-view"
            command = (
                f"printf changed >> {pod}/datasets/tiny/a.png && "
                f"touch {pod}/{lock_view}/ai-toolkit/tiny/stray.png && "
                f"touch {pod}/{lock_view}/ai-toolkit/tiny/_latent_cache.safetensors.json"
            )

            result = self._run_job(pod, remote_dir, record, command=command)

            self.assertEqual(result.returncode, 0, result.stderr)
            postflight = self._postflight(pod)
            self.assertEqual(postflight["status"], "changed")
            self.assertEqual(postflight["source_changes"], ["transferred source changed: datasets/tiny/a.png"])
            self.assertEqual(len(postflight["view_changes"]), 1)
            self.assertIn("stray.png", postflight["view_changes"][0])


    def test_postflight_catches_same_size_rewrites_with_restored_mtime_and_symlinks(self) -> None:
        cases = {
            "same size, mtime restored": (
                "python -c \"import os; p='{pod}/datasets/tiny/a.png'; s=os.stat(p); "
                "open(p,'r+b').write(b'X'); os.utime(p, ns=(s.st_atime_ns, s.st_mtime_ns))\""
            ),
            "replaced by a symlink": (
                "python -c \"import os; p='{pod}/datasets/tiny/a.png'; os.rename(p, p + '.orig'); "
                "os.symlink(p + '.orig', p)\""
            ),
        }
        for name, command in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                pod, remote_dir, record = self._pod(Path(directory))

                result = self._run_job(pod, remote_dir, record, command=command.format(pod=pod))

                self.assertEqual(result.returncode, 0, result.stderr)
                postflight = self._postflight(pod)
                self.assertEqual(postflight["status"], "changed")
                self.assertEqual(postflight["source_changes"], ["transferred source changed: datasets/tiny/a.png"])

    def test_publication_is_all_or_nothing_and_retryable_on_the_same_pod(self) -> None:
        namespace = {"__name__": "__test__"}
        exec(script_source("runpod_input_verify.py"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            pod, remote_dir, record = self._pod(Path(directory))
            real_rename = os.rename
            calls = {"count": 0}

            def failing_second_rename(source, target):
                calls["count"] += 1
                if calls["count"] == 2:
                    raise OSError("simulated rename failure")
                return real_rename(source, target)

            env = {
                "KURA_WORKSPACE": str(pod), "KURA_RUN_ID": "example", "KURA_REALIZATION_ID": "real-1",
                "KURA_KNOWN_MEDIA_SUFFIXES": json.dumps([".png"]),
            }
            argv = ["verify", str(remote_dir / record["archive_name"]),
                    str(remote_dir / Path(record["manifest"]).name), self.manifest_sha256]
            with (
                patch.dict(os.environ, env),
                patch.object(sys, "argv", argv),
                patch.object(namespace["os"], "rename", side_effect=failing_second_rename),
                self.assertRaises(SystemExit),
            ):
                namespace["main"]()
            self.assertEqual(self._verify_record(pod)["status"], "failed")
            self.assertFalse((pod / "runs" / "example" / "run.yaml").exists())
            self.assertFalse((pod / "runs" / "example" / "resolved").exists())
            self.assertFalse((pod / "datasets").exists())

            with patch.dict(os.environ, env), patch.object(sys, "argv", argv):
                namespace["main"]()
            self.assertEqual(self._verify_record(pod)["status"], "verified")
            self.assertTrue((pod / "datasets" / "tiny" / "a.png").is_file())


    def test_failure_after_views_exist_rolls_back_and_keeps_the_upload_for_retry(self) -> None:
        namespace = {"__name__": "__test__"}
        exec(script_source("runpod_input_verify.py"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            pod, remote_dir, record = self._pod(Path(directory))
            env = {
                "KURA_WORKSPACE": str(pod), "KURA_RUN_ID": "example", "KURA_REALIZATION_ID": "real-1",
                "KURA_KNOWN_MEDIA_SUFFIXES": json.dumps([".png"]),
            }
            argv = ["verify", str(remote_dir / record["archive_name"]),
                    str(remote_dir / Path(record["manifest"]).name), self.manifest_sha256]
            real_baseline = namespace["source_baseline"]
            namespace["source_baseline"] = lambda *args: (_ for _ in ()).throw(OSError("simulated baseline failure"))
            with patch.dict(os.environ, env), patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
                namespace["main"]()
            self.assertEqual(self._verify_record(pod)["status"], "failed")
            for published in ("runs/example/resolved", "runs/example/run.yaml", "datasets", "runs/example/cache/dataset-view"):
                self.assertFalse((pod / published).exists(), published)
            self.assertTrue((remote_dir / record["archive_name"]).is_file())

            namespace["source_baseline"] = real_baseline
            with patch.dict(os.environ, env), patch.object(sys, "argv", argv):
                namespace["main"]()
            self.assertEqual(self._verify_record(pod)["status"], "verified")
            self.assertFalse((remote_dir / record["archive_name"]).exists())


@posix_only(DATASET_IO)
class RunPodDownloadFinalizeTests(_CompiledRunFixture, unittest.TestCase):
    def _downloaded(self, root: Path, remote_status: str) -> tuple[Path, Path]:
        run_dir, _ = self._compiled(root)
        (run_dir / "status.json").write_text(json.dumps({
            "state": "running", "last_realization": "realizations/real-1.json",
        }), encoding="utf-8")
        downloaded = run_dir / "downloads" / "example"
        (downloaded / "realizations").mkdir(parents=True)
        (downloaded / "realizations" / "real-1.runpod-input.json").write_text(
            json.dumps({"status": "verified"}), encoding="utf-8",
        )
        (downloaded / "realizations" / "real-1.runpod-input-postflight.json").write_text(json.dumps({
            "status": remote_status,
            "source_stat_verification": remote_status,
            "view_link_verification": "matched",
        }), encoding="utf-8")
        return run_dir, downloaded

    def test_download_promotes_records_and_projects_postflight_once(self) -> None:
        from kura.executors.runpod import project_runpod_dataset_handoff as finalize_runpod_dataset_handoff

        with tempfile.TemporaryDirectory() as directory:
            run_dir, downloaded = self._downloaded(Path(directory), "matched")

            first = finalize_runpod_dataset_handoff(run_dir, downloaded, "real-1")
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            status["dataset_input_postflight"] = first
            (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
            second = finalize_runpod_dataset_handoff(run_dir, downloaded, "real-1")

            self.assertEqual(first, second)
            self.assertEqual(first["status"], "matched")
            # A crash between the event and the status projection: finalize
            # runs again without the projection and must not duplicate it.
            status.pop("dataset_input_postflight")
            (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
            finalize_runpod_dataset_handoff(run_dir, downloaded, "real-1")
            finalize_runpod_dataset_handoff(run_dir, downloaded, "real-1")
            self.assertNotIn("warning", first)
            self.assertTrue((run_dir / "realizations" / "real-1.runpod-input.json").is_file())
            events = [
                json.loads(line)
                for line in (run_dir / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([item["event"] for item in events].count("dataset_input_postflight"), 1)

    def test_snapshot_records_with_controller_owned_names_are_not_promoted(self) -> None:
        from kura.executors.runpod import project_runpod_dataset_handoff

        with tempfile.TemporaryDirectory() as directory:
            run_dir, downloaded = self._downloaded(Path(directory), "matched")
            forged = {
                "schema_version": 1, "realization_id": "real-1", "observed_at": "2026-01-01T00:00:00+00:00",
                "status": "matched", "source_stat_verification": "matched", "view_link_verification": "matched",
            }
            for name in ("real-1.dataset-input-postflight.json", "stage-forged.json", "real-1.transfer-manifest.json"):
                (downloaded / "realizations" / name).write_text(json.dumps(forged), encoding="utf-8")
            (Path(directory) / "datasets" / "tiny" / "a.png").write_bytes(b"edited after compile")

            projected = project_runpod_dataset_handoff(run_dir, downloaded, "real-1")

            # The controller inspected the host itself instead of trusting the forged record.
            self.assertEqual(projected["status"], "changed")
            self.assertFalse((run_dir / "realizations" / "stage-forged.json").exists())
            self.assertFalse((run_dir / "realizations" / "real-1.transfer-manifest.json").exists())
            self.assertTrue((run_dir / "realizations" / "real-1.runpod-input.json").is_file())

    def test_local_or_remote_drift_projects_a_warning_and_never_overwrites_records(self) -> None:
        from kura.executors.runpod import project_runpod_dataset_handoff as finalize_runpod_dataset_handoff

        with tempfile.TemporaryDirectory() as directory:
            run_dir, downloaded = self._downloaded(Path(directory), "changed")
            (run_dir / "realizations").mkdir(exist_ok=True)
            (run_dir / "realizations" / "real-1.runpod-input.json").write_text("local", encoding="utf-8")
            (Path(directory) / "datasets" / "tiny" / "a.png").write_bytes(b"edited after compile")

            projected = finalize_runpod_dataset_handoff(run_dir, downloaded, "real-1")

            self.assertEqual(projected["status"], "changed")
            self.assertIn("inputs changed", projected["warning"])
            record = json.loads((run_dir / projected["record"]).read_text(encoding="utf-8"))
            self.assertEqual(record["source_stat_verification"], "changed")
            self.assertEqual(record["remote_source_stat_verification"], "changed")
            self.assertEqual(record["record_conflicts"], ["real-1.runpod-input.json"])
            self.assertEqual((run_dir / "realizations" / "real-1.runpod-input.json").read_text(encoding="utf-8"), "local")


    def test_malformed_records_become_an_uncheckable_record_never_an_exception(self) -> None:
        from kura.executors.runpod import project_runpod_dataset_handoff

        malformed = {
            "empty object": "{}",
            "json list": "[]",
            "missing required fields": json.dumps({"schema_version": 1, "status": "matched"}),
            "not json": "{",
        }
        for name, content in malformed.items():
            with self.subTest(existing_local_record=name), tempfile.TemporaryDirectory() as directory:
                run_dir, downloaded = self._downloaded(Path(directory), "matched")
                (run_dir / "realizations").mkdir(exist_ok=True)
                (run_dir / "realizations" / "real-1.dataset-input-postflight.json").write_text(content, encoding="utf-8")

                projected = project_runpod_dataset_handoff(run_dir, downloaded, "real-1")

                self.assertEqual(projected["status"], "uncheckable")
                record = json.loads((run_dir / projected["record"]).read_text(encoding="utf-8"))
                self.assertEqual(record["status"], "uncheckable")
                self.assertIn(projected["record"], [
                    item.get("record") for item in run_events_of(run_dir)
                ])

        for name, content in malformed.items():
            with self.subTest(remote_record=name), tempfile.TemporaryDirectory() as directory:
                run_dir, downloaded = self._downloaded(Path(directory), "matched")
                (downloaded / "realizations" / "real-1.runpod-input-postflight.json").write_text(content, encoding="utf-8")

                projected = project_runpod_dataset_handoff(run_dir, downloaded, "real-1")

                self.assertEqual(projected["status"], "uncheckable")
                self.assertEqual(projected["record"], "realizations/real-1.dataset-input-postflight.json")

    def test_status_carries_the_fact_only_when_no_record_can_be_written(self) -> None:
        from kura.executors.runpod import project_runpod_dataset_handoff

        with tempfile.TemporaryDirectory() as directory:
            run_dir, downloaded = self._downloaded(Path(directory), "matched")
            with patch("kura.executors.runpod._write_json", side_effect=OSError("read-only filesystem")):
                projected = project_runpod_dataset_handoff(run_dir, downloaded, "real-1")

            self.assertEqual(projected["status"], "uncheckable")
            self.assertNotIn("record", projected)
            self.assertIn("could not be written", projected["error"])


    def test_an_event_failure_never_hides_a_written_record(self) -> None:
        from kura.executors.runpod import project_runpod_dataset_handoff

        for name, content in {"normal record": None, "fallback record": "[]"}.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                run_dir, downloaded = self._downloaded(Path(directory), "matched")
                if content is not None:
                    (run_dir / "realizations").mkdir(exist_ok=True)
                    (run_dir / "realizations" / "real-1.dataset-input-postflight.json").write_text(content, encoding="utf-8")
                with patch("kura.executors.runpod.append_run_event", side_effect=OSError("events log unwritable")):
                    projected = project_runpod_dataset_handoff(run_dir, downloaded, "real-1")

                self.assertTrue((run_dir / projected["record"]).is_file())
                self.assertIn("event could not be appended", projected["error"])
                self.assertNotIn("could not be written", projected["error"])
                self.assertIs(projected["event_recorded"], False)

                # The retry, with the failed projection in status, backfills the event once.
                status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
                status["dataset_input_postflight"] = projected
                (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
                retried = project_runpod_dataset_handoff(run_dir, downloaded, "real-1")
                status["dataset_input_postflight"] = retried
                (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
                project_runpod_dataset_handoff(run_dir, downloaded, "real-1")

                self.assertIs(retried["event_recorded"], True)
                if content is None:
                    # The same record is kept; only its missing event is backfilled.
                    self.assertEqual(retried["record"], projected["record"])
                # A malformed local record stays malformed, so each attempt
                # writes its own append-only uncheckable record instead.
                events = [item for item in run_events_of(run_dir) if item.get("event") == "dataset_input_postflight"]
                self.assertEqual(len([item for item in events if item.get("record") == retried["record"]]), 1)


def run_events_of(run_dir: Path) -> list[dict]:
    from kura.executors.common import run_events

    return run_events(run_dir)


class RunEventReaderTests(unittest.TestCase):
    def test_events_with_unicode_line_separators_are_still_found(self) -> None:
        from kura.executors.common import _event_exists, append_run_event

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            append_run_event(run_dir, {"event": "note", "detail": "caption\u2028second"})
            append_run_event(run_dir, {
                "event": "dataset_input_postflight", "realization_id": "r", "record": "realizations/r.json",
                "detail": "path\u2029with separator",
            })
            with (run_dir / "logs" / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("not json\n")

            self.assertTrue(_event_exists(
                run_dir, event="dataset_input_postflight", realization_id="r", record="realizations/r.json",
            ))


if __name__ == "__main__":
    unittest.main()


class DockerFinalizeRecordTests(unittest.TestCase):
    def _run_dir(self, root: Path, publication_state: str) -> Path:
        run_dir = root / "runs" / "example"
        (run_dir / "resolved").mkdir(parents=True)
        (run_dir / "realizations").mkdir()
        (run_dir / "resolved" / "dataset-input.lock.json").write_text(
            json.dumps({"schema_version": 2, "input_sha256": "sha256:" + "0" * 64}), encoding="utf-8",
        )
        (run_dir / "status.json").write_text(json.dumps({
            "state": "completed", "last_realization": "realizations/real-1.json",
            "publication_state": publication_state,
        }), encoding="utf-8")
        return run_dir

    def test_unreadable_postflight_record_is_kept_and_projected_as_uncheckable(self) -> None:
        from kura.executors.docker import _finalize_dataset_handoff

        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory), "failed")
            record = run_dir / "realizations" / "real-1.dataset-input-postflight.json"
            record.write_text("{truncated", encoding="utf-8")

            status = _finalize_dataset_handoff(run_dir, "realizations/real-1.json", "real-1", execution_ended_at=None)

            self.assertEqual(status["dataset_input_postflight"]["status"], "uncheckable")
            events = (run_dir / "logs" / "events.jsonl").read_text(encoding="utf-8")
            self.assertIn('"status": "uncheckable"', events)
            self.assertEqual(record.read_text(encoding="utf-8"), "{truncated")

    def test_unreadable_cleanup_record_is_kept_and_cleanup_is_recorded_beside_it(self) -> None:
        from kura.executors import docker

        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory), "completed")
            (run_dir / "realizations" / "real-1.dataset-input-postflight.json").write_text(json.dumps({
                "schema_version": 1, "realization_id": "real-1", "observed_at": "2026-01-01T00:00:00+00:00",
                "status": "matched", "source_stat_verification": "matched", "view_link_verification": "matched",
            }), encoding="utf-8")
            cleanup = run_dir / "realizations" / "real-1.dataset-view-cleanup.json"
            cleanup.write_text("[]", encoding="utf-8")

            with patch.object(docker, "remove_dataset_views", return_value={"status": "removed"}):
                status = docker._finalize_dataset_handoff(run_dir, "realizations/real-1.json", "real-1", execution_ended_at=None)

            projected = status["dataset_input_postflight"]
            self.assertEqual(projected["status"], "matched")
            self.assertEqual(projected["view_cleanup"], "removed")
            self.assertIn("dataset-view-cleanup-attempt-", projected["cleanup_record"])
            self.assertEqual(cleanup.read_text(encoding="utf-8"), "[]")

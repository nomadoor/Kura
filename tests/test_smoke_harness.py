"""Regression tests for the developer real-smoke harness."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from kura import cli
from kura.dataset_manifest import validate_manifest
from tests.platform_support import DATASET_IO, posix_only


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("real_smoke", ROOT / "scripts" / "real_smoke.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _in_process_kura(workspace: Path, *args: str, timeout: float = 0) -> subprocess.CompletedProcess[str]:
    """Run the Kura CLI in this process, the way the harness runs it as a subprocess."""
    del timeout
    out, err = io.StringIO(), io.StringIO()
    previous = Path.cwd()
    os.chdir(workspace)
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), patch.object(sys, "argv", ["kura", *args]):
            try:
                cli.main()
                code = 0
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    finally:
        os.chdir(previous)
    return subprocess.CompletedProcess(list(args), code, out.getvalue(), err.getvalue())


def _fake_video_dataset(root: Path, dataset_id: str) -> None:
    # The container encodes the real MP4; compile needs only a selected file.
    (root / "0001.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
    (root / "0001.txt").write_text("a tiny synthetic smoke-test video\n", encoding="utf-8")
    MODULE._write_manifest(root, dataset_id, [{"id": "0001", "files": [{"type": "file", "role": "target", "path": "0001.mp4"}], "caption": MODULE._caption("0001.txt")}])


class RealSmokeHarnessTests(unittest.TestCase):
    @posix_only(DATASET_IO)
    def test_generated_datasets_are_valid_manifest_v2(self) -> None:
        for dataset_id in (MODULE.IMAGE_DATASET, MODULE.CONTROL_DATASET):
            with self.subTest(dataset=dataset_id), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / dataset_id
                root.mkdir()
                MODULE._CREATORS[dataset_id](root, dataset_id)
                count, errors = validate_manifest(root)
                self.assertEqual(yaml.safe_load((root / "dataset.yaml").read_text(encoding="utf-8"))["id"], dataset_id)
                self.assertEqual((count, errors), (1, []))
                row = json.loads((root / "items.jsonl").read_text(encoding="utf-8"))
                roles = [item["role"] for item in row["files"]]
                self.assertEqual(roles, ["target", "control"] if dataset_id == MODULE.CONTROL_DATASET else ["target"])

    def test_a_created_dataset_records_its_final_id_not_the_staging_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            ok = subprocess.CompletedProcess([], 0, "dataset valid", "")
            with patch.object(MODULE, "_kura", return_value=ok):
                self.assertEqual(MODULE.ensure_dataset(workspace, MODULE.IMAGE_DATASET), "created")
            manifest = yaml.safe_load((workspace / "datasets" / MODULE.IMAGE_DATASET / "dataset.yaml").read_text(encoding="utf-8"))
            self.assertEqual(manifest["id"], MODULE.IMAGE_DATASET)

    def test_an_existing_dataset_is_validated_and_never_rewritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            dataset = workspace / "datasets" / MODULE.IMAGE_DATASET
            dataset.mkdir(parents=True)
            (dataset / "items.jsonl").write_text("authored\n", encoding="utf-8")
            ok = subprocess.CompletedProcess([], 0, "dataset valid", "")
            with patch.object(MODULE, "_kura", return_value=ok):
                self.assertEqual(MODULE.ensure_dataset(workspace, MODULE.IMAGE_DATASET), "existing")
            invalid = subprocess.CompletedProcess([], 1, "", "bad manifest")
            with patch.object(MODULE, "_kura", return_value=invalid), self.assertRaises(SystemExit):
                MODULE.ensure_dataset(workspace, MODULE.IMAGE_DATASET)
            self.assertEqual(sorted(path.name for path in dataset.iterdir()), ["items.jsonl"])
            self.assertEqual((dataset / "items.jsonl").read_text(encoding="utf-8"), "authored\n")

    def test_an_interrupted_creation_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "datasets" / f".{MODULE.IMAGE_DATASET}.creating").mkdir(parents=True)
            with self.assertRaisesRegex(SystemExit, "interrupted creation"):
                MODULE.ensure_dataset(workspace, MODULE.IMAGE_DATASET)
            self.assertFalse((workspace / "datasets" / MODULE.IMAGE_DATASET).exists())

    def test_every_smoke_compiles_through_the_normal_kura_cli(self) -> None:
        # A backend surface change that invalidates a smoke must fail here,
        # not after a paid Pod has started.
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            self.assertEqual(_in_process_kura(workspace, "init").returncode, 0)
            creators = {**MODULE._CREATORS, MODULE.VIDEO_DATASET: _fake_video_dataset, MODULE.FPS30_VIDEO_DATASET: _fake_video_dataset}
            with patch.object(MODULE, "_kura", side_effect=_in_process_kura), patch.dict(MODULE._CREATORS, creators):
                for smoke_id in sorted(MODULE.SMOKES):
                    with self.subTest(smoke=smoke_id):
                        run_id = MODULE.prepare(workspace, smoke_id)
                        run_dir = workspace / "runs" / run_id
                        self.assertTrue((run_dir / "resolved" / "dataset-projection.lock.json").is_file())
                        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
                        self.assertEqual(status["state"], "compiled")

    def test_gpu_override_reaches_runpod_compute_and_refuses_local_smokes(self) -> None:
        fields = MODULE.build_run_fields("ai-toolkit-hidream", MODULE.SMOKES["ai-toolkit-hidream"], gpu="NVIDIA A100 80GB PCIe")
        self.assertEqual(fields["compute"]["gpu"], "NVIDIA A100 80GB PCIe")
        self.assertEqual(fields["compute"]["capacity"]["mode"], "wait")
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(SystemExit, "selects a RunPod GPU"):
            MODULE.prepare(Path(directory), "ai-toolkit-sd1", gpu="NVIDIA A100 80GB PCIe")

    def test_verify_requires_a_finished_published_step_and_a_stopped_pod(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run_dir = workspace / "runs" / "20260101-0000_musubi-zimage_abcd"
            (run_dir / "outputs").mkdir(parents=True)
            (run_dir / "logs").mkdir()
            (run_dir / "resolved").mkdir()
            (run_dir / "resolved" / "backend-command.lock.json").write_text(json.dumps({"argv": ["zimage_train_network.py"]}), encoding="utf-8")
            (run_dir / "outputs" / "nested").mkdir()
            (run_dir / "outputs" / "nested" / "adapter.safetensors").write_bytes(b"x")
            (run_dir / "logs" / "stdout.log").write_text("steps: 1/1 avr_loss=0.123\n", encoding="utf-8")
            status = {
                "state": "completed", "exit_code": 0, "last_step": 1, "total_steps": 1, "host": "runpod",
                "publication_state": "completed", "dataset_input_postflight": {"status": "matched"},
                "pod_stopped_at": "2026-01-01T00:00:00+00:00", "outputs": ["outputs/nested/adapter.safetensors"],
            }
            (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
            self.assertTrue(MODULE.verify(workspace, run_dir.name)["ok"])
            for key, value in (("pod_stopped_at", None), ("publication_state", "blocked"), ("dataset_input_postflight", {"status": "changed"})):
                with self.subTest(key=key):
                    (run_dir / "status.json").write_text(json.dumps({**status, key: value}), encoding="utf-8")
                    self.assertFalse(MODULE.verify(workspace, run_dir.name)["ok"])
            (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
            for log in ("avr_loss=nan\n", "loss=0.12\navr_loss=nan\n", "loss: 0.5\nloss: inf\n", "avr_loss=-Infinity\n"):
                with self.subTest(log=log):
                    (run_dir / "logs" / "stdout.log").write_text(log, encoding="utf-8")
                    self.assertFalse(MODULE.verify(workspace, run_dir.name)["checks"]["finite_loss"])

    def test_evidence_binds_identities_and_refuses_an_unverified_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run_id = "20260101-0000_musubi-zimage_abcd"
            run_dir = workspace / "runs" / run_id
            for relative in ("outputs", "logs", "resolved", "realizations"):
                (run_dir / relative).mkdir(parents=True)
            (run_dir / "resolved" / "backend-command.lock.json").write_text(json.dumps({"argv": ["zimage_train_network.py"]}), encoding="utf-8")
            (run_dir / "outputs" / "adapter.safetensors").write_bytes(b"x")
            (run_dir / "logs" / "stdout.log").write_text("avr_loss=0.5\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text("backend:\n  config:\n    architecture: zimage\n", encoding="utf-8")
            (run_dir / "realizations" / "r1.json").write_text(json.dumps({
                "executor": "runpod",
                "adapter_source": {"kind": "source-tree-sha256", "value": "a" * 64},
                "image_identity": {"reference": "image@sha256:" + "b" * 64, "pinning": {"value": "sha256:" + "b" * 64}},
                "pod": {"machine": {"gpu_display_name": "A40"}, "cost_per_h": 0.49},
            }), encoding="utf-8")
            status = {
                "state": "completed", "exit_code": 0, "last_step": 1, "total_steps": 1, "host": "runpod",
                "publication_state": "completed", "dataset_input_postflight": {"status": "matched"},
                "pod_stopped_at": "2026-01-01T00:10:00+00:00", "last_realization": "realizations/r1.json",
                "outputs": ["outputs/adapter.safetensors"],
            }
            (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")

            record, summary = MODULE.evidence(workspace, run_id, artifact="smoke-evidence/x.yaml")

            self.assertEqual(record["id"], "musubi-zimage-2026-01-01-0000")
            self.assertEqual(record["adapter_source"]["value"], "a" * 64)
            self.assertEqual(record["runtime_image"]["value"], "sha256:" + "b" * 64)
            self.assertEqual(record["native_path"]["transfer"], "selected-files")
            self.assertIn("executor_source", record)
            self.assertEqual(summary["gpu"], "A40")
            (run_dir / "status.json").write_text(json.dumps({**status, "publication_state": "blocked"}), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "did not pass verify"):
                MODULE.evidence(workspace, run_id, artifact="smoke-evidence/x.yaml")


if __name__ == "__main__":
    unittest.main()

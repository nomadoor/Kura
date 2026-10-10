"""Regression tests for the developer real-smoke harness."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

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


def _safetensors(path: Path, values: bytes) -> None:
    header = json.dumps({"__metadata__": {"ss_output_name": path.parent.name}, "lora.weight": {"dtype": "F32", "shape": [len(values) // 4], "data_offsets": [0, len(values)]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + values)


def _torch_archive(path: Path, value: object) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{path.stem}/data.pkl", pickle.dumps(value, protocol=2))
        archive.writestr(f"{path.stem}/.data/serialization_id", os.urandom(8).hex())


def _state(payload: Path, step: int, *, weight: bytes = b"\0\0\x80?") -> None:
    payload.mkdir(parents=True)
    _safetensors(payload / "model.safetensors", weight)
    _torch_archive(payload / "optimizer.bin", {"state": {0: {"step": step}}})
    _torch_archive(payload / "scheduler.bin", {"last_epoch": step, "_step_count": step + 1, "lr_lambdas": [None]})
    (payload / "train_state.json").write_text(json.dumps({"current_step": step}), encoding="utf-8")


def _finished_run(workspace: Path, run_id: str, *, weights: list[str], states: list[int], last_step: int) -> Path:
    """A finished run as Kura records it: status, locks, realization, outputs, and published states."""
    run_dir = workspace / "runs" / run_id
    for relative in ("outputs", "resolved", "realizations"):
        (run_dir / relative).mkdir(parents=True)
    for name in weights:
        _safetensors(run_dir / "outputs" / name, b"\0\0\0\0")
    (run_dir / "resolved" / "env.lock").write_text(yaml.safe_dump({"selected_image": "kura-sd-scripts@sha256:" + "a" * 64}), encoding="utf-8")
    (run_dir / "realizations" / "r1.json").write_text(json.dumps({"image_identity": {"reference": "kura-sd-scripts@sha256:" + "a" * 64}}), encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({
        "state": "completed", "exit_code": 0, "last_step": last_step, "host": "docker",
        "publication_state": "completed", "dataset_input_postflight": {"status": "matched"},
        "last_realization": "realizations/r1.json", "outputs": [f"outputs/{name}" for name in weights],
    }), encoding="utf-8")
    for step in states:
        artifact = workspace / "artifacts" / "training-state" / f"state-step-{step:08d}-{run_id[-4:]}0000000a"
        _state(artifact / "payload", step)
        (artifact / "manifest.json").write_text(json.dumps({"id": artifact.name, "source_run": run_id, "observed_step": step}), encoding="utf-8")
    return run_dir


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

    @posix_only(DATASET_IO)
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


    def test_expected_saves_follow_each_trainers_cadence_and_names(self) -> None:
        # sd-scripts and Musubi Tuner save after update m when m % cadence == 0 and name it m;
        # AI-Toolkit saves after the update at index i (i + 1 updates), names it i, and never
        # saves on the index it started at. Every trainer adds unnamed final weights.
        for backend in ("sd-scripts", "musubi-tuner"):
            with self.subTest(backend=backend):
                self.assertEqual(MODULE.expected_saves(backend, 0, 7, 2), [(2, 2), (4, 4), (6, 6), (7, None)])
                self.assertEqual(MODULE.expected_saves(backend, 4, 7, 2), [(6, 6), (7, None)])
                self.assertEqual(MODULE.expected_saves(backend, 0, 4, 2), [(2, 2), (4, 4), (4, None)])
        self.assertEqual(MODULE.expected_saves("ai-toolkit", 0, 7, 2), [(3, 2), (5, 4), (7, 6), (7, None)])
        self.assertEqual(MODULE.expected_saves("ai-toolkit", 4, 7, 2), [(7, 6), (7, None)])
        self.assertEqual(MODULE.expected_saves("ai-toolkit", 2, 4, 2), [(4, None)])
        self.assertEqual(MODULE.expected_state_steps(MODULE.expected_saves("sd-scripts", 0, 7, 2)), [6, 7])
        self.assertEqual(MODULE.expected_state_steps(MODULE.expected_saves("ai-toolkit", 2, 4, 2)), [4])

    def test_the_scenario_crosses_epochs_and_starts_resumes_on_the_cadence(self) -> None:
        runs = {run.key: run for run in MODULE.SCENARIO}
        steps_per_epoch = len(MODULE.BUCKET_SIZES)
        self.assertGreater(runs["multi-split"].steps, steps_per_epoch)
        self.assertNotEqual(runs["multi-resume"].steps % steps_per_epoch, 0)
        for run in MODULE.SCENARIO:
            if run.resume_of:
                self.assertEqual(runs[run.resume_of].steps % MODULE.CADENCE, 0)
                compared, _ = MODULE.COMPARISONS[run.key]
                self.assertEqual(runs[compared].steps, run.steps)

    def test_check_run_passes_a_run_that_kept_its_promises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run_id = "20260101-0000_conf-sd-multi-resume_abcd"
            _finished_run(workspace, run_id, weights=[f"{run_id}-step00000006.safetensors", f"{run_id}.safetensors"], states=[6, 7], last_step=7)
            self.assertEqual(MODULE.check_run(workspace, run_id, "sd-scripts", 4, 7), {"P1": [], "P2": [], "P4": [], "P7": []})

    def test_check_run_reports_steps_names_and_records_that_break_a_promise(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            # A Resume from 4 that named its checkpoint by the steps it added (2, not 6),
            # recorded a step it did not reach, and published no state at its target.
            run_id = "20260101-0000_conf-musubi-multi-resume_abcd"
            run_dir = _finished_run(workspace, run_id, weights=[f"{run_id}-step00000002.safetensors", f"{run_id}.safetensors"], states=[6], last_step=5)
            problems = MODULE.check_run(workspace, run_id, "musubi-tuner", 4, 7)
            self.assertEqual(len(problems["P1"]), 2)
            self.assertIn("weight file steps [None, 2], expected [None, 6]", problems["P2"][0])
            self.assertIn("published state steps [6], expected [6, 7]", problems["P2"][1])
            # A state whose own counters disagree with the step it is published at.
            payload = next((workspace / "artifacts" / "training-state").glob("*/payload"))
            (payload / "train_state.json").write_text(json.dumps({"current_step": 5}), encoding="utf-8")
            self.assertIn("state at step 6 records train_state.json:current_step = 5", MODULE.check_run(workspace, run_id, "musubi-tuner", 4, 7)["P2"])
            # The image that ran is not the compiled one, and an output vanished.
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            (run_dir / "realizations" / "r1.json").write_text(json.dumps({"image_identity": {"reference": "other:dev"}}), encoding="utf-8")
            (run_dir / "outputs" / f"{run_id}.safetensors").unlink()
            problems = MODULE.check_run(workspace, run_id, "musubi-tuner", 4, 7)
            self.assertTrue(problems["P4"] and problems["P7"])
            self.assertNotIn("P6", problems)
            (run_dir / "status.json").write_text(json.dumps({**status, "host": "runpod"}), encoding="utf-8")
            self.assertEqual(MODULE.check_run(workspace, run_id, "musubi-tuner", 4, 7)["P6"], ["no pod_stopped_at recorded"])

    def test_compare_states_ignores_metadata_and_save_ids_but_not_learned_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _state(root / "control", 7)
            _state(root / "same", 7)
            self.assertEqual(MODULE.compare_states(root / "same", root / "control", learned=True), [])
            _state(root / "drifted", 7, weight=b"\0\0\0@")
            self.assertEqual(MODULE.compare_states(root / "drifted", root / "control", learned=True), ["model.safetensors: 1 of 1 entries differ"])
            self.assertEqual(MODULE.compare_states(root / "drifted", root / "control", learned=False), [])
            _state(root / "short", 6)
            differences = MODULE.compare_states(root / "short", root / "control", learned=False)
            self.assertIn("scheduler.bin:last_epoch: 6 vs 7", differences)
            self.assertIn("scheduler.bin: 1 of 1 entries differ", differences)

    def test_a_scheduler_holding_anything_but_plain_values_is_not_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory)
            _torch_archive(payload / "scheduler.bin", {"last_epoch": Path("x")})
            self.assertTrue(str(MODULE.state_counters(payload)["scheduler.bin:last_epoch"]).startswith("unreadable"))

    def test_disk_problems_compare_the_peak_with_the_launch_estimate(self) -> None:
        self.assertEqual(MODULE.disk_problems({"count": 3, "bytes": 10}, {"count": 3, "bytes": 3 * 1024**3}), [])
        self.assertEqual(len(MODULE.disk_problems({"count": 4, "bytes": 10}, {"count": 3, "bytes": 3 * 1024**3})), 1)

    @posix_only(DATASET_IO)
    def test_every_conformance_run_compiles_and_the_bucket_dataset_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            self.assertEqual(_in_process_kura(workspace, "init").returncode, 0)
            fake_videos = {MODULE.VIDEO_DATASET: _fake_video_dataset, MODULE.VIDEO_BUCKET_DATASET: _fake_video_dataset}
            with patch.object(MODULE, "_kura", side_effect=_in_process_kura), patch.dict(MODULE._CREATORS, fake_videos):
                for backend in MODULE.CONFORMANCE_BACKENDS:
                    for dataset in MODULE.conformance_datasets(backend).values():
                        MODULE.ensure_dataset(workspace, dataset)
                self.assertEqual(validate_manifest(workspace / "datasets" / MODULE.BUCKET_DATASET), (3, []))
                for backend in MODULE.CONFORMANCE_BACKENDS:
                    for run in MODULE.SCENARIO:
                        if run.resume_of:
                            continue
                        with self.subTest(backend=backend, run=run.key):
                            smoke = MODULE.conformance_smoke(backend, run.dataset)
                            run_id = MODULE._create_run(workspace, "conformance", f"c-{run.key}", smoke, MODULE.build_run_fields("c", smoke, steps=run.steps))
                            status = json.loads((workspace / "runs" / run_id / "status.json").read_text(encoding="utf-8"))
                            self.assertEqual(status["state"], "compiled")
                            self.assertGreater(MODULE._checkpoint_estimate(workspace / "runs" / run_id)["count"], 0)

    def test_conformance_refuses_this_checkout_and_launches_nothing_without_yes(self) -> None:
        with self.assertRaisesRegex(SystemExit, "separate workspace"):
            MODULE.conformance(ROOT, ["sd-scripts"], runpod=False, gpu="x", yes=False)
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "workspace.yaml").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "one backend"):
                MODULE.conformance(workspace, ["sd-scripts", "ai-toolkit"], runpod=True, gpu="x", yes=True)
            with patch.object(MODULE, "_kura") as kura, contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(MODULE.conformance(workspace, list(MODULE.CONFORMANCE_BACKENDS), runpod=False, gpu="x", yes=False), 0)
            kura.assert_not_called()
            self.assertIn("Nothing launched", out.getvalue())
            self.assertEqual(sorted(path.name for path in workspace.iterdir()), ["workspace.yaml"])


if __name__ == "__main__":
    unittest.main()

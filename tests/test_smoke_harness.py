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


def _safetensors(path: Path, values: bytes, *, metadata: dict[str, str] | None = None) -> None:
    header = json.dumps({"__metadata__": metadata or {"ss_output_name": path.parent.name}, "lora.weight": {"dtype": "F32", "shape": [len(values) // 4], "data_offsets": [0, len(values)]}}).encode()
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
        self.assertEqual(fields["compute"]["gpu"], ["NVIDIA A100 80GB PCIe"])
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
            self.assertNotIn("executor_source", record)
            self.assertEqual(summary["gpu"], "A40")
            (run_dir / "status.json").write_text(json.dumps({**status, "publication_state": "blocked"}), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "did not pass verify"):
                MODULE.evidence(workspace, run_id, artifact="smoke-evidence/x.yaml")


    def test_expected_saves_follow_the_cadence_and_name_the_held_step(self) -> None:
        # Every trainer saves after update m when m % cadence == 0, names it m, and adds
        # unnamed final weights.
        self.assertEqual(MODULE.expected_saves(0, 7, 2), [(2, 2), (4, 4), (6, 6), (7, None)])
        self.assertEqual(MODULE.expected_saves(4, 7, 2), [(6, 6), (7, None)])
        self.assertEqual(MODULE.expected_saves(0, 4, 2), [(2, 2), (4, 4), (4, None)])
        self.assertEqual(MODULE.expected_state_steps(MODULE.expected_saves(0, 7, 2)), [6, 7])

    def test_the_scenario_crosses_epochs_and_starts_resumes_on_the_cadence(self) -> None:
        runs = {run.key: run for run in MODULE.SCENARIO}
        steps_per_epoch = len(MODULE.BUCKET_SIZES)
        self.assertGreater(runs["multi-split"].steps, steps_per_epoch)
        self.assertNotEqual(runs["multi-resume"].steps % steps_per_epoch, 0)
        for run in MODULE.SCENARIO:
            if run.resume_of:
                self.assertEqual(runs[run.resume_of].steps % MODULE.CADENCE, 0)
                for control in MODULE.COMPARISONS[run.key]:
                    self.assertEqual(runs[control].steps, run.steps)
                    self.assertIsNone(runs[control].resume_of)
        # Every backend runs one fresh run with more step saves than AI-Toolkit's own default keeps
        # (five), so a trainer that prunes a run setting no retention fails P2's weight-file check.
        for backend in MODULE.CONFORMANCE_BACKENDS:
            most = max(len(MODULE.expected_saves(0, run.steps, MODULE.CADENCE)) - 1 for run in MODULE.scenario_runs(backend) if not run.resume_of)
            self.assertGreater(most, 5, backend)
        # The one-item Resume has two uninterrupted controls, so a nondeterministic trainer shows itself.
        self.assertEqual(len(MODULE.COMPARISONS["one-resume"]), 2)

    def test_check_run_passes_a_run_that_kept_its_promises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run_id = "20260101-0000_conf-sd-multi-resume_abcd"
            _finished_run(workspace, run_id, weights=[f"{run_id}-step00000006.safetensors", f"{run_id}.safetensors"], states=[6, 7], last_step=7)
            self.assertEqual(MODULE.check_run(workspace, run_id, 4, 7), {"P1": [], "P2": [], "P4": [], "P7": []})

    def test_check_run_reports_steps_names_and_records_that_break_a_promise(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            # A Resume from 4 that named its checkpoint by the steps it added (2, not 6),
            # recorded a step it did not reach, and published no state at its target.
            run_id = "20260101-0000_conf-musubi-multi-resume_abcd"
            run_dir = _finished_run(workspace, run_id, weights=[f"{run_id}-step00000002.safetensors", f"{run_id}.safetensors"], states=[6], last_step=5)
            problems = MODULE.check_run(workspace, run_id, 4, 7)
            self.assertEqual(len(problems["P1"]), 2)
            self.assertIn("weight file steps [None, 2], expected [None, 6]", problems["P2"][0])
            self.assertIn("published state steps [6], expected [6, 7]", problems["P2"][1])
            # A state whose own counters disagree with the step it is published at.
            payload = next((workspace / "artifacts" / "training-state").glob("*/payload"))
            (payload / "train_state.json").write_text(json.dumps({"current_step": 5}), encoding="utf-8")
            self.assertIn("state at step 6 records train_state.json:current_step = 5", MODULE.check_run(workspace, run_id, 4, 7)["P2"])
            # The image that ran is not the compiled one, and an output vanished.
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            (run_dir / "realizations" / "r1.json").write_text(json.dumps({"image_identity": {"reference": "other:dev"}}), encoding="utf-8")
            (run_dir / "outputs" / f"{run_id}.safetensors").unlink()
            problems = MODULE.check_run(workspace, run_id, 4, 7)
            self.assertTrue(problems["P4"] and problems["P7"])
            self.assertNotIn("P6", problems)
            (run_dir / "status.json").write_text(json.dumps({**status, "host": "runpod"}), encoding="utf-8")
            self.assertEqual(MODULE.check_run(workspace, run_id, 4, 7)["P6"], ["no pod_stopped_at recorded"])

    def test_check_run_holds_a_weight_to_the_update_count_its_trainer_recorded(self) -> None:
        # AI-Toolkit writes training_info.step into each weight file: its own count of updates.
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run_id = "20260101-0000_conf-ai-multi-resume_abcd"
            run_dir = _finished_run(workspace, run_id, weights=[f"{run_id}_000000006.safetensors", f"{run_id}.safetensors"], states=[6, 7], last_step=7)
            for name, step in ((f"{run_id}_000000006.safetensors", 6), (f"{run_id}.safetensors", 7)):
                _safetensors(run_dir / "outputs" / name, b"\0\0\0\0", metadata={"training_info": json.dumps({"step": step, "epoch": 0})})
            self.assertEqual(MODULE.check_run(workspace, run_id, 4, 7)["P2"], [])
            _safetensors(run_dir / "outputs" / f"{run_id}.safetensors", b"\0\0\0\0", metadata={"training_info": json.dumps({"step": 9})})
            self.assertEqual(MODULE.check_run(workspace, run_id, 4, 7)["P2"], [f"{run_id}.safetensors records training_info.step 9, holds step 7"])

    def test_check_run_compares_image_digests_not_reference_spellings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run_id = "20260101-0000_conf-sd-one-resume_abcd"
            run_dir = _finished_run(workspace, run_id, weights=[f"{run_id}.safetensors"], states=[4], last_step=4)
            observed = "sha256:" + "b" * 64
            (run_dir / "resolved" / "env.lock").write_text(yaml.safe_dump({
                "selected_image": "nomadoor/kura-sd-scripts@sha256:" + "a" * 64,
                "selected_image_identity": {"reference": "x", "pinning": {"strength": "content-hash", "value": observed}},
            }), encoding="utf-8")
            realization = run_dir / "realizations" / "r1.json"
            # A Docker run records the bare digest it launched: the reference's, or (a local
            # Resume) the image ID compile observed for it.
            for used, ok in (("sha256:" + "a" * 64, True), (observed, True), ("sha256:" + "c" * 64, False), ("other:dev", False)):
                with self.subTest(used=used):
                    realization.write_text(json.dumps({"image_identity": {"reference": used}}), encoding="utf-8")
                    self.assertEqual(MODULE.check_run(workspace, run_id, 2, 4)["P4"] == [], ok)

    def test_compare_states_ignores_metadata_and_save_ids_but_not_counters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _state(root / "control", 7)
            _state(root / "same", 7)
            with patch.object(MODULE.subprocess, "run") as run:
                self.assertEqual(MODULE.compare_states(root / "same", root / "control"), [])
                _state(root / "drifted", 7, weight=b"\0\0\0@")
                self.assertEqual(MODULE.compare_states(root / "drifted", root / "control"), [])
            run.assert_not_called()
            _state(root / "short", 6)
            differences = MODULE.compare_states(root / "short", root / "control")
            self.assertIn("scheduler.bin:last_epoch: 6 vs 7", differences)
            self.assertIn("scheduler.bin: 1 of 1 entries differ", differences)

    def test_compare_states_compares_learned_values_in_the_trainer_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _state(root / "a", 7)
            _state(root / "b", 7)
            image = "nomadoor/kura-sd-scripts@sha256:" + "a" * 64
            answer = subprocess.CompletedProcess([], 0, json.dumps({"model.safetensors": [0, 4], "optimizer.bin": [2, 9]}) + "\n", "")
            with patch.object(MODULE.subprocess, "run", return_value=answer) as run:
                self.assertEqual(MODULE.compare_states(root / "a", root / "b", image=image), ["optimizer.bin: 2 of 9 entries differ"])
            argv = run.call_args.args[0]
            self.assertEqual(argv[:3], ["docker", "run", "--rm"])
            self.assertIn("--network", argv)
            self.assertEqual(argv[argv.index("--network") + 1], "none")
            self.assertIn(f"{(root / 'a').resolve()}:/a:ro", argv)
            self.assertIn(f"{(root / 'b').resolve()}:/b:ro", argv)
            self.assertIn(f"{MODULE.COMPARE_VALUES.parent}:/c:ro", argv)
            self.assertEqual(argv[argv.index(image) + 1:], ["-I", f"/c/{MODULE.COMPARE_VALUES.name}", "/a", "/b", "model.safetensors", "optimizer.bin"])
            failed = subprocess.CompletedProcess([], 1, "", "no such image")
            with patch.object(MODULE.subprocess, "run", return_value=failed):
                self.assertEqual(MODULE.compare_states(root / "a", root / "b", image=image), ["cannot compare values in the trainer image: no such image"])

    def test_resume_is_held_to_the_controls_only_when_they_agree(self) -> None:
        resumed, first, second = Path("r"), Path("c1"), Path("c2")
        image = "img@sha256:" + "a" * 64
        calls = []

        def compare(left: Path, right: Path, *, image: str | None = None) -> list[str]:
            calls.append((left, right, image))
            return outcomes[(left, image is not None)]

        with patch.object(MODULE, "compare_states", side_effect=compare):
            # Deterministic controls: the Resume must equal them in learned state.
            outcomes = {(second, True): [], (resumed, True): ["optimizer.bin: 1 of 9 entries differ"]}
            self.assertEqual(MODULE.resume_problems(resumed, [first, second], image), (["optimizer.bin: 1 of 9 entries differ"], []))
            # Controls that differ: only step counts and the scheduler are checked, and that is said.
            outcomes = {(second, True): ["model.safetensors: 3 of 4 entries differ"], (resumed, False): []}
            self.assertEqual(MODULE.resume_problems(resumed, [first, second], image), ([], ["controls differ: exact Resume not checkable"]))
            # A comparison that could not run is a failure, not a nondeterministic trainer.
            outcomes = {(second, True): [f"{MODULE.CANNOT_COMPARE}: no such image"], (resumed, False): []}
            self.assertEqual(MODULE.resume_problems(resumed, [first, second], image), ([f"{MODULE.CANNOT_COMPARE}: no such image"], []))
            # One control (the multi-item dataset): step counts and the scheduler only.
            calls.clear()
            outcomes = {(resumed, False): ["scheduler.bin: 1 of 1 entries differ"]}
            self.assertEqual(MODULE.resume_problems(resumed, [first], image), (["scheduler.bin: 1 of 1 entries differ"], []))
            self.assertEqual(calls, [(resumed, first, None)])

    def test_the_value_comparison_counts_differing_leaves(self) -> None:
        spec = importlib.util.spec_from_file_location("compare_state_values", MODULE.COMPARE_VALUES)
        assert spec is not None and spec.loader is not None
        compare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compare)

        class Tensor:
            def __init__(self, values: list[float], dtype: str = "f32") -> None:
                self.values, self.dtype, self.shape = values, dtype, (len(values),)

        class FakeTorch:
            @staticmethod
            def is_tensor(value: object) -> bool:
                return isinstance(value, Tensor)

            @staticmethod
            def equal(a: Tensor, b: Tensor) -> bool:
                return a.values == b.values

        state = {"state": {0: {"step": Tensor([7.0]), "exp_avg": Tensor([0.5, 0.25])}}, "param_groups": [{"lr": 1e-4, "params": [0]}]}
        same = {"state": {0: {"step": Tensor([7.0]), "exp_avg": Tensor([0.5, 0.25])}}, "param_groups": [{"lr": 1e-4, "params": [0]}]}
        self.assertEqual(compare.leaves(state, same, torch=FakeTorch), (0, 4))
        drifted = {"state": {0: {"step": Tensor([7.0]), "exp_avg": Tensor([0.5, 0.75])}}, "param_groups": [{"lr": 1e-4, "params": [0]}]}
        self.assertEqual(compare.leaves(state, drifted, torch=FakeTorch), (1, 4))
        self.assertEqual(compare.leaves({"a": Tensor([1.0])}, {"a": Tensor([1.0], "bf16")}, torch=FakeTorch), (1, 1))
        self.assertEqual(compare.leaves({"a": 1}, {"b": 1}), (2, 2))
        self.assertEqual(compare.leaves([1, 2], [1, 2, 3]), (1, 1))

    def test_a_scheduler_holding_anything_but_plain_values_is_not_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory)
            _torch_archive(payload / "scheduler.bin", {"last_epoch": Path("x")})
            self.assertTrue(str(MODULE.state_counters(payload)["scheduler.bin:last_epoch"]).startswith("unreadable"))

    def test_disk_problems_compare_the_peak_bytes_with_the_bound_and_the_watch(self) -> None:
        bound = {"rank": 4, "bytes": 3 * 1024**3, "weights_files": 12}
        self.assertEqual(MODULE.disk_problems({"bytes": 3 * 1024**3}, bound), [])
        self.assertEqual(len(MODULE.disk_problems({"bytes": 3 * 1024**3 + 1}, bound)), 1)
        self.assertIn("disk watch stopped", MODULE.disk_problems({"bytes": 1}, bound, "stopped by the disk watch: low")[0])
        self.assertIn("no rank", MODULE.disk_problems({"bytes": 1}, {"rank": None, "bytes": 0, "weights_files": 3})[0])

    def test_a_peak_sample_counts_weights_sized_files_and_survives_a_folder_that_disappears(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run_dir = workspace / "runs" / "r1"
            (run_dir / "outputs" / "r1-step00000002-state").mkdir(parents=True)
            _safetensors(run_dir / "outputs" / "r1-step00000002.safetensors", b"\0\0\0\0")
            (run_dir / "outputs" / "r1-step00000002-state" / "optimizer.bin").write_bytes(b"x" * 100)
            (run_dir / "outputs" / "sample.png").write_bytes(b"x" * 1000)
            store = workspace / "artifacts" / "training-state" / "state-a" / "payload"
            store.mkdir(parents=True)
            (store / "optimizer.bin").write_bytes(b"x" * 50)
            peak = {"bytes": 0, "store_baseline": 20}
            with patch.object(Path, "rglob", side_effect=FileNotFoundError("staging folder moved")):
                MODULE._sample_peak(run_dir, peak)
            self.assertEqual(peak["bytes"], 0)
            MODULE._sample_peak(run_dir, peak)
            weights = (run_dir / "outputs" / "r1-step00000002.safetensors").stat().st_size
            self.assertEqual(peak["bytes"], weights + 100 + 30)

    def test_the_ai_toolkit_model_is_held_to_its_pinned_revision(self) -> None:
        smoke = MODULE.conformance_smoke("ai-toolkit", "one")
        self.assertEqual(smoke.model_revision, "3ee6c9f225f088ad5d35b624b6514b091e6a4849")
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            self.assertIn("not cached", MODULE.model_revision_problems(cache, smoke)[0])
            ref = cache / "hub" / "models--hf-internal-testing--tiny-stable-diffusion-pipe" / "refs" / "main"
            ref.parent.mkdir(parents=True)
            ref.write_text("0" * 40, encoding="utf-8")
            self.assertIn("0" * 40, MODULE.model_revision_problems(cache, smoke)[0])
            self.assertIn("main", MODULE._cache_state(cache, smoke))
            ref.write_text(smoke.model_revision, encoding="utf-8")
            self.assertEqual(MODULE.model_revision_problems(cache, smoke), [])
        self.assertEqual(MODULE.model_revision_problems(Path(directory), MODULE.conformance_smoke("sd-scripts", "one")), [])

    def test_the_runpod_pass_leases_each_pod_for_an_hour_and_prices_that_lease(self) -> None:
        from unittest.mock import MagicMock

        captured: list[list[str]] = []

        def popen(argv, **_kwargs):
            captured.append(list(argv))
            process = MagicMock()
            process.poll.return_value = 0
            process.returncode = 0
            return process

        with tempfile.TemporaryDirectory() as directory, patch.object(MODULE.subprocess, "Popen", side_effect=popen):
            MODULE._execute(Path(directory), "run-1", runpod=True)
            MODULE._execute(Path(directory), "run-2", runpod=False)
        self.assertEqual(captured[0][-3:], ["--yes", "--max-lease", "1h"])
        self.assertNotIn("--max-lease", captured[1])

        measurement = {"status": "ok", "candidates": [
            {"gpu_type_id": "NVIDIA RTX A4500", "clouds": [{"cloud_type": "COMMUNITY", "available": True, "price_per_hour": 0.19}]},
            {"gpu_type_id": "NVIDIA RTX A5000", "clouds": [{"cloud_type": "COMMUNITY", "available": False, "price_per_hour": None, "stock_status": None}]},
        ]}
        payload = {"measurement": measurement, "selected_gpu_type_ids": ["NVIDIA RTX A4500", "NVIDIA RTX A5000"]}
        with tempfile.TemporaryDirectory() as directory, \
                patch("kura.run_commands.plan._runpod_capacity_payload", return_value=payload), patch("kura.secrets.load_secrets"):
            ceiling = MODULE.runpod_cost_ceiling(Path(directory), {}, MODULE.replace(MODULE.conformance_smoke("ai-toolkit", "one"), executor="runpod"))
        self.assertEqual(ceiling["max_lease_sec"], MODULE.CONFORMANCE_MAX_LEASE_SEC)
        self.assertAlmostEqual(ceiling["max_cost"], 0.19)
        # A GPU with no current price is left out of the runs, so the ceiling covers every GPU they ask for.
        self.assertEqual(ceiling["priced_gpu_type_ids"], ["NVIDIA RTX A4500"])
        self.assertEqual(ceiling["unpriced"], [])
        self.assertEqual(MODULE.build_run_fields("c", MODULE.replace(MODULE.conformance_smoke("ai-toolkit", "one"), executor="runpod", gpu=MODULE.CONFORMANCE_GPUS))["compute"]["gpu"], list(MODULE.CONFORMANCE_GPUS))

    def test_the_runpod_pass_shows_its_ceiling_before_creating_anything_and_needs_a_known_one(self) -> None:
        known = {"max_lease": "12h", "max_lease_sec": 43200, "hourly_price": 0.5, "max_cost": 6.0, "basis": "the highest current quote", "unpriced": [], "reason": None}
        unknown = {**known, "hourly_price": None, "max_cost": None, "reason": "RUNPOD_API_KEY is not set"}
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "workspace.yaml").write_text("{}\n", encoding="utf-8")
            with patch.object(MODULE, "runpod_cost_ceiling", return_value=known), patch.object(MODULE, "_kura") as kura, contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(MODULE.conformance(workspace, ["ai-toolkit"], runpod=True, gpu="NVIDIA A40", yes=False), 0)
            kura.assert_not_called()
            self.assertIn("at most about $6.00", out.getvalue())
            self.assertIn("Nothing launched", out.getvalue())
            with patch.object(MODULE, "runpod_cost_ceiling", return_value=unknown), patch.object(MODULE, "_kura") as kura, contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(MODULE.conformance(workspace, ["ai-toolkit"], runpod=True, gpu="NVIDIA A40", yes=False), 0)
            self.assertIn("unknown: RUNPOD_API_KEY is not set", out.getvalue())
            self.assertNotIn("Add --yes", out.getvalue())
            with patch.object(MODULE, "runpod_cost_ceiling", return_value=unknown), patch.object(MODULE, "_kura") as kura, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(SystemExit, "known cost ceiling.*RUNPOD_API_KEY"):
                    MODULE.conformance(workspace, ["ai-toolkit"], runpod=True, gpu="NVIDIA A40", yes=True)
            kura.assert_not_called()
            self.assertEqual(sorted(path.name for path in workspace.iterdir()), ["workspace.yaml"])

    def test_conformance_refuses_a_workspace_on_a_windows_drive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "workspace.yaml").write_text("{}\n", encoding="utf-8")
            with patch("kura.storage.is_wsl", return_value=True), patch("kura.storage._findmnt_for", return_value={"available": True, "fstype": "9p"}):
                with self.assertRaisesRegex(SystemExit, "Windows drive.*WSL disk"):
                    MODULE.conformance(workspace, ["sd-scripts"], runpod=False, gpu="x", yes=False)
            with patch("kura.storage.is_wsl", return_value=True), patch("kura.storage._findmnt_for", return_value={"available": True, "fstype": "ext4"}):
                self.assertIsNone(MODULE.windows_drive_problem(workspace))
            with patch("kura.storage.is_wsl", return_value=True), patch("kura.storage._findmnt_for", return_value={"available": False}):
                self.assertIn(str(Path("/mnt/d/kura")), str(MODULE.windows_drive_problem(Path("/mnt/d/kura"))))
            # findmnt's filesystem wins over the path: an ext4 disk mounted under /mnt/<letter> is fine.
            with patch("kura.storage.is_wsl", return_value=True), patch("kura.storage._findmnt_for", return_value={"available": True, "fstype": "ext4"}):
                self.assertIsNone(MODULE.windows_drive_problem(Path("/mnt/d/kura")))
        with patch("kura.storage.is_wsl", return_value=False):
            self.assertIsNone(MODULE.windows_drive_problem(Path("/mnt/d/kura")))

    def test_a_resume_that_does_not_run_fails_p3(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "workspace.yaml").write_text("{}\n", encoding="utf-8")
            passing = {"P1": [], "P2": [], "P4": [], "P7": []}
            with (
                patch.object(MODULE, "ensure_dataset"), patch.object(MODULE, "_tree_digest", return_value="d"),
                patch.object(MODULE, "_create_run", side_effect=lambda workspace, experiment, slug, smoke, fields: slug),
                patch.object(MODULE, "_kura", side_effect=lambda workspace, *args, **kwargs: subprocess.CompletedProcess(args, 0, args[args.index("--slug") + 1] + "\n", "")),
                patch.object(MODULE, "_compile"),
                patch.object(MODULE, "_execute", side_effect=lambda workspace, run_id, runpod: (1 if "resume" in run_id else 0, {"bytes": 1}, "trainer failed")),
                patch.object(MODULE, "check_run", return_value=passing), patch.object(MODULE, "_own_write_bound", return_value={"rank": 4, "bytes": 5, "weights_files": 1}),
                patch.object(MODULE, "_stop_reason", return_value=None),
                contextlib.redirect_stdout(io.StringIO()) as out,
            ):
                self.assertEqual(MODULE.conformance(workspace, ["sd-scripts"], runpod=False, gpu="x", yes=True), 1)
            rows = {line.split()[0]: line.split()[1] for line in out.getvalue().splitlines() if line[:2] in {"P1", "P3"} and len(line.split()) == 2}
            self.assertEqual(rows, {"P1": "FAIL", "P3": "FAIL"})
            self.assertIn("sd-scripts P3 one-resume: not checked: one-resume did not complete", out.getvalue())

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
                    for run in MODULE.scenario_runs(backend):
                        if run.resume_of:
                            continue
                        with self.subTest(backend=backend, run=run.key):
                            smoke = MODULE.conformance_smoke(backend, run.dataset, accumulation=run.accumulation)
                            run_id = MODULE._create_run(workspace, "conformance", f"c-{run.key}", smoke, MODULE.build_run_fields("c", smoke, steps=run.steps))
                            status = json.loads((workspace / "runs" / run_id / "status.json").read_text(encoding="utf-8"))
                            self.assertEqual(status["state"], "compiled")
                            # Every scenario run keeps training state; compile froze that with its command.
                            lock = json.loads((workspace / "runs" / run_id / "resolved" / "backend-command.lock.json").read_text(encoding="utf-8"))
                            self.assertIs(lock["training_state_managed"], True)
                            self.assertGreater(MODULE._own_write_bound(workspace / "runs" / run_id)["bytes"], 0)

    def test_the_accumulation_case_runs_on_ai_toolkit_and_sd_scripts_only(self) -> None:
        accumulated = [run for run in MODULE.SCENARIO if run.accumulation != 1]
        self.assertEqual([(run.key, run.accumulation, run.resume_of) for run in accumulated], [("multi-accum-2", 2, None)])
        for backend in MODULE.CONFORMANCE_BACKENDS:
            with self.subTest(backend=backend):
                keys = [run.key for run in MODULE.scenario_runs(backend)]
                self.assertEqual("multi-accum-2" in keys, backend in ("ai-toolkit", "sd-scripts"))
        for backend in ("ai-toolkit", "sd-scripts"):
            smoke = MODULE.conformance_smoke(backend, "multi", accumulation=2)
            self.assertEqual(MODULE.build_run_fields("c", smoke, steps=4)["backend"]["config"]["gradient_accumulation_steps"], 2)
            self.assertNotIn("gradient_accumulation_steps", MODULE.conformance_smoke(backend, "multi").config)

    def test_accumulation_problems_hold_the_optimizer_counter_to_the_recipe_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run_id = "20260101-0000_conf-ai-multi-accum-2_abcd"
            _finished_run(workspace, run_id, weights=[f"{run_id}.safetensors"], states=[2, 4], last_step=4)
            answer = subprocess.CompletedProcess([], 0, json.dumps({"optimizer.bin": [4]}) + "\n", "")
            with patch.object(MODULE.subprocess, "run", return_value=answer) as run:
                self.assertEqual(MODULE.accumulation_problems(workspace, run_id, 4), [])
            argv = run.call_args.args[0]
            self.assertEqual(argv[:3], ["docker", "run", "--rm"])
            self.assertEqual(argv[argv.index("--network") + 1], "none")
            self.assertEqual(argv[argv.index("kura-sd-scripts@sha256:" + "a" * 64) + 1:], ["-I", f"/c/{MODULE.COMPARE_VALUES.name}", "--optimizer-steps", "/a", "optimizer.bin"])
            # The legacy AI-Toolkit accumulation: 4 steps made 2 optimizer updates.
            halved = subprocess.CompletedProcess([], 0, json.dumps({"optimizer.bin": [2]}) + "\n", "")
            with patch.object(MODULE.subprocess, "run", return_value=halved):
                self.assertEqual(MODULE.accumulation_problems(workspace, run_id, 4), ["optimizer.bin holds [2] optimizer updates, recipe steps 4"])
            failed = subprocess.CompletedProcess([], 1, "", "no such image")
            with patch.object(MODULE.subprocess, "run", return_value=failed):
                self.assertEqual(MODULE.accumulation_problems(workspace, run_id, 4), ["cannot read the optimizer's update counter in the trainer image: no such image"])
            self.assertEqual(MODULE.accumulation_problems(workspace, run_id, 6), ["no training state at step 6 to read the optimizer's update counter from"])

    def test_the_optimizer_counter_reader_returns_each_distinct_step(self) -> None:
        spec = importlib.util.spec_from_file_location("compare_state_values", MODULE.COMPARE_VALUES)
        assert spec is not None and spec.loader is not None
        compare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compare)

        class Scalar:
            def __init__(self, value: float) -> None:
                self.value = value

            def item(self) -> float:
                return self.value

        self.assertEqual(compare.optimizer_steps({"state": {0: {"step": Scalar(4.0)}, 1: {"step": 4}}}), [4])
        self.assertEqual(compare.optimizer_steps({"state": {0: {"step": 2}, 1: {"step": 4}}}), [2, 4])
        self.assertEqual(compare.optimizer_steps({"state": {}}), [])

    def test_one_scenario_run_can_be_run_alone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "workspace.yaml").write_text("{}\n", encoding="utf-8")
            passing = {"P1": [], "P2": [], "P4": [], "P7": []}
            created = []
            with (
                patch.object(MODULE, "ensure_dataset"), patch.object(MODULE, "_tree_digest", return_value="d"),
                patch.object(MODULE, "_create_run", side_effect=lambda workspace, experiment, slug, smoke, fields: created.append((slug, fields)) or slug),
                patch.object(MODULE, "_execute", return_value=(0, {"bytes": 1}, "")),
                patch.object(MODULE, "check_run", return_value=passing), patch.object(MODULE, "_own_write_bound", return_value={"rank": 4, "bytes": 5, "weights_files": 1}),
                patch.object(MODULE, "_stop_reason", return_value=None),
                patch.object(MODULE, "model_revision_problems", return_value=[]),
                patch.object(MODULE, "accumulation_problems", return_value=["optimizer.bin holds [2] optimizer updates, recipe steps 4"]) as accumulation,
                contextlib.redirect_stdout(io.StringIO()) as out,
            ):
                self.assertEqual(MODULE.conformance(workspace, ["ai-toolkit"], runpod=False, gpu="x", yes=True, runs=["multi-accum-2"]), 1)
            self.assertEqual([slug for slug, _ in created], ["conf-ai-multi-accum-2"])
            self.assertEqual(created[0][1]["backend"]["config"]["gradient_accumulation_steps"], 2)
            accumulation.assert_called_once_with(workspace, "conf-ai-multi-accum-2", 4)
            rows = {line.split()[0]: line.split()[1] for line in out.getvalue().splitlines() if line[:1] == "P" and len(line.split()) == 2}
            self.assertEqual((rows["P1"], rows["P3"]), ("FAIL", "-"))
            with self.assertRaisesRegex(SystemExit, "no scenario run"):
                MODULE.conformance(workspace, ["musubi-tuner"], runpod=False, gpu="x", yes=False, runs=["multi-accum-2"])

    def test_conformance_without_a_backend_runs_the_backends_that_have_the_selected_runs(self) -> None:
        self.assertEqual(MODULE.conformance_backends(None, None), list(MODULE.CONFORMANCE_BACKENDS))
        accum = MODULE.conformance_backends(None, ["multi-accum-2"])
        self.assertTrue(accum)
        self.assertNotIn("musubi-tuner", accum)
        self.assertTrue(all(MODULE.scenario_runs(backend, ["multi-accum-2"]) for backend in accum))
        # A named backend is kept as named, so conformance refuses one without the runs.
        self.assertEqual(MODULE.conformance_backends(["musubi-tuner"], ["multi-accum-2"]), ["musubi-tuner"])
        with patch.object(MODULE, "scenario_runs", return_value=[]), self.assertRaisesRegex(SystemExit, "no backend has scenario run"):
            MODULE.conformance_backends(None, ["multi-accum-2"])

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

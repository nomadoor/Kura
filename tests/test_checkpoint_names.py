"""One parser reads the step from a checkpoint name for every backend, wherever Kura needs it."""

from __future__ import annotations

import unittest

NAMES = {
    # sd-scripts and Musubi Tuner
    "vivi-step00000100.safetensors": 100,
    "vivi-step00000100-state": 100,
    # AI-Toolkit
    "vivi_000000500.safetensors": 500,
    "my_lora_v2_000001000.safetensors": 1000,
    # AI-Toolkit names checkpoints after the run ID, which ends in 4 hex digits
    "20261008-2356_smoke-test-gc_7868_000000025.safetensors": 25,
    # final weights carry no step, even when the run ID ends in digits
    "20261008-2356_smoke-test-gc_7868.safetensors": None,
    "vivi.safetensors": None,
    "my_lora_v2.safetensors": None,
}


class CheckpointNameTests(unittest.TestCase):
    def test_every_reader_takes_the_step_from_the_same_parser(self) -> None:
        from kura import monitor
        from kura.run_commands import experiment, render_completion, runpod_ssh
        from kura.training_artifacts import checkpoint_step

        for name, step in NAMES.items():
            with self.subTest(name=name):
                self.assertEqual(checkpoint_step(name), step)
        for module in (monitor, experiment, render_completion, runpod_ssh):
            with self.subTest(module=module.__name__):
                self.assertIs(module.checkpoint_step, checkpoint_step)


def _safetensors(path, header: str, data_length: int) -> None:
    raw = header.encode("utf-8")
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + b"\0" * data_length)


class OneCheckpointReadingTests(unittest.TestCase):
    def test_every_weight_file_is_checked_by_one_validator(self) -> None:
        from kura import artifact_publication, training_artifacts
        from kura.run_commands import runpod_ssh

        owner = training_artifacts.validate_safetensors_file
        for module in (artifact_publication, runpod_ssh):
            with self.subTest(module=module.__name__):
                self.assertIs(module.validate_safetensors_file, owner)

    def test_the_validator_keeps_every_check_either_copy_had(self) -> None:
        import tempfile
        from pathlib import Path

        from kura.training_artifacts import validate_safetensors_file

        cases = {
            "valid": ('{"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}', 4, None),
            # A dtype Kura has no size for is still a valid file.
            "newer dtype": ('{"w": {"dtype": "F8_E8M0", "shape": [4], "data_offsets": [0, 4]}}', 4, None),
            "short tensor": ('{"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}', 4, "byte size"),
            "duplicate key": ('{"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}, "w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}', 4, "duplicate"),
            "metadata": ('{"__metadata__": {"a": 1}, "w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}', 4, "metadata"),
        }
        for name, (header, length, refusal) in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.safetensors"
                _safetensors(path, header, length)
                if refusal is None:
                    validate_safetensors_file(path)
                else:
                    with self.assertRaisesRegex(ValueError, refusal):
                        validate_safetensors_file(path)

    def test_a_runpod_pull_reads_steps_as_the_local_side_does(self) -> None:
        from kura.run_commands.runpod_ssh import _select_remote_outputs

        # A stale or foreign step field never decides: the name does, through checkpoint_step.
        items = [
            {"name": "flux_lora_2048.safetensors", "path": "/w/a", "size": 1, "step": 2048},
            {"name": "vivi-step00000100.safetensors", "path": "/w/b", "size": 1},
            {"name": "vivi_000000200.safetensors", "path": "/w/c", "size": 1},
        ]
        self.assertEqual([item["name"] for item in _select_remote_outputs(items)], ["vivi_000000200.safetensors"])
        self.assertIsNone(next(item for item in items if item["name"] == "flux_lora_2048.safetensors")["step"])

    def test_the_pod_listing_carries_the_step_every_mirror_records(self) -> None:
        import json
        import subprocess
        from unittest.mock import patch

        from kura.run_commands.runpod_ssh import _runpod_remote_outputs

        # The Pod reports names only; every caller, including the mid-run mirror, gets the step from them.
        listed = [{"path": "/w/vivi-step00000250.safetensors", "name": "vivi-step00000250.safetensors", "size": 1, "mtime_ns": 1},
                  {"path": "/w/vivi.safetensors", "name": "vivi.safetensors", "size": 1, "mtime_ns": 1}]
        with patch("kura.run_commands.runpod_ssh.subprocess.run", return_value=subprocess.CompletedProcess([], 0, json.dumps(listed), "")), \
                patch("kura.run_commands.runpod_ssh._ssh_base", return_value=["ssh"]):
            items = _runpod_remote_outputs({"ip": "h", "port": 22, "key": "k"}, workspace="/workspace", run_id="vivi")
        self.assertEqual([item["step"] for item in items], [250, None])


if __name__ == "__main__":
    unittest.main()

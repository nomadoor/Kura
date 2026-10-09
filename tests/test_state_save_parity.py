"""Which state directories a run publishes, and how often its trainer saves state, are each decided once."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml

import kura.backends.musubi_command as musubi_command
import kura.backends.sd_scripts as sd_scripts
import kura.training_artifacts as training_artifacts
from kura.run_commands.runpod_ssh import _pull_remote_training_state_items
from kura.training_artifacts import (
    FINAL_STATE_STEP,
    managed_state_save_args,
    publish_completed_training_states,
    run_output_name,
    state_directory_step,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tests.test_resume_steps import as_resume, musubi_run, sd_scripts_run  # noqa: E402
from tests.test_training_resume import _safetensors_bytes, _torch_archive_bytes, _write_state_marker  # noqa: E402


def _cadence(run: dict[str, Any]) -> tuple[str, str]:
    """The state save cadence and retained window each trainer is given."""
    if run["backend"]["name"] == "musubi-tuner":
        script = musubi_command.command_musubi_tuner(run)["argv"][2]
        every = re.findall(r"--save_every_n_steps (\d+)", script)
        window = re.findall(r"--save_last_n_steps_state (\d+)", script)
    else:
        script = sd_scripts.command_sd_scripts(run)["argv"][2]
        every = re.findall(r'"--save_every_n_steps","(\d+)"', script)
        window = re.findall(r'"--save_last_n_steps_state","(\d+)"', script)
    assert len(every) == 1 and len(window) == 1, (every, window)
    return every[0], window[0]


class StateDirectoryNameTests(unittest.TestCase):
    def test_one_rule_reads_a_state_directory_name(self) -> None:
        self.assertEqual(state_directory_step("run-a-step0200-state", "run-a", allow_final=False), 200)
        self.assertEqual(state_directory_step("run-a-step00000200-state", "run-a", allow_final=False), 200)
        self.assertIsNone(state_directory_step("other-step0300-state", "run-a", allow_final=False))
        self.assertIsNone(state_directory_step("run-a-step020-state", "run-a", allow_final=False))
        self.assertIsNone(state_directory_step("run-a-state", "run-a", allow_final=False))
        self.assertEqual(state_directory_step("run-a-state", "run-a", allow_final=True), FINAL_STATE_STEP)

    def test_output_name_is_the_run_id_on_resume_else_the_configured_name(self) -> None:
        run = sd_scripts_run()
        self.assertEqual(run_output_name(run), "derived")
        run["backend"]["config"]["output_name"] = "named"
        self.assertEqual(run_output_name(run), "named")
        self.assertEqual(run_output_name(as_resume(run)), "derived")


class ManagedStateCadenceTests(unittest.TestCase):
    def test_musubi_and_sd_scripts_save_state_on_one_cadence(self) -> None:
        cases = {
            "fresh": ({}, None, 2, ("1000", "1000")),
            "fresh-configured": ({"save_every_n_steps": 100}, None, 2, ("100", "100")),
            "fresh-one-generation": ({}, None, 1, ("1000", "1")),
            "resume": ({}, 200, 2, ("200", "200")),
            "resume-configured": ({"save_every_n_steps": 50}, 200, 2, ("50", "50")),
            "resume-one-generation": ({}, 200, 1, ("200", "1")),
        }
        for build in (musubi_run, sd_scripts_run):
            for label, (config, additional, keep, expected) in cases.items():
                run = build()
                run["backend"]["config"].update(config)
                run["recovery"] = {"training_state": {"enabled": True, "keep_generations": keep}}
                if additional is not None:
                    run = as_resume(run, additional=additional)
                name = run["backend"]["name"]
                with self.subTest(backend=name, case=label), patch.object(
                    musubi_command if name == "musubi-tuner" else sd_scripts,
                    "managed_state_save_args",
                    wraps=managed_state_save_args,
                ) as owner:
                    self.assertEqual(_cadence(run), expected)
                    owner.assert_called_once()

    def test_epoch_save_flags_are_refused_by_the_shared_rule(self) -> None:
        with self.assertRaisesRegex(ValueError, "epoch.*training-state"):
            managed_state_save_args(sd_scripts_run(), None, ["--save_every_n_epochs", "1"])


class StatePublicationParityTests(unittest.TestCase):
    NAMES = ("derived-step0200-state", "other-step0300-state", "derived-state")
    STEPS = {"derived-step0200-state": 200, "other-step0300-state": 300, "derived-state": 400}

    def _write_state(self, directory: Path, step: int) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl"):
            (directory / name).write_bytes(
                _safetensors_bytes(f"{name}-{step}".encode()) if name.endswith(".safetensors") else _torch_archive_bytes(f"{name}-{step}".encode())
            )
        (directory / "train_state.json").write_text(json.dumps({"current_epoch": 1, "current_step": step}) + "\n", encoding="utf-8")
        _write_state_marker(directory, "sd-scripts", step)
        return directory

    def _run_dir(self, root: Path) -> Path:
        run = sd_scripts_run()
        run["recovery"] = {"training_state": {"enabled": True, "keep_generations": 2}}
        run_dir = root / "runs" / "derived"
        (run_dir / "resolved").mkdir(parents=True)
        (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
        (run_dir / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
        return run_dir

    def test_docker_and_runpod_publish_the_same_state_directories_mid_run(self) -> None:
        import kura.run_commands.runpod_ssh as runpod_ssh

        published: dict[str, list[int]] = {}
        for executor in ("docker", "runpod"):
            with tempfile.TemporaryDirectory() as directory, patch.object(
                training_artifacts, "state_directory_step", wraps=state_directory_step,
            ) as local_owner, patch.object(runpod_ssh, "state_directory_step", wraps=state_directory_step) as remote_owner:
                root = Path(directory)
                run_dir = self._run_dir(root)
                if executor == "docker":
                    for name in self.NAMES:
                        self._write_state(run_dir / "outputs" / name, self.STEPS[name])
                    items = publish_completed_training_states(root, run_dir)
                    local_owner.assert_called()
                else:
                    sources = {name: self._write_state(root / "remote" / name, self.STEPS[name]) for name in self.NAMES}
                    listing = [
                        {
                            "path": f"/workspace/runs/derived/outputs/{name}",
                            "name": name,
                            "files": [
                                {"path": path.name, "size": path.stat().st_size, "mtime_ns": 1}
                                for path in sorted(source.iterdir())
                            ],
                        }
                        for name, source in sources.items()
                    ]

                    def fake_scp(command: list[str], **_: object) -> subprocess.CompletedProcess:
                        name = command[-2].rstrip("/.").rsplit("/", 1)[-1]
                        destination = Path(command[-1])
                        for path in sources[name].iterdir():
                            (destination / path.name).write_bytes(path.read_bytes())
                        return subprocess.CompletedProcess(command, 0, "", "")

                    with patch.object(runpod_ssh, "_run_bounded", side_effect=fake_scp), patch.object(
                        runpod_ssh, "_runpod_remote_training_states", return_value=deepcopy(listing),
                    ), patch.object(runpod_ssh, "_workspace_config", return_value={}):
                        items = _pull_remote_training_state_items(
                            run_dir, {"ip": "example", "port": 22, "key": root / "key"}, workspace="/workspace", items=deepcopy(listing),
                        )
                    remote_owner.assert_called()
                published[executor] = sorted(entry["observed_step"] for entry in items)
        self.assertEqual(published, {"docker": [200], "runpod": [200]})


if __name__ == "__main__":
    unittest.main()

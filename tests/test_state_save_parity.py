"""Which state directories a run publishes, and how often its trainer saves state, are each decided once."""

from __future__ import annotations

import json
import os
import re
import shutil
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
    managed_state_cadence,
    managed_state_save_args,
    publish_completed_training_states,
    run_output_name,
    state_directory_step,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tests.test_resume_steps import as_resume, musubi_run, sd_scripts_run  # noqa: E402
from tests.test_training_resume import _safetensors_bytes, _torch_archive_bytes, _write_state_marker  # noqa: E402


def _cadence(run: dict[str, Any]) -> tuple[str | None, str]:
    """The state save cadence (None when the trainer gets no flag) and retained window each trainer is given."""
    if run["backend"]["name"] == "musubi-tuner":
        script = musubi_command.command_musubi_tuner(run)["argv"][2]
        every = re.findall(r"--save_every_n_steps (\d+)", script)
        window = re.findall(r"--save_last_n_steps_state (\d+)", script)
    else:
        script = sd_scripts.command_sd_scripts(run)["argv"][2]
        every = re.findall(r'"--save_every_n_steps","(\d+)"', script)
        window = re.findall(r'"--save_last_n_steps_state","(\d+)"', script)
    assert len(every) <= 1 and len(window) == 1, (every, window)
    return (every[0] if every else None), window[0]


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
    def test_a_fresh_run_keeps_each_backends_cadence_and_a_resume_caps_it_at_the_added_steps(self) -> None:
        # Fresh runs: each backend's own default (Musubi always names a cadence, sd-scripts only a
        # configured one). Process-local Resume: both save within the steps the run adds.
        cases = {
            "fresh": ({}, None, 2, {"musubi-tuner": ("1000", "1000"), "sd-scripts": (None, "1000")}),
            "fresh-configured": ({"save_every_n_steps": 100}, None, 2, ("100", "100")),
            "fresh-one-generation": ({}, None, 1, {"musubi-tuner": ("1000", "1"), "sd-scripts": (None, "1")}),
            "resume": ({}, 200, 2, ("200", "200")),
            "resume-configured-below": ({"save_every_n_steps": 50}, 200, 2, ("50", "50")),
            "resume-configured-above": ({"save_every_n_steps": 500}, 200, 2, ("200", "200")),
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
                wanted = expected[name] if isinstance(expected, dict) else expected
                with self.subTest(backend=name, case=label), patch.object(
                    musubi_command if name == "musubi-tuner" else sd_scripts,
                    "managed_state_save_args",
                    wraps=managed_state_save_args,
                ) as owner:
                    self.assertEqual(_cadence(run), wanted)
                    owner.assert_called_once()

    def test_a_fresh_sd_scripts_run_keeps_its_save_flags_where_they_were(self) -> None:
        # A fresh run's argv stays byte-identical: a configured cadence keeps its place
        # before save_last_n_steps, and the state flags follow them.
        run = sd_scripts_run()
        run["backend"]["config"].update({"save_every_n_steps": 50, "save_last_n_steps": 100})
        run["recovery"] = {"training_state": {"enabled": True, "keep_generations": 2}}
        script = sd_scripts.command_sd_scripts(run)["argv"][2]
        self.assertIn(
            '"--save_every_n_steps","50","--save_last_n_steps","100","--save_state","--save_state_on_train_end",'
            '"--save_last_n_steps_state","50"',
            script,
        )

    def test_a_capped_resume_passes_the_checkpoint_safety_preflight_as_a_fresh_run_does(self) -> None:
        # The preflight counts checkpoints from the recipe's steps and the configured cadence;
        # the cap on a Resume's state saves must not make it refuse the Resume.
        from kura.run_commands.plan import _checkpoint_safety_preflight

        for build in (musubi_run, sd_scripts_run):
            run = build()
            run["backend"]["config"]["save_every_n_steps"] = 200
            run["recovery"] = {"training_state": {"enabled": True, "keep_generations": 2}}
            with self.subTest(backend=run["backend"]["name"]):
                _checkpoint_safety_preflight(run)
                _checkpoint_safety_preflight(as_resume(run, additional=50))

    def test_every_accelerate_trainer_names_its_outputs_through_one_rule(self) -> None:
        for module, build, command in (
            (musubi_command, musubi_run, musubi_command.command_musubi_tuner),
            (sd_scripts, sd_scripts_run, sd_scripts.command_sd_scripts),
        ):
            with self.subTest(module=module.__name__), patch.object(module, "run_output_name", wraps=run_output_name) as owner:
                command(as_resume(build()))
                owner.assert_called()

    def test_managed_state_cadence_changes_only_a_process_local_resume(self) -> None:
        self.assertIsNone(managed_state_cadence(sd_scripts_run(), None))
        self.assertEqual(managed_state_cadence(sd_scripts_run(), 100), 100)
        self.assertEqual(managed_state_cadence(as_resume(sd_scripts_run()), None), 200)
        self.assertEqual(managed_state_cadence(as_resume(sd_scripts_run()), 500), 200)
        no_steps = as_resume(sd_scripts_run())
        del no_steps["recipe"]["steps"]
        with self.assertRaisesRegex(ValueError, "recipe.steps"):
            managed_state_cadence(no_steps, None)

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

    def _run_dir(self, root: Path, *, resume: bool = False, build=sd_scripts_run) -> Path:
        run = build()
        run["recovery"] = {"training_state": {"enabled": True, "keep_generations": 2}}
        if resume:
            run = as_resume(run)
        run_dir = root / "runs" / "derived"
        (run_dir / "resolved").mkdir(parents=True)
        (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
        (run_dir / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
        return run_dir

    def test_docker_and_runpod_publish_the_same_state_directories_mid_run(self) -> None:
        self.assertEqual(self._publish(resume=False), {"docker": [200], "runpod": [200]})

    def test_docker_and_runpod_publish_the_same_resume_states_mid_run(self) -> None:
        # Resume +200 from 1000: sd-scripts names its states by process-local step, and its
        # verified marker carries the logical step.
        logical = {"derived-step0200-state": 1200, "other-step0300-state": 1300, "derived-state": 1250}
        self.assertEqual(self._publish(resume=True, logical=logical), {"docker": [1200], "runpod": [1200]})

    def test_docker_and_runpod_place_a_state_by_its_marker_when_its_name_counts_logical_steps(self) -> None:
        # A multi-item sd-scripts Resume names its states by the logical step; the marker decides.
        logical = {"derived-step1200-state": 1200, "other-step0300-state": 1300, "derived-state": 1250}
        names = ("derived-step1200-state", "other-step0300-state", "derived-state")
        self.assertEqual(self._publish(resume=True, logical=logical, names=names), {"docker": [1200], "runpod": [1200]})

    def test_runpod_does_not_copy_a_state_already_published_at_its_marked_step(self) -> None:
        import kura.run_commands.runpod_ssh as runpod_ssh

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._run_dir(root, resume=True)
            self._write_state(run_dir / "outputs" / "derived-step1200-state", 1200)
            self.assertEqual([item["observed_step"] for item in publish_completed_training_states(root, run_dir)], [1200])
            item = {
                "path": "/workspace/runs/derived/outputs/derived-step1200-state",
                "name": "derived-step1200-state",
                "marked_step": 1200,
                "files": [{"path": "model.safetensors", "size": 1, "mtime_ns": 1}],
            }
            with patch.object(runpod_ssh, "_run_bounded", side_effect=AssertionError("a published state must not be copied again")), \
                    patch.object(runpod_ssh, "state_logical_step", wraps=training_artifacts.state_logical_step) as owner:
                items = _pull_remote_training_state_items(
                    run_dir, {"ip": "example", "port": 22, "key": root / "key"}, workspace="/workspace", items=[item],
                )
            self.assertEqual([entry["observed_step"] for entry in items], [1200])
            owner.assert_called()

    @unittest.skipUnless(os.name == "posix" and shutil.which("bash") and shutil.which("python"), "a Pod runs this under Linux bash and python")
    def test_the_pod_listing_reports_the_step_the_contracts_marker_records(self) -> None:
        import kura.run_commands.runpod_ssh as runpod_ssh

        with tempfile.TemporaryDirectory() as directory:
            outputs = Path(directory) / "runs" / "derived" / "outputs"
            self._write_state(outputs / "derived-step1200-state", 1200)
            (outputs / "derived-step0300-state").mkdir()
            (outputs / "derived-step0300-state" / "kura-state-info.json").write_text("[1]", encoding="utf-8")
            scripts: list[str] = []
            real_run = subprocess.run

            def run_on_pod(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
                scripts.append(command[-1])
                return real_run(["bash", "-c", command[-1]], **kwargs)

            marker = sd_scripts.training_state_contract_sd_scripts(sd_scripts_run())["state_step"]
            with patch.object(runpod_ssh, "_ssh_base", return_value=["ssh"]), patch.object(runpod_ssh.subprocess, "run", side_effect=run_on_pod):
                listed = runpod_ssh._runpod_remote_training_states({}, workspace=directory, run_id="derived", marker=marker)
                unmarked = runpod_ssh._runpod_remote_training_states({}, workspace=directory, run_id="derived", marker=None)
        self.assertEqual(len(scripts), 2)
        self.assertEqual({item["name"]: item["marked_step"] for item in listed}, {"derived-step0300-state": None, "derived-step1200-state": 1200})
        self.assertEqual({item["marked_step"] for item in unmarked}, {None})

    def test_runpod_reports_a_malformed_continuation_instead_of_pulling_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Musubi has no logical step marker, so its states are placed through the Resume steps.
            run_dir = self._run_dir(root, resume=True, build=musubi_run)
            run = yaml.safe_load((run_dir / "resolved" / "manifest.lock.yaml").read_text(encoding="utf-8"))
            run["continuation"]["target_step"] = 1300
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
            item = {"path": "/workspace/runs/derived/outputs/derived-step0150-state", "name": "derived-step0150-state", "files": [{"path": "x", "size": 1, "mtime_ns": 1}]}
            with self.assertRaisesRegex(ValueError, "target_step does not match"):
                _pull_remote_training_state_items(
                    run_dir, {"ip": "example", "port": 22, "key": root / "key"}, workspace="/workspace", items=[item],
                )

    def _publish(self, *, resume: bool, logical: dict[str, int] | None = None, names: tuple[str, ...] | None = None) -> dict[str, list[int]]:
        import kura.run_commands.runpod_ssh as runpod_ssh

        logical = logical or self.STEPS
        names = names or self.NAMES
        published: dict[str, list[int]] = {}
        for executor in ("docker", "runpod"):
            with tempfile.TemporaryDirectory() as directory, patch.object(
                training_artifacts, "state_logical_step", wraps=training_artifacts.state_logical_step,
            ) as local_owner, patch.object(runpod_ssh, "state_logical_step", wraps=training_artifacts.state_logical_step) as remote_owner:
                root = Path(directory)
                run_dir = self._run_dir(root, resume=resume)
                if executor == "docker":
                    for name in names:
                        self._write_state(run_dir / "outputs" / name, logical[name])
                    items = publish_completed_training_states(root, run_dir)
                    local_owner.assert_called()
                else:
                    sources = {name: self._write_state(root / "remote" / name, logical[name]) for name in names}
                    listing = [
                        {
                            "path": f"/workspace/runs/derived/outputs/{name}",
                            "name": name,
                            # What the Pod reads from the marker the backend's contract names.
                            "marked_step": logical[name],
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
        return published


if __name__ == "__main__":
    unittest.main()

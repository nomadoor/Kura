"""One rule decides whether a completed run must leave recoverable training state.

Docker and RunPod must decide the same for the same compiled run, and a backend
must not save state Kura cannot use (capability "unsupported").
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from kura.backends.sd_scripts import command_sd_scripts
from kura.training_artifacts import training_state_capture_required, training_state_contract, training_state_managed
from tests.test_sd_scripts_backend import base_run

SD_SCRIPTS_KINDS = (("sd15", "lora"), ("sdxl", "lora"), ("flux1", "lora"), ("anima", "lora"), ("anima", "controlnet_lllite"))


def _compiled(directory: str, run: dict) -> Path:
    run_dir = Path(directory) / "runs" / run["id"]
    (run_dir / "resolved").mkdir(parents=True)
    (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
    return run_dir


def _ai_toolkit(**config) -> dict:
    return {"id": "ai-run", "type": "train", "backend": {"name": "ai-toolkit", "config": dict(config)},
            "model": {"base": "example/model"}, "recipe": {"steps": 10, "seed": 1}}


def _musubi() -> dict:
    return {"id": "musubi-run", "type": "train", "backend": {"name": "musubi-tuner", "config": {
        "architecture": "flux2", "model_version": "klein-base-4b",
        "model_paths": {"dit": "/models/dit", "vae": "/models/vae", "text_encoder": "/models/text"},
    }}, "model": {"base": "example/model"}, "datasets": [{"id": "tiny"}], "recipe": {"steps": 10, "seed": 1}}


# Every backend, including kinds Kura declares it cannot resume; "saves" is how its command shows a managed state.
ALL_BACKEND_RUNS = (
    *((f"sd-scripts {architecture}/{mode}", lambda architecture=architecture, mode=mode: base_run(architecture, mode),
       lambda argv: "--save_state" in argv) for architecture, mode in SD_SCRIPTS_KINDS),
    ("ai-toolkit adamw8bit", lambda: _ai_toolkit(), lambda argv: "state_root" in argv),
    ("ai-toolkit accumulation 2", lambda: _ai_toolkit(gradient_accumulation_steps=2), lambda argv: "state_root" in argv),
    ("ai-toolkit prodigy", lambda: _ai_toolkit(optimizer_type="prodigy"), lambda argv: "state_root" in argv),
    ("musubi flux2", _musubi, lambda argv: "--save_state" in argv),
)


class OneRuleTests(unittest.TestCase):
    def test_docker_and_runpod_collection_apply_the_same_rule(self) -> None:
        from kura.executors import docker
        from kura.run_commands import runpod_ssh

        # Both executors call the one shared function; neither keeps its own copy.
        self.assertIs(docker.training_state_capture_required, training_state_capture_required)
        self.assertIs(runpod_ssh.training_state_capture_required, training_state_capture_required)
        self.assertFalse(hasattr(runpod_ssh, "_directory_training_state_sync_enabled"))

    def test_a_completed_run_must_leave_state_exactly_when_its_backend_saves_usable_state(self) -> None:
        for architecture, mode in SD_SCRIPTS_KINDS:
            for enabled in (True, False):
                with self.subTest(architecture=architecture, mode=mode, enabled=enabled), tempfile.TemporaryDirectory() as directory:
                    run = base_run(architecture, mode)
                    run["recovery"] = {"training_state": {"enabled": enabled, "keep_generations": 2}}
                    run_dir = _compiled(directory, run)
                    saves_state = "--save_state" in " ".join(command_sd_scripts(run)["argv"])
                    self.assertEqual(training_state_capture_required(run_dir), saves_state)
                    self.assertEqual(training_state_managed(run), saves_state)

    def test_no_state_is_saved_where_kura_cannot_use_it(self) -> None:
        for architecture, mode in SD_SCRIPTS_KINDS:
            with self.subTest(architecture=architecture, mode=mode):
                run = base_run(architecture, mode)
                run["recovery"] = {"training_state": {"enabled": True, "keep_generations": 2}}
                argv = " ".join(command_sd_scripts(run)["argv"])
                supported = training_state_contract(run).get("capability") != "unsupported"
                self.assertEqual("--save_state" in argv, supported)


class EveryBackendTests(unittest.TestCase):
    def test_every_backend_saves_state_exactly_when_a_finished_run_must_leave_it(self) -> None:
        from kura.backends import get_backend

        for label, make, saves in ALL_BACKEND_RUNS:
            for enabled in (True, False):
                with self.subTest(run=label, enabled=enabled), tempfile.TemporaryDirectory() as directory:
                    run = make()
                    run["recovery"] = {"training_state": {"enabled": enabled, "keep_generations": 2}}
                    run_dir = _compiled(directory, run)
                    argv = " ".join(get_backend(run["backend"]["name"]).command(run)["argv"])
                    self.assertEqual(saves(argv), training_state_capture_required(run_dir))


if __name__ == "__main__":
    unittest.main()

"""One rule decides whether a completed run must leave recoverable training state.

Docker and RunPod must decide the same for the same compiled run, and a backend
must not save state Kura cannot use (capability "unsupported").
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from kura.backends.sd_scripts import command_sd_scripts
from kura.run_commands.common import backend_command_lock
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



class PublisherRuleTests(unittest.TestCase):
    def test_the_publishers_keep_state_exactly_when_the_compiled_command_was_built_to(self) -> None:
        # Collection holds a run to what its frozen command does, even when the live rule (here
        # the run's own policy) says otherwise, as after a Kura upgrade between compile and collection.
        from tests.test_state_save_parity import StatePublicationParityTests
        from tests.test_resume_steps import sd_scripts_run
        from kura.training_artifacts import publish_completed_training_states, publish_training_state_candidate

        for frozen, live in ((False, True), (True, False)):
            with self.subTest(frozen=frozen, live=live), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run = sd_scripts_run()
                run["recovery"] = {"training_state": {"enabled": live, "keep_generations": 2}}
                run_dir = root / "runs" / "derived"
                (run_dir / "resolved").mkdir(parents=True)
                (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
                (run_dir / "resolved" / "backend-command.lock.json").write_text(json.dumps({"training_state_managed": frozen}), encoding="utf-8")
                (run_dir / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
                candidate = StatePublicationParityTests._write_state(None, run_dir / "outputs" / "derived-step0200-state", 200)
                self.assertEqual(training_state_capture_required(run_dir), frozen)
                single = publish_training_state_candidate(root, run_dir, candidate, 200)
                self.assertEqual(single is not None, frozen)
                self.assertEqual([item["observed_step"] for item in publish_completed_training_states(root, run_dir)], [200] if frozen else [])

class RuleEdgeTests(unittest.TestCase):
    def test_a_custom_command_and_a_manifest_compiled_before_managed_state_keep_no_state(self) -> None:
        custom = base_run("sdxl", "lora")
        custom["backend"]["config"] = {"command": {"cwd": "/opt", "argv": ["python", "train.py"]}}
        custom["recovery"] = {"training_state": {"enabled": True, "keep_generations": 2}}
        self.assertFalse(training_state_managed(custom))
        legacy = base_run("sdxl", "lora")  # frozen without `recovery`
        self.assertFalse(training_state_managed(legacy, frozen=True))
        self.assertTrue(training_state_managed(legacy))  # authored run.yaml: the policy default applies
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(training_state_capture_required(_compiled(directory, legacy)))


class EveryBackendTests(unittest.TestCase):
    def test_every_backend_saves_state_exactly_when_a_finished_run_must_leave_it(self) -> None:
        from kura.backends import get_backend

        for label, make, saves in ALL_BACKEND_RUNS:
            for enabled in (True, False):
                with self.subTest(run=label, enabled=enabled), tempfile.TemporaryDirectory() as directory:
                    run = make()
                    run["recovery"] = {"training_state": {"enabled": enabled, "keep_generations": 2}}
                    run_dir = _compiled(directory, run)
                    command = get_backend(run["backend"]["name"]).command(run)
                    # Compile freezes the decision with the command, and collection reads it there.
                    lock = backend_command_lock(run, command, {"kind": "test"})
                    self.assertEqual(lock["training_state_managed"], saves(" ".join(command["argv"])))
                    (run_dir / "resolved" / "backend-command.lock.json").write_text(json.dumps(lock), encoding="utf-8")
                    self.assertEqual(training_state_capture_required(run_dir), saves(" ".join(command["argv"])))

    def test_a_lock_written_before_compile_froze_the_decision_reads_the_rule_of_its_time(self) -> None:
        # AI-Toolkit did not capture state under gradient accumulation before it compiled the
        # trainer's real accumulation; its frozen command saves none, so none is required.
        for accumulation, required in ((2, False), (1, True)):
            with self.subTest(accumulation=accumulation), tempfile.TemporaryDirectory() as directory:
                run = _ai_toolkit(gradient_accumulation_steps=accumulation)
                run["recovery"] = {"training_state": {"enabled": True, "keep_generations": 2}}
                run_dir = _compiled(directory, run)
                lock = {"backend": "ai-toolkit", "cwd": "/opt/ai-toolkit", "argv": ["python", "run.py", "x.yaml"], "env": {}, "adapter_source": {}}
                (run_dir / "resolved" / "backend-command.lock.json").write_text(json.dumps(lock), encoding="utf-8")
                self.assertEqual(training_state_capture_required(run_dir), required)

    def test_the_plan_reads_an_old_lock_by_the_rule_of_its_time(self) -> None:
        # The plan's capability, warning, and saved answer all follow the lock's contract.
        import os

        import kura.run_commands.plan as plan
        import kura.training_artifacts as training_artifacts

        for accumulation, saved in ((2, False), (1, True)):
            with self.subTest(accumulation=accumulation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
                run = _ai_toolkit(model_arch="sdxl", gradient_accumulation_steps=accumulation)
                run["compute"] = {"executor": "docker"}
                run["recovery"] = {"training_state": {"enabled": True, "keep_generations": 2}}
                run_dir = _compiled(str(root), run)
                (run_dir / "run.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
                lock = {"backend": "ai-toolkit", "cwd": "/opt/ai-toolkit", "argv": ["python", "run.py", "x.yaml"], "env": {}, "adapter_source": {}}
                (run_dir / "resolved" / "backend-command.lock.json").write_text(json.dumps(lock), encoding="utf-8")
                previous = Path.cwd()
                os.chdir(root)
                try:
                    with patch.object(plan, "frozen_training_state_contract", wraps=training_artifacts.frozen_training_state_contract) as owner:
                        state = plan.plan_run(run["id"])["training_state"]
                finally:
                    os.chdir(previous)
                owner.assert_called()
                self.assertEqual(state["saved"], saved)
                self.assertEqual(state["capability"] == "unsupported", not saved)
                self.assertEqual(state["warning"] is not None and state["warning"].startswith("not saved"), not saved)


if __name__ == "__main__":
    unittest.main()

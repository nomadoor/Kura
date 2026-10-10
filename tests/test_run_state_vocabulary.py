"""Every run state Kura names comes from one vocabulary, and each decision over states uses one shared set."""

from __future__ import annotations

import contextlib
import io
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / "src" / "kura"


class RunStateVocabularyTests(unittest.TestCase):
    def test_the_decisions_over_states_use_the_shared_sets(self) -> None:
        from kura import runner
        from kura.executors import common
        from kura.run_commands import launch, plan

        # Whether a run may stage and start is decided by one function over the shared set.
        self.assertIs(launch.can_start, common.can_start)
        self.assertIs(plan.can_start, common.can_start)
        self.assertIs(runner.EXIT_FOR_STATE, common.EXIT_CODE_FOR_STATE)
        for name in ("STARTING_STATES", "OBSERVABLE_STATES", "TERMINAL_STATES", "UNFINISHED_STATES", "CLEANUP_ELIGIBLE_STATES", "UNSUCCESSFUL_STATES"):
            with self.subTest(name=name):
                self.assertLessEqual(set(getattr(common, name)), common.RUN_STATES)
        self.assertLessEqual(set(common.EXIT_CODE_FOR_STATE), common.RUN_STATES)

    def test_a_run_starts_only_from_compiled_or_its_own_capacity_wait(self) -> None:
        from kura.executors.common import can_start

        self.assertTrue(can_start({"state": "compiled"}))
        self.assertTrue(can_start({"state": "queued", "capacity_wait": {"started_at": "2026-10-09T00:00:00+09:00"}}))
        for state in ("failed", "interrupted", "unknown", "launch_failed", "completed", "draft", "queued", "running"):
            with self.subTest(state=state):
                self.assertFalse(can_start({"state": state}))

    def test_stage_and_launch_send_an_ended_run_to_a_new_run_from_its_settings(self) -> None:
        from kura.run_commands import launch, plan

        def stage(run_dir: Path, state: str):
            return (lambda: plan.stage_run("example")), (
                patch.object(plan, "_run_path", return_value=run_dir),
                patch.object(plan, "_load_yaml", return_value={"datasets": [{"id": "dataset"}]}),
                patch.object(plan, "_workspace_config", return_value={}),
                patch.object(plan, "stage_runpod", return_value={}),
                patch.object(plan, "observe_run", return_value={"state": state}),
            )

        def start(run_dir: Path, state: str):
            return (lambda: launch.launch_run("example", executor="docker", dry_run=False, check_only=True)), (
                patch.object(launch, "_run_path", return_value=run_dir),
                patch.object(launch, "_load_yaml", return_value={"compute": {"executor": "docker"}}),
                patch.object(launch, "_workspace_config", return_value={}),
                patch.object(launch, "unresolved_create_intents", return_value=[]),
                patch.object(launch, "unstopped_recovered_pod", return_value=None),
                patch.object(launch, "observe_run", return_value={"state": state}),
            )

        for state in ("failed", "interrupted", "unknown", "launch_failed"):
            for name, attempt in (("stage", stage), ("launch", start)):
                with self.subTest(state=state, command=name), tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
                    run_dir = Path(directory) / "runs" / "example"
                    run_dir.mkdir(parents=True)
                    call, patches = attempt(run_dir, state)
                    for item in patches:
                        stack.enter_context(item)
                    stderr = stack.enter_context(patch("sys.stderr", new_callable=io.StringIO))
                    self.assertEqual(call(), 1)
                    self.assertIn(f"run example ended {state}", stderr.getvalue())
                    self.assertIn("kura run new --from example", stderr.getvalue())

    def test_viewers_decide_whether_a_run_is_unfinished_with_the_runner_rule(self) -> None:
        from kura import monitor, tui
        from kura.executors import common

        # One rule says something is still happening to a run; the runner, status, monitor, and TUI all call it.
        from kura import cli

        self.assertFalse(hasattr(common, "ACTIVE_STATES"))
        self.assertLessEqual(common.STARTING_STATES, common.UNFINISHED_STATES)
        self.assertIs(monitor.run_finished, common.run_finished)
        self.assertIs(cli.run_finished, common.run_finished)
        self.assertFalse(hasattr(monitor, "UNFINISHED_STATES"))
        self.assertFalse(hasattr(tui, "UNFINISHED_STATES"))
        self.assertFalse(hasattr(cli, "UNFINISHED_STATES"))
        self.assertIs(tui.STARTING_STATES, common.STARTING_STATES)

    def test_no_module_keeps_its_own_list_of_run_states(self) -> None:
        # A literal set or tuple of two or more run states outside executors/common.py is a second copy.
        from kura.executors.common import RUN_STATES

        # Decisions made in exactly one place, about that place's own step.
        local = {
            "render.py": ["compiled", "running"],  # a RunPod render starts after its session marks the run running
            "executors/docker.py": ["completed", "failed"],  # the outcomes a container's exit code gives
        }
        literal = re.compile(r"[{(]\s*((?:\"[a-z_]+\"\s*,\s*)+\"[a-z_]+\")\s*,?\s*[})]")
        offenders = []
        for path in SRC.rglob("*.py"):
            if path.name == "common.py" and path.parent.name == "executors":
                continue
            for match in literal.finditer(path.read_text(encoding="utf-8")):
                names = re.findall(r"\"([a-z_]+)\"", match.group(1))
                if len(names) >= 2 and set(names) <= RUN_STATES and len(set(names) & {"completed", "failed", "interrupted", "compiled", "running", "launch_failed"}) >= 2:
                    if local.get(path.relative_to(SRC).as_posix()) != names:
                        offenders.append(f"{path.relative_to(SRC).as_posix()}: {names}")
        self.assertEqual(offenders, [])


class StartRefusalTests(unittest.TestCase):
    def test_every_ended_run_is_sent_to_a_new_run(self) -> None:
        from kura.executors.common import can_start, start_refusal

        for state in ("completed", "stopped", "failed", "interrupted", "unknown", "launch_failed"):
            with self.subTest(state=state):
                self.assertFalse(can_start({"state": state}))
                self.assertIn("kura run new --from r1", start_refusal("r1", {"state": state}, action="launch"))
        # A run waiting for a person, or not yet compiled, is not "ended".
        for state in ("recovery_required", "draft"):
            with self.subTest(state=state):
                self.assertNotIn("run new --from", start_refusal("r1", {"state": state}, action="launch"))

    def test_an_unknown_run_is_first_sent_to_reconcile_or_stop(self) -> None:
        from kura.executors.common import start_refusal

        # A RunPod run ends unknown while its exited Pod may still exist and bill for its disk.
        message = start_refusal("r1", {"state": "unknown"}, action="launch")
        self.assertIn("kura run reconcile r1", message)
        self.assertIn("kura run stop r1", message)
        for state in ("completed", "failed", "interrupted", "launch_failed"):
            with self.subTest(state=state):
                self.assertNotIn("kura run reconcile", start_refusal("r1", {"state": state}, action="launch"))


if __name__ == "__main__":
    unittest.main()

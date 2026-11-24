"""A run whose status.json is gone cannot be stopped or reconciled by Kura, so both commands
name the RunPod Pods its records hold, for the user to delete by hand.

`kura run reconcile` and `kura run stop` say the same thing from one helper
(`run_without_status_problem`); neither sends a request or changes a file."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kura.cli import cmd_run_reconcile
from kura.run_commands.plan import stop_run
from tests.user_config import isolate_user_config

RID_1 = "20261022-101010-000001"
RID_2 = "20261023-101010-000002"
RID_OLD = "20260901-101010-000003"
ENV_VALUE = "env-value-that-must-never-be-printed"

COMMANDS = (
    ("kura run reconcile", lambda run_id: cmd_run_reconcile(argparse.Namespace(run_id=run_id))),
    ("kura run stop", lambda run_id: stop_run(run_id, yes=True)),
)


def _record(run_dir: Path, name: str, value: dict) -> None:
    (run_dir / "realizations").mkdir(parents=True, exist_ok=True)
    (run_dir / "realizations" / name).write_text(json.dumps(value), encoding="utf-8")


def _runpod_run_without_status(root: Path) -> Path:
    run_dir = root / "runs" / "train"
    request = {"name": f"kura-train-{RID_1}", "env": {"HF_TOKEN": ENV_VALUE, "KURA_LOG_PATH": "/workspace/pod-log-path"}}
    # A launch whose Pod is recorded, with the duplicate a recovery found beside it.
    _record(run_dir, f"{RID_1}.create-intent.json", {"kind": "pod_create_intent", "schema_version": 1, "realization_id": RID_1,
                                                      "executor": "runpod", "pod_name": f"kura-train-{RID_1}", "request": request})
    _record(run_dir, f"{RID_1}.json", {"id": RID_1, "executor": "runpod", "state": "running", "request": request,
                                        "pod": {"id": "pod-1", "name": f"kura-train-{RID_1}"}, "duplicate_pod_ids": ["pod-1b"]})
    # A launch that stopped before recording whether its Pod was created; an intent from before names were recorded.
    _record(run_dir, f"{RID_2}.create-intent.json", {"realization_id": RID_2, "request": request})
    # A launch from before create intents existed.
    _record(run_dir, f"{RID_OLD}.json", {"id": RID_OLD, "executor": "runpod", "state": "completed", "pod": {"id": "pod-old", "name": "kura-old-name"}})
    _record(run_dir, f"{RID_1}.observed-20261022-101500-000000.json", {"kind": "observation", "state": "running", "pod": {"id": "pod-1"}})
    return run_dir


def _docker_run_without_status(root: Path) -> Path:
    run_dir = root / "runs" / "train"
    _record(run_dir, f"{RID_1}.create-intent.json", {"kind": "container_create_intent", "schema_version": 1, "realization_id": RID_1,
                                                      "executor": "docker", "container_name": "kura-train"})
    _record(run_dir, f"{RID_1}.json", {"id": RID_1, "executor": "docker", "state": "running", "container": {"id": "c1", "name": "kura-train"}})
    return run_dir


def _files(root: Path) -> dict[str, bytes]:
    """Every file but the empty advisory lock files a stop creates when it looks for a runner."""
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file() and path.suffix != ".lock"}


class LostStatusTests(unittest.TestCase):
    def _run(self, command, make_run=None, run_id: str = "train") -> tuple[int, str, str]:
        """Run one command in a workspace; fails if it sends a request, runs a command, or changes a record."""
        with contextlib.ExitStack() as stack:
            root = Path(stack.enter_context(tempfile.TemporaryDirectory())).resolve()
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            (root / "runs").mkdir()
            if make_run is not None:
                make_run(root)
            before = _files(root)
            previous = Path.cwd()
            os.chdir(root)
            stack.callback(os.chdir, previous)
            stack.enter_context(patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret-value"}, clear=False))
            requested = stack.enter_context(patch("kura.executors.runpod._runpod_graphql", side_effect=AssertionError("no RunPod request")))
            docker = stack.enter_context(patch("subprocess.run", side_effect=AssertionError("no command")))
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = command(run_id)
            requested.assert_not_called()
            docker.assert_not_called()
            self.assertEqual(_files(root), before)
            return code, out.getvalue(), err.getvalue()

    def test_a_run_that_does_not_exist_is_named_as_such(self) -> None:
        for label, command in COMMANDS:
            with self.subTest(label):
                code, out, err = self._run(command, run_id="trian")
                self.assertEqual(code, 1)
                self.assertIn("there is no run trian", err)
                self.assertNotIn("restore", err)
                self.assertNotIn("Pod", err)
                self.assertEqual(out, "")

    def test_a_runpod_run_without_status_names_every_pod_its_records_hold(self) -> None:
        for label, command in COMMANDS:
            with self.subTest(label):
                code, out, err = self._run(command, _runpod_run_without_status)
                self.assertEqual(code, 1)
                self.assertEqual(out, "")
                self.assertIn("runs/train/status.json is missing", err)
                self.assertIn("restore it", err)
                # Every name a launch used, with the ids its records hold; the second launch recorded none.
                self.assertIn(f"kura-train-{RID_1} (id pod-1, pod-1b)", err)
                self.assertIn(f"kura-train-{RID_2} (no id recorded)", err)
                self.assertIn("kura-old-name (id pod-old)", err)
                self.assertIn("RunPod console", err)
                # Deleting by hand loses what was only on the Pod, and the message says so.
                self.assertIn("not collected", err)
                for hidden in (ENV_VALUE, "api-secret-value", "pod-log-path", "HF_TOKEN"):
                    self.assertNotIn(hidden, err)

    def test_a_docker_run_without_status_names_no_pod(self) -> None:
        for label, command in COMMANDS:
            with self.subTest(label):
                code, _out, err = self._run(command, _docker_run_without_status)
                self.assertEqual(code, 1)
                self.assertIn("runs/train/status.json is missing", err)
                self.assertIn("restore it", err)
                self.assertIn("name no RunPod Pod", err)
                self.assertNotIn("RunPod console", err)
                self.assertNotIn("kura-train", err)

    def test_both_commands_say_the_same(self) -> None:
        messages = [self._run(command, _runpod_run_without_status)[2].split(": ", 1)[1] for _label, command in COMMANDS]
        self.assertEqual(messages[0], messages[1])


def setUpModule() -> None:
    isolate_user_config()


if __name__ == "__main__":
    unittest.main()

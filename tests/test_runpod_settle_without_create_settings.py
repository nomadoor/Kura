"""Deleting, observing, and settling a RunPod Pod need only the API key and the run's records, never Pod-create settings."""

from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from kura.executors import runpod
from kura.run_commands.plan import cmd_run_stop

RID = "20261019-101010-000001"
# Settings that could never create a Pod: no GPU listed.
NO_CREATE_SETTINGS = "schema_version: 2\n"


@contextmanager
def _run(status: dict, *, intent: bool = False):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "workspace.yaml").write_text(NO_CREATE_SETTINGS, encoding="utf-8")
        run_dir = root / "runs" / "example"
        (run_dir / "realizations").mkdir(parents=True)
        if intent:
            (run_dir / "realizations" / f"{RID}.create-intent.json").write_text(json.dumps(
                {"kind": "pod_create_intent", "schema_version": 1, "realization_id": RID, "pod_name": f"kura-example-{RID}",
                 "requested_at": "2026-10-19T10:10:10+09:00", "request": {}}), encoding="utf-8")
        else:
            (run_dir / "realizations" / f"{RID}.json").write_text(json.dumps({"id": RID, "executor": "runpod", "pod": {"id": "pod-1"}}), encoding="utf-8")
        (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
        previous = Path.cwd()
        os.chdir(root)
        try:
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False), patch("kura.executors.runpod.time.sleep"):
                yield run_dir
        finally:
            os.chdir(previous)


def _status(run_dir: Path) -> dict:
    return json.loads((run_dir / "status.json").read_text(encoding="utf-8"))


RUNNING = {"state": "running", "host": "runpod", "pod_id": "pod-1", "last_realization": f"realizations/{RID}.json"}


class SettleWithoutCreateSettingsTests(unittest.TestCase):
    def test_kura_run_stop_deletes_the_pod_when_workspace_yaml_names_no_gpu(self) -> None:
        with _run(RUNNING) as run_dir:
            with patch("kura.run_commands.plan._stop_through_runner", return_value=None), \
                 patch("kura.executors.runpod._runpod_request", return_value={}) as request, \
                 patch("sys.stdout", io.StringIO()), patch("sys.stderr", io.StringIO()) as err:
                code = cmd_run_stop(argparse.Namespace(run_id="example", yes=True))
            self.assertEqual(code, 0, err.getvalue())
            self.assertEqual(request.call_args.args[:2], ("DELETE", "/pods/pod-1"))
            self.assertTrue(_status(run_dir)["pod_stopped_at"])

    def test_kura_run_reconcile_observes_the_pod_when_workspace_yaml_names_no_gpu(self) -> None:
        from kura.cli import cmd_run_reconcile

        with _run(RUNNING) as run_dir:
            with patch("kura.executors.runpod._runpod_request", return_value={"id": "pod-1", "desiredStatus": "RUNNING"}), \
                 patch("kura.cli._try_sync_runpod_remote_stdout", return_value=True), patch("kura.cli._try_observe_runpod_remote_exit"), \
                 patch("sys.stdout", io.StringIO()), patch("sys.stderr", io.StringIO()) as err:
                code = cmd_run_reconcile(argparse.Namespace(run_id="example"))
            self.assertEqual(code, 0, err.getvalue())
            self.assertTrue(_status(run_dir)["last_observation"])

    def test_stop_reconcile_and_settle_take_only_the_run(self) -> None:
        with _run(RUNNING) as run_dir:
            with patch("kura.executors.runpod._runpod_request", return_value={"id": "pod-1", "desiredStatus": "RUNNING"}):
                self.assertEqual(runpod.reconcile_runpod(run_dir)["state"], "running")
            with patch("kura.executors.runpod._runpod_request", return_value={}):
                self.assertTrue(runpod.stop_runpod(run_dir)["pod_stopped_at"])
        with _run({"state": "launching", "host": "runpod"}, intent=True) as run_dir:
            with patch("kura.executors.runpod._runpod_pods_named", return_value=[]):
                lines = runpod.resolve_runpod_create_intents(run_dir)
            self.assertEqual(len(lines), 1)
            self.assertEqual(_status(run_dir)["state"], "launch_failed")

    def test_the_runner_deletes_a_pod_whose_request_carries_no_runpod_settings(self) -> None:
        from kura import runner

        with _run({**RUNNING, "state": "completed"}) as run_dir:
            with patch("kura.executors.runpod._runpod_request", return_value={}) as request:
                self.assertTrue(runner._delete_pod(run_dir, why="test"))
            self.assertEqual(request.call_args.args[:2], ("DELETE", "/pods/pod-1"))


if __name__ == "__main__":
    unittest.main()

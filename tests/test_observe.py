"""Run-state observation and materialization regressions."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kura.cli import cmd_run_status
from kura.executors import observe_run
from kura.executors.common import _run_operation_lock
from kura.monitor import RunProgress, RunSummary, collect_run_summaries
from kura.run_commands.plan import stage_run
from kura.tui import _progress_text


class ObserveRunTests(unittest.TestCase):
    def _docker_run(self, root: Path, *, state: str = "running", step: int | None = None, total: int | None = None) -> Path:
        run_dir = root / "runs" / "example"
        (run_dir / "realizations").mkdir(parents=True)
        (run_dir / "logs").mkdir()
        (run_dir / "run.yaml").write_text(
            "id: example\ntype: train\nrecipe: {steps: 10}\ncompute: {executor: docker}\n",
            encoding="utf-8",
        )
        status = {
            "state": state,
            "started": "2026-01-01T00:00:00Z",
            "ended": None,
            "last_step": step,
            "total_steps": total,
            "exit_code": None,
            "outputs": [],
            "last_realization": "realizations/r1.json",
            "container_id": "container-1",
        }
        (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
        (run_dir / "realizations" / "r1.json").write_text(
            json.dumps({"id": "r1", "executor": "docker", "container": {"id": "container-1"}}),
            encoding="utf-8",
        )
        return run_dir

    def test_terminal_observe_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._docker_run(Path(directory), state="completed", step=10, total=10)
            before = (run_dir / "status.json").stat().st_mtime_ns
            with patch("kura.executors.observe.reconcile_docker") as reconcile:
                first = observe_run(run_dir)
                second = observe_run(run_dir)
            self.assertEqual(first, second)
            reconcile.assert_not_called()
            self.assertEqual((run_dir / "status.json").stat().st_mtime_ns, before)
            self.assertFalse(list((run_dir / "realizations").glob("*.observed-*.json")))
            self.assertFalse((run_dir / "logs" / "events.jsonl").exists())

    def test_progress_only_observation_updates_status_without_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._docker_run(Path(directory))
            (run_dir / "logs" / "stdout.log").write_text(
                "example: 50%|#####| 5/10 [00:05<00:05, loss: 0.5]\n",
                encoding="utf-8",
            )
            result = subprocess.CompletedProcess([], 0, '{"Running": true, "ExitCode": 0}', "")
            with patch("kura.executors.docker.subprocess.run", return_value=result):
                status = observe_run(run_dir)
            self.assertEqual((status["last_step"], status["total_steps"]), (5, 10))
            self.assertFalse(list((run_dir / "realizations").glob("*.observed-*.json")))
            self.assertFalse((run_dir / "logs" / "events.jsonl").exists())

    def test_unchanged_observation_does_not_rewrite_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._docker_run(Path(directory))
            before = (run_dir / "status.json").stat().st_mtime_ns
            result = subprocess.CompletedProcess([], 0, '{"Running": true, "ExitCode": 0}', "")
            with patch("kura.executors.docker.subprocess.run", return_value=result):
                observe_run(run_dir)
            self.assertEqual((run_dir / "status.json").stat().st_mtime_ns, before)
            self.assertFalse(list((run_dir / "realizations").glob("*.observed-*.json")))

    def test_progress_is_monotonic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._docker_run(Path(directory), step=8, total=10)
            (run_dir / "logs" / "stdout.log").write_text(
                "example: 50%|#####| 5/9 [00:05<00:05, loss: 0.5]\n",
                encoding="utf-8",
            )
            result = subprocess.CompletedProcess([], 0, '{"Running": true, "ExitCode": 0}', "")
            with patch("kura.executors.docker.subprocess.run", return_value=result):
                status = observe_run(run_dir)
            self.assertEqual((status["last_step"], status["total_steps"]), (8, 10))

    def test_busy_observation_returns_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._docker_run(Path(directory))
            with _run_operation_lock(run_dir, "observe"):
                status = observe_run(run_dir)
            self.assertEqual(status["state"], "running")

    def test_busy_status_mutation_returns_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._docker_run(Path(directory))
            result = subprocess.CompletedProcess([], 0, '{"Running": true, "ExitCode": 0}', "")
            with _run_operation_lock(run_dir, "status"), patch("kura.executors.docker.subprocess.run", return_value=result):
                status = observe_run(run_dir)
            self.assertEqual(status["state"], "running")
            self.assertFalse(list((run_dir / "realizations").glob("*.observed-*.json")))

    def test_automatic_runpod_observation_uses_short_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._docker_run(Path(directory))
            (run_dir / "realizations" / "r1.json").write_text(
                json.dumps({"id": "r1", "executor": "runpod", "pod": {"id": "pod-1"}}),
                encoding="utf-8",
            )
            snapshot = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            with patch("kura.executors.observe.reconcile_runpod", return_value=snapshot) as reconcile:
                self.assertEqual(observe_run(run_dir, config={}), snapshot)
            reconcile.assert_called_once_with(run_dir, {}, timeout=2.0, blocking=False, source="automatic")

    def test_missing_or_malformed_status_is_not_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            run_dir.mkdir(parents=True)
            with self.assertRaises(FileNotFoundError):
                observe_run(run_dir)
            (run_dir / "status.json").write_text("{", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                observe_run(run_dir)

    def test_non_observable_states_do_not_call_executor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._docker_run(Path(directory))
            with patch("kura.executors.observe.reconcile_docker") as reconcile:
                for state in ("queued", "staged"):
                    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
                    status["state"] = state
                    (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
                    self.assertEqual(observe_run(run_dir)["state"], state)
            reconcile.assert_not_called()

    def test_status_command_and_monitor_share_persisted_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._docker_run(root)
            (root / "index.jsonl").write_text(json.dumps({"id": "example"}) + "\n", encoding="utf-8")
            (run_dir / "logs" / "stdout.log").write_text(
                "steps: 100%|##########| 10/10 [00:10<00:00, 2.0it/s, avr_loss=0.5]\n",
                encoding="utf-8",
            )
            result = subprocess.CompletedProcess(
                [],
                0,
                '{"Running": false, "ExitCode": 0, "FinishedAt": "2026-01-01T01:02:03Z"}',
                "",
            )
            stdout = io.StringIO()
            with (
                patch("kura.executors.docker.subprocess.run", return_value=result),
                patch("kura.cli._run_path", return_value=run_dir),
                patch("kura.cli._workspace_config", return_value={}),
                patch("sys.stdout", stdout),
            ):
                self.assertEqual(cmd_run_status(argparse.Namespace(run_id="example")), 0)
            payload = json.loads(stdout.getvalue())
            summary = collect_run_summaries(root)[0]
            self.assertEqual((payload["state"], payload["last_step"], payload["total_steps"]), ("completed", 10, 10))
            self.assertEqual((summary.state, summary.progress.step, summary.progress.total), ("completed", 10, 10))
            self.assertEqual(payload["seconds_per_iter"], 0.5)
            self.assertEqual(summary.progress.seconds_per_iter, 0.5)
            self.assertEqual(payload["ended"], "2026-01-01T01:02:03Z")
            self.assertEqual(payload["latest_observation"]["source"], "automatic")

    def test_unknown_step_is_not_displayed_as_zero(self) -> None:
        summary = RunSummary(id="draft", experiment=None, type="train", executor=None, state="draft", progress=RunProgress(step=None, total=10))
        rendered = str(_progress_text(summary))
        self.assertIn("step unknown/10", rendered)
        self.assertNotIn("step 0/10", rendered)

    def test_stage_guard_rejects_observed_completed_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            run_dir.mkdir(parents=True)
            with (
                patch("kura.run_commands.plan._run_path", return_value=run_dir),
                patch("kura.run_commands.plan._load_yaml", return_value={"datasets": [{"id": "dataset"}]}),
                patch("kura.run_commands.plan._workspace_config", return_value={}),
                patch("kura.run_commands.plan.observe_run", return_value={"state": "completed"}),
                patch("kura.run_commands.plan.stage_runpod") as stage,
                patch("sys.stderr", new_callable=io.StringIO),
            ):
                self.assertEqual(stage_run("example"), 1)
            stage.assert_not_called()


if __name__ == "__main__":
    unittest.main()

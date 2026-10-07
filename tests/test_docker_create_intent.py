"""A Docker launch records its container before creating it, and a crash in between is settled by discovery."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr
from pathlib import Path
from unittest.mock import patch

from kura.executors.common import CREATE_INTENT_SUFFIX, unresolved_create_intents
from kura.executors.docker import DOCKER_LAUNCH_LOCK, launch_docker, resolve_docker_create_intents
from kura.fsio import file_lock

SPEC = {"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}}


def _run(root: Path) -> Path:
    (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
    run_dir = root / "runs" / "example"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(json.dumps({"run_id": "example", "state": "compiled"}), encoding="utf-8")
    return run_dir


def _status(run_dir: Path) -> dict:
    return json.loads((run_dir / "status.json").read_text(encoding="utf-8"))


def _realization(run_dir: Path) -> dict:
    return json.loads((run_dir / _status(run_dir)["last_realization"]).read_text(encoding="utf-8"))


REAL_RUN = subprocess.run


@contextmanager
def _docker(replies):
    """Answer `docker` commands from `replies` (a list, a callable, or an exception); run anything else."""
    queue = list(replies) if isinstance(replies, list) else None
    calls = []

    def run(command, *args, **kwargs):
        if not command or command[0] != "docker":
            return REAL_RUN(command, *args, **kwargs)
        calls.append(command)
        reply = queue.pop(0) if queue is not None else replies
        if isinstance(reply, type) and issubclass(reply, BaseException) or isinstance(reply, BaseException):
            raise reply
        return reply(command) if callable(reply) else reply

    with (
        patch("kura.executors.docker._docker_image_id", return_value="sha256:" + "a" * 64),
        patch("kura.executors.docker.docker_preflight", return_value={}),
        patch("kura.executors.docker.kura_provenance", return_value={}),
        patch("kura.executors.docker.subprocess.run", side_effect=run),
    ):
        yield calls


def _ps(*lines: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], 0, "".join(f"{line}\n" for line in lines), "")


class DockerCreateIntentTests(unittest.TestCase):
    def test_the_intent_is_written_before_the_container_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _run(root)
            seen = {}

            def docker(command):
                seen["intents"] = [path.name for path in unresolved_create_intents(run_dir, "docker")]
                seen["state"] = _status(run_dir)["state"]
                return subprocess.CompletedProcess(command, 0, "container-1\n", "")

            with _docker(docker):
                _, realization_id = launch_docker(workspace=root, run_dir=run_dir, spec=SPEC, image="example:image", mounts=[], gpu=False)

            self.assertEqual(seen, {"intents": [f"{realization_id}{CREATE_INTENT_SUFFIX}"], "state": "launching"})
            realization = _realization(run_dir)
            self.assertEqual((realization["state"], realization["container"]["id"]), ("running", "container-1"))
            self.assertEqual(realization["create_intent"], f"{realization_id}{CREATE_INTENT_SUFFIX}")
            self.assertEqual((realization["kind"], realization["schema_version"]), ("realization", 1))
            self.assertEqual((_status(run_dir)["state"], _status(run_dir)["container_id"]), ("running", "container-1"))
            self.assertEqual(unresolved_create_intents(run_dir), [])

    def test_a_container_created_but_not_started_is_recorded_as_a_failed_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _run(root)
            replies = [subprocess.CompletedProcess([], 125, "", "could not select device driver"), _ps("container-2 created")]
            with _docker(replies), self.assertRaisesRegex(ValueError, "could not select device driver"):
                launch_docker(workspace=root, run_dir=run_dir, spec=SPEC, image="example:image", mounts=[], gpu=True)

            realization = _realization(run_dir)
            self.assertEqual((realization["state"], realization["container"]["id"]), ("launch_failed", "container-2"))
            self.assertEqual(_status(run_dir)["state"], "launch_failed")
            self.assertEqual(unresolved_create_intents(run_dir), [])

    def test_a_crash_after_the_create_is_recovered_by_label_without_starting_again(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _run(root)

            # The process dies right after Docker started the container.
            with _docker(KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
                launch_docker(workspace=root, run_dir=run_dir, spec=SPEC, image="example:image", mounts=[], gpu=False)
            self.assertEqual(len(unresolved_create_intents(run_dir, "docker")), 1)

            with _docker([_ps("container-3 running")]) as docker:
                lines = resolve_docker_create_intents(run_dir)

            self.assertEqual([command[:3] for command in docker], [["docker", "ps", "--all"]])
            self.assertIn("label=io.kura.run_id=example", docker[0])
            self.assertIn("recorded and reconcile now follows it", lines[0])
            realization = _realization(run_dir)
            self.assertEqual((realization["state"], realization["container"]["id"]), ("running", "container-3"))
            self.assertTrue(realization["recovered_from_intent"].endswith(CREATE_INTENT_SUFFIX))
            self.assertEqual(realization["local_image"], "example:image")
            self.assertEqual(_status(run_dir)["container_id"], "container-3")
            self.assertEqual(unresolved_create_intents(run_dir), [])

    def test_an_unreachable_daemon_keeps_both_errors_and_leaves_the_launch_unsettled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _run(root)
            replies = [subprocess.CompletedProcess([], 1, "", "Cannot connect to the Docker daemon"),
                       subprocess.CompletedProcess([], 1, "", "Cannot connect to the Docker daemon")]
            with _docker(replies), self.assertRaisesRegex(ValueError, "Cannot connect.*could not check.*kura run reconcile example"):
                launch_docker(workspace=root, run_dir=run_dir, spec=SPEC, image="example:image", mounts=[], gpu=False)
            self.assertEqual(_status(run_dir)["state"], "launching")
            self.assertEqual(len(unresolved_create_intents(run_dir, "docker")), 1)

    def test_no_container_found_records_the_launch_as_failed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _run(root)
            with _docker(KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
                launch_docker(workspace=root, run_dir=run_dir, spec=SPEC, image="example:image", mounts=[], gpu=False)
            with _docker([_ps()]):
                lines = resolve_docker_create_intents(run_dir)

            self.assertIn("recorded as failed", lines[0])
            self.assertEqual((_realization(run_dir)["state"], _status(run_dir)["state"]), ("launch_failed", "launch_failed"))

    def test_reconcile_waits_while_a_launch_still_holds_its_lock(self) -> None:
        from kura.cli import cmd_run_reconcile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _run(root)
            with _docker(KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
                launch_docker(workspace=root, run_dir=run_dir, spec=SPEC, image="example:image", mounts=[], gpu=False)
            stderr = io.StringIO()
            with (
                patch("kura.cli._run_path", return_value=run_dir),
                file_lock(run_dir / ".locks" / DOCKER_LAUNCH_LOCK, blocking=False),
                _docker([]) as docker,
                redirect_stderr(stderr),
            ):
                self.assertEqual(cmd_run_reconcile(argparse.Namespace(run_id="example")), 1)
            self.assertEqual(docker, [])
            self.assertIn("still creating its container", stderr.getvalue())
            self.assertEqual(len(unresolved_create_intents(run_dir, "docker")), 1)


    def test_a_launch_that_stopped_before_status_followed_its_realization_is_settled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _run(root)

            from kura.executors.common import _mutate_run_status as real_mutate

            calls = []

            def die_on_second_status_write(*args, **kwargs):
                # The intent's status write happens; the process dies before the started one.
                calls.append(1)
                if len(calls) == 1:
                    return real_mutate(*args, **kwargs)
                raise KeyboardInterrupt

            with _docker(lambda command: subprocess.CompletedProcess(command, 0, "container-9\n", "")), \
                 patch("kura.executors.docker._mutate_run_status", side_effect=die_on_second_status_write), \
                 self.assertRaises(KeyboardInterrupt):
                launch_docker(workspace=root, run_dir=run_dir, spec=SPEC, image="example:image", mounts=[], gpu=False)
            self.assertEqual(_status(run_dir)["state"], "launching")
            self.assertEqual(len(unresolved_create_intents(run_dir, "docker")), 1)
            with _docker([]) as docker:
                lines = resolve_docker_create_intents(run_dir)
            self.assertEqual(docker, [])  # settled from the record, nothing looked up or started
            self.assertIn("status now does", lines[0])
            self.assertEqual((_status(run_dir)["state"], _status(run_dir)["container_id"]), ("running", "container-9"))
            self.assertEqual(unresolved_create_intents(run_dir), [])

    def test_a_second_launch_that_passed_the_checks_does_not_start_another_container(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _run(root)
            with _docker(lambda command: subprocess.CompletedProcess(command, 0, "container-1\n", "")):
                launch_docker(workspace=root, run_dir=run_dir, spec=SPEC, image="example:image", mounts=[], gpu=False)
            with _docker([]) as docker, self.assertRaisesRegex(ValueError, "started first"):
                launch_docker(workspace=root, run_dir=run_dir, spec=SPEC, image="example:image", mounts=[], gpu=False)
            self.assertEqual([command[:2] for command in docker], [])

if __name__ == "__main__":
    unittest.main()

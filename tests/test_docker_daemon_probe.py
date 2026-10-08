"""Whether the Docker daemon can be used is one question with one answer, and the answer says why not."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kura.executors import docker


class DockerDaemonProblemTests(unittest.TestCase):
    def test_an_answering_daemon_has_no_problem(self) -> None:
        with patch("kura.executors.docker.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "ok", "")):
            self.assertIsNone(docker.docker_daemon_problem())

    def test_a_slow_daemon_is_reported_as_slow_not_as_absent(self) -> None:
        with patch("kura.executors.docker.subprocess.run", side_effect=subprocess.TimeoutExpired(["docker", "info"], 30)):
            self.assertIn("did not answer within", docker.docker_daemon_problem())

    def test_a_refusing_daemon_says_what_docker_said(self) -> None:
        refused = subprocess.CompletedProcess([], 1, "", "Cannot connect to the Docker daemon at unix:///var/run/docker.sock\n")
        with patch("kura.executors.docker.subprocess.run", return_value=refused):
            self.assertIn("Cannot connect to the Docker daemon", docker.docker_daemon_problem())

    def test_the_probe_runs_outside_the_workspace(self) -> None:
        # A docker helper that outlives the probe would otherwise hold the workspace open on Windows.
        with patch("kura.executors.docker.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "", "")) as run:
            docker.docker_daemon_problem()
        self.assertEqual(Path(run.call_args.kwargs["cwd"]), Path.home())
        self.assertTrue(run.call_args.kwargs["timeout"])

    def test_init_and_launch_report_the_same_reason(self) -> None:
        from kura.doctor import readiness_gaps

        with tempfile.TemporaryDirectory() as directory, \
                patch("kura.doctor.shutil.which", return_value="docker"), \
                patch("kura.executors.docker.subprocess.run", side_effect=subprocess.TimeoutExpired(["docker", "info"], 30)):
            gaps = readiness_gaps(Path(directory))
            with self.assertRaises(ValueError) as launch:
                docker.docker_preflight(Path(directory), [])
        self.assertTrue(any("did not answer within" in gap for gap in gaps))
        self.assertIn("did not answer within", str(launch.exception))

    def test_every_caller_asks_the_one_owner(self) -> None:
        from kura import doctor

        self.assertIs(doctor.docker_daemon_problem, docker.docker_daemon_problem)


if __name__ == "__main__":
    unittest.main()

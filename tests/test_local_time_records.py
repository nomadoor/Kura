"""Kura writes the times it records, and the names it gives records, in the host's local time with
its offset; a time another clock wrote (a Pod's UTC) is shown in local time too. Old records keep
their UTC names and values, and readers read both."""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from kura.run_commands.runpod_ssh import _runpod_remote_job_script


class RemoteExitNameTests(unittest.TestCase):
    def test_the_pod_names_its_exit_record_in_the_hosts_local_time(self) -> None:
        script = _runpod_remote_job_script(workspace="/workspace", run_id="example", realization_id="r1", remote_secret_path="/tmp/s.env",
                                           archive_name="b.tar.gz", remote_archive="/workspace/b.tar.gz", cwd="/opt", command="true")
        python = script.split("python - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        offset = datetime.now().astimezone().utcoffset()
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "runs" / "example" / "realizations").mkdir(parents=True)
            # The Pod's clock is UTC whatever the host's zone is.
            env = {"KURA_RUN_ID": "example", "KURA_WORKSPACE": directory, "KURA_EXIT_CODE": "0", "TZ": "UTC",
                   "KURA_UNATTENDED_WAIT_SEC": "0", "RUNPOD_POD_ID": "", "PATH": "/usr/bin:/bin"}
            subprocess.run([sys.executable, "-c", python], env=env, check=True, capture_output=True, text=True, timeout=30)
            [record] = (Path(directory) / "runs" / "example" / "realizations").glob("remote-exit-*.json")
            stamp = json.loads(record.read_text(encoding="utf-8"))["timestamp"]
        self.assertEqual(datetime.fromisoformat(stamp).utcoffset(), offset)
        sign = "+" if offset.total_seconds() >= 0 else "-"
        minutes = abs(int(offset.total_seconds())) // 60
        self.assertTrue(record.name.endswith(f"{sign}{minutes // 60:02d}{minutes % 60:02d}.json"), record.name)


class BillingConfirmationTests(unittest.TestCase):
    def test_an_old_utc_confirmation_is_shown_in_local_time(self) -> None:
        from kura.executors.common import host_time

        shown = host_time("2026-10-18T23:02:13.972222+00:00")
        self.assertEqual(datetime.fromisoformat(shown).utcoffset(), datetime.now().astimezone().utcoffset())
        source = Path(__file__).resolve().parents[1] / "src" / "kura" / "run_commands" / "launch.py"
        self.assertNotRegex(source.read_text(encoding="utf-8"), r"datetime\.now\(timezone\.utc\)\.isoformat\(\)")


if __name__ == "__main__":
    unittest.main()

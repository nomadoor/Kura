"""The newest launch request and the newest launch are the ones whose records say they came
last, not the ones whose ids sort last: ids are local clock times, which repeat or run
backwards when the clock is set back."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class NewestByRecordedTimeTests(unittest.TestCase):
    def test_the_latest_launch_request_is_the_one_written_last(self) -> None:
        from kura import runner

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            requests = runner.requests_dir(run_dir)
            requests.mkdir(parents=True)
            # Written after the clock went back an hour: its id sorts first.
            (requests / "20261101-013000-000000-aaaa.launch.json").write_text(json.dumps({"written_at": "2026-11-01T01:30:00-05:00"}), encoding="utf-8")
            (requests / "20261101-014500-000000-bbbb.launch.json").write_text(json.dumps({"written_at": "2026-11-01T01:45:00-04:00"}), encoding="utf-8")
            self.assertEqual(runner.latest_request(run_dir).name, "20261101-013000-000000-aaaa.launch.json")

    def test_the_monitor_falls_back_to_the_launch_its_records_say_came_last(self) -> None:
        from kura.monitor import _latest_realization

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            realizations = run_dir / "realizations"
            realizations.mkdir(parents=True)
            (realizations / "20261101-013000-000000.json").write_text(json.dumps({"id": "later", "launched_at": "2026-11-01T01:30:00-05:00"}), encoding="utf-8")
            (realizations / "20261101-014500-000000.json").write_text(json.dumps({"id": "earlier", "launched_at": "2026-11-01T01:45:00-04:00"}), encoding="utf-8")
            self.assertEqual(_latest_realization(run_dir, {})["id"], "later")


if __name__ == "__main__":
    unittest.main()

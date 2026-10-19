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

    def test_the_monitor_shows_no_realization_while_the_newest_launch_has_none_yet(self) -> None:
        from kura.monitor import _latest_realization

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            realizations = run_dir / "realizations"
            realizations.mkdir(parents=True)
            (realizations / "20261101-010000-000000.json").write_text(json.dumps({"id": "old", "launched_at": "2026-11-01T01:00:00-04:00"}), encoding="utf-8")
            (realizations / "20261101-020000-000000.create-intent.json").write_text(json.dumps({"requested_at": "2026-11-01T02:00:00-05:00"}), encoding="utf-8")
            self.assertEqual(_latest_realization(run_dir, {}), {})

    def test_the_monitor_falls_back_to_the_newest_launchs_newest_observation(self) -> None:
        from kura.monitor import _latest_observation

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            realizations = run_dir / "realizations"
            realizations.mkdir(parents=True)
            (realizations / "20261101-013000-000000.json").write_text(json.dumps({"launched_at": "2026-11-01T01:30:00-05:00"}), encoding="utf-8")
            (realizations / "20261101-014500-000000.json").write_text(json.dumps({"launched_at": "2026-11-01T01:45:00-04:00"}), encoding="utf-8")
            (realizations / "20261101-014500-000000.observed-20261101-015000-000000.json").write_text(json.dumps({"id": "older launch", "observed_at": "2026-11-01T01:50:00-04:00"}), encoding="utf-8")
            (realizations / "20261101-013000-000000.observed-20261101-013100-000000.json").write_text(json.dumps({"id": "first", "observed_at": "2026-11-01T01:31:00-05:00"}), encoding="utf-8")
            (realizations / "20261101-013000-000000.observed-20261101-013000-500000.json").write_text(json.dumps({"id": "second", "observed_at": "2026-11-01T01:40:00-05:00"}), encoding="utf-8")
            self.assertEqual(_latest_observation(run_dir, {})["id"], "second")

    def test_the_runner_takes_pending_requests_in_the_order_they_were_written(self) -> None:
        from kura import runner

        with tempfile.TemporaryDirectory() as directory:
            requests = []
            for run_id, name, written in (("a", "20261101-014500-000000-aaaa", "2026-11-01T01:45:00-04:00"),
                                          ("b", "20261101-013000-000000-bbbb", "2026-11-01T01:30:00-05:00")):
                folder = runner.requests_dir(Path(directory) / "runs" / run_id)
                folder.mkdir(parents=True)
                (folder / f"{name}.launch.json").write_text(json.dumps({"written_at": written}), encoding="utf-8")
                requests.append(folder / f"{name}.launch.json")
            self.assertEqual(sorted(reversed(requests), key=runner._written_order), requests)


if __name__ == "__main__":
    unittest.main()

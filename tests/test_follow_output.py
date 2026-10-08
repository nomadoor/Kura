"""`kura run execute` follows a run with a short progress line, not the trainer's raw log, and shows the log's end when the run fails."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from kura import runner
from kura.fsio import file_lock


def _follow(run_dir: Path, states: list[dict], *, log: str = "", earlier: str = "") -> str:
    """Follow a run whose log holds `earlier` from a previous attempt and gains `log` while it is followed."""
    root = run_dir.parent.parent
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "realizations").mkdir()
    (run_dir / "run.yaml").write_text("id: example\ntype: train\nrecipe: {steps: 200}\n", encoding="utf-8")
    (run_dir / "logs" / "stdout.log").write_text(earlier, encoding="utf-8")
    request = runner.write_launch_request(run_dir, executor="docker")
    runner.claim_request(request, 1)
    (run_dir / "realizations" / "r1.json").write_text(json.dumps({"controlled_by": {"request": request.name, "epoch": 1}}), encoding="utf-8")
    clock = {"now": 0.0}

    def show(index: int) -> None:
        (run_dir / "status.json").write_text(json.dumps({"last_realization": "realizations/r1.json", **states[index]}), encoding="utf-8")

    show(0)
    polls = {"n": 0}

    def sleep(seconds: float) -> None:
        clock["now"] += 31
        polls["n"] += 1
        if polls["n"] == 1:
            with (run_dir / "logs" / "stdout.log").open("a", encoding="utf-8") as handle:
                handle.write(log)
        show(min(polls["n"], len(states) - 1))

    out = io.StringIO()
    with file_lock(root / ".kura" / "runner" / "runner.lock", blocking=False):
        runner.follow(root, run_dir, request, sleep=sleep, out=out, clock=lambda: clock["now"])
    return out.getvalue()


class FollowOutputTests(unittest.TestCase):
    def test_progress_is_a_short_line_and_the_raw_log_stays_in_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            text = _follow(run_dir, [
                {"state": "running", "last_step": 10, "total_steps": 200, "seconds_per_iter": 1.2},
                {"state": "running", "last_step": 120, "total_steps": 200, "seconds_per_iter": 1.2},
                {"state": "completed", "publication_state": "not-required", "last_step": 200, "total_steps": 200},
            ], log="Downloading model.safetensors: 45%|####5     | 6.3G/14G\n" * 50)
        self.assertIn("120/200", text)
        self.assertIn("1.20s/it", text)
        self.assertNotIn("Downloading model.safetensors", text)

    def test_a_failed_run_shows_the_end_of_its_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            text = _follow(run_dir, [
                {"state": "running", "last_step": 0, "total_steps": 200},
                {"state": "failed", "publication_state": "not-required", "exit_code": 1},
            ], log="".join(f"line {n}\n" for n in range(500)) + "RuntimeError: CUDA out of memory\n")
        self.assertIn("RuntimeError: CUDA out of memory", text)
        self.assertNotIn("line 100\n", text)
        self.assertIn("logs/stdout.log", text)


    def test_a_failure_before_any_new_output_does_not_show_an_earlier_attempts_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            text = _follow(run_dir, [
                {"state": "launching"},
                {"state": "launch_failed", "publication_state": "not-required"},
            ], earlier="RuntimeError: from the previous attempt\n")
        self.assertNotIn("previous attempt", text)

if __name__ == "__main__":
    unittest.main()

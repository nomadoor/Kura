from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kura.log_tail import show, tail


def _show(path: Path) -> str:
    stdout = io.StringIO()
    with patch("sys.stdout", stdout):
        show(path)
    return stdout.getvalue()


class LogTailTests(unittest.TestCase):
    def test_a_long_log_shows_its_end_and_says_where_the_rest_is(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stdout.log"
            path.write_text("".join(f"line {number}\n" for number in range(1, 1001)), encoding="utf-8")
            output = _show(path)
        lines = output.splitlines()
        self.assertEqual(lines[0], "line 801")
        self.assertEqual(lines[-2], "line 1000")
        self.assertEqual(lines[-1], f"[showing lines 801-1000 of 1000; full log: {path}]")

    def test_a_short_log_is_shown_whole(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stdout.log"
            path.write_text("a\nb\nunfinished", encoding="utf-8")
            output = _show(path)
        self.assertEqual(output.splitlines(), ["a", "b", "unfinished", f"[all 3 lines; log: {path}]"])

    def test_bytes_are_capped_and_progress_bars_keep_their_last_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stdout.log"
            bar = "".join(f"\rsteps {step}/100" for step in range(100)) + "\n"
            path.write_text("x" * 200 + "\n" + bar * 3 + "y" * 30_000 + "\n" + "z" * 30_000 + "\n", encoding="utf-8")
            lines, first, total, _ = tail(path, max_bytes=50_000)
            self.assertEqual(lines, ["z" * 30_000])
            self.assertEqual((first, total), (6, 6))
            lines, first, _, _ = tail(path, max_lines=4, max_bytes=10**6)
            self.assertEqual(lines[0], "steps 99/100")
            self.assertEqual(first, 3)

    def test_a_progress_bar_larger_than_the_cap_shows_its_last_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stdout.log"
            path.write_bytes(b"start\n" + b"".join(f"\rsteps {step}/20000 loss=0.1".encode() for step in range(20000)))
            output = _show(path)
        self.assertEqual(output.splitlines(), ["steps 19999/20000 loss=0.1", f"[showing lines 2-2 of 2; full log: {path}]"])

    def test_follow_streams_progress_frames_and_starts_where_the_tail_ended(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stdout.log"
            path.write_bytes(b"first\n")
            sleeps = iter([None, None, KeyboardInterrupt()])

            def fake_sleep(_: float) -> None:
                step = next(sleeps)
                if isinstance(step, BaseException):
                    raise step
                with path.open("ab") as handle:
                    handle.write(b"\rstep 1/9\rstep 2/9" if path.stat().st_size == 6 else b"\rstep 3/9\ndone\n")

            stdout = io.StringIO()
            with patch("sys.stdout", stdout), patch("kura.log_tail.time.sleep", fake_sleep):
                self.assertEqual(show(path, follow=True), 0)
        self.assertEqual(stdout.getvalue().splitlines()[2:], ["step 1/9", "step 3/9", "done"])

    def test_follow_completes_an_unfinished_last_line_and_notices_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stdout.log"
            path.write_bytes("first\nhalf of a ".encode())
            steps = iter(["append", "replace", "stop"])

            def fake_sleep(_: float) -> None:
                step = next(steps)
                if step == "append":
                    with path.open("ab") as handle:
                        handle.write("line ✓\n".encode())
                elif step == "replace":
                    replacement = path.with_name("new.log")
                    replacement.write_bytes(b"fresh\n")
                    os.replace(replacement, path)
                else:
                    raise KeyboardInterrupt

            stdout = io.StringIO()
            with patch("sys.stdout", stdout), patch("kura.log_tail.time.sleep", fake_sleep):
                show(path, follow=True)
        lines = stdout.getvalue().splitlines()
        self.assertEqual(lines[1], "half of a ")
        self.assertEqual(lines[3], "half of a line ✓")
        self.assertIn("replaced", lines[4])
        self.assertEqual(lines[5], "fresh")

    def test_the_byte_cap_counts_utf8_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stdout.log"
            path.write_text("あ" * 100 + "\n" + "い" * 10 + "\n", encoding="utf-8")
            lines, _, _, _ = tail(path, max_bytes=100)
        self.assertEqual(lines, ["い" * 10])

    def test_a_closed_pipe_is_not_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stdout.log"
            path.write_text("a\n", encoding="utf-8")
            with patch("kura.log_tail._show", side_effect=BrokenPipeError), patch("kura.log_tail.os.dup2"):
                self.assertEqual(show(path), 0)

    def test_an_empty_log_says_so(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stdout.log"
            path.write_text("", encoding="utf-8")
            self.assertIn("the log is empty", _show(path))


if __name__ == "__main__":
    unittest.main()

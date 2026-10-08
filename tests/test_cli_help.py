"""An agent with no context learns from --help what each run option means and in which order the run commands go."""

from __future__ import annotations

import contextlib
import io
import unittest

from unittest.mock import patch

from kura.cli import main


def _help(*argv: str) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.suppress(SystemExit), patch("sys.argv", ["kura", *argv, "--help"]):
        main()
    return out.getvalue()


class RunHelpTests(unittest.TestCase):
    def test_run_new_explains_every_option(self) -> None:
        text = _help("run", "new")
        for phrase in ("groups runs", "run ID", "kura run capabilities", "RunPod GPU type"):
            self.assertIn(phrase, " ".join(text.split()))

    def test_run_help_gives_the_order_of_a_training_run(self) -> None:
        text = " ".join(_help("run").split())
        text = text[text.index("A training run, in order"):]
        positions = [text.index(step) for step in ("run new", "run.yaml", "run compile", "run plan", "run execute")]
        self.assertEqual(positions, sorted(positions))


if __name__ == "__main__":
    unittest.main()

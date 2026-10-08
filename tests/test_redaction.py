"""A secret's value never reaches output, whatever name the workspace gave it."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

VALUE = "Kq7" + "zz81" * 4


class RedactionTests(unittest.TestCase):
    def _redact(self, env: dict[str, str], workspace_yaml: str = "schema_version: 2\n") -> str:
        from kura.executors.common import _redact_secret_text

        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "workspace.yaml").write_text(workspace_yaml, encoding="utf-8")
            previous = Path.cwd()
            os.chdir(directory)
            try:
                with patch.dict(os.environ, env):
                    return _redact_secret_text(f"failed with {VALUE}")
            finally:
                os.chdir(previous)

    def test_a_key_under_a_name_workspace_yaml_chose_is_hidden(self) -> None:
        text = self._redact({"MY_RUNPOD": VALUE}, "schema_version: 2\nrunpod:\n  api_key_env: MY_RUNPOD\n")
        self.assertNotIn(VALUE, text)

    def test_the_ntfy_topic_is_hidden(self) -> None:
        # Anyone who knows the topic can read the run's notifications.
        self.assertNotIn(VALUE, self._redact({"KURA_NTFY_TOPIC": VALUE}))

    def test_an_ordinary_variable_is_left_alone(self) -> None:
        self.assertIn(VALUE, self._redact({"KURA_NTFY_PRIORITY": VALUE}))

    def test_one_rule_decides_which_values_are_hidden(self) -> None:
        from kura import secrets
        from kura.executors import common

        self.assertIs(common.secret_values, secrets.secret_values)


if __name__ == "__main__":
    unittest.main()

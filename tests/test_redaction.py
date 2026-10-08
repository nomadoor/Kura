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

    def test_a_word_the_user_chose_for_a_non_key_is_never_cut_out_of_records(self) -> None:
        # An ntfy topic is a name the user picks, such as "train"; hiding it would rewrite
        # "training" and run IDs in every record.
        from kura.executors.common import _redact_secrets

        with patch.dict(os.environ, {"KURA_NTFY_TOPIC": "train"}):
            self.assertEqual(_redact_secrets({"state": "training"}), {"state": "training"})

    def test_an_ordinary_variable_is_left_alone(self) -> None:
        self.assertIn(VALUE, self._redact({"KURA_NTFY_PRIORITY": VALUE}))

    def test_one_rule_decides_which_values_are_hidden(self) -> None:
        from kura import secrets
        from kura.executors import common

        self.assertIs(common.secret_values, secrets.secret_values)

    def test_values_are_looked_up_once_per_record(self) -> None:
        from kura.executors import common

        with patch.object(common, "secret_values", return_value=[VALUE]) as lookup:
            common._redact_secrets({"a": [f"x {VALUE}", "y", {"b": VALUE}]})
        self.assertEqual(lookup.call_count, 1)


if __name__ == "__main__":
    unittest.main()

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

    def test_a_word_the_user_chose_for_a_non_key_is_never_cut_out_of_records(self) -> None:
        # An ntfy topic is a name the user picks, such as "train"; hiding it would rewrite
        # "training" and run IDs in every record.
        from kura.executors.common import _redact_secrets

        with patch.dict(os.environ, {"KURA_NTFY_TOPIC": "train"}):
            self.assertEqual(_redact_secrets({"state": "training"}), {"state": "training"})

    def test_a_short_setting_under_a_secret_looking_name_is_not_a_secret(self) -> None:
        from kura.executors.common import _redact_secrets

        with patch.dict(os.environ, {"HF_HUB_DISABLE_IMPLICIT_TOKEN": "true"}):
            self.assertEqual(_redact_secrets({"cmd": ["--use_xformers=true"]}), {"cmd": ["--use_xformers=true"]})

    def test_an_unreadable_workspace_yaml_does_not_break_redaction(self) -> None:
        text = self._redact({"HF_TOKEN": VALUE})
        self.assertNotIn(VALUE, text)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "workspace.yaml").write_bytes(b"runpod: \xff\n")
            previous = Path.cwd()
            os.chdir(directory)
            try:
                from kura.executors.common import _redact_secret_text

                with patch.dict(os.environ, {"HF_TOKEN": VALUE}):
                    self.assertNotIn(VALUE, _redact_secret_text(f"x {VALUE}"))
            finally:
                os.chdir(previous)

    def test_an_ordinary_variable_is_left_alone(self) -> None:
        self.assertIn(VALUE, self._redact({"KURA_NTFY_PRIORITY": VALUE}))

    def test_one_rule_decides_which_values_are_hidden(self) -> None:
        from kura import secrets
        from kura.executors import common

        self.assertIs(common._redact_secrets, secrets.redact)
        self.assertIs(common._redact_secret_text, secrets.redact_text)

    def test_values_are_looked_up_once_per_record(self) -> None:
        from kura import secrets

        with patch.object(secrets, "secret_values", return_value=[VALUE]) as lookup:
            secrets.redact({"a": [f"x {VALUE}", "y", {"b": VALUE}]})
        self.assertEqual(lookup.call_count, 1)


class RecordRedactionTests(unittest.TestCase):
    """Redaction lives in the one place every record is built, and in the one notification sender."""

    def test_every_record_is_built_without_a_secret(self) -> None:
        from kura.records import record

        with patch.dict(os.environ, {"HF_TOKEN": VALUE}):
            built = record("launch_failed", {"error": f"401 for token {VALUE}", "nested": [{"detail": VALUE}]})
        self.assertNotIn(VALUE, str(built))
        self.assertEqual(built["kind"], "launch_failed")

    def test_records_written_outside_the_executor_writers_hold_no_secret(self) -> None:
        # The render realization and the runner's request outcomes are written with the plain
        # JSON writer; the record they are built from is already redacted.
        from kura import runner
        from kura.render import _write_realization_file

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"HF_TOKEN": VALUE}):
            run_dir = Path(directory) / "runs" / "render-1"
            run_dir.mkdir(parents=True)
            reference = _write_realization_file(run_dir, error=f"ComfyUI refused {VALUE}")
            request = run_dir / "requests" / "r1.launch.json"
            request.parent.mkdir()
            runner._record_request_outcome(request, ".launch-failed.json", "launch_failed", error=f"boom {VALUE}")
            written = [(run_dir / reference).read_text(encoding="utf-8"),
                       (request.parent / "r1.launch-failed.json").read_text(encoding="utf-8")]
        for text in written:
            self.assertIn("***", text)
            self.assertNotIn(VALUE, text)

    def test_the_runner_log_never_carries_a_secret(self) -> None:
        import contextlib
        import io as _io

        from kura import runner

        out = _io.StringIO()
        with patch.dict(os.environ, {"HF_TOKEN": VALUE}), contextlib.redirect_stdout(out):
            runner._log(f"follower failed: 401 for {VALUE}")
        self.assertNotIn(VALUE, out.getvalue())
        self.assertIn("follower failed: 401 for ***", out.getvalue())

    def test_a_notification_never_carries_a_secret(self) -> None:
        from kura import notifications

        sent = []
        with patch.dict(os.environ, {"HF_TOKEN": VALUE, "KURA_NTFY_TOPIC": "kura-topic"}), \
                patch.object(notifications, "send_ntfy_notification", side_effect=lambda subject, body, priority=None: sent.append((subject, body))):
            notifications.notify(["ntfy"], subject=f"failed {VALUE}", body=f"Render could not start: {VALUE}")
        self.assertEqual(sent, [("failed ***", "Render could not start: ***")])


if __name__ == "__main__":
    unittest.main()

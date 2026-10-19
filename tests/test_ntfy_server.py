"""The ntfy server is decided once, for the controller and for the Pod's own notification."""

from __future__ import annotations

import io
import os
import unittest
from unittest.mock import patch

from kura.notifications import ntfy_server
from kura.run_commands.runpod_ssh import _runpod_secret_env_payload


class NtfyServerTests(unittest.TestCase):
    def test_one_rule_and_one_default(self) -> None:
        clean = {key: value for key, value in os.environ.items() if key != "KURA_NTFY_SERVER"}
        with patch.dict(os.environ, clean, clear=True):
            self.assertEqual(ntfy_server(), "https://ntfy.sh")
        with patch.dict(os.environ, {"KURA_NTFY_SERVER": "https://ntfy.example.com/"}):
            self.assertEqual(ntfy_server(), "https://ntfy.example.com")
        for bad in ("file:///etc/passwd", "ntfy.example.com", "ftp://ntfy.example.com"):
            with self.subTest(bad=bad), patch.dict(os.environ, {"KURA_NTFY_SERVER": bad}), self.assertRaisesRegex(ValueError, "KURA_NTFY_SERVER"):
                ntfy_server()

    def test_the_pod_gets_the_checked_server_or_no_notification(self) -> None:
        env = {"KURA_NTFY_TOPIC": "kura-topic"}
        clean = {key: value for key, value in os.environ.items() if key not in ("KURA_NTFY_SERVER", "HF_TOKEN")}
        with patch.dict(os.environ, {**clean, **env}, clear=True):
            payload = _runpod_secret_env_payload(remote_notify=True) or ""
        self.assertIn("export KURA_NTFY_SERVER=https://ntfy.sh", payload)
        self.assertIn("export KURA_REMOTE_NOTIFY_NTFY=1", payload)
        with patch.dict(os.environ, {**clean, **env, "KURA_NTFY_SERVER": "file:///etc/passwd"}, clear=True), \
                patch("sys.stderr", new_callable=io.StringIO) as stderr:
            payload = _runpod_secret_env_payload(remote_notify=True) or ""
        self.assertIn("the Pod will not send its own notification", stderr.getvalue())
        self.assertNotIn("KURA_REMOTE_NOTIFY_NTFY", payload)
        self.assertNotIn("file://", payload)

    def test_the_pod_script_holds_no_server_rule_of_its_own(self) -> None:
        from kura.run_commands import runpod_ssh

        source = open(runpod_ssh.__file__, encoding="utf-8").read()
        self.assertNotIn('os.environ.get("KURA_NTFY_SERVER", "https://ntfy.sh")', source)


if __name__ == "__main__":
    unittest.main()

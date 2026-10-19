"""A token is sent only to the host it is meant for: urllib copies ordinary headers onto a redirect."""

from __future__ import annotations

import io
import os
import unittest
import urllib.request
from unittest.mock import patch

TOKEN = "hf_" + "x" * 20


def _redirected(request: urllib.request.Request) -> urllib.request.Request:
    handler = urllib.request.HTTPRedirectHandler()
    return handler.redirect_request(request, io.BytesIO(), 302, "Found", {}, "https://cdn.example.com/blob")


class TokenRedirectTests(unittest.TestCase):
    def test_the_helper_sends_the_token_to_the_first_host_only(self) -> None:
        from kura.secrets import bearer_request

        request = bearer_request("https://huggingface.co/x", TOKEN, method="HEAD")
        self.assertEqual(request.get_header("Authorization"), f"Bearer {TOKEN}")
        self.assertIsNone(_redirected(request).get_header("Authorization"))
        self.assertFalse(bearer_request("https://huggingface.co/x", None, method="HEAD").has_header("Authorization"))

    def test_the_hugging_face_probe_and_ntfy_never_forward_their_token(self) -> None:
        from kura import notifications
        from kura.run_commands.plan import _hf_file_size_probe

        sent = []

        def capture(request, timeout):
            sent.append(request)
            raise OSError("offline")

        with patch.dict(os.environ, {"HF_TOKEN": TOKEN, "KURA_NTFY_TOPIC": "topic", "KURA_NTFY_TOKEN": TOKEN}), \
                patch("urllib.request.urlopen", side_effect=capture):
            _hf_file_size_probe({"repo_id": "org/model", "filename": "a.safetensors"})
            with self.assertRaises(OSError):
                notifications.send_ntfy_notification("subject", "body")
        self.assertEqual(len(sent), 2)
        for request in sent:
            with self.subTest(url=request.full_url):
                self.assertEqual(request.get_header("Authorization"), f"Bearer {TOKEN}")
                self.assertIsNone(_redirected(request).get_header("Authorization"))

    def test_the_pod_notification_script_never_forwards_its_token(self) -> None:
        from kura.run_commands import runpod_ssh

        source = open(runpod_ssh.__file__, encoding="utf-8").read()
        self.assertNotIn('headers["Authorization"]', source)
        self.assertIn('add_unredirected_header("Authorization"', source)


if __name__ == "__main__":
    unittest.main()

"""The file secret scan asks kura.secrets which names hold secrets, and a hint never hides a real value."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kura.checks import secret_findings

# Built at runtime so this file itself does not trip the repository secret scan.
VALUE = "Ab3" + "x9Kq" * 5
HF = "hf" + "_" + "A" * 24


def _scan(text: str) -> list[str]:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "notes.txt"
        path.write_text(text + "\n", encoding="utf-8")
        return secret_findings([path], Path(directory))


class SecretScanTests(unittest.TestCase):
    def test_every_secret_name_kura_knows_is_caught(self) -> None:
        from kura.secrets import SECRET_NAME_PARTS

        for part in SECRET_NAME_PARTS:
            for name in (f"MY_{part}", f"R2_{part}_ID"):
                with self.subTest(name=name):
                    self.assertTrue(_scan(f"{name}={VALUE}"), name)

    def test_a_known_variable_name_on_the_line_does_not_hide_its_value(self) -> None:
        for line in (f"HF_TOKEN={HF}", f"RUNPOD_API_KEY={VALUE}", f"KURA_NTFY_TOKEN: {VALUE}"):
            with self.subTest(line=line[:12]):
                self.assertTrue(_scan(line))

    def test_names_and_placeholders_are_not_values(self) -> None:
        for line in (
            "api_key_env: RUNPOD_API_KEY",
            "token = os.environ.get('HF_TOKEN')",
            "HF_TOKEN=<your-token>",
            "password: your-app-password",
            "secret_ref: ${R2_SECRET_ACCESS_KEY}",
            'hf_token = declared_secret("HF_TOKEN")',
            "RUNPOD_API_KEY=pod-scoped-key",
        ):
            with self.subTest(line=line):
                self.assertEqual(_scan(line), [])


if __name__ == "__main__":
    unittest.main()

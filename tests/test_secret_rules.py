"""Which names hold secrets is decided by whole words; which files leak them, by the exact values Kura holds."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Built at runtime so this file itself does not trip the repository secret scan.
VALUE = "Kq7" + "zz81" * 4
HF = "hf" + "_" + "A" * 24

NAMES = {
    "HF_TOKEN": True, "RUNPOD_API_KEY": True, "R2_SECRET_ACCESS_KEY": True, "AWS_ACCESS_KEY_ID": True,
    "SSH_PRIVATE_KEY": True, "DB_PASSWORD": True, "apiKey": True, "api-key": True, "accessToken": True,
    "TOKENIZERS_PARALLELISM": False, "tokenizer_path": False, "KEYFRAME_COUNT": False, "PATH": False,
    "KURA_NTFY_PRIORITY": False,
}


class SecretNameTests(unittest.TestCase):
    def test_a_name_is_a_secret_by_its_whole_words(self) -> None:
        from kura.secrets import is_secret_name

        for name, expected in NAMES.items():
            with self.subTest(name=name):
                self.assertEqual(is_secret_name(name), expected)

    def test_doctor_asks_the_same_rule(self) -> None:
        from kura import doctor, secrets

        self.assertIs(doctor.is_secret_name, secrets.is_secret_name)


class SecretFileTests(unittest.TestCase):
    def _scan(self, text: str, env: dict[str, str] | None = None) -> list[str]:
        from kura.checks import secret_findings

        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            path = Path(directory) / "notes.txt"
            path.write_text(text + "\n", encoding="utf-8")
            previous = Path.cwd()
            os.chdir(directory)
            try:
                with patch.dict(os.environ, env or {}):
                    return secret_findings([path], Path(directory))
            finally:
                os.chdir(previous)

    def test_a_value_kura_holds_is_found_under_any_name(self) -> None:
        self.assertTrue(self._scan(f"note = {VALUE}", {"HF_TOKEN": VALUE}))

    def test_a_token_shape_is_found_even_next_to_a_known_name(self) -> None:
        self.assertTrue(self._scan(f"HF_TOKEN={HF}"))

    def test_names_and_guesses_are_not_findings(self) -> None:
        for line in ("password: correcthorse", "api_key_env: RUNPOD_API_KEY", 'hf_token = declared_secret("HF_TOKEN")'):
            with self.subTest(line=line):
                self.assertEqual(self._scan(line), [])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import re
import subprocess
import tempfile
import tomllib
import unittest
from pathlib import Path

from kura import __version__
from kura.install_source import describe_install, kura_provenance

ROOT = Path(__file__).resolve().parents[1]


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args], check=True, capture_output=True, text=True,
    ).stdout.strip()


class InstallSourceTests(unittest.TestCase):
    def test_git_install_records_url_and_commit_without_credentials(self) -> None:
        source = describe_install({
            "url": "https://user:secret-token@github.com/nomadoor/Kura",
            "vcs_info": {"vcs": "git", "commit_id": "a" * 40, "requested_revision": "main"},
        })
        self.assertEqual(source, {
            "kind": "git", "url": "https://github.com/nomadoor/Kura",
            "commit": "a" * 40, "requested_revision": "main",
        })

    def test_ssh_user_name_is_kept_and_only_secrets_are_dropped(self) -> None:
        ssh = describe_install({"url": "ssh://git@github.com/nomadoor/Kura", "vcs_info": {"vcs": "git", "commit_id": "c"}})
        with_user_credential = describe_install({"url": "https://deploy-credential@github.com/nomadoor/Kura", "vcs_info": {"vcs": "git", "commit_id": "c"}})
        self.assertEqual(ssh["url"], "ssh://git@github.com/nomadoor/Kura")
        self.assertEqual(with_user_credential["url"], "https://github.com/nomadoor/Kura")

    def test_editable_install_records_checkout_and_its_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory)
            _git(checkout, "init", "-q")
            (checkout / "README.md").write_text("x\n", encoding="utf-8")
            _git(checkout, "add", "README.md")
            _git(checkout, "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", "commit", "-q", "-m", "init")
            head = _git(checkout, "rev-parse", "HEAD")
            clean = describe_install({"url": checkout.as_uri(), "dir_info": {"editable": True}})
            (checkout / "README.md").write_text("changed\n", encoding="utf-8")
            dirty = describe_install({"url": checkout.as_uri(), "dir_info": {"editable": True}})

        self.assertEqual(clean, {"kind": "editable", "path": str(checkout), "commit": head, "dirty": False})
        self.assertEqual(dirty["dirty"], True)

    def test_editable_install_outside_git_has_unknown_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = describe_install({"url": Path(directory).as_uri(), "dir_info": {"editable": True}})
        self.assertEqual(source, {"kind": "editable", "path": directory, "commit": None, "dirty": None})

    def test_missing_or_unrecognized_metadata_is_unknown(self) -> None:
        self.assertEqual(describe_install(None), {"kind": "unknown"})
        self.assertEqual(describe_install({"url": "https://example.com/kura.whl", "archive_info": {}}), {"kind": "unknown"})

    def test_provenance_carries_version_and_source(self) -> None:
        provenance = kura_provenance()
        self.assertEqual(provenance["kura_version"], __version__)
        self.assertIn(provenance["kura_source"]["kind"], {"git", "editable", "unknown"})


class PackageMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    def test_version_has_one_source(self) -> None:
        self.assertNotIn("version", self.project)
        self.assertIn("version", self.project.get("dynamic", []))

    def test_every_dependency_has_an_upper_bound(self) -> None:
        requirements = list(self.project["dependencies"])
        for extra in self.project.get("optional-dependencies", {}).values():
            requirements.extend(extra)
        unbounded = [requirement for requirement in requirements if not re.search(r"<(?!=)", requirement)]
        self.assertEqual(unbounded, [])


if __name__ == "__main__":
    unittest.main()

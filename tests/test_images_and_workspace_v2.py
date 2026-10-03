from __future__ import annotations

import argparse
import io
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import yaml

from kura.cli import cmd_image_build, cmd_image_publish, cmd_init, cmd_run_status, cmd_workspace_migrate
from kura.images import PINNED_IMAGES, effective_image, launch_image, launch_image_warnings, mutable_override_warning
from kura.run_commands.plan import _image_preflight_report
from kura.workspace import WORKSPACE_SCHEMA_VERSION, migrate_workspace_config, require_workspace

ROOT = Path(__file__).resolve().parents[1]

# The shape the maintainer's workspace had before schema 2.
V1_WORKSPACE = {
    "schema_version": 1,
    "name": "kura",
    "docker": {
        "images": {
            "ai-toolkit": {"local": "nomadoor/kura-ai-toolkit:dev", "remote": "nomadoor/kura-ai-toolkit:h3-checkpoint-static-20260921", "dockerfile": "docker/ai-toolkit/Dockerfile", "context": "."},
            "musubi-tuner": {"local": "nomadoor/kura-musubi-tuner:dev", "remote": "nomadoor/kura-musubi-tuner:resume-e2e-20260827", "dockerfile": "docker/musubi-tuner/Dockerfile", "context": "."},
            "comfyui": {"local": "nomadoor/kura-comfyui:dev", "remote": "nomadoor/kura-comfyui:dev", "dockerfile": "docker/comfyui/Dockerfile", "context": "."},
        },
        "workspace_target": "/workspace",
        "gpu": True,
    },
    "runpod": {
        "default_image": {
            "ai-toolkit": PINNED_IMAGES["ai-toolkit"],
            "musubi-tuner": PINNED_IMAGES["musubi-tuner"],
            "sd-scripts": "nomadoor/kura-sd-scripts@sha256:" + "f" * 64,
        },
        "api_key_env": "RUNPOD_API_KEY",
    },
}


@contextmanager
def _inside(directory: Path):
    previous = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


class PinnedImageTests(unittest.TestCase):
    def test_every_image_is_pinned_by_digest(self) -> None:
        self.assertEqual(set(PINNED_IMAGES), {"ai-toolkit", "musubi-tuner", "sd-scripts", "comfyui"})
        for reference in PINNED_IMAGES.values():
            self.assertRegex(reference, r"@sha256:[0-9a-f]{64}$")

    def test_override_replaces_the_pinned_default(self) -> None:
        self.assertEqual(effective_image({}, "sd-scripts"), {"name": "sd-scripts", "reference": PINNED_IMAGES["sd-scripts"], "origin": "pinned"})
        override = effective_image({"images": {"sd-scripts": "example/sd:test"}}, "sd-scripts")
        self.assertEqual(override["reference"], "example/sd:test")
        self.assertEqual(override["origin"], "override")

    def test_only_a_mutable_override_warns(self) -> None:
        self.assertIsNone(mutable_override_warning(effective_image({}, "comfyui")))
        self.assertIsNone(mutable_override_warning(effective_image({"images": {"comfyui": "x@sha256:" + "0" * 64}}, "comfyui")))
        self.assertIn("mutable tag", mutable_override_warning(effective_image({"images": {"comfyui": "x:dev"}}, "comfyui")))


class LaunchImageTests(unittest.TestCase):
    def test_a_compiled_run_keeps_its_frozen_image_and_the_plan_says_so(self) -> None:
        override = {"images": {"ai-toolkit": "example/ai@sha256:" + "1" * 64}}
        compiled = {"selected_image": PINNED_IMAGES["ai-toolkit"], "image_origin": "pinned"}
        image = launch_image(override, "ai-toolkit", compiled)
        self.assertEqual(image["reference"], PINNED_IMAGES["ai-toolkit"])
        self.assertIn("recompile to use the current image", "\n".join(launch_image_warnings(image)))
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "resolved").mkdir()
            (run_dir / "resolved" / "env.lock").write_text(yaml.safe_dump(compiled), encoding="utf-8")
            records = _image_preflight_report({"backend": {"name": "ai-toolkit"}}, override, run_dir)
        self.assertIn(f"{PINNED_IMAGES['ai-toolkit']} (pinned by Kura, frozen at compile)", records[0]["fact"])
        self.assertEqual([record["severity"] for record in records], ["info", "warning"])

    def test_a_run_compiled_before_pinning_warns_about_its_mutable_tag(self) -> None:
        image = launch_image({}, "musubi-tuner", {"selected_image": "nomadoor/kura-musubi-tuner:dev"})
        self.assertEqual(image["origin"], "compiled")
        self.assertIn("the image frozen at compile uses mutable tag", "\n".join(launch_image_warnings(image)))

    def test_an_uncompiled_run_uses_the_current_image(self) -> None:
        image = launch_image({}, "sd-scripts", {})
        self.assertEqual((image["reference"], image["frozen"]), (PINNED_IMAGES["sd-scripts"], False))

    def test_an_empty_override_is_refused_as_a_configuration_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "delete the line"):
            effective_image({"images": {"comfyui": " "}}, "comfyui")
        from kura.workspace import validate_workspace_config

        with self.assertRaisesRegex(ValueError, "images.comfyui must name an image"):
            validate_workspace_config({"schema_version": 2, "images": {"comfyui": ""}})
        from kura.cli import cmd_doctor_workspace

        with tempfile.TemporaryDirectory() as directory, _inside(Path(directory)):
            Path("workspace.yaml").write_text("schema_version: 2\nimages:\n  comfyui: ''\n", encoding="utf-8")
            stdout = io.StringIO()
            with patch("sys.stdout", stdout):
                self.assertEqual(cmd_doctor_workspace(argparse.Namespace()), 1)
            self.assertIn("images.comfyui must name an image", stdout.getvalue())


class WorkspaceSchemaTests(unittest.TestCase):
    def test_init_writes_the_current_schema_without_image_definitions(self) -> None:
        with tempfile.TemporaryDirectory() as directory, _inside(Path(directory)):
            self.assertEqual(cmd_init(argparse.Namespace()), 0)
            config = yaml.safe_load(Path("workspace.yaml").read_text(encoding="utf-8"))
            self.assertEqual(config["schema_version"], WORKSPACE_SCHEMA_VERSION)
            self.assertNotIn("images", config.get("docker", {}))
            self.assertNotIn("default_image", config.get("runpod", {}))
            self.assertFalse(list(Path(directory).rglob("Dockerfile")))

    def test_older_schema_is_refused_with_the_migrate_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory, _inside(Path(directory)):
            Path("workspace.yaml").write_text(yaml.safe_dump(V1_WORKSPACE), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "kura workspace migrate"):
                require_workspace()
            stderr = io.StringIO()
            with patch("sys.stderr", stderr), patch("sys.stdout", io.StringIO()):
                self.assertNotEqual(cmd_run_status(argparse.Namespace(run_id="any", json=False)), 0)
            self.assertIn("kura workspace migrate", stderr.getvalue())

    def test_migration_keeps_only_digests_that_differ_from_the_pinned_table(self) -> None:
        migrated, notes = migrate_workspace_config(V1_WORKSPACE)
        self.assertEqual(migrated["schema_version"], WORKSPACE_SCHEMA_VERSION)
        self.assertEqual(migrated["images"], {"sd-scripts": "nomadoor/kura-sd-scripts@sha256:" + "f" * 64})
        self.assertNotIn("images", migrated["docker"])
        self.assertEqual(migrated["docker"]["workspace_target"], "/workspace")
        self.assertNotIn("default_image", migrated["runpod"])
        self.assertEqual(migrated["runpod"]["api_key_env"], "RUNPOD_API_KEY")
        joined = "\n".join(notes)
        self.assertIn("nomadoor/kura-musubi-tuner:resume-e2e-20260827", joined)
        self.assertIn("dockerfile", joined)

    def test_migration_edge_cases(self) -> None:
        minimal, _ = migrate_workspace_config({"schema_version": 1, "name": "x"})
        self.assertEqual(minimal, {"schema_version": WORKSPACE_SCHEMA_VERSION, "name": "x"})
        comfy, notes = migrate_workspace_config({"schema_version": 1, "comfyui": {"runpod": {"default_image": {"comfyui": "c@sha256:" + "2" * 64}}}})
        self.assertEqual(comfy["images"], {"comfyui": "c@sha256:" + "2" * 64})
        self.assertNotIn("default_image", comfy["comfyui"]["runpod"])
        unknown, notes = migrate_workspace_config({"schema_version": 1, "docker": {"images": {"custom": {"local": "x@sha256:" + "3" * 64}}}})
        self.assertNotIn("images", unknown)
        self.assertIn("no image named 'custom'", "\n".join(notes))
        conflict, notes = migrate_workspace_config({"schema_version": 1, "docker": {"images": {"sd-scripts": {"local": "a@sha256:" + "4" * 64}}}, "runpod": {"default_image": {"sd-scripts": "b@sha256:" + "5" * 64}}})
        self.assertEqual(conflict["images"], {"sd-scripts": "a@sha256:" + "4" * 64})
        self.assertIn("local and RunPod now share one image", "\n".join(notes))
        broken, notes = migrate_workspace_config({"schema_version": 1, "docker": {"images": "nonsense"}})
        self.assertIn("not a mapping", "\n".join(notes))
        for section in ({"runpod": {"default_image": "x@sha256:" + "6" * 64}}, {"comfyui": {"runpod": {"default_image": ["x"]}}}):
            scalar, notes = migrate_workspace_config({"schema_version": 1, **section})
            self.assertNotIn("images", scalar)
            self.assertIn("default_image: it was not a mapping", "\n".join(notes))
        string_version, notes = migrate_workspace_config({"schema_version": "2"})
        self.assertEqual(string_version["schema_version"], WORKSPACE_SCHEMA_VERSION)
        with self.assertRaisesRegex(ValueError, "Edit workspace.yaml by hand"):
            migrate_workspace_config({"schema_version": 1, "runpod": {"container_cwd": "/x"}})
        again, notes = migrate_workspace_config(migrated := migrate_workspace_config(V1_WORKSPACE)[0])
        self.assertEqual((again, notes), (migrated, []))

    def test_migrate_previews_then_applies_only_when_confirmed(self) -> None:
        with tempfile.TemporaryDirectory() as directory, _inside(Path(directory)):
            path = Path("workspace.yaml")
            path.write_text(yaml.safe_dump(V1_WORKSPACE), encoding="utf-8")
            before = path.read_text(encoding="utf-8")
            stdout = io.StringIO()
            with patch("sys.stdout", stdout), patch("sys.stdin.isatty", return_value=False):
                self.assertEqual(cmd_workspace_migrate(argparse.Namespace(yes=False)), 1)
            self.assertEqual(path.read_text(encoding="utf-8"), before)
            self.assertIn("schema_version", stdout.getvalue())
            with patch("sys.stdout", io.StringIO()):
                self.assertEqual(cmd_workspace_migrate(argparse.Namespace(yes=True)), 0)
            self.assertEqual(yaml.safe_load(path.read_text(encoding="utf-8"))["schema_version"], WORKSPACE_SCHEMA_VERSION)
            self.assertEqual(Path("workspace.yaml.v1").read_text(encoding="utf-8"), before)
            require_workspace()


class DevelopmentImageTests(unittest.TestCase):
    def test_build_refuses_outside_an_editable_checkout(self) -> None:
        stderr = io.StringIO()
        with patch("kura.cli.development_checkout", return_value=None), patch("sys.stderr", stderr):
            self.assertEqual(cmd_image_build(argparse.Namespace(name="sd-scripts", ref=None, allow_large_build_cache=True)), 1)
        self.assertIn(PINNED_IMAGES["sd-scripts"], stderr.getvalue())

    def test_build_uses_the_checkout_definitions_from_any_workspace(self) -> None:
        calls: list[list[str]] = []

        def run(command: list[str], *, capture: bool = False):
            calls.append(command)
            import subprocess
            return subprocess.CompletedProcess(command, 0, "sha256:built\n", "")

        with patch("kura.cli.development_checkout", return_value=ROOT), patch("kura.cli._docker_run", side_effect=run), patch("sys.stdout", io.StringIO()):
            self.assertEqual(cmd_image_build(argparse.Namespace(name="sd-scripts", ref=None, allow_large_build_cache=True)), 0)
        build = calls[0]
        self.assertEqual(build[build.index("--file") + 1], str(ROOT / "docker" / "sd-scripts" / "Dockerfile"))
        self.assertEqual(build[-1], str(ROOT))
        self.assertIn("SD_SCRIPTS_REF=37a1cbbc5725ed2a3575506e7bd2001c9908ac92", build)
        self.assertEqual(build[build.index("--tag") + 1], "kura-sd-scripts:dev")

    def test_publish_needs_a_target_tag(self) -> None:
        stdout = io.StringIO()
        with patch("kura.cli.development_checkout", return_value=ROOT), patch("sys.stdout", stdout):
            self.assertEqual(cmd_image_publish(argparse.Namespace(name="sd-scripts", tag="example/sd:1", dry_run=True)), 0)
        self.assertIn("example/sd:1", stdout.getvalue())
        self.assertIn("kura-sd-scripts:dev", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()

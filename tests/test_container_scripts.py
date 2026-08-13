from __future__ import annotations

import importlib
from dataclasses import replace
import io
import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kura.container_scripts import script_source
from kura.backends import BACKENDS, BackendSurface
from kura.init_templates import SD_SCRIPTS_DOCKERFILE_TEMPLATE, SD_SCRIPTS_SYMLINK_PATCH_TEMPLATE
from kura.provenance import adapter_source_identity, legacy_adapter_source_identity


class ContainerScriptTests(unittest.TestCase):
    def test_sd_scripts_init_template_carries_the_managed_symlink_patch(self) -> None:
        root = Path(__file__).resolve().parents[1]
        patch_text = (root / "docker/sd-scripts/patches/0001-preserve-safetensors-symlink-name.patch").read_text(encoding="utf-8")
        self.assertEqual(SD_SCRIPTS_SYMLINK_PATCH_TEMPLATE, patch_text)
        self.assertIn("git apply --check", SD_SCRIPTS_DOCKERFILE_TEMPLATE)
        self.assertIn('io.kura.patch.symlink-safetensors="preserve-input-filename-v2"', SD_SCRIPTS_DOCKERFILE_TEMPLATE)

    def test_sd_scripts_probe_fails_closed_for_each_checkpoint_loader(self) -> None:
        namespace = {"__name__": "__test__"}
        exec(script_source("sd_scripts_probe.py"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "library"
            library.mkdir()
            model_io = library / "model_io.py"
            sdxl = library / "sdxl_train_util.py"
            model_io.write_text("safe", encoding="utf-8")
            sdxl.write_text("safe", encoding="utf-8")
            self.assertEqual(namespace["symlink_compatibility"](root), {"sd_checkpoint_symlink_safe": True, "sdxl_checkpoint_symlink_safe": True})
            sdxl.write_text("os.readlink(name_or_path)", encoding="utf-8")
            self.assertFalse(namespace["symlink_compatibility"](root)["sdxl_checkpoint_symlink_safe"])
            model_io.write_text("os.path.realpath(name_or_path)", encoding="utf-8")
            self.assertFalse(namespace["symlink_compatibility"](root)["sd_checkpoint_symlink_safe"])

    def test_adapter_identity_ignores_unrelated_registry_changes(self) -> None:
        baseline = {name: adapter_source_identity(name)["value"] for name in ("ai-toolkit", "musubi-tuner")}
        original = Path.read_bytes

        def changed_registry(path):
            payload = original(path)
            return payload + (b"\n# unrelated registry entry\n" if path.name == "registry.py" else b"")

        with patch.object(Path, "read_bytes", changed_registry):
            changed = {name: adapter_source_identity(name)["value"] for name in baseline}

        self.assertEqual(baseline, changed)

    def test_adapter_identity_tracks_only_imported_shared_helper(self) -> None:
        ai_baseline = adapter_source_identity("ai-toolkit")["value"]
        musubi_baseline = adapter_source_identity("musubi-tuner")["value"]
        original = Path.read_bytes

        def changed_truthy(path):
            payload = original(path)
            if path.name == "shared.py":
                return payload.replace(b"def _truthy(value: Any)", b"def _truthy(value: Any)  ")
            return payload

        with patch.object(Path, "read_bytes", changed_truthy):
            self.assertEqual(ai_baseline, adapter_source_identity("ai-toolkit")["value"])
            self.assertNotEqual(musubi_baseline, adapter_source_identity("musubi-tuner")["value"])

    def test_adapter_identity_tracks_declared_surface(self) -> None:
        baseline = adapter_source_identity("ai-toolkit")
        adapter = BACKENDS["ai-toolkit"]
        changed_surface = BackendSurface(
            fields=adapter.surface.fields | {"future_field"},
            escape_hatches=adapter.surface.escape_hatches,
        )
        with patch.dict(BACKENDS, {"ai-toolkit": replace(adapter, surface=changed_surface)}):
            changed = adapter_source_identity("ai-toolkit")

        self.assertEqual(baseline["scope"], "selected-adapter-v2")
        self.assertNotEqual(baseline["value"], changed["value"])

    def test_legacy_adapter_identity_remains_available(self) -> None:
        identity = legacy_adapter_source_identity("musubi-tuner")

        self.assertEqual(identity["scope"], "legacy-whole-files")
        self.assertEqual(len(identity["value"]), 64)

    def test_musubi_adapter_identity_includes_embedded_runtime_helpers(self) -> None:
        baseline = adapter_source_identity("musubi-tuner")["value"]
        original = Path.read_bytes

        def changed_helper(path):
            payload = original(path)
            return payload + (b"changed" if path.name == "hf_download.py" else b"")

        with patch.object(Path, "read_bytes", changed_helper):
            changed = adapter_source_identity("musubi-tuner")["value"]

        self.assertNotEqual(baseline, changed)

    def test_sd_scripts_adapter_identity_includes_anima_runtime_publisher(self) -> None:
        baseline = adapter_source_identity("sd-scripts")["value"]
        original = Path.read_bytes

        def changed_helper(path):
            payload = original(path)
            return payload + (b"changed" if path.name == "sd_scripts_publish_anima.py" else b"")

        with patch.object(Path, "read_bytes", changed_helper):
            changed = adapter_source_identity("sd-scripts")["value"]

        self.assertNotEqual(baseline, changed)

    def test_container_scripts_compile(self) -> None:
        for name in (
            "hf_download.py",
            "safetensors_validator.py",
            "prune_checkpoints.py",
            "musubi_probe.py",
            "musubi_dataset_assert.py",
            "sd_scripts_publish_anima.py",
        ):
            with self.subTest(name=name):
                compile(script_source(name), name, "exec")

    def test_musubi_dataset_assert_counts_video_inputs(self) -> None:
        namespace = {"__name__": "__test__"}
        exec(script_source("musubi_dataset_assert.py"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.mp4").write_bytes(b"video")
            (root / "caption.txt").write_text("caption", encoding="utf-8")

            count = namespace["media_count"](
                root,
                namespace["VIDEO_SUFFIXES"],
                "video_directory",
            )

        self.assertEqual(count, 1)

    def test_musubi_dataset_assert_dispatches_video_jsonl_and_defers_unknown_sources(self) -> None:
        namespace = {"__name__": "__test__"}
        exec(script_source("musubi_dataset_assert.py"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_jsonl = root / "videos.jsonl"
            video_jsonl.write_text('{}\n', encoding="utf-8")
            config = root / "dataset.toml"
            config.write_text(
                '[[datasets]]\nvideo_jsonl_file = "' + video_jsonl.as_posix() + '"\n'
                '[[datasets]]\nfuture_native_source = "opaque"\n',
                encoding="utf-8",
            )

            with patch.object(sys, "argv", ["musubi_dataset_assert.py", str(config)]):
                namespace["main"]()

    def test_hf_download_child_script_compiles(self) -> None:
        module = importlib.import_module("kura.container_scripts.hf_download")

        compile(module.CHILD, "hf_download.CHILD", "exec")

    def test_hf_download_preflight_measures_remote_metadata_and_disk(self) -> None:
        namespace = {"__name__": "__test__"}
        exec(script_source("hf_download.py"), namespace)
        fake_hub = SimpleNamespace(
            try_to_load_from_cache=lambda *args, **kwargs: None,
            hf_hub_url=lambda **kwargs: "https://huggingface.invalid/file",
            get_hf_file_metadata=lambda *args, **kwargs: SimpleNamespace(size=1234),
        )
        item = {
            "key": "dit",
            "repo_id": "owner/model",
            "filename": "weights.safetensors",
            "link_path": "/workspace/cache/models/owner--model/dit/weights.safetensors",
        }
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory) / "hf"
            cache = home / "hub"
            mapping = json.dumps([{"container": str(cache), "workspace": "/workspace/cache/huggingface"}])
            usage = SimpleNamespace(total=10_000, used=1_000, free=9_000)
            with (
                patch.dict(os.environ, {"HF_HOME": str(home), "HF_HUB_CACHE": str(cache), "KURA_WORKSPACE_PATH_MAPS": mapping}, clear=True),
                patch.dict(sys.modules, {"huggingface_hub": fake_hub}),
                patch.object(namespace["shutil"], "disk_usage", return_value=usage),
            ):
                namespace["DOWNLOAD_RESERVE_BYTES"] = 100
                namespace["preflight_downloads"]([item])
        self.assertEqual(item["_size_bytes"], 1234)

    def test_hf_download_progress_is_scoped_and_capped_to_the_item(self) -> None:
        namespace = {"__name__": "__test__"}
        exec(script_source("hf_download.py"), namespace)

        self.assertEqual(namespace["progress_bytes"](1050, 1000, 200), 50)
        self.assertEqual(namespace["progress_bytes"](1400, 1000, 200), 200)
        self.assertEqual(namespace["progress_bytes"](900, 1000, 200), 0)

    def test_hf_download_progress_uses_the_hub_cache_layout(self) -> None:
        namespace = {"__name__": "__test__"}
        exec(script_source("hf_download.py"), namespace)
        item = {"repo_id": "owner/model"}

        directories = namespace["repo_cache_dirs"]("/cache/hf", item)

        self.assertEqual(
            directories,
            [
                "/cache/hf/models--owner--model",
                "/cache/hf/.locks/models--owner--model",
            ],
        )

    def test_hf_download_starts_all_model_parts_in_parallel(self) -> None:
        namespace = {"__name__": "__test__"}
        exec(script_source("hf_download.py"), namespace)
        items = [{"key": "dit"}, {"key": "vae"}, {"key": "text_encoder"}]
        seen: list[str] = []
        barrier = threading.Barrier(len(items))
        namespace["preflight_downloads"] = lambda values: None

        def observe_parallel_start(item):
            seen.append(item["key"])
            barrier.wait(timeout=2)

        namespace["run_one"] = observe_parallel_start

        with patch.object(sys, "argv", ["hf_download.py", json.dumps(items)]):
            namespace["main"]()

        self.assertCountEqual(seen, ["dit", "vae", "text_encoder"])

    def test_hf_download_caps_internal_part_concurrency(self) -> None:
        namespace = {"__name__": "__test__"}
        exec(script_source("hf_download.py"), namespace)
        observed = {}

        class ImmediateFuture:
            def result(self):
                return None

        class RecordingExecutor:
            def __init__(self, max_workers):
                observed["max_workers"] = max_workers

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def submit(self, function, item):
                function(item)
                return ImmediateFuture()

        namespace["ThreadPoolExecutor"] = RecordingExecutor
        namespace["preflight_downloads"] = lambda values: None
        namespace["run_one"] = lambda item: None
        items = [{"key": str(index)} for index in range(7)]

        with patch.object(sys, "argv", ["hf_download.py", json.dumps(items)]):
            namespace["main"]()

        self.assertEqual(observed["max_workers"], 4)

    def test_hf_download_retry_preserves_shared_incomplete_files(self) -> None:
        namespace = {"__name__": "__test__"}
        exec(script_source("hf_download.py"), namespace)

        class FinishedProcess:
            def __init__(self, returncode, output):
                self.returncode = returncode
                self.stdout = io.StringIO(output)

            def poll(self):
                return self.returncode

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            home = root / "hf"
            cache = home / "hub"
            repo = cache / "models--owner--model" / "blobs"
            repo.mkdir(parents=True)
            incomplete = repo / "shared.incomplete"
            incomplete.write_bytes(b"partial")
            target = repo / "complete"
            target.write_bytes(b"complete")
            link = root / "models" / "weights.safetensors"
            item = {"key": "dit", "repo_id": "owner/model", "filename": "weights.safetensors", "link_path": str(link), "_size_bytes": 8}
            mapping = json.dumps([{"container": str(cache), "workspace": "/workspace/cache/huggingface/hub"}])
            processes = [FinishedProcess(1, "temporary failure\n"), FinishedProcess(0, str(target) + "\n")]

            with (
                patch.dict(os.environ, {"HF_HOME": str(home), "HF_HUB_CACHE": str(cache), "KURA_WORKSPACE_PATH_MAPS": mapping}, clear=True),
                patch.object(namespace["subprocess"], "Popen", side_effect=processes),
                patch.object(namespace["time"], "sleep", return_value=None),
            ):
                namespace["stable_link_target"] = lambda path, link_path: path
                namespace["run_one"](item)

            self.assertEqual(incomplete.read_bytes(), b"partial")
            self.assertTrue(link.is_symlink())

    def test_hf_download_preflight_rejects_insufficient_disk_before_download(self) -> None:
        namespace = {"__name__": "__test__"}
        exec(script_source("hf_download.py"), namespace)
        fake_hub = SimpleNamespace(
            try_to_load_from_cache=lambda *args, **kwargs: None,
            hf_hub_url=lambda **kwargs: "https://huggingface.invalid/file",
            get_hf_file_metadata=lambda *args, **kwargs: SimpleNamespace(size=1234),
        )
        item = {
            "key": "dit",
            "repo_id": "owner/model",
            "filename": "weights.safetensors",
            "link_path": "/workspace/cache/models/owner--model/dit/weights.safetensors",
        }
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory) / "hf"
            cache = home / "hub"
            mapping = json.dumps([{"container": str(cache), "workspace": "/workspace/cache/huggingface"}])
            usage = SimpleNamespace(total=1_000, used=900, free=100)
            with (
                patch.dict(os.environ, {"HF_HOME": str(home), "HF_HUB_CACHE": str(cache), "KURA_WORKSPACE_PATH_MAPS": mapping}, clear=True),
                patch.dict(sys.modules, {"huggingface_hub": fake_hub}),
                patch.object(namespace["shutil"], "disk_usage", return_value=usage),
            ):
                namespace["DOWNLOAD_RESERVE_BYTES"] = 10
                with self.assertRaisesRegex(SystemExit, "insufficient disk"):
                    namespace["preflight_downloads"]([item])

    def test_hf_download_preflight_classifies_auth_and_missing_artifact(self) -> None:
        namespace = {"__name__": "__test__"}
        exec(script_source("hf_download.py"), namespace)

        auth = RuntimeError("denied")
        auth.response = SimpleNamespace(status_code=403)  # type: ignore[attr-defined]
        missing = RuntimeError("missing")
        missing.response = SimpleNamespace(status_code=404)  # type: ignore[attr-defined]

        self.assertEqual(namespace["metadata_failure_kind"](auth), "authentication")
        self.assertEqual(namespace["metadata_failure_kind"](missing), "missing-artifact")

    def test_loader_rejects_unknown_script(self) -> None:
        with self.assertRaises(FileNotFoundError):
            script_source("missing.py")


if __name__ == "__main__":
    unittest.main()

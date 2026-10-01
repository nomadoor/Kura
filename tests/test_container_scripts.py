from __future__ import annotations

import ast
import importlib
from dataclasses import replace
import io
import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from kura.container_scripts import script_source
from kura.backends import BACKENDS, BackendSurface
from kura.backends.ai_toolkit import AI_TOOLKIT_VIDEO_SUFFIXES
from kura.backends.musubi_datasets import MUSUBI_AUDIO_SUFFIXES, MUSUBI_IMAGE_SUFFIXES, MUSUBI_VIDEO_SUFFIXES
from kura.init_templates import SD_SCRIPTS_DOCKERFILE_TEMPLATE, SD_SCRIPTS_SYMLINK_PATCH_TEMPLATE
from kura.media_types import frozen_suffixes
import kura.provenance as provenance
from kura.provenance import adapter_source_identity, legacy_adapter_source_identity
from tests.platform_support import POSIX_PATHS, posix_only


MUSUBI_MEDIA_ENV = {
    "KURA_MUSUBI_IMAGE_SUFFIXES": frozen_suffixes(MUSUBI_IMAGE_SUFFIXES),
    "KURA_MUSUBI_VIDEO_SUFFIXES": frozen_suffixes(MUSUBI_VIDEO_SUFFIXES),
    "KURA_MUSUBI_AUDIO_SUFFIXES": frozen_suffixes(MUSUBI_AUDIO_SUFFIXES),
}


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

    def test_adapter_identity_tracks_core_media_registry(self) -> None:
        baseline = {
            name: adapter_source_identity(name)["value"]
            for name in ("ai-toolkit", "musubi-tuner", "sd-scripts")
        }
        original = Path.read_bytes

        def changed_registry(path):
            payload = original(path)
            return payload + (b"\n# changed media registry\n" if path.name == "media_types.py" else b"")

        with patch.object(Path, "read_bytes", changed_registry):
            changed = {name: adapter_source_identity(name)["value"] for name in baseline}

        for name in baseline:
            self.assertNotEqual(baseline[name], changed[name], name)

    def test_adapter_identity_tracks_backend_loader_suffix_sets(self) -> None:
        cases = (
            ("ai-toolkit", "ai_toolkit.py", b"AI_TOOLKIT_VIDEO_SUFFIXES"),
            ("musubi-tuner", "musubi_datasets.py", b"MUSUBI_VIDEO_SUFFIXES"),
            ("sd-scripts", "sd_scripts_datasets.py", b"SD_SCRIPTS_IMAGE_SUFFIXES"),
        )
        original = Path.read_bytes
        for backend, filename, marker in cases:
            with self.subTest(backend=backend):
                baseline = adapter_source_identity(backend)["value"]

                def changed_suffixes(path, *, target=filename, needle=marker):
                    payload = original(path)
                    return payload.replace(needle, needle + b"_CHANGED", 1) if path.name == target else payload

                with patch.object(Path, "read_bytes", changed_suffixes):
                    changed = adapter_source_identity(backend)["value"]
                self.assertNotEqual(baseline, changed)

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

    def test_musubi_adapter_identity_includes_video_frame_preflight(self) -> None:
        baseline = adapter_source_identity("musubi-tuner")["value"]
        original = Path.read_bytes

        def changed_helper(path):
            payload = original(path)
            return payload + (b"changed" if path.name == "musubi_dataset_assert.py" else b"")

        with patch.object(Path, "read_bytes", changed_helper):
            changed = adapter_source_identity("musubi-tuner")["value"]

        self.assertNotEqual(baseline, changed)

    def test_ai_toolkit_adapter_identity_includes_video_audio_preflight(self) -> None:
        baseline = adapter_source_identity("ai-toolkit")["value"]
        original = Path.read_bytes

        def changed_helper(path):
            payload = original(path)
            return payload + (b"changed" if path.name == "ai_toolkit_video_assert.py" else b"")

        with patch.object(Path, "read_bytes", changed_helper):
            changed = adapter_source_identity("ai-toolkit")["value"]

        self.assertNotEqual(baseline, changed)

    def test_ai_toolkit_adapter_identity_includes_dataset_profiles(self) -> None:
        baseline = adapter_source_identity("ai-toolkit")["value"]
        original = Path.read_bytes

        def changed_helper(path):
            payload = original(path)
            return payload + (b"changed" if path.name == "dataset_profiles.py" else b"")

        with patch.object(Path, "read_bytes", changed_helper):
            changed = adapter_source_identity("ai-toolkit")["value"]

        self.assertNotEqual(baseline, changed)

    def test_adapter_identities_cover_imported_backend_dependencies(self) -> None:
        package_root = Path(provenance.__file__).resolve().parent
        backend_root = package_root / "backends"
        container_root = package_root / "container_scripts"
        seeds = {
            "ai-toolkit": [backend_root / "ai_toolkit.py"],
            "musubi-tuner": sorted(backend_root.glob("musubi_*.py")),
            "sd-scripts": sorted(backend_root.glob("sd_scripts*.py")),
        }
        uncovered: list[str] = []

        def top_level_definitions(path: Path) -> dict[str, ast.AST]:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            definitions: dict[str, ast.AST] = {}
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    definitions[node.name] = node
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target, ast.Name):
                            definitions[target.id] = node
            return definitions

        def closure_names(path: Path, root: str) -> set[str]:
            definitions = top_level_definitions(path)
            pending = [root]
            closure: set[str] = set()
            while pending:
                name = pending.pop()
                if name in closure or name not in definitions:
                    continue
                closure.add(name)
                pending.extend(
                    child.id for child in ast.walk(definitions[name])
                    if isinstance(child, ast.Name)
                    and isinstance(child.ctx, ast.Load)
                    and child.id in definitions
                )
            return closure

        def probe_symbol_source(path: Path, symbol: str) -> bytes:
            text = path.read_text(encoding="utf-8")
            node = top_level_definitions(path)[symbol]
            source = ast.get_source_segment(text, node)
            assert source is not None
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                marker = f"def {node.name}"
                changed = source.replace(marker, f"def  {node.name}", 1)
            elif isinstance(node, ast.ClassDef):
                marker = f"class {node.name}"
                changed = source.replace(marker, f"class  {node.name}", 1)
            else:
                separator = source.index("=") + 1
                changed = source[:separator] + " " + source[separator:]
            assert changed != source
            return text.replace(source, changed, 1).encode("utf-8")

        for backend, initial_paths in seeds.items():
            module_paths: set[Path] = set()
            imported_symbols: set[tuple[Path, str]] = set()
            runtime_paths: set[Path] = set()
            pending = list(initial_paths)
            while pending:
                path = pending.pop()
                if path in module_paths:
                    continue
                module_paths.add(path)
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module == "kura.backends.shared":
                        imported_symbols.update((backend_root / "shared.py", alias.name) for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module == "kura.run_envelope":
                        imported_symbols.update((package_root / "run_envelope.py", alias.name) for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("kura.backends."):
                        dependency = backend_root / f"{node.module.rsplit('.', 1)[-1]}.py"
                        if dependency.is_file() and dependency.name not in {"shared.py", "registry.py"}:
                            pending.append(dependency)
                    elif (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "script_source"
                        and node.args
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)
                    ):
                        runtime_paths.add(container_root / node.args[0].value)

            baseline = adapter_source_identity(backend)["value"]
            original_read_bytes = Path.read_bytes
            for dependency in sorted(module_paths | runtime_paths):
                def changed_file(path: Path, *, target: Path = dependency) -> bytes:
                    payload = original_read_bytes(path)
                    return payload + (b"\n# identity coverage probe\n" if path == target else b"")

                with patch.object(Path, "read_bytes", changed_file):
                    if adapter_source_identity(backend)["value"] == baseline:
                        uncovered.append(f"{backend}: file {dependency.relative_to(package_root)}")

            original_source_symbol = provenance._source_symbol
            for dependency, symbol in sorted(imported_symbols, key=lambda item: (str(item[0]), item[1])):
                def changed_symbol(path: Path, name: str, *, target: Path = dependency, target_name: str = symbol) -> bytes:
                    payload = original_source_symbol(path, name)
                    return payload + (b"\n# identity coverage probe\n" if path == target and name == target_name else b"")

                with patch.object(provenance, "_source_symbol", changed_symbol):
                    if adapter_source_identity(backend)["value"] == baseline:
                        uncovered.append(f"{backend}: symbol {dependency.relative_to(package_root)}:{symbol}")
                for closure_symbol in sorted(closure_names(dependency, symbol)):
                    def changed_closure_file(path: Path, *, target: Path = dependency, target_name: str = closure_symbol) -> bytes:
                        return probe_symbol_source(target, target_name) if path == target else original_read_bytes(path)

                    with patch.object(Path, "read_bytes", changed_closure_file):
                        if adapter_source_identity(backend)["value"] == baseline:
                            uncovered.append(
                                f"{backend}: closure {dependency.relative_to(package_root)}:{symbol}->{closure_symbol}"
                            )

        self.assertEqual([], uncovered, "adapter source identity misses imported dependencies:\n" + "\n".join(uncovered))

    def test_sd_scripts_adapter_identity_includes_anima_runtime_publisher(self) -> None:
        baseline = adapter_source_identity("sd-scripts")["value"]
        original = Path.read_bytes

        def changed_helper(path):
            payload = original(path)
            return payload + (b"changed" if path.name == "sd_scripts_publish_anima.py" else b"")

        with patch.object(Path, "read_bytes", changed_helper):
            changed = adapter_source_identity("sd-scripts")["value"]

        self.assertNotEqual(baseline, changed)

    def test_sd_scripts_adapter_identity_includes_training_state_runner(self) -> None:
        baseline = adapter_source_identity("sd-scripts")["value"]
        original = Path.read_bytes

        def changed_helper(path):
            payload = original(path)
            return payload + (b"changed" if path.name == "sd_scripts_state.py" else b"")

        with patch.object(Path, "read_bytes", changed_helper):
            changed = adapter_source_identity("sd-scripts")["value"]

        self.assertNotEqual(baseline, changed)

    def test_container_scripts_compile(self) -> None:
        for name in (
            "hf_download.py",
            "ai_toolkit_video_assert.py",
            "safetensors_validator.py",
            "prune_checkpoints.py",
            "musubi_probe.py",
            "musubi_dataset_assert.py",
            "sd_scripts_publish_anima.py",
        ):
            with self.subTest(name=name):
                compile(script_source(name), name, "exec")

    def test_ai_toolkit_audio_preflight_uses_pinned_file_item_loader(self) -> None:
        namespace = {"__name__": "__test__"}
        exec(script_source("ai_toolkit_video_assert.py"), namespace)
        calls = []

        class FakeDatasetConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeAudio:
            @staticmethod
            def numel():
                return 4

        class FakeFileItem:
            def __init__(self, **kwargs):
                calls.append(("init", kwargs))
                self.audio_tensor = None

            def load_and_process_video(self, transform):
                calls.append(("load", transform))
                self.audio_tensor = FakeAudio()

        class FakeToTensor:
            pass

        class FakeRescale:
            pass

        class FakeCompose:
            def __init__(self, steps):
                self.steps = steps

        config_module = ModuleType("toolkit.config_modules")
        config_module.DatasetConfig = FakeDatasetConfig
        data_module = ModuleType("toolkit.data_transfer_object.data_loader")
        data_module.FileItemDTO = FakeFileItem
        loader_module = ModuleType("toolkit.data_loader")
        loader_module.RescaleTransform = FakeRescale
        torchvision_module = ModuleType("torchvision")
        transforms_module = ModuleType("torchvision.transforms")
        transforms_module.Compose = FakeCompose
        transforms_module.ToTensor = FakeToTensor
        torchvision_module.transforms = transforms_module
        with patch.dict(sys.modules, {
            "toolkit": ModuleType("toolkit"),
            "toolkit.config_modules": config_module,
            "toolkit.data_loader": loader_module,
            "toolkit.data_transfer_object": ModuleType("toolkit.data_transfer_object"),
            "toolkit.data_transfer_object.data_loader": data_module,
            "torchvision": torchvision_module,
            "torchvision.transforms": transforms_module,
        }):
            namespace["_probe_with_pinned_loader"](
                Path("/workspace/view/clip.mp4"),
                {"num_frames": 49, "fps": 24, "do_audio": True},
            )

        # The pinned loader cannot stack frames without its dataset's tensor
        # transform (observed in a real container: "expected Tensor ... got Image").
        transform = calls[1][1]
        self.assertIsInstance(transform, FakeCompose)
        self.assertEqual([type(step) for step in transform.steps], [FakeToTensor, FakeRescale])
        self.assertEqual(calls[0][1]["scale_to_width"], 64)

    def test_ai_toolkit_audio_preflight_records_and_rejects_missing_audio(self) -> None:
        namespace = {"__name__": "__test__"}
        exec(script_source("ai_toolkit_video_assert.py"), namespace)

        class FakeDatasetConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeFileItem:
            def __init__(self, **_kwargs):
                self.audio_tensor = None

            def load_and_process_video(self, _transform):
                self.audio_tensor = None

        config_module = ModuleType("toolkit.config_modules")
        config_module.DatasetConfig = FakeDatasetConfig
        data_module = ModuleType("toolkit.data_transfer_object.data_loader")
        data_module.FileItemDTO = FakeFileItem
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            view = workspace / "runs" / "video" / "cache" / "dataset-view" / "ai-toolkit" / "tiny"
            view.mkdir(parents=True)
            (view / "clip.mp4").write_bytes(b"video")
            config = workspace / "runs" / "video" / "resolved" / "ai-toolkit.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "config:\n  process:\n    - datasets:\n"
                f"        - folder_path: {view}\n"
                "          num_frames: 49\n          fps: 24\n          do_audio: true\n",
                encoding="utf-8",
            )
            resolved = workspace / "runs" / "video" / "resolved"
            (resolved / "dataset-input.lock.json").write_text(json.dumps({
                "semantic": {"datasets": [{"dataset": "clips", "samples": [
                    {"id": "clip-sample"},
                ]}]},
                "views": [{"links": [{
                    "path": str((view / "clip.mp4").relative_to(workspace)),
                    "target": "/workspace/datasets/clips/clip.mp4",
                    "input_id": "opaque-input",
                    "dataset": "clips",
                    "sample": "clip-sample",
                }]}],
            }), encoding="utf-8")
            environment = {
                "KURA_WORKSPACE": str(workspace),
                "KURA_RUN_ID": "video",
                "KURA_REALIZATION_ID": "r1",
                "KURA_AI_TOOLKIT_VIDEO_SUFFIXES": frozen_suffixes(AI_TOOLKIT_VIDEO_SUFFIXES),
            }
            loader_module = ModuleType("toolkit.data_loader")
            loader_module.RescaleTransform = object
            transforms_module = ModuleType("torchvision.transforms")
            transforms_module.Compose = lambda steps: steps
            transforms_module.ToTensor = object
            torchvision_module = ModuleType("torchvision")
            torchvision_module.transforms = transforms_module
            with patch.dict(sys.modules, {
                "toolkit": ModuleType("toolkit"),
                "toolkit.config_modules": config_module,
                "toolkit.data_loader": loader_module,
                "toolkit.data_transfer_object": ModuleType("toolkit.data_transfer_object"),
                "toolkit.data_transfer_object.data_loader": data_module,
                "torchvision": torchvision_module,
                "torchvision.transforms": transforms_module,
            }), patch.dict(os.environ, environment), patch.object(
                sys, "argv", ["ai_toolkit_video_assert.py", str(config)],
            ):
                with self.assertRaisesRegex(SystemExit, "produced no usable audio tensor"):
                    namespace["main"]()

            record = json.loads(
                (workspace / "runs" / "video" / "realizations" / "r1.ai-toolkit-video-preflight.json")
                .read_text(encoding="utf-8")
            )
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["videos"][0]["status"], "unusable")
        self.assertEqual(record["videos"][0]["sample_id"], "clip-sample")
        self.assertEqual(record["videos"][0]["source"], "/workspace/datasets/clips/clip.mp4")

    def test_musubi_dataset_assert_counts_video_inputs(self) -> None:
        namespace = {"__name__": "__test__"}
        exec(script_source("musubi_dataset_assert.py"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.mp4").write_bytes(b"video")
            (root / "caption.txt").write_text("caption", encoding="utf-8")

            count = namespace["media_count"](
                root,
                MUSUBI_VIDEO_SUFFIXES,
                "video_directory",
            )

        self.assertEqual(count, 1)

    def test_musubi_video_preflight_lists_all_short_videos_and_records_measurements(self) -> None:
        namespace = {"__name__": "__test__"}
        exec(script_source("musubi_dataset_assert.py"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            view = workspace / "runs" / "video-run" / "cache" / "dataset-view" / "musubi" / "tiny"
            view.mkdir(parents=True)
            first = view / "first.mp4"
            second = view / "second.mp4"
            source = workspace / "datasets" / "clips"
            source.mkdir(parents=True)
            first_source = source / "one.mp4"
            second_source = source / "two.mp4"
            first_source.write_bytes(b"first")
            second_source.write_bytes(b"second")
            first.symlink_to(first_source)
            second.symlink_to(second_source)
            config = workspace / "dataset.toml"
            config.write_text(
                '[[datasets]]\nvideo_directory = "' + view.as_posix() + '"\n'
                'target_frames = [1, 25]\nsource_fps = 30.0\n',
                encoding="utf-8",
            )
            counts = {str(first): 12, str(second): 8}
            resolved = workspace / "runs" / "video-run" / "resolved"
            resolved.mkdir(parents=True)
            (resolved / "dataset-input.lock.json").write_text(json.dumps({
                "semantic": {"datasets": [{"dataset": "clips", "samples": [
                    {"id": "sample-one"}, {"id": "sample-two"},
                ]}]},
                "views": [{"links": [
                    {"path": str(first.relative_to(workspace)), "target": "/workspace/datasets/clips/one.mp4", "input_id": "opaque-one", "dataset": "clips", "sample": "sample-one"},
                    {"path": str(second.relative_to(workspace)), "target": "/workspace/datasets/clips/two.mp4", "input_id": "opaque-two", "dataset": "clips", "sample": "sample-two"},
                ]}],
            }), encoding="utf-8")
            media_utils = ModuleType("musubi_tuner.dataset.media_utils")
            calls: list[tuple[str, int, float | None, float | None]] = []

            def fake_load_video(path, start_frame, end_frame, **kwargs):
                calls.append((path, end_frame, kwargs.get("source_fps"), kwargs.get("target_fps")))
                return [object()] * counts[path]

            media_utils.load_video = fake_load_video  # type: ignore[attr-defined]
            modules = {
                "musubi_tuner": ModuleType("musubi_tuner"),
                "musubi_tuner.dataset": ModuleType("musubi_tuner.dataset"),
                "musubi_tuner.dataset.media_utils": media_utils,
            }
            env = {
                **MUSUBI_MEDIA_ENV,
                "KURA_WORKSPACE": str(workspace),
                "KURA_RUN_ID": "video-run",
                "KURA_REALIZATION_ID": "realization-1",
                "KURA_MUSUBI_ARCHITECTURE": "wan",
                "KURA_MUSUBI_TARGET_FPS": "16.0",
                "KURA_MUSUBI_FPS_RESAMPLE_MODE": "source-fps-when-declared",
                "KURA_MUSUBI_PROFILES": "wan-video",
            }
            with (
                patch.dict(sys.modules, modules),
                patch.dict(os.environ, env, clear=True),
                patch.object(sys, "argv", ["musubi_dataset_assert.py", str(config)]),
                self.assertRaises(SystemExit) as raised,
            ):
                namespace["main"]()

            message = str(raised.exception)
            self.assertIn("first.mp4: 12 converted frames", message)
            self.assertIn("second.mp4: 8 converted frames", message)
            self.assertIn("sample sample-one", message)
            self.assertIn("/workspace/datasets/clips/one.mp4", message)
            self.assertEqual(calls, [(str(first), 25, 30.0, 16.0), (str(second), 25, 30.0, 16.0)])
            record = json.loads(
                (workspace / "runs" / "video-run" / "realizations" / "realization-1.musubi-video-preflight.json").read_text(encoding="utf-8")
            )
            self.assertEqual(record["status"], "failed")
            self.assertEqual([item["effective_frames"] for item in record["videos"]], [12, 8])
            self.assertEqual(record["videos"][0]["sample_id"], "sample-one")
            self.assertEqual(record["videos"][0]["source"], "/workspace/datasets/clips/one.mp4")
            self.assertEqual(record["architecture"], "wan")
            self.assertEqual(record["target_fps"], 16.0)

    def test_musubi_video_preflight_records_success_without_resampling_when_source_fps_is_absent(self) -> None:
        namespace = {"__name__": "__test__"}
        exec(script_source("musubi_dataset_assert.py"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            view = workspace / "videos"
            view.mkdir()
            video = view / "enough.mp4"
            video.write_bytes(b"video")
            config = workspace / "dataset.toml"
            config.write_text(
                '[[datasets]]\nvideo_directory = "' + view.as_posix() + '"\n'
                'target_frames = [1, 25]\n',
                encoding="utf-8",
            )
            media_utils = ModuleType("musubi_tuner.dataset.media_utils")
            calls = []

            def fake_load_video(path, start_frame, end_frame, **kwargs):
                calls.append((path, start_frame, end_frame, kwargs))
                return [object()] * 25

            media_utils.load_video = fake_load_video  # type: ignore[attr-defined]
            modules = {
                "musubi_tuner": ModuleType("musubi_tuner"),
                "musubi_tuner.dataset": ModuleType("musubi_tuner.dataset"),
                "musubi_tuner.dataset.media_utils": media_utils,
            }
            env = {
                **MUSUBI_MEDIA_ENV,
                "KURA_WORKSPACE": str(workspace),
                "KURA_RUN_ID": "video-run",
                "KURA_REALIZATION_ID": "realization-2",
                "KURA_MUSUBI_ARCHITECTURE": "wan",
                "KURA_MUSUBI_TARGET_FPS": "16.0",
                "KURA_MUSUBI_FPS_RESAMPLE_MODE": "source-fps-when-declared",
                "KURA_MUSUBI_PROFILES": "wan-video",
            }
            output = io.StringIO()
            with (
                patch.dict(sys.modules, modules),
                patch.dict(os.environ, env, clear=True),
                patch.object(sys, "argv", ["musubi_dataset_assert.py", str(config)]),
                patch("sys.stdout", output),
            ):
                namespace["main"]()

            self.assertEqual(calls[0][1:3], (0, 25))
            self.assertIsNone(calls[0][3]["source_fps"])
            self.assertIsNone(calls[0][3]["target_fps"])
            self.assertEqual(calls[0][3]["bucket_reso"], (64, 64))
            record = json.loads(
                (workspace / "runs" / "video-run" / "realizations" / "realization-2.musubi-video-preflight.json").read_text(encoding="utf-8")
            )
            self.assertEqual(record["status"], "passed")
            self.assertEqual(record["videos"][0]["effective_frames"], 25)

    def test_musubi_framepack_full_preflight_rounds_and_rejects_short_video(self) -> None:
        namespace = {"__name__": "__test__"}
        exec(script_source("musubi_dataset_assert.py"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            video = workspace / "short.mp4"
            video.write_bytes(b"video")
            video_jsonl = workspace / "items.jsonl"
            video_jsonl.write_text(
                json.dumps({"video_path": str(video), "caption": "caption"}) + "\n",
                encoding="utf-8",
            )
            config = workspace / "dataset.toml"
            config.write_text(
                '[[datasets]]\nvideo_jsonl_file = "' + video_jsonl.as_posix() + '"\n'
                'target_frames = [37]\nframe_extraction = "full"\nmax_frames = 129\n'
                'fp_latent_window_size = 9\n',
                encoding="utf-8",
            )
            media_utils = ModuleType("musubi_tuner.dataset.media_utils")
            media_utils.load_video = lambda *_args, **_kwargs: [object()] * 36  # type: ignore[attr-defined]
            architectures = ModuleType("musubi_tuner.dataset.architectures")
            observed_architectures = []

            def round_down_frame_count(count, architecture, stride):
                observed_architectures.append(architecture)
                return 1 + ((count - 1) // stride) * stride

            architectures.round_down_frame_count = round_down_frame_count  # type: ignore[attr-defined]
            modules = {
                "musubi_tuner": ModuleType("musubi_tuner"),
                "musubi_tuner.dataset": ModuleType("musubi_tuner.dataset"),
                "musubi_tuner.dataset.media_utils": media_utils,
                "musubi_tuner.dataset.architectures": architectures,
            }
            env = {
                **MUSUBI_MEDIA_ENV,
                "KURA_WORKSPACE": str(workspace),
                "KURA_RUN_ID": "video-run",
                "KURA_REALIZATION_ID": "framepack-full",
                "KURA_MUSUBI_ARCHITECTURE": "framepack",
                "KURA_MUSUBI_NATIVE_DATASET_ARCHITECTURE": "fp",
                "KURA_MUSUBI_TARGET_FPS": "30.0",
                "KURA_MUSUBI_FPS_RESAMPLE_MODE": "source-fps-when-declared",
                "KURA_MUSUBI_PROFILES": "framepack-video",
            }
            with (
                patch.dict(sys.modules, modules),
                patch.dict(os.environ, env, clear=True),
                patch.object(sys, "argv", ["musubi_dataset_assert.py", str(config)]),
                self.assertRaisesRegex(SystemExit, r"(?s)shorter than FramePack's full-window minimum.*33 converted frames"),
            ):
                namespace["main"]()
            self.assertEqual(observed_architectures, ["fp"])

    def test_musubi_h3_video_preflight_uses_the_pinned_timestamp_resampling_path(self) -> None:
        namespace = {"__name__": "__test__"}
        exec(script_source("musubi_dataset_assert.py"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            video = workspace / "clip.mp4"
            video.write_bytes(b"video")
            jsonl = workspace / "items.jsonl"
            jsonl.write_text(json.dumps({"video_path": str(video), "caption": "caption"}) + "\n", encoding="utf-8")
            config = workspace / "dataset.toml"
            config.write_text(
                '[[datasets]]\nvideo_jsonl_file = "' + jsonl.as_posix() + '"\n'
                'target_frames = [124]\n',
                encoding="utf-8",
            )
            media_utils = ModuleType("musubi_tuner.dataset.media_utils")
            calls = []

            def fake_load_video(path, start_frame, end_frame, **kwargs):
                calls.append((path, start_frame, end_frame, kwargs))
                return [object()] * 124

            media_utils.load_video = fake_load_video  # type: ignore[attr-defined]
            modules = {
                "musubi_tuner": ModuleType("musubi_tuner"),
                "musubi_tuner.dataset": ModuleType("musubi_tuner.dataset"),
                "musubi_tuner.dataset.media_utils": media_utils,
            }
            env = {
                **MUSUBI_MEDIA_ENV,
                "KURA_WORKSPACE": str(workspace),
                "KURA_RUN_ID": "video-run",
                "KURA_REALIZATION_ID": "h3-realization",
                "KURA_MUSUBI_ARCHITECTURE": "minimax_h3",
                "KURA_MUSUBI_TARGET_FPS": "24.0",
                "KURA_MUSUBI_FPS_RESAMPLE_MODE": "timestamps",
                "KURA_MUSUBI_PROFILES": "h3-video-t2va",
            }
            with (
                patch.dict(sys.modules, modules),
                patch.dict(os.environ, env, clear=True),
                patch.object(sys, "argv", ["musubi_dataset_assert.py", str(config)]),
            ):
                namespace["main"]()

            self.assertEqual(calls[0][1:3], (0, 124))
            self.assertEqual(calls[0][3]["target_fps"], 24.0)
            self.assertEqual(calls[0][3]["fps_resample_mode"], "timestamps")
            self.assertNotIn("source_fps", calls[0][3])
            self.assertEqual(calls[0][3]["bucket_reso"], (64, 64))

    def test_musubi_audio_sidecar_lookup_scans_each_directory_once(self) -> None:
        namespace = {"__name__": "__test__"}
        exec(script_source("musubi_dataset_assert.py"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(20):
                (root / f"clip{index}.mp4").write_bytes(b"video")
            (root / "clip3.wav").write_bytes(b"audio")
            (root / "clip3.txt").write_text("caption", encoding="utf-8")
            lookup = namespace["audio_sidecar_index"](frozenset({".wav"}))
            original = Path.iterdir
            scans: list[Path] = []

            def counting_iterdir(path: Path):
                scans.append(path)
                return original(path)

            with patch.object(Path, "iterdir", counting_iterdir):
                found = {index: lookup(root / f"clip{index}.mp4") for index in range(20)}

            self.assertEqual(scans, [root])
            self.assertEqual(found[3], [root / "clip3.wav"])
            self.assertTrue(all(found[index] == [] for index in found if index != 3))

    def test_musubi_h3_preflight_rejects_an_implicit_sidecar_beside_the_symlink_target(self) -> None:
        namespace = {"__name__": "__test__"}
        exec(script_source("musubi_dataset_assert.py"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "datasets" / "clips"
            source.mkdir(parents=True)
            source_video = source / "clip.mp4"
            source_video.write_bytes(b"video")
            (source / "clip.wav").write_bytes(b"audio")
            view = workspace / "view"
            view.mkdir()
            view_video = view / "000000-hash.mp4"
            view_video.symlink_to(source_video)
            jsonl = view / "items.jsonl"
            jsonl.write_text(json.dumps({"video_path": str(view_video), "caption": "caption"}) + "\n", encoding="utf-8")
            config = workspace / "dataset.toml"
            config.write_text(
                '[[datasets]]\nvideo_jsonl_file = "' + jsonl.as_posix() + '"\n'
                'target_frames = [124]\n',
                encoding="utf-8",
            )
            media_utils = ModuleType("musubi_tuner.dataset.media_utils")
            media_utils.load_video = lambda *_args, **_kwargs: [object()] * 124  # type: ignore[attr-defined]
            modules = {
                "musubi_tuner": ModuleType("musubi_tuner"),
                "musubi_tuner.dataset": ModuleType("musubi_tuner.dataset"),
                "musubi_tuner.dataset.media_utils": media_utils,
            }
            env = {
                **MUSUBI_MEDIA_ENV,
                "KURA_WORKSPACE": str(workspace),
                "KURA_RUN_ID": "video-run",
                "KURA_REALIZATION_ID": "h3-sidecar",
                "KURA_MUSUBI_ARCHITECTURE": "minimax_h3",
                "KURA_MUSUBI_TARGET_FPS": "24.0",
                "KURA_MUSUBI_FPS_RESAMPLE_MODE": "timestamps",
                "KURA_MUSUBI_PROFILES": "h3-video-t2va",
            }
            with (
                patch.dict(sys.modules, modules),
                patch.dict(os.environ, env, clear=True),
                patch.object(sys, "argv", ["musubi_dataset_assert.py", str(config)]),
                self.assertRaisesRegex(SystemExit, "implicit same-stem audio sidecar.*clip.wav"),
            ):
                namespace["main"]()

            record = json.loads(
                (workspace / "runs" / "video-run" / "realizations" / "h3-sidecar.musubi-video-preflight.json").read_text(encoding="utf-8")
            )
            self.assertEqual(record["status"], "failed")
            self.assertIn("clip.wav", record["errors"][0]["error"])

    def test_musubi_dataset_assert_counts_symlinked_image_view(self) -> None:
        namespace = {"__name__": "__test__"}
        exec(script_source("musubi_dataset_assert.py"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            source.write_bytes(b"image")
            view = root / "view"
            view.mkdir()
            (view / "000000-deadbeef.png").symlink_to(source)
            config = root / "dataset.toml"
            config.write_text(
                '[[datasets]]\nimage_directory = "' + view.as_posix() + '"\n',
                encoding="utf-8",
            )
            output = io.StringIO()

            with patch.dict(os.environ, MUSUBI_MEDIA_ENV, clear=True), patch.object(
                sys, "argv", ["musubi_dataset_assert.py", str(config)],
            ), patch("sys.stdout", output):
                namespace["main"]()

        self.assertIn('"images": 1', output.getvalue())

    def test_musubi_dataset_assert_dispatches_video_jsonl_and_defers_unknown_sources(self) -> None:
        namespace = {"__name__": "__test__"}
        exec(script_source("musubi_dataset_assert.py"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_jsonl = root / "videos.jsonl"
            video = root / "selected.mp4"
            video.write_bytes(b"video")
            video_jsonl.write_text(json.dumps({"video_path": str(video), "caption": "caption"}) + "\n", encoding="utf-8")
            config = root / "dataset.toml"
            config.write_text(
                '[[datasets]]\nvideo_jsonl_file = "' + video_jsonl.as_posix() + '"\n'
                'target_frames = [1, 25]\n'
                '[[datasets]]\nfuture_native_source = "opaque"\n',
                encoding="utf-8",
            )
            media_utils = ModuleType("musubi_tuner.dataset.media_utils")
            calls = []

            def fake_load_video(path, start_frame, end_frame, **kwargs):
                calls.append(path)
                return [object()] * 25

            media_utils.load_video = fake_load_video  # type: ignore[attr-defined]
            modules = {
                "musubi_tuner": ModuleType("musubi_tuner"),
                "musubi_tuner.dataset": ModuleType("musubi_tuner.dataset"),
                "musubi_tuner.dataset.media_utils": media_utils,
            }
            env = {
                **MUSUBI_MEDIA_ENV,
                "KURA_WORKSPACE": str(root),
                "KURA_RUN_ID": "video-run",
                "KURA_REALIZATION_ID": "jsonl-realization",
                "KURA_MUSUBI_ARCHITECTURE": "wan",
                "KURA_MUSUBI_TARGET_FPS": "16.0",
                "KURA_MUSUBI_FPS_RESAMPLE_MODE": "source-fps-when-declared",
                "KURA_MUSUBI_PROFILES": "wan-video",
            }
            with patch.dict(sys.modules, modules), patch.dict(os.environ, env, clear=True), patch.object(
                sys, "argv", ["musubi_dataset_assert.py", str(config)],
            ):
                namespace["main"]()
            self.assertEqual(calls, [str(video)])

    def test_musubi_fun_control_preflight_measures_the_jsonl_control_before_training(self) -> None:
        namespace = {"__name__": "__test__"}
        exec(script_source("musubi_dataset_assert.py"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "datasets" / "clips"
            source.mkdir(parents=True)
            target_source = source / "target.mp4"
            control_source = source / "control.mp4"
            target_source.write_bytes(b"target")
            control_source.write_bytes(b"control")
            view = workspace / "view"
            view.mkdir()
            target = view / "target.mp4"
            control = view / "control.mp4"
            target.symlink_to(target_source)
            control.symlink_to(control_source)
            jsonl = view / "items.jsonl"
            jsonl.write_text(json.dumps({
                "video_path": str(target),
                "control_path": str(control),
                "caption": "caption",
            }) + "\n", encoding="utf-8")
            config = workspace / "dataset.toml"
            config.write_text(
                '[[datasets]]\nvideo_jsonl_file = "' + jsonl.as_posix() + '"\n'
                'target_frames = [1, 25]\n',
                encoding="utf-8",
            )
            resolved = workspace / "runs" / "video-run" / "resolved"
            resolved.mkdir(parents=True)
            (resolved / "dataset-input.lock.json").write_text(json.dumps({
                "semantic": {"datasets": [{"dataset": "clips", "samples": [{"id": "pair"}]}]},
                "views": [{"links": [
                    {
                        "path": str(target.relative_to(workspace)),
                        "target": "/workspace/datasets/clips/target.mp4",
                        "input_id": "opaque-target",
                        "dataset": "clips",
                        "sample": "pair",
                    },
                    {
                        "path": str(control.relative_to(workspace)),
                        "target": "/workspace/datasets/clips/control.mp4",
                        "input_id": "opaque-control",
                        "dataset": "clips",
                        "sample": "pair",
                    },
                ]}],
            }), encoding="utf-8")
            media_utils = ModuleType("musubi_tuner.dataset.media_utils")
            calls = []

            def fake_load_video(path, start_frame, end_frame, **kwargs):
                calls.append((path, start_frame, end_frame, kwargs))
                return [object()] * (25 if path == str(target) else 12)

            media_utils.load_video = fake_load_video  # type: ignore[attr-defined]
            modules = {
                "musubi_tuner": ModuleType("musubi_tuner"),
                "musubi_tuner.dataset": ModuleType("musubi_tuner.dataset"),
                "musubi_tuner.dataset.media_utils": media_utils,
            }
            env = {
                **MUSUBI_MEDIA_ENV,
                "KURA_WORKSPACE": str(workspace),
                "KURA_RUN_ID": "video-run",
                "KURA_REALIZATION_ID": "fun-control",
                "KURA_MUSUBI_ARCHITECTURE": "wan",
                "KURA_MUSUBI_TARGET_FPS": "16.0",
                "KURA_MUSUBI_FPS_RESAMPLE_MODE": "source-fps-when-declared",
                "KURA_MUSUBI_PROFILES": "wan-fun-control-video",
            }
            with (
                patch.dict(sys.modules, modules),
                patch.dict(os.environ, env, clear=True),
                patch.object(sys, "argv", ["musubi_dataset_assert.py", str(config)]),
            ):
                namespace["main"]()

            self.assertEqual([call[0] for call in calls], [str(target), str(control)])
            record = json.loads(
                (workspace / "runs" / "video-run" / "realizations" / "fun-control.musubi-video-preflight.json").read_text(encoding="utf-8")
            )
            self.assertEqual(record["status"], "passed")
            self.assertEqual(record["videos"][0]["control"]["video"], str(control))
            self.assertEqual(record["videos"][0]["control"]["source"], "/workspace/datasets/clips/control.mp4")
            self.assertEqual(record["videos"][0]["control"]["sample_id"], "pair")
            self.assertEqual(record["videos"][0]["control"]["loaded_frames"], 12)
            self.assertTrue(record["videos"][0]["control"]["passed"])

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

    @posix_only(POSIX_PATHS)
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

    def test_hf_download_notices_a_finished_child_without_a_full_poll_interval(self) -> None:
        namespace = {"__name__": "__test__"}
        exec(script_source("hf_download.py"), namespace)

        class CachedHit:
            returncode = 0

            def __init__(self, output):
                self.stdout = io.StringIO(output)
                self.polls = 0

            def poll(self):
                self.polls += 1
                return None if self.polls == 1 else 0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            home = root / "hf"
            cache = home / "hub"
            target = cache / "models--owner--model" / "blobs" / "complete"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"complete")
            item = {"key": "dit", "repo_id": "owner/model", "filename": "weights.safetensors", "link_path": str(root / "models" / "weights.safetensors"), "_size_bytes": 8}
            mapping = json.dumps([{"container": str(cache), "workspace": "/workspace/cache/huggingface/hub"}])
            slept: list[float] = []
            with (
                patch.dict(os.environ, {"HF_HOME": str(home), "HF_HUB_CACHE": str(cache), "KURA_WORKSPACE_PATH_MAPS": mapping, "KURA_HF_DOWNLOAD_POLL_SEC": "15"}, clear=True),
                patch.object(namespace["subprocess"], "Popen", return_value=CachedHit(str(target) + "\n")),
                patch.object(namespace["time"], "sleep", side_effect=slept.append),
            ):
                namespace["stable_link_target"] = lambda path, link_path: path
                namespace["run_one"](item)

        self.assertLess(sum(slept), 1.0)

    def test_hf_download_hardlink_mode_preserves_named_safetensors_path(self) -> None:
        namespace = {"__name__": "__test__"}
        exec(script_source("hf_download.py"), namespace)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blob = root / "cache" / "blobs" / "extensionless"
            blob.parent.mkdir(parents=True)
            blob.write_bytes(b"weights")
            snapshot = root / "cache" / "snapshots" / "model.safetensors"
            snapshot.parent.mkdir(parents=True)
            snapshot.symlink_to(blob)
            published = root / "models" / "model.safetensors"
            published.parent.mkdir(parents=True)

            namespace["publish_download"](
                str(snapshot),
                str(published),
                {"link_mode": "hardlink"},
            )

            self.assertFalse(published.is_symlink())
            self.assertTrue(published.samefile(blob))
            self.assertEqual(published.resolve().suffix, ".safetensors")

    @posix_only(POSIX_PATHS)
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

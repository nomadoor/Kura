from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kura.container_scripts import script_source


class ContainerScriptTests(unittest.TestCase):
    def test_container_scripts_compile(self) -> None:
        for name in (
            "hf_download.py",
            "safetensors_validator.py",
            "prune_checkpoints.py",
            "musubi_probe.py",
        ):
            with self.subTest(name=name):
                compile(script_source(name), name, "exec")

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
            cache = Path(directory) / "hf"
            mapping = json.dumps([{"container": str(cache), "workspace": "/workspace/cache/huggingface"}])
            usage = SimpleNamespace(total=10_000, used=1_000, free=9_000)
            with (
                patch.dict(os.environ, {"HF_HOME": str(cache), "KURA_WORKSPACE_PATH_MAPS": mapping}, clear=True),
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
            cache = Path(directory) / "hf"
            mapping = json.dumps([{"container": str(cache), "workspace": "/workspace/cache/huggingface"}])
            usage = SimpleNamespace(total=1_000, used=900, free=100)
            with (
                patch.dict(os.environ, {"HF_HOME": str(cache), "KURA_WORKSPACE_PATH_MAPS": mapping}, clear=True),
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

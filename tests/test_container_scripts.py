from __future__ import annotations

import importlib
import unittest

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

    def test_loader_rejects_unknown_script(self) -> None:
        with self.assertRaises(FileNotFoundError):
            script_source("missing.py")


if __name__ == "__main__":
    unittest.main()

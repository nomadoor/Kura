"""Static executor/container environment contract tests."""

from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from kura.backends import command_musubi_tuner
from kura.executors.docker import docker_command
from kura.executors.runpod import _runpod_session_env, _runpod_training_env
from kura.run_commands.runpod_ssh import _runpod_remote_job_script


ROOT = Path(__file__).resolve().parents[1]
CONTAINER_SCRIPT_PATHS = sorted((ROOT / "src" / "kura" / "container_scripts").glob("*.py"))
COMFYUI_PREPARE_PATH = ROOT / "docker" / "comfyui" / "kura_comfy_prepare.py"
SECRET_OPTIONAL = {"HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "KURA_REMOTE_NOTIFY_NTFY"}
DEFAULTED_OPTIONAL = {"COMFYUI_ROOT"}
RETRY_OPTIONAL = {"KURA_HF_DOWNLOAD_ATTEMPTS", "KURA_HF_DOWNLOAD_POLL_SEC", "KURA_HF_DOWNLOAD_NO_PROGRESS_SEC"}


def _literal_env_name(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _is_os_environ(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "environ"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def consumed_env_names(paths: list[Path]) -> set[str]:
    names: set[str] = set()
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and _is_os_environ(node.value):
                name = _literal_env_name(node.slice)
                if name:
                    names.add(name)
                continue
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "get" and _is_os_environ(func.value) and node.args:
                name = _literal_env_name(node.args[0])
                if name:
                    names.add(name)
            if isinstance(func, ast.Name) and func.id == "env_int" and node.args:
                name = _literal_env_name(node.args[0])
                if name:
                    names.add(name)
    return names


def required_env_names(paths: list[Path]) -> set[str]:
    return {
        name
        for name in consumed_env_names(paths)
        if name not in DEFAULTED_OPTIONAL
        and name not in RETRY_OPTIONAL
        and name not in SECRET_OPTIONAL
        and not name.startswith("KURA_NTFY_")
    }


def _posix_prefix(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix.rstrip("/") + "/")


def _hf_home_has_workspace_mapping(hf_home: str, mappings: list[dict[str, str]]) -> bool:
    if _posix_prefix(hf_home, "/workspace"):
        return True
    return any(_posix_prefix(hf_home, item["container"]) for item in mappings)


def _minimal_flux2_run() -> dict[str, Any]:
    return {
        "id": "contract-run",
        "model": {"base": "black-forest-labs/FLUX.2-klein-base-4B"},
        "params": {"steps": 1},
        "backend_overrides": {
            "musubi-tuner": {
                "architecture": "flux2",
                "model_version": "klein-base-4b",
                "model_downloads": {
                    "dit": {"repo": "repo/dit", "filename": "dit.safetensors"},
                    "vae": {"repo": "repo/vae", "filename": "vae.safetensors"},
                    "text_encoder": {"repo": "repo/text", "filename": "text.safetensors"},
                },
                "precache": False,
                "validate_models": False,
            }
        },
    }


class LaunchEnvironmentContractTests(unittest.TestCase):
    def test_container_env_inventory_is_derived_from_sources(self) -> None:
        self.assertEqual(required_env_names(CONTAINER_SCRIPT_PATHS), {"HF_HOME", "KURA_WORKSPACE_PATH_MAPS"})
        self.assertEqual(required_env_names([COMFYUI_PREPARE_PATH]), {"HF_HOME", "KURA_WORKSPACE"})

    def test_local_docker_env_satisfies_container_script_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "cache" / "huggingface").mkdir(parents=True)
            run_dir = workspace / "runs" / "contract-run"
            mounts = [{"source": "./cache/huggingface", "target": "/root/.cache/huggingface"}]
            _, runtime_env, _ = docker_command(
                workspace,
                run_dir,
                {"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}},
                "example:image",
                mounts,
                True,
                "r1",
            )
        self.assertTrue(required_env_names(CONTAINER_SCRIPT_PATHS) <= set(runtime_env))
        self.assertIn("KURA_LOG_PATH", runtime_env)
        mappings = json.loads(runtime_env["KURA_WORKSPACE_PATH_MAPS"])
        self.assertTrue(_hf_home_has_workspace_mapping(runtime_env["HF_HOME"], mappings))

    def test_runpod_pod_env_satisfies_training_and_session_contracts(self) -> None:
        training_env = _runpod_training_env({}, workspace_path="/workspace", run_id="contract-run")
        self.assertEqual(training_env["HF_HOME"], "/workspace/cache/huggingface")
        self.assertEqual(training_env["KURA_WORKSPACE"], "/workspace")
        self.assertEqual(training_env["KURA_RUN_ID"], "contract-run")
        self.assertIn("KURA_LOG_PATH", training_env)
        self.assertTrue(_posix_prefix(training_env["HF_HOME"], "/workspace"))

        session_env = _runpod_session_env(workspace_path="/workspace", run_id="contract-run")
        self.assertEqual(session_env["HF_HOME"], "/workspace/cache/huggingface")
        self.assertEqual(session_env["KURA_WORKSPACE"], "/workspace")
        self.assertEqual(session_env["KURA_RUN_ID"], "contract-run")
        self.assertIn("KURA_MAX_LEASE_SEC", session_env)
        self.assertTrue(required_env_names([COMFYUI_PREPARE_PATH]) <= set(session_env))

    def test_runpod_ssh_remote_job_exports_cache_contract_before_work(self) -> None:
        script = _runpod_remote_job_script(
            workspace="/workspace",
            run_id="contract-run",
            remote_secret_path="/tmp/kura-secrets/contract-run.env",
            archive_name="bundle.tar.gz",
            remote_archive="/workspace/bundle.tar.gz",
            cwd="/opt/musubi",
            command="python -c hf_download.py",
        )
        lines = script.splitlines()

        def line_index(needle: str) -> int:
            for index, line in enumerate(lines):
                if needle in line:
                    return index
            self.fail(f"missing line containing {needle!r}")

        export_hf = line_index('export HF_HOME="$KURA_WORKSPACE/cache/huggingface"')
        mkdir_hf = line_index('mkdir -p "$HF_HOME" "$KURA_WORKSPACE/cache/models"')
        contract_check = line_index("HF_HOME must be under KURA_WORKSPACE before remote job start")
        unpack = line_index("tar -xzf")
        backend_command = line_index("python -c hf_download.py")
        self.assertLess(export_hf, mkdir_hf)
        self.assertLess(mkdir_hf, contract_check)
        self.assertLess(contract_check, unpack)
        self.assertLess(contract_check, backend_command)

    def test_musubi_container_command_asserts_dataset_before_download(self) -> None:
        script = command_musubi_tuner(_minimal_flux2_run())["argv"][2]
        self.assertLess(script.index("musubi_dataset_assert.py"), script.index("hf_hub_download"))


if __name__ == "__main__":
    unittest.main()

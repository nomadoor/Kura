"""Static executor/container environment contract tests."""

from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from kura.backends import command_musubi_tuner
from kura.backends.musubi_datasets import MUSUBI_PROJECTION_PROFILES
from kura.backends.ai_toolkit import command_ai_toolkit
from kura.executors.docker import docker_command
from kura.executors.runpod import _runpod_session_env, _runpod_training_env
from kura.run_commands.common import _load_frozen_command
from kura.run_commands.runpod_ssh import _runpod_remote_job_script


ROOT = Path(__file__).resolve().parents[1]
CONTAINER_SCRIPT_PATHS = sorted((ROOT / "src" / "kura" / "container_scripts").glob("*.py"))
COMFYUI_PREPARE_PATH = ROOT / "docker" / "comfyui" / "kura_comfy_prepare.py"
SECRET_OPTIONAL = {"HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "KURA_REMOTE_NOTIFY_NTFY"}
DEFAULTED_OPTIONAL = {"COMFYUI_ROOT", "SD_SCRIPTS_ROOT"}
RETRY_OPTIONAL = {"KURA_HF_DOWNLOAD_ATTEMPTS", "KURA_HF_DOWNLOAD_POLL_SEC", "KURA_HF_DOWNLOAD_NO_PROGRESS_SEC"}
BACKEND_SCOPED = {
    "KURA_MUSUBI_ARCHITECTURE",
    "KURA_MUSUBI_TARGET_FPS",
    "KURA_MUSUBI_FPS_RESAMPLE_MODE",
    "KURA_MUSUBI_PROFILES",
}


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
        and name not in BACKEND_SCOPED
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
        "recipe": {"steps": 1, "seed": 1},
        "backend": {"name": "musubi-tuner", "config": {
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
    def test_ai_toolkit_audio_preflight_runs_before_the_trainer(self) -> None:
        spec = command_ai_toolkit({
            "id": "audio-video",
            "backend": {"name": "ai-toolkit", "config": {
                "model_arch": "ltx2.5",
                "dataset_config": {
                    "num_frames": 49, "fps": 24, "do_audio": True,
                },
            }},
            "model": {"base": "example/model"},
            "recipe": {"steps": 1, "seed": 1},
        })

        script = " ".join(spec["argv"])
        self.assertLess(
            script.index("AI-Toolkit embedded-audio preflight"),
            script.index("ai_toolkit_state"),
        )

    def test_ai_toolkit_declares_its_backend_managed_model_write_root(self) -> None:
        spec = command_ai_toolkit({
            "id": "contract-run",
            "backend": {"name": "ai-toolkit", "config": {}},
            "model": {"base": "example/model"},
            "recipe": {"steps": 1, "seed": 1},
        })
        self.assertEqual(spec["env"]["MODELS_PATH"], "/workspace/cache/ai-toolkit/models")
        self.assertEqual(spec["write_roots"], [{
            "role": "model-cache",
            "path": "/workspace/cache/ai-toolkit/models",
            "env": "MODELS_PATH",
        }])

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run_dir = workspace / "runs" / "contract-run"
            run_dir.mkdir(parents=True)
            docker_argv, _, _ = docker_command(workspace, run_dir, spec, "example:image", [], True, "r1")
        wrapper = docker_argv[docker_argv.index("kura-job") - 1]
        self.assertIn('"/workspace/cache/ai-toolkit/models"', wrapper)
        self.assertIn('test -w "/workspace/cache/ai-toolkit/models"', wrapper)

        remote = _runpod_remote_job_script(
            workspace="/workspace",
            run_id="contract-run",
            realization_id="r1",
            remote_secret_path="/tmp/contract.env",
            archive_name="contract.tar.gz",
            remote_archive="/workspace/contract.tar.gz",
            cwd="/app/ai-toolkit",
            command="python run.py config.yaml",
            write_roots=spec["write_roots"],
        )
        self.assertIn('mkdir -p "/workspace/cache/ai-toolkit/models"', remote)
        self.assertIn('test -w "/workspace/cache/ai-toolkit/models"', remote)

    def test_musubi_builtin_command_requires_a_trained_adapter_output(self) -> None:
        spec = command_musubi_tuner(_minimal_flux2_run())
        self.assertEqual(spec["output_contract"], {
            "required": [{"role": "trained-adapter", "suffix": ".safetensors", "minimum": 1}],
        })

    def test_ai_toolkit_explicit_command_keeps_model_cache_managed(self) -> None:
        run = {
            "id": "contract-run", "backend": {"name": "ai-toolkit", "config": {
                "command": {"cwd": "/app/ai-toolkit", "argv": ["python", "run.py"], "env": {}},
            }},
        }
        spec = command_ai_toolkit(run)
        self.assertEqual(spec["env"]["MODELS_PATH"], "/workspace/cache/ai-toolkit/models")
        self.assertEqual(spec["write_roots"][0]["role"], "model-cache")
        run["backend"]["config"]["command"]["env"]["MODELS_PATH"] = "/app/ai-toolkit/models"
        with self.assertRaisesRegex(ValueError, "MODELS_PATH"):
            command_ai_toolkit(run)

    def test_launch_requires_a_frozen_backend_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "recompile the run"):
                _load_frozen_command(
                    Path(directory),
                    {"backend": {"name": "ai-toolkit"}},
                )

    def test_container_env_inventory_is_derived_from_sources(self) -> None:
        self.assertEqual(required_env_names(CONTAINER_SCRIPT_PATHS), {
            "HF_HOME", "HF_HUB_CACHE", "KURA_REALIZATION_ID", "KURA_RUN_ID", "KURA_WORKSPACE", "KURA_WORKSPACE_PATH_MAPS",
        })
        self.assertEqual(required_env_names([COMFYUI_PREPARE_PATH]), {"HF_HUB_CACHE", "KURA_WORKSPACE"})

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
        self.assertEqual(runtime_env["KURA_REALIZATION_ID"], "r1")
        mappings = json.loads(runtime_env["KURA_WORKSPACE_PATH_MAPS"])
        self.assertTrue(_hf_home_has_workspace_mapping(runtime_env["HF_HOME"], mappings))
        self.assertEqual(runtime_env["HF_HUB_CACHE"], "/workspace/cache/huggingface/hub")

    def test_runpod_pod_env_satisfies_training_and_session_contracts(self) -> None:
        training_env = _runpod_training_env({}, workspace_path="/workspace", run_id="contract-run", realization_id="r1")
        self.assertEqual(training_env["HF_HOME"], "/workspace/cache/huggingface")
        self.assertEqual(training_env["HF_HUB_CACHE"], "/workspace/cache/huggingface/hub")
        self.assertEqual(training_env["KURA_WORKSPACE"], "/workspace")
        self.assertEqual(training_env["KURA_RUN_ID"], "contract-run")
        self.assertEqual(training_env["KURA_REALIZATION_ID"], "r1")
        self.assertIn("KURA_LOG_PATH", training_env)
        self.assertTrue(_posix_prefix(training_env["HF_HOME"], "/workspace"))

        session_env = _runpod_session_env(workspace_path="/workspace", run_id="contract-run")
        self.assertEqual(session_env["HF_HOME"], "/workspace/cache/huggingface")
        self.assertEqual(session_env["HF_HUB_CACHE"], "/workspace/cache/huggingface/hub")
        self.assertEqual(session_env["KURA_WORKSPACE"], "/workspace")
        self.assertEqual(session_env["KURA_RUN_ID"], "contract-run")
        self.assertIn("KURA_MAX_LEASE_SEC", session_env)
        self.assertTrue(required_env_names([COMFYUI_PREPARE_PATH]) <= set(session_env))

    def test_runpod_ssh_remote_job_exports_cache_contract_before_work(self) -> None:
        script = _runpod_remote_job_script(
            workspace="/workspace",
            run_id="contract-run",
            realization_id="r1",
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
        self.assertIn("export KURA_REALIZATION_ID=r1", script)
        export_hub = line_index('export HF_HUB_CACHE="$HF_HOME/hub"')
        mkdir_hf = line_index('mkdir -p "$HF_HUB_CACHE" "$KURA_WORKSPACE/cache/models"')
        contract_check = line_index("HF_HOME must be under KURA_WORKSPACE before remote job start")
        unpack = line_index("tar -xzf")
        backend_command = line_index("python -c hf_download.py")
        self.assertLess(export_hf, export_hub)
        self.assertLess(export_hub, mkdir_hf)
        self.assertLess(mkdir_hf, contract_check)
        self.assertLess(contract_check, unpack)
        self.assertLess(contract_check, backend_command)

    def test_musubi_container_command_asserts_dataset_before_download(self) -> None:
        script = command_musubi_tuner(_minimal_flux2_run())["argv"][2]
        self.assertLess(script.index("musubi_dataset_assert.py"), script.index("hf_hub_download"))

    def test_musubi_video_profiles_own_preflight_frame_rates(self) -> None:
        self.assertEqual(MUSUBI_PROJECTION_PROFILES["wan-video"]["target_fps"], 16.0)
        self.assertEqual(MUSUBI_PROJECTION_PROFILES["hunyuan-video"]["target_fps"], 24.0)
        self.assertEqual(MUSUBI_PROJECTION_PROFILES["hunyuan-video-1.5-video"]["target_fps"], 24.0)
        self.assertEqual(MUSUBI_PROJECTION_PROFILES["framepack-video"]["target_fps"], 30.0)
        self.assertEqual(MUSUBI_PROJECTION_PROFILES["framepack-f1-video"]["target_fps"], 30.0)
        self.assertNotIn("hunyuan-video-1.5-image", MUSUBI_PROJECTION_PROFILES)
        self.assertEqual(MUSUBI_PROJECTION_PROFILES["h3-video-t2va"]["target_fps"], 24.0)
        self.assertNotIn("target_fps", MUSUBI_PROJECTION_PROFILES["ordinary-image"])


if __name__ == "__main__":
    unittest.main()

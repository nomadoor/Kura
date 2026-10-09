"""The environment Kura owns inside a container is the same on every executor, and the
names that count as secrets are the same everywhere."""

from __future__ import annotations

import tempfile
import unittest
import unittest.mock
from copy import deepcopy
from pathlib import Path

from kura.executors.docker import docker_command
from kura.executors.runpod import _runpod_training_env

KURA_OWNED = ("KURA_LOG_PATH", "KURA_WORKSPACE", "KURA_RUN_ID", "KURA_REALIZATION_ID", "HF_HOME", "HF_HUB_CACHE", "PYTHONUNBUFFERED")


class EnvParityTests(unittest.TestCase):
    def test_kura_owned_variables_win_over_a_command_env_on_every_executor(self) -> None:
        spec_env = {"HF_HOME": "/opt/elsewhere", "HF_HUB_CACHE": "/opt/elsewhere/hub", "PYTHONUNBUFFERED": "0", "OWN": "kept"}
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            _, docker_env, _ = docker_command(workspace, workspace / "runs" / "r", {"cwd": "/opt", "argv": ["x"], "env": deepcopy(spec_env)}, "image", [], "r1")
        runpod_env = _runpod_training_env(deepcopy(spec_env), workspace_path="/workspace", run_id="r", realization_id="r1")
        self.assertEqual({key: docker_env[key] for key in KURA_OWNED}, {key: runpod_env[key] for key in KURA_OWNED})
        self.assertEqual(docker_env["HF_HOME"], "/workspace/cache/huggingface")
        self.assertEqual((docker_env["OWN"], runpod_env["OWN"]), ("kept", "kept"))


class SecretNameTests(unittest.TestCase):
    def test_every_backend_refuses_the_same_secret_names_in_a_custom_command(self) -> None:
        from kura.backends.musubi_command import command_musubi_tuner
        from kura.backends.sd_scripts import command_sd_scripts

        for name in ("HF_TOKEN", "MY_SECRET", "DB_PASSWORD", "OPENAI_API_KEY", "R2_ACCESS_KEY_ID", "SSH_PRIVATE_KEY"):
            command = {"cwd": "/opt", "argv": ["python", "train.py"], "env": {name: "x"}}
            for backend, build in (("sd-scripts", command_sd_scripts), ("musubi-tuner", command_musubi_tuner)):
                with self.subTest(name=name, backend=backend):
                    run = {"id": "r", "backend": {"name": backend, "config": {"command": command}},
                           "recovery": {"training_state": {"enabled": False}}}
                    with self.assertRaisesRegex(ValueError, "secret"):
                        build(run)


if __name__ == "__main__":
    unittest.main()


class SharedMemoryTests(unittest.TestCase):
    def test_a_local_training_container_has_room_in_dev_shm(self) -> None:
        # Docker gives a container 64 MiB of /dev/shm; PyTorch data-loader workers pass
        # whole images through it and fail at real resolutions. RunPod Pods already have more.
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            argv, _, _ = docker_command(workspace, workspace / "runs" / "r", {"cwd": "/opt", "argv": ["x"], "env": {}}, "image", [], "r1")
        self.assertEqual(argv[argv.index("--shm-size") + 1], "16g")
        self.assertLess(argv.index("--shm-size"), argv.index("image"))
        self.assertNotIn("--ipc=host", argv)


class LocalGpuAndRunPodKeyTests(unittest.TestCase):
    def test_a_local_training_container_always_requests_the_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            argv, _, _ = docker_command(workspace, workspace / "runs" / "r", {"cwd": "/opt", "argv": ["x"], "env": {}}, "image", [], "r1")
        self.assertEqual(argv[argv.index("--gpus") + 1], "all")
        self.assertLess(argv.index("--gpus"), argv.index("image"))

    def test_the_runpod_key_is_read_only_by_its_owner(self) -> None:
        import ast
        import kura.executors.runpod as runpod

        source = Path(runpod.__file__).read_text(encoding="utf-8")
        reads = [
            node for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Constant) and node.value == "RUNPOD_API_KEY"
        ]
        # One read and one message, both inside runpod_api_key.
        self.assertEqual(len(reads), 2)
        owner = next(node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef) and node.name == "runpod_api_key")
        self.assertTrue(all(owner.lineno <= node.lineno <= owner.end_lineno for node in reads))
        with unittest.mock.patch.dict("os.environ", {}, clear=True):
            unavailable = runpod.runpod_gpu_availability({"gpu_type_ids": ["NVIDIA A40"]}, ["NVIDIA A40"])
        self.assertEqual(unavailable["status"], "unavailable")
        self.assertIn("kura secrets set RUNPOD_API_KEY", unavailable["reason"])

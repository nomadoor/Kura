from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kura.paths import inspect_workspace_symlinks, local_docker_mounts, local_hf_cache, to_container, to_host, to_host_path, to_workspace_relative
from tests.platform_support import POSIX_PATHS, posix_only


class PathNamespaceTests(unittest.TestCase):
    def test_workspace_relative_helpers_reject_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with self.assertRaisesRegex(ValueError, "unsafe workspace-relative path"):
                to_host("/etc/passwd", workspace)
            with self.assertRaisesRegex(ValueError, "unsafe workspace-relative path"):
                to_container("/etc/passwd")

    @posix_only(POSIX_PATHS)
    def test_symlink_inspection_ignores_malformed_mount_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            link = workspace / "cache" / "models" / "model.safetensors"
            link.parent.mkdir(parents=True)
            link.symlink_to("/root/.cache/huggingface/model.safetensors")

            payload = inspect_workspace_symlinks(workspace, mounts=[False, *local_docker_mounts(workspace, {})])

            self.assertEqual(payload["unsafe"][0]["workspace_target"], "cache/huggingface/model.safetensors")


class HuggingFaceCacheLocationTests(unittest.TestCase):
    def test_the_cache_defaults_to_the_workspace_and_follows_docker_hf_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as elsewhere:
            workspace = Path(directory).resolve()
            outside = Path(elsewhere).resolve() / "hf"
            self.assertEqual(local_hf_cache(workspace, {}), workspace / "cache" / "huggingface")
            self.assertEqual(local_hf_cache(workspace, {"docker": {"hf_cache": "./shared/hf"}}), workspace / "shared" / "hf")
            self.assertEqual(local_hf_cache(workspace, {"docker": {"hf_cache": str(outside)}}), outside)

    def test_the_cache_is_one_mount_beside_the_configured_ones(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as elsewhere:
            workspace = Path(directory).resolve()
            outside = Path(elsewhere).resolve() / "hf"
            extra = {"source": "./loras", "target": "/workspace/loras", "mode": "ro"}
            mounts = local_docker_mounts(workspace, {"docker": {"hf_cache": str(outside), "mounts": [extra]}})
            self.assertEqual(mounts, [extra, {"source": str(outside), "target": "/workspace/cache/huggingface", "mode": "rw"}])
            # The default location is already inside the workspace's cache/ mount.
            self.assertEqual(local_docker_mounts(workspace, {"docker": {"mounts": [extra]}}), [extra])

    @posix_only(POSIX_PATHS)
    def test_a_default_cache_linked_elsewhere_is_mounted_at_its_real_location(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as elsewhere:
            workspace = Path(directory).resolve()
            outside = Path(elsewhere).resolve()
            (workspace / "cache").mkdir()
            (workspace / "cache" / "huggingface").symlink_to(outside)
            self.assertEqual(local_hf_cache(workspace, {}), outside)
            self.assertEqual(local_docker_mounts(workspace, {}), [{"source": str(outside), "target": "/workspace/cache/huggingface", "mode": "rw"}])

    def test_container_paths_map_to_the_cache_wherever_it_lives(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as elsewhere:
            workspace = Path(directory).resolve()
            outside = Path(elsewhere).resolve() / "hf"
            mounts = local_docker_mounts(workspace, {"docker": {"hf_cache": str(outside)}})
            blob = "hub/models--repo/snapshots/abc/model.safetensors"
            self.assertEqual(to_host_path(f"/workspace/cache/huggingface/{blob}", workspace=workspace, mounts=mounts), outside / blob)
            # Links that containers wrote before docker.hf_cache still point at the old target.
            self.assertEqual(to_host_path(f"/root/.cache/huggingface/{blob}", workspace=workspace, mounts=mounts), outside / blob)
            self.assertEqual(to_host_path("/workspace/datasets/a/1.png", workspace=workspace, mounts=mounts), workspace / "datasets" / "a" / "1.png")
            self.assertIsNone(to_host_path("/root/elsewhere/file", workspace=workspace, mounts=mounts))
            self.assertIsNone(to_host_path("/workspace/cache/huggingface/../../../etc/passwd", workspace=workspace, mounts=mounts))
            # Outside the workspace there is no workspace-relative form to repair a link with.
            self.assertIsNone(to_workspace_relative(f"/workspace/cache/huggingface/{blob}", workspace=workspace, mounts=mounts))
            inside = local_docker_mounts(workspace, {})
            self.assertEqual(to_workspace_relative(f"/root/.cache/huggingface/{blob}", workspace=workspace, mounts=inside), f"cache/huggingface/{blob}")


if __name__ == "__main__":
    unittest.main()

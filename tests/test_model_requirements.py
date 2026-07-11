from __future__ import annotations

import unittest

from kura.model_requirements import model_requirements


class ModelRequirementsTests(unittest.TestCase):
    def test_ai_toolkit_keeps_backend_managed_repository(self) -> None:
        requirements = model_requirements(
            {
                "backend": {"name": "ai-toolkit"},
                "model": {"base": "example/model", "revision": "abc123"},
            }
        )

        self.assertEqual(len(requirements), 1)
        requirement = requirements[0]
        self.assertEqual(requirement["role"], "base_model")
        self.assertEqual(requirement["acquisition"], "backend")
        self.assertEqual(requirement["identity"], {"kind": "huggingface-repository", "repo_id": "example/model", "revision": "abc123"})
        self.assertEqual(requirement["measurement"]["scope"], "backend-runtime")

    def test_ai_toolkit_preserves_explicit_local_path(self) -> None:
        requirements = model_requirements(
            {
                "backend": {"name": "ai-toolkit"},
                "model": {"base": "./models/example"},
            }
        )

        self.assertEqual(requirements[0]["acquisition"], "local-path")
        self.assertEqual(requirements[0]["identity"], {"kind": "path", "path": "./models/example"})

    def test_musubi_projects_kura_downloads_and_explicit_paths(self) -> None:
        run = {
            "backend": {"name": "musubi-tuner"},
            "backend_overrides": {"musubi-tuner": {"model_paths": {"vae": "/models/vae.safetensors"}}},
        }
        estimate = {
            "items": [
                {
                    "key": "dit",
                    "repo_id": "example/model",
                    "filename": "dit.safetensors",
                    "revision": "def456",
                    "runtime_reference": "/workspace/cache/models/example/dit.safetensors",
                    "size_status": "ok",
                    "size_bytes": 123,
                    "cached": False,
                }
            ]
        }

        requirements = model_requirements(run, estimate)

        self.assertEqual([item["role"] for item in requirements], ["dit", "vae"])
        self.assertEqual(requirements[0]["acquisition"], "kura")
        self.assertEqual(requirements[0]["measurement"], {"scope": "controller", "status": "ok", "size_bytes": 123, "cached": False})
        self.assertEqual(requirements[1]["acquisition"], "local-path")
        self.assertEqual(requirements[1]["measurement"]["scope"], "compile")


if __name__ == "__main__":
    unittest.main()

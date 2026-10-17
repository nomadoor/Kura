from __future__ import annotations

import unittest

from kura.model_requirements import declared_model_requirements, model_requirements, requirement_source


class ModelRequirementsTests(unittest.TestCase):
    def test_ai_toolkit_records_the_selector_the_trainer_resolves(self) -> None:
        requirements = model_requirements(
            {
                "backend": {"name": "ai-toolkit"},
                "model": {"base": "example/model"},
            }
        )

        self.assertEqual(len(requirements), 1)
        requirement = requirements[0]
        self.assertEqual(requirement["role"], "base_model")
        self.assertEqual(requirement["acquisition"], "backend")
        self.assertEqual(requirement["identity"], {"kind": "backend-resolved-selector", "value": "example/model"})
        self.assertEqual(requirement["settings"], "resolved/ai-toolkit.yaml")
        self.assertEqual(requirement["measurement"]["scope"], "backend-runtime")

    def test_requirement_source_reads_every_identity_kind(self) -> None:
        self.assertEqual(requirement_source({"kind": "huggingface-repository", "repo_id": "org/old"}), "org/old")
        self.assertEqual(requirement_source({"kind": "backend-resolved-selector", "value": "org/new"}), "org/new")
        self.assertEqual(requirement_source({"kind": "huggingface-file", "repo_id": "org/x", "filename": "a"}), "org/x")
        self.assertEqual(requirement_source({"kind": "path", "path": "./models/x"}), "./models/x")
        self.assertIsNone(requirement_source({}))
        self.assertIsNone(requirement_source(None))

    def test_ai_toolkit_preserves_explicit_local_path(self) -> None:
        requirements = model_requirements(
            {
                "backend": {"name": "ai-toolkit"},
                "model": {"base": "./models/example"},
            }
        )

        self.assertEqual(requirements[0]["acquisition"], "local-path")
        self.assertEqual(requirements[0]["identity"], {"kind": "path", "path": "./models/example"})
        self.assertEqual(requirements[0]["expected_format"], "backend-native-path")

    def test_musubi_projects_kura_downloads_and_explicit_paths(self) -> None:
        run = {
                        "backend": {"name": "musubi-tuner", "config": {"model_paths": {"vae": "/models/vae.safetensors"}}},
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

    def test_declared_musubi_requirements_do_not_need_network_measurement(self) -> None:
        requirements = declared_model_requirements(
            {
                                "model": {"base": "krea/Krea-2-Raw"},
                "backend": {"name": "musubi-tuner", "config": {"architecture": "krea2", "model_bundle": "krea2-raw"}},
            }
        )

        self.assertEqual([item["role"] for item in requirements], ["dit", "vae", "text_encoder"])
        self.assertEqual({item["acquisition"] for item in requirements}, {"kura"})
        self.assertEqual({item["measurement"]["scope"] for item in requirements}, {"compile"})
        self.assertEqual({item["measurement"]["status"] for item in requirements}, {"not-measured"})


if __name__ == "__main__":
    unittest.main()


class PlanRequirementSourceTests(unittest.TestCase):
    """The plan names the source of an old repository lock and of a new selector lock alike."""

    REQUIREMENTS = [
        {"role": "base_model", "acquisition": "backend", "runtime_reference": "org/old",
         "identity": {"kind": "huggingface-repository", "repo_id": "org/old"}},
        {"role": "model_extras", "acquisition": "backend", "runtime_reference": "org/new",
         "identity": {"kind": "backend-resolved-selector", "value": "org/new"}},
    ]

    def test_artifact_sources_read_both_kinds(self) -> None:
        from kura.run_commands.plan import _model_artifact_filenames

        self.assertEqual(
            [item["source"] for item in _model_artifact_filenames(self.REQUIREMENTS)],
            ["org/old", "org/new"],
        )

    def test_plan_text_names_both_sources(self) -> None:
        from kura.run_commands.plan import format_run_plan

        text = format_run_plan({"resources": {"model": {"requirements": self.REQUIREMENTS}}})
        sources = [line.split()[-1] for line in text.splitlines() if line.startswith("        source")]
        self.assertEqual(sources, ["org/old", "org/new"])


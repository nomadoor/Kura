from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from kura.comfyui_models import DEFAULT_MODEL_REGISTRY, required_model_refs, resolve_model_specs
from kura.init_templates import COMFYUI_DOCKERFILE_TEMPLATE
from kura.render import _cleanup_lora_stage, _lora_insert_from_sidecar, _materialize_lora_stage, _model_patch_stage_plan, insert_lora_loader, launch_render, patch_workflow


ROOT = Path(__file__).resolve().parents[1]


class ComfyUIModelPatchTests(unittest.TestCase):
    def test_model_patch_loader_is_discovered_and_resolved(self) -> None:
        workflow = {"4": {"class_type": "ModelPatchLoader", "inputs": {"name": "control.safetensors"}}}
        registry = {"model_patches": {"control.safetensors": {"repo": "owner/repo", "filename": "control.safetensors", "revision": "a" * 40}}}

        specs, unknown = resolve_model_specs(workflow, registry)

        self.assertEqual(unknown, [])
        self.assertEqual(specs[0]["target_dir"], "model_patches")
        self.assertEqual(required_model_refs(workflow)[0]["input"], "name")

    def test_default_anima_registry_is_immutable_and_uses_text_encoders(self) -> None:
        self.assertEqual(DEFAULT_MODEL_REGISTRY["clip"]["qwen_3_06b_base.safetensors"]["target_dir"], "text_encoders")
        for section, name in (("diffusion_models", "anima-base-v1.0.safetensors"), ("clip", "qwen_3_06b_base.safetensors"), ("vae", "qwen_image_vae.safetensors")):
            self.assertEqual(len(DEFAULT_MODEL_REGISTRY[section][name]["revision"]), 40)

    def test_model_patch_checkpoint_stages_outside_lora_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "runs" / "train" / "outputs" / "lllite.safetensors"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"weights")
            run_dir = workspace / "runs" / "render"
            run_dir.mkdir(parents=True)
            frozen = {
                "workflow_patches": {"model_patch": {"node": "4", "field": "inputs.name"}},
                "comfyui": {"model_patches_dir": "managed-comfy/models/model_patches", "model_patch_stage_mode": "symlink", "model_patch_stage_cleanup": "remove_after_render"},
            }
            plan = _model_patch_stage_plan(workspace, run_dir, frozen, {"path": "runs/train/outputs/lllite.safetensors"})
            assert plan is not None
            _materialize_lora_stage(plan)
            target = Path(plan["target"])
            exists_during_render = target.is_symlink()
            _cleanup_lora_stage(plan)
            exists_after_cleanup = target.exists() or target.is_symlink()
        self.assertTrue(exists_during_render)
        self.assertFalse(exists_after_cleanup)
        self.assertIn("Kura_tmp/", plan["model_patch_name"])

    def test_model_patch_requires_non_empty_local_stage_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "runs" / "train" / "outputs" / "lllite.safetensors"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"weights")
            frozen = {"workflow_patches": {"model_patch": {"node": "4", "field": "inputs.name"}}, "comfyui": {"model_patches_dir": ""}}
            with self.assertRaisesRegex(ValueError, "model_patches_dir is required"):
                _model_patch_stage_plan(workspace, workspace / "runs" / "render", frozen, {"path": "runs/train/outputs/lllite.safetensors"})

    def test_runpod_model_patch_fails_before_using_host_checkpoint_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run_dir = workspace / "runs" / "render"
            resolved = run_dir / "resolved"
            resolved.mkdir(parents=True)
            (resolved / "manifest.lock.yaml").write_text(yaml.safe_dump({"generator": {"name": "comfyui"}, "executor": {"name": "runpod"}, "workflow_patches": {"model_patch": {"node": "4", "field": "inputs.name"}}}), encoding="utf-8")
            (resolved / "workflow_used.json").write_text("{}", encoding="utf-8")
            (run_dir / "status.json").write_text('{"state":"compiled"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not supported for the runpod executor"):
                launch_render(workspace, run_dir, executor_name="runpod", manage_lora_stage=False)

    def test_model_patch_patch_value_is_separate_from_lora_value(self) -> None:
        workflow = {"4": {"inputs": {"name": "old"}}, "5": {"inputs": {"lora_name": "old"}}}
        patched = patch_workflow(workflow, {"model_patch": {"node": "4", "field": "inputs.name"}, "lora": {"node": "5", "field": "inputs.lora_name"}}, prompt="", negative_prompt="", seed=1, checkpoint="lora.safetensors", model_patch="patch.safetensors")
        self.assertEqual(patched["4"]["inputs"]["name"], "patch.safetensors")
        self.assertEqual(patched["5"]["inputs"]["lora_name"], "lora.safetensors")

    def test_managed_comfyui_pin_contains_core_anima_lllite_merge(self) -> None:
        dockerfile = (ROOT / "docker" / "comfyui" / "Dockerfile").read_text(encoding="utf-8")
        revision = "0f42ba51463174fb255f2c4605ae0e0b441fe6d7"
        self.assertIn(revision, dockerfile)
        self.assertIn(revision, COMFYUI_DOCKERFILE_TEMPLATE)

    def test_authored_lllite_workflow_loads_then_applies_patch(self) -> None:
        workflow = json.loads((ROOT / "examples" / "sd-scripts-anima-smoke" / "anima-lllite-api.json").read_text(encoding="utf-8"))
        self.assertEqual(workflow["4"]["class_type"], "ModelPatchLoader")
        self.assertEqual(workflow["6"]["class_type"], "AnimaLLLiteApply")
        self.assertEqual(workflow["6"]["inputs"]["model_patch"], ["4", 0])

    def test_authored_anima_lora_sidecar_inserts_and_routes_model_loader(self) -> None:
        workflow = json.loads((ROOT / "examples" / "sd-scripts-anima-smoke" / "anima-lora-api.json").read_text(encoding="utf-8"))
        sidecar = yaml.safe_load((ROOT / "examples" / "sd-scripts-anima-smoke" / "anima-lora-api.kura.yaml").read_text(encoding="utf-8"))
        spec = _lora_insert_from_sidecar(sidecar)

        patched = insert_lora_loader(workflow, spec, "Kura_tmp/trained-anima.safetensors")

        loaders = [(node_id, node) for node_id, node in patched.items() if node.get("class_type") == "LoraLoaderModelOnly"]
        self.assertEqual(len(loaders), 1)
        node_id, loader = loaders[0]
        self.assertEqual(loader["inputs"]["model"], ["1", 0])
        self.assertEqual(loader["inputs"]["lora_name"], "Kura_tmp/trained-anima.safetensors")
        self.assertEqual(patched["7"]["inputs"]["model"], [node_id, 0])


if __name__ == "__main__":
    unittest.main()

"""Importing a site workflow as a sample: the checks and the sidecar, without the network."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("import_site_workflow", ROOT / "scripts" / "import_site_workflow.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

HF = "https://huggingface.co/Comfy-Org/MOSS-Image/resolve/main"
ENTRY = {
    "url": "https://comfyui.nomadoor.net/workflows/basic-workflows/moss-image/moss_image_text2image.json",
    "updated": "2026-10-17",
    "models": [
        {"name": "moss_image_sft_int8_convrot.safetensors", "directory": "diffusion_models", "url": f"{HF}/diffusion_models/moss_image_sft_int8_convrot.safetensors"},
        {"name": "qwen3.5_9b_int8_convrot.safetensors", "directory": "text_encoders", "url": f"{HF}/text_encoders/qwen3.5_9b_int8_convrot.safetensors"},
        {"name": "flux2-vae.safetensors", "directory": "vae", "url": f"{HF}/vae/flux2-vae.safetensors"},
    ],
    "custom_nodes": [],
}
API = {
    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "moss_image_sft_int8_convrot.safetensors"}},
    "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3.5_9b_int8_convrot.safetensors"}},
    "3": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
    "4": {"class_type": "KSampler", "inputs": {"model": ["1", 0]}},
}
UI = {"nodes": [{"id": 1, "type": "UNETLoader"}, {"id": 2, "type": "CLIPLoader"}, {"id": 3, "type": "VAELoader"},
                {"id": 4, "type": "KSampler"}, {"id": 9, "type": "Note"}]}


class ImportSiteWorkflowTests(unittest.TestCase):
    def test_a_hugging_face_url_names_its_repo_file_and_revision(self) -> None:
        self.assertEqual(MODULE.hf_file(f"{HF}/text_encoders/qwen3.5_9b_int8_convrot.safetensors"), {
            "repo": "Comfy-Org/MOSS-Image", "revision": "main", "filename": "text_encoders/qwen3.5_9b_int8_convrot.safetensors",
        })
        with self.assertRaises(ValueError):
            MODULE.hf_file("https://example.com/model.safetensors")

    def test_the_export_must_come_from_the_ui_workflow(self) -> None:
        # A UI-only node (a note) is not exported; that is fine.
        self.assertEqual(MODULE.api_problems(API, UI), [])
        changed = {**API, "4": {"class_type": "SamplerCustom", "inputs": {}}}
        self.assertEqual(MODULE.api_problems(changed, UI), ["API node 4 is SamplerCustom, the UI node KSampler"])
        extra = {**API, "7": {"class_type": "KSampler", "inputs": {}}}
        self.assertIn("API node 7 (KSampler) is not in the UI workflow", MODULE.api_problems(extra, UI))
        self.assertTrue(MODULE.api_problems(UI, UI))

    def test_the_sidecar_pins_each_repo_and_resolves_every_model(self) -> None:
        asked: list[tuple[str, str]] = []

        def commit(repo: str, revision: str) -> str:
            asked.append((repo, revision))
            return "abc123"

        sidecar, notes = MODULE.build_sidecar(API, ENTRY, imported_on="2026-10-27", commit=commit)
        self.assertEqual(asked, [("Comfy-Org/MOSS-Image", "main")])
        self.assertEqual(sidecar["source"], ENTRY["url"])
        self.assertIn("2026-10-27", sidecar["provenance"])
        self.assertEqual(sidecar["models"]["clip"]["qwen3.5_9b_int8_convrot.safetensors"], {
            "repo": "Comfy-Org/MOSS-Image", "filename": "text_encoders/qwen3.5_9b_int8_convrot.safetensors",
            "revision": "abc123", "target_dir": "text_encoders",
        })
        # The default folder of a type is not repeated.
        self.assertNotIn("target_dir", sidecar["models"]["vae"]["flux2-vae.safetensors"])
        self.assertEqual(sidecar["lora_insert"], {"kind": "model_only", "model_node": "1", "strength_model": 0.8})
        self.assertEqual(notes, [])
        MODULE.check_sidecar(API, sidecar)

    def test_the_sidecar_is_one_kura_accepts(self) -> None:
        from kura.render import SIDECAR_KEYS, _lora_insert_from_sidecar

        sidecar, _ = MODULE.build_sidecar(API, ENTRY, imported_on="2026-10-27", commit=lambda repo, revision: "abc123")
        self.assertLessEqual(set(sidecar), SIDECAR_KEYS)
        self.assertIsNotNone(_lora_insert_from_sidecar(sidecar))

    def test_a_checkpoint_loader_takes_the_lora_on_model_and_text_encoder(self) -> None:
        api = {"4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "v1-5.safetensors"}}}
        entry = {"url": "u", "models": [{"name": "v1-5.safetensors", "directory": "checkpoints", "url": f"{HF}/v1-5.safetensors"}], "custom_nodes": []}
        sidecar, _ = MODULE.build_sidecar(api, entry, imported_on="d", commit=lambda r, v: "c")
        self.assertEqual(sidecar["lora_insert"], {"kind": "model_clip", "model_node": "4", "clip_node": "4", "strength_model": 0.8, "strength_clip": 0.8})

    def test_a_model_named_inside_the_authors_folders_takes_the_sites_file_name(self) -> None:
        import copy

        api = copy.deepcopy(API)
        api["1"]["inputs"]["unet_name"] = "MOSS\\moss_image_sft_int8_convrot.safetensors"
        self.assertEqual(MODULE.site_model_names(api, ENTRY), ["MOSS\\moss_image_sft_int8_convrot.safetensors -> moss_image_sft_int8_convrot.safetensors"])
        self.assertEqual(api, API)
        self.assertEqual(MODULE.site_model_names(api, ENTRY), [])

    def test_what_the_import_cannot_do_is_said(self) -> None:
        with self.assertRaisesRegex(ValueError, "custom nodes"):
            MODULE.build_sidecar(API, {**ENTRY, "custom_nodes": [{"id": "x"}]}, imported_on="d", commit=lambda r, v: "c")
        with self.assertRaisesRegex(ValueError, "no download URL for flux2-vae"):
            MODULE.build_sidecar(API, {**ENTRY, "models": ENTRY["models"][:2]}, imported_on="d", commit=lambda r, v: "c")
        # A model only an unknown loader reads, and no single model loader for a LoRA, become notes.
        extra = {**ENTRY, "models": ENTRY["models"] + [{"name": "upscaler.pth", "directory": "upscale_models", "url": f"{HF}/u.pth"}]}
        two_loaders = {**API, "5": {"class_type": "UNETLoader", "inputs": {"unet_name": "moss_image_sft_int8_convrot.safetensors"}}}
        sidecar, notes = MODULE.build_sidecar(two_loaders, extra, imported_on="d", commit=lambda r, v: "c")
        self.assertNotIn("lora_insert", sidecar)
        self.assertTrue(any("upscaler.pth" in note for note in notes))
        self.assertTrue(any("lora_insert left out" in note for note in notes))


if __name__ == "__main__":
    unittest.main()

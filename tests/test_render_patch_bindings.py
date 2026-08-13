import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kura.render import (  # noqa: E402
    compile_render,
    image_patch_names,
    launch_render,
    patch_workflow,
    reconcile_promptset,
    validate_patch_bindings,
)

WORKFLOW = {
    "3": {"inputs": {"seed": 0}},
    "6": {"inputs": {"text": ""}},
    "7": {"inputs": {"text": ""}},
    "12": {"inputs": {"lora_name": "old.safetensors"}},
    "15": {"inputs": {"image": "placeholder.png"}},
    "17": {"inputs": {"strength": 0.8}},
}

BASE_PATCHES = {
    "prompt": {"node": "6", "field": "inputs.text"},
    "negative_prompt": {"node": "7", "field": "inputs.text"},
    "seed": {"node": "3", "field": "inputs.seed"},
    "lora": {"node": "12", "field": "inputs.lora_name"},
}


class PatchBindingTest(unittest.TestCase):
    def test_binding_to_missing_node_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as caught:
            validate_patch_bindings(WORKFLOW, {"strength": {"node": "99", "field": "inputs.strength"}})
        self.assertIn("workflow_patches.strength", str(caught.exception))

    def test_binding_to_missing_field_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_patch_bindings(WORKFLOW, {"strength": {"node": "17", "field": "inputs.nope"}})

    def test_binding_shape_and_type_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            validate_patch_bindings(WORKFLOW, {"strength": {"node": "17"}})
        with self.assertRaises(ValueError):
            validate_patch_bindings(WORKFLOW, {"control": {"node": "15", "field": "inputs.image", "type": "video"}})

    def test_arbitrary_bound_name_is_applied_from_the_item(self) -> None:
        patches = {**BASE_PATCHES, "strength": {"node": "17", "field": "inputs.strength"}}
        item = {"id": "a", "prompt": "hello", "strength": 0.5}
        patched = patch_workflow(
            WORKFLOW, patches, prompt="hello", negative_prompt="", seed=7,
            checkpoint="lora.safetensors", item=item,
        )
        self.assertEqual(patched["17"]["inputs"]["strength"], 0.5)
        self.assertEqual(patched["3"]["inputs"]["seed"], 7)
        self.assertEqual(WORKFLOW["17"]["inputs"]["strength"], 0.8)

    def test_bound_name_missing_from_item_fails_loudly(self) -> None:
        patches = {**BASE_PATCHES, "strength": {"node": "17", "field": "inputs.strength"}}
        with self.assertRaises(ValueError) as caught:
            patch_workflow(
                WORKFLOW, patches, prompt="hello", negative_prompt="", seed=7,
                checkpoint="lora.safetensors", item={"id": "a", "prompt": "hello"},
            )
        self.assertIn("strength", str(caught.exception))

    def test_legacy_callers_without_item_still_patch_reserved_names(self) -> None:
        patched = patch_workflow(
            WORKFLOW, {**BASE_PATCHES, "strength": {"node": "17", "field": "inputs.strength"}},
            prompt="hi", negative_prompt="no", seed=3, checkpoint="lora.safetensors",
        )
        self.assertEqual(patched["6"]["inputs"]["text"], "hi")
        self.assertEqual(patched["17"]["inputs"]["strength"], 0.8)

    def test_image_patch_names(self) -> None:
        patches = {**BASE_PATCHES, "control_image": {"node": "15", "field": "inputs.image", "type": "image"}}
        self.assertEqual(image_patch_names(patches), ["control_image"])
        self.assertEqual(image_patch_names(BASE_PATCHES), [])


class ReconcilePromptsetTest(unittest.TestCase):
    def test_unbound_key_is_rejected_with_actionable_message(self) -> None:
        items = [{"id": "a", "prompt": "x", "width": 512, "height": 768}]
        with self.assertRaises(ValueError) as caught:
            reconcile_promptset(items, BASE_PATCHES)
        message = str(caught.exception)
        self.assertIn("height, width", message)
        self.assertIn("meta", message)

    def test_meta_is_reserved_for_provenance(self) -> None:
        items = [{"id": "a", "prompt": "x", "meta": {"width": 512, "spec": "grid-v1"}}]
        reconcile_promptset(items, BASE_PATCHES)

    def test_bound_key_must_be_present_in_every_item(self) -> None:
        patches = {**BASE_PATCHES, "strength": {"node": "17", "field": "inputs.strength"}}
        with self.assertRaises(ValueError) as caught:
            reconcile_promptset([{"id": "a", "prompt": "x", "strength": 0.5}, {"id": "b", "prompt": "y"}], patches)
        self.assertIn("'b'", str(caught.exception))

    def test_run_sourced_patches_need_no_item_key(self) -> None:
        patches = {**BASE_PATCHES, "model_patch": {"node": "12", "field": "inputs.lora_name"}}
        reconcile_promptset([{"id": "a", "prompt": "x", "seeds": [1]}], patches)


class RenderImageBindingTest(unittest.TestCase):
    def _workspace(self, root: Path, *, patches: dict[str, Any], items: list[dict[str, Any]], input_dir: Path | None = None) -> Path:
        run_dir = root / "runs" / "render-1"
        (run_dir / "resolved").mkdir(parents=True)
        (root / "workflows").mkdir()
        (root / "promptsets" / "grid").mkdir(parents=True)
        train_out = root / "runs" / "train-1" / "outputs"
        train_out.mkdir(parents=True)
        (train_out.parent / "run.yaml").write_text("id: train-1\ntype: train\n", encoding="utf-8")
        (train_out / "example.safetensors").write_bytes(b"fake-lora")
        comfyui = {"lora_dir": str(root / "loras")}
        if input_dir is not None:
            comfyui.update({"input_dir": str(input_dir), "input_stage_subdir": "Kura_tmp", "input_stage_mode": "copy"})
        (root / "workspace.yaml").write_text(yaml.safe_dump({"comfyui": comfyui}), encoding="utf-8")
        (root / "workflows" / "wf.json").write_text(json.dumps(WORKFLOW), encoding="utf-8")
        (root / "promptsets" / "grid" / "prompts.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in items), encoding="utf-8",
        )
        (run_dir / "run.yaml").write_text(
            yaml.safe_dump({
                "schema_version": 1,
                "type": "render",
                "inputs": {
                    "train_run": "train-1",
                    "checkpoint": {"path": "runs/train-1/outputs/example.safetensors", "hash": None},
                    "workflow": {"path": "workflows/wf.json", "digest": None},
                    "promptset": {"path": "promptsets/grid/prompts.jsonl", "digest": None},
                },
                "generator": {"name": "comfyui", "endpoint": ""},
                "executor": {"name": "local"},
                "workflow_patches": patches,
                "render": {"output_dir": "samples/images", "timeout_sec": 5, "default_seed": 42},
            }),
            encoding="utf-8",
        )
        (run_dir / "status.json").write_text(json.dumps({"state": "draft"}), encoding="utf-8")
        return run_dir

    def test_compile_rejects_promptset_key_the_workflow_cannot_accept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = self._workspace(
                root, patches=BASE_PATCHES,
                items=[{"id": "a", "prompt": "x", "seeds": [1], "width": 512, "height": 768}],
            )
            with self.assertRaises(ValueError) as caught:
                compile_render(root, run_dir)
            self.assertIn("height, width", str(caught.exception))
            self.assertFalse((run_dir / "resolved" / "promptset_used.jsonl").exists())

    def test_compile_freezes_control_images_and_repoints_the_promptset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            patches = {**BASE_PATCHES, "control_image": {"node": "15", "field": "inputs.image", "type": "image"}}
            run_dir = self._workspace(
                root, patches=patches,
                items=[{"id": "a", "prompt": "x", "seeds": [1], "control_image": "a/control.png"}],
            )
            source = root / "promptsets" / "grid" / "a" / "control.png"
            source.parent.mkdir()
            source.write_bytes(b"control-a")
            compile_render(root, run_dir)
            frozen_image = run_dir / "resolved" / "images" / "control_image" / "a.png"
            self.assertEqual(frozen_image.read_bytes(), b"control-a")
            used = [json.loads(line) for line in (run_dir / "resolved" / "promptset_used.jsonl").read_text().splitlines()]
            self.assertEqual(used[0]["control_image"], "resolved/images/control_image/a.png")
            manifest = yaml.safe_load((run_dir / "resolved" / "manifest.lock.yaml").read_text(encoding="utf-8"))
            self.assertEqual(manifest["promptset_images"][0]["prompt_id"], "a")
            self.assertEqual(manifest["promptset_images"][0]["source"], str(source.resolve()))
            source.write_bytes(b"changed-after-compile")
            self.assertEqual(frozen_image.read_bytes(), b"control-a")

    def test_compile_cannot_write_outside_the_frozen_image_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            patches = {**BASE_PATCHES, "control_image": {"node": "15", "field": "inputs.image", "type": "image"}}
            run_dir = self._workspace(
                root, patches=patches,
                items=[{"id": "../../../run", "prompt": "x", "seeds": [1], "control_image": "payload.yaml"}],
            )
            run_yaml = run_dir / "run.yaml"
            before = run_yaml.read_bytes()
            (root / "promptsets" / "grid" / "payload.yaml").write_bytes(b"type: pwned\n")
            with self.assertRaises(ValueError) as caught:
                compile_render(root, run_dir)
            self.assertIn("single safe file name", str(caught.exception))
            self.assertEqual(run_yaml.read_bytes(), before)

    def test_compile_rejects_a_missing_or_escaping_control_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            patches = {**BASE_PATCHES, "control_image": {"node": "15", "field": "inputs.image", "type": "image"}}
            run_dir = self._workspace(
                root, patches=patches,
                items=[{"id": "a", "prompt": "x", "seeds": [1], "control_image": "a/missing.png"}],
            )
            with self.assertRaises(ValueError) as caught:
                compile_render(root, run_dir)
            self.assertIn("does not exist", str(caught.exception))

    def test_launch_stages_images_and_patches_each_item(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "comfy-input"
            input_dir.mkdir()
            patches = {**BASE_PATCHES, "control_image": {"node": "15", "field": "inputs.image", "type": "image"}}
            run_dir = self._workspace(
                root, patches=patches, input_dir=input_dir,
                items=[
                    {"id": "a", "prompt": "x", "seeds": [1], "control_image": "a/control.png"},
                    {"id": "b", "prompt": "y", "seeds": [1], "control_image": "b/control.png"},
                ],
            )
            for case, payload in (("a", b"control-a"), ("b", b"control-b")):
                path = root / "promptsets" / "grid" / case / "control.png"
                path.parent.mkdir()
                path.write_bytes(payload)
            compile_render(root, run_dir)
            queued: list[dict[str, Any]] = []
            staged_during_queue: list[bool] = []

            class FakeClient:
                def __init__(self, endpoint: str, timeout: int) -> None:
                    pass

                def input_image_names(self) -> set[str]:
                    return {f"Kura_tmp/{path.name}" for path in (input_dir / "Kura_tmp").glob("*")}

                def queue(self, workflow: dict[str, Any]) -> str:
                    queued.append(workflow)
                    name = workflow["15"]["inputs"]["image"]
                    staged_during_queue.append((input_dir / name).is_file())
                    return f"prompt-{len(queued)}"

                def wait(self, prompt_id: str) -> list[dict[str, Any]]:
                    return [{"filename": "out.png", "subfolder": "", "type": "output"}]

                def download(self, image: dict[str, Any]) -> bytes:
                    return b"image-bytes"

            import kura.render as render_module

            original = render_module.ComfyUIClient
            render_module.ComfyUIClient = FakeClient  # type: ignore[assignment]
            try:
                exit_code = launch_render(root, run_dir, endpoint_override="http://127.0.0.1:8188")
            finally:
                render_module.ComfyUIClient = original  # type: ignore[assignment]
            self.assertEqual(exit_code, 0)
            self.assertEqual(len(queued), 2)
            names = [workflow["15"]["inputs"]["image"] for workflow in queued]
            self.assertTrue(all(name.startswith("Kura_tmp/") for name in names))
            self.assertEqual(len(set(names)), 2, "each case must render its own control image")
            self.assertEqual(staged_during_queue, [True, True])
            self.assertEqual(list((input_dir / "Kura_tmp").glob("*")), [], "staged images are removed after render")
            records = [json.loads(line) for line in (run_dir / "samples" / "images.jsonl").read_text().splitlines()]
            self.assertEqual(records[0]["patch_inputs"]["control_image"], "resolved/images/control_image/a.png")


if __name__ == "__main__":
    unittest.main()


class PromptIdSafetyTest(unittest.TestCase):
    def _write(self, root: Path, items: list[dict[str, Any]]) -> Path:
        path = root / "prompts.jsonl"
        path.write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")
        return path

    def test_traversing_id_is_rejected(self) -> None:
        from kura.render import promptset

        with tempfile.TemporaryDirectory() as tmp:
            for bad in ("../../../run", "a/b", "..", ".", "a\\b", ".hidden"):
                path = self._write(Path(tmp), [{"id": bad, "prompt": "x"}])
                with self.assertRaises(ValueError, msg=bad) as caught:
                    promptset(path)
                self.assertIn("single safe file name", str(caught.exception))

    def test_duplicate_ids_are_rejected(self) -> None:
        from kura.render import promptset

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), [{"id": "a", "prompt": "x"}, {"id": "a", "prompt": "y"}])
            with self.assertRaises(ValueError) as caught:
                promptset(path)
            self.assertIn("duplicate id", str(caught.exception))

    def test_ordinary_ids_still_load(self) -> None:
        from kura.render import promptset

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), [{"id": "01_portrait", "prompt": "x"}, {"id": "b-2.v1", "prompt": "y"}])
            self.assertEqual([item["id"] for item in promptset(path)], ["01_portrait", "b-2.v1"])


class CoreKeyReconciliationTest(unittest.TestCase):
    ITEMS = [{"id": "a", "prompt": "USER PROMPT", "seeds": [123]}]

    def test_unbound_prompt_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as caught:
            reconcile_promptset(self.ITEMS, {})
        self.assertIn("no prompt binding", str(caught.exception))

    def test_unbound_seed_is_rejected(self) -> None:
        patches = {"prompt": {"node": "6", "field": "inputs.text"}}
        with self.assertRaises(ValueError) as caught:
            reconcile_promptset(self.ITEMS, patches)
        self.assertIn("no seed binding", str(caught.exception))

    def test_unbound_negative_prompt_is_rejected_only_when_used(self) -> None:
        patches = {"prompt": {"node": "6", "field": "inputs.text"}, "seed": {"node": "3", "field": "inputs.seed"}}
        reconcile_promptset(self.ITEMS, patches)
        with self.assertRaises(ValueError) as caught:
            reconcile_promptset([{**self.ITEMS[0], "negative_prompt": "blurry"}], patches)
        self.assertIn("no negative_prompt binding", str(caught.exception))

    def test_default_seed_alone_requires_a_seed_binding(self) -> None:
        patches = {"prompt": {"node": "6", "field": "inputs.text"}}
        with self.assertRaises(ValueError):
            reconcile_promptset([{"id": "a", "prompt": "x"}], patches, default_seed=42)
        reconcile_promptset([{"id": "a", "prompt": "x"}], patches)

    def test_workflow_fixed_declares_a_deliberately_hardcoded_parameter(self) -> None:
        reconcile_promptset(self.ITEMS, {}, workflow_fixed=["prompt", "seed"])

    def test_workflow_fixed_rejects_unknown_and_conflicting_names(self) -> None:
        with self.assertRaises(ValueError) as caught:
            reconcile_promptset(self.ITEMS, {}, workflow_fixed=["prompt", "seed", "strength"])
        self.assertIn("strength", str(caught.exception))
        patches = {"prompt": {"node": "6", "field": "inputs.text"}, "seed": {"node": "3", "field": "inputs.seed"}}
        with self.assertRaises(ValueError) as caught:
            reconcile_promptset(self.ITEMS, patches, workflow_fixed=["prompt"])
        self.assertIn("both claim", str(caught.exception))

    def test_run_sourced_names_are_rejected_as_item_keys(self) -> None:
        patches = {"prompt": {"node": "6", "field": "inputs.text"}, "seed": {"node": "3", "field": "inputs.seed"}}
        for key, value in (("seed", 456), ("lora", "other.safetensors"), ("model_patch", "p.safetensors"), ("checkpoint", "c")):
            with self.assertRaises(ValueError, msg=key) as caught:
                reconcile_promptset([{**self.ITEMS[0], key: value}], patches)
            self.assertIn("comes from the run", str(caught.exception))


class ImageVisibilityAndErrorSurfaceTest(unittest.TestCase):
    PLANS = [{"image_name": "Kura_tmp/a.png"}]

    def test_unavailable_listing_warns_instead_of_blocking(self) -> None:
        from kura.render import _ensure_image_stage_visible

        class Broken:
            def input_image_names(self):
                raise RuntimeError("LoadImage object_info query failed")

        _ensure_image_stage_visible(Broken(), "http://127.0.0.1:8188", self.PLANS)

    def test_empty_listing_warns_instead_of_blocking(self) -> None:
        from kura.render import _ensure_image_stage_visible

        class Empty:
            def input_image_names(self):
                return set()

        _ensure_image_stage_visible(Empty(), "http://127.0.0.1:8188", self.PLANS)

    def test_populated_listing_missing_the_name_still_only_warns(self) -> None:
        from kura.render import _ensure_image_stage_visible

        class Other:
            def input_image_names(self):
                return {"someone-elses.png"}

        _ensure_image_stage_visible(Other(), "http://127.0.0.1:8188", self.PLANS)

    def test_http_error_body_is_surfaced(self) -> None:
        import urllib.error
        import urllib.request
        from io import BytesIO

        from kura.render import ComfyUIClient

        body = b'{"error": {"message": "Prompt outputs failed validation"}, "node_errors": {"1": {"errors": [{"details": "image - Invalid image file: Kura_tmp/a.png"}]}}}'

        def raise_http(request, timeout=None):
            raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, BytesIO(body))

        original = urllib.request.urlopen
        urllib.request.urlopen = raise_http
        try:
            with self.assertRaises(urllib.error.HTTPError) as caught:
                ComfyUIClient("http://127.0.0.1:8188", 5).queue({})
        finally:
            urllib.request.urlopen = original
        self.assertIn("Invalid image file: Kura_tmp/a.png", str(caught.exception))


if __name__ == "__main__":
    unittest.main()

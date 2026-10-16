"""A render run.yaml and its workflow sidecar are closed and read in one form."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml

from kura.render import _lora_insert_from_sidecar, compile_render

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_render_cases import PATCHES, WORKFLOW, _case, _workspace  # noqa: E402


def _edit_run(run_dir: Path, change: Any) -> None:
    path = run_dir / "run.yaml"
    run = yaml.safe_load(path.read_text(encoding="utf-8"))
    change(run)
    path.write_text(yaml.safe_dump(run, sort_keys=False), encoding="utf-8")


class RenderSurfaceTests(unittest.TestCase):
    def _run(self, root: Path) -> Path:
        return _workspace(
            root,
            cases=[_case("a", step=None, prompt="first", seed=1, strength=1.0, cfg=4.0)],
            patches={key: value for key, value in PATCHES.items() if key != "lora"},
        )

    def test_an_unknown_key_at_any_render_level_is_refused(self) -> None:
        for change, message in (
            (lambda run: run.update(notes="x"), "notes is not a render run setting"),
            (lambda run: run["inputs"].update(prompts="x"), "inputs.prompts is not a render run setting"),
            (lambda run: run["inputs"]["workflow"].update(sha="x"), "inputs.workflow.sha is not a render run setting"),
            (lambda run: run["generator"].update(port=8188), "generator.port is not a render run setting"),
            (lambda run: run["executor"].update(gpu="A40"), "executor.gpu is not a render run setting"),
            (lambda run: run["executor"].update(name="docker"), "executor.name must be local or runpod"),
            (lambda run: run.update(compute={"gpu": ["NVIDIA A40"]}), "a local run uses the local GPU"),
            (lambda run: run["generator"].update(name="a1111"), "generator.name must be comfyui"),
            (lambda run: run.update(executor="runpod"), "executor must be a mapping of name, not 'runpod'"),
            (lambda run: run.update(compute=["NVIDIA A40"]), "compute must be a mapping of gpu"),
            (lambda run: run.update(generator="comfyui"), "generator must be a mapping of endpoint, name"),
            (lambda run: run["inputs"].update(workflow="workflows/wf.json"), "inputs.workflow must be a mapping of digest, path"),
            (lambda run: run["inputs"].update(workflow=None), "inputs.workflow must be a mapping of digest, path, not None"),
            (lambda run: run["inputs"].pop("workflow"), "inputs.workflow must be a mapping of digest, path, not None"),
            (lambda run: run.update(inputs=None), "inputs must be a mapping of .*, not None"),
            (lambda run: run.pop("inputs"), "inputs must be a mapping of .*, not None"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                run_dir = self._run(Path(directory))
                _edit_run(run_dir, change)
                with self.assertRaisesRegex(ValueError, message):
                    compile_render(Path(directory), run_dir)

    def test_one_compile_names_every_unknown_key_compiled_only_key_and_render_setting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run(Path(directory))
            _edit_run(run_dir, lambda run: (run.update(notes="x", lora_insert={}), run["render"].update(lora_stage="off")))
            with self.assertRaises(ValueError) as caught:
                compile_render(Path(directory), run_dir)
        message = str(caught.exception)
        self.assertTrue(message.startswith("render run.yaml: "), message)
        self.assertIn("notes is not a render run setting", message)
        self.assertIn("lora_insert is written by compile from the workflow, its sidecar, and workspace.yaml; to change the inserted LoRA's strength, set render.lora_strength", message)
        self.assertIn("render.lora_stage is not a render run setting", message)

    def test_an_authored_digest_must_match_its_file(self) -> None:
        for name in ("workflow", "cases"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                run_dir = self._run(Path(directory))
                _edit_run(run_dir, lambda run: run["inputs"][name].update(digest="sha256:" + "0" * 64))
                with self.assertRaisesRegex(ValueError, rf"inputs.{name}.digest sha256:0+ does not match .* \(sha256:[0-9a-f]{{64}}\); set it to null"):
                    compile_render(Path(directory), run_dir)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._run(root)
            import hashlib

            actual = "sha256:" + hashlib.sha256((root / "workflows" / "wf.json").read_bytes()).hexdigest()
            _edit_run(run_dir, lambda run: run["inputs"]["workflow"].update(digest=actual))
            compile_render(root, run_dir)
            self.assertEqual(yaml.safe_load((run_dir / "resolved" / "manifest.lock.yaml").read_text(encoding="utf-8"))["inputs"]["workflow"]["digest"], actual)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run(Path(directory))
            _edit_run(run_dir, lambda run: run["inputs"]["workflow"].update(digest=5))
            with self.assertRaisesRegex(ValueError, "inputs.workflow.digest must be a string or null"):
                compile_render(Path(directory), run_dir)

    def test_the_output_directory_stays_inside_the_run_and_the_timeout_keeps_its_value(self) -> None:
        for render, message in (
            ({"output_dir": "/tmp/out"}, "render.output_dir must be a folder inside the run"),
            ({"output_dir": "../elsewhere"}, "render.output_dir must be a folder inside the run"),
            ({"output_dir": "C:out"}, "render.output_dir must be a folder inside the run"),
            ({"timeout_sec": 0}, "render.timeout_sec must be a whole number of seconds of at least 1, not 0"),
            ({"timeout_sec": "600"}, "render.timeout_sec must be a whole number"),
        ):
            with self.subTest(render=render), tempfile.TemporaryDirectory() as directory:
                run_dir = self._run(Path(directory))
                _edit_run(run_dir, lambda run: run["render"].update(render))
                with self.assertRaisesRegex(ValueError, message):
                    compile_render(Path(directory), run_dir)

    def test_seed_values_reach_the_workflow_or_are_refused(self) -> None:
        for change, message in (
            (lambda run: (run["workflow_patches"].pop("seed"), run["render"].update(default_seed=42)), "render.default_seed needs a workflow_patches.seed binding"),
            (lambda run: run["render"].update(workflow_fixed=["seed"]), "render.workflow_fixed and workflow_patches both claim seed"),
            (lambda run: (run["workflow_patches"].pop("seed"), run["render"].update(workflow_fixed=["seed"], default_seed=42)), "includes seed, so render.default_seed must be null"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                run_dir = self._run(Path(directory))
                (Path(directory) / "cases" / "render.jsonl").write_text(json.dumps({"id": "a", "values": {"prompt": "x", "negative_prompt": ""}}) + "\n", encoding="utf-8")
                _edit_run(run_dir, change)
                with self.assertRaisesRegex(ValueError, message):
                    compile_render(Path(directory), run_dir)

    def test_an_unknown_sidecar_key_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._run(root)
            (root / "workflows" / "wf.kura.yaml").write_text("lora:\n  insert: {kind: model_only, model_node: '3'}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"wf.kura.yaml has unsupported key\(s\): lora; use lora_insert, models, provenance, source"):
                compile_render(root, run_dir)

    def test_lora_insert_has_one_shape(self) -> None:
        self.assertEqual(
            _lora_insert_from_sidecar({"lora_insert": {"kind": "model_clip", "model_node": "4"}}),
            {"class_type": "LoraLoader", "model_node": "4", "model_output": 0, "strength_model": 0.8, "clip_node": "4", "clip_output": 1, "strength_clip": 0.8},
        )
        for raw, message in (
            ({"type": "model_only", "model_node": "4"}, "kind is model_only or model_clip"),
            ({"kind": "LoraLoader", "model_node": "4"}, "kind is model_only or model_clip"),
            ({"kind": "full", "model_node": "4"}, "kind is model_only or model_clip"),
            ({"model_node": "4"}, "kind is model_only or model_clip"),
            ({"kind": "model_only", "model": {"node": "4"}}, "takes kind, model_node, model_output, strength_model, not model"),
            ({"kind": "model_only", "model_node": 4}, "written as a string, such as '37'"),
            ({"kind": "model_only", "model_node": "4", "clip_node": "5"}, "not clip_node"),
            ({"kind": "model_only", "model_node": "4", "strength_model": "0.8"}, "strength_model must be a number"),
        ):
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, message):
                _lora_insert_from_sidecar({"lora_insert": raw})

    def test_a_local_compile_refuses_a_model_comfyui_does_not_have(self) -> None:
        workflow = {**WORKFLOW, "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "missing.safetensors"}}}
        object_info = {"CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["present.safetensors"]]}}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._run(root)
            (root / "workflows" / "wf.json").write_text(json.dumps(workflow), encoding="utf-8")
            _edit_run(run_dir, lambda run: run["generator"].update(endpoint="http://127.0.0.1:8188"))
            with patch("kura.render._fetch_endpoint_object_info", return_value=object_info), \
                    self.assertRaisesRegex(ValueError, r"does not have model\(s\) this workflow loads: CheckpointLoaderSimple.ckpt_name=missing.safetensors"):
                compile_render(root, run_dir)
            self.assertFalse((run_dir / "resolved" / "manifest.lock.yaml").exists())


if __name__ == "__main__":
    unittest.main()

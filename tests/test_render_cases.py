from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml
from rich.console import Console

import kura.render as render_module
from kura.monitor import collect_run_summaries, render_monitor
from kura.render import authored_cases, compile_render, launch_render


WORKFLOW = {
    "3": {"inputs": {"seed": 0}},
    "6": {"inputs": {"text": ""}},
    "7": {"inputs": {"text": ""}},
    "12": {"inputs": {"lora_name": "old.safetensors"}},
    "17": {"inputs": {"strength": 0.0}},
    "18": {"inputs": {"cfg": 1.0}},
    "19": {"inputs": {"image": "placeholder.png"}},
}

PATCHES = {
    "prompt": {"node": "6", "field": "inputs.text"},
    "negative_prompt": {"node": "7", "field": "inputs.text"},
    "seed": {"node": "3", "field": "inputs.seed"},
    "lora": {"node": "12", "field": "inputs.lora_name"},
    "strength": {"node": "17", "field": "inputs.strength"},
    "cfg": {"node": "18", "field": "inputs.cfg"},
}


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _checkpoint(step: int) -> dict[str, Any]:
    return {
        "id": f"step-{step:04d}",
        "path": f"runs/train-1/outputs/model-step{step:08d}.safetensors",
        "hash": None,
    }


def _case(case_id: str, *, step: int | None, prompt: str, seed: int, strength: float, cfg: float) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": case_id,
        "values": {
            "prompt": prompt,
            "negative_prompt": "text, watermark",
            "seed": seed,
            "strength": strength,
            "cfg": cfg,
        },
        "meta": {
            "step": step,
            "prompt_role": case_id.split("-")[-1],
            "authored_by": "test",
        },
    }
    if step is not None:
        row["checkpoint"] = _checkpoint(step)
    return row


def _workspace(
    root: Path,
    *,
    cases: list[dict[str, Any]] | None,
    prompt_items: list[dict[str, Any]] | None = None,
    shared_checkpoint: dict[str, Any] | None = None,
    patches: dict[str, Any] | None = None,
) -> Path:
    run_dir = root / "runs" / "render-1"
    train_outputs = root / "runs" / "train-1" / "outputs"
    for path in (run_dir / "resolved", train_outputs, root / "workflows", root / "cases", root / "promptsets"):
        path.mkdir(parents=True, exist_ok=True)
    (train_outputs.parent / "run.yaml").write_text("id: train-1\ntype: train\n", encoding="utf-8")
    (root / "workspace.yaml").write_text(
        yaml.safe_dump({
            "comfyui": {
                "lora_dir": str(root / "comfyui" / "models" / "loras"),
                "input_dir": str(root / "comfyui" / "input"),
            },
        }),
        encoding="utf-8",
    )
    (root / "workflows" / "wf.json").write_text(json.dumps(WORKFLOW), encoding="utf-8")
    inputs: dict[str, Any] = {
        "train_run": "train-1",
        "workflow": {"path": "workflows/wf.json", "digest": None},
    }
    if cases is not None:
        cases_path = root / "cases" / "render.jsonl"
        cases_path.write_text("".join(json.dumps(row) + "\n" for row in cases), encoding="utf-8")
        inputs["cases"] = {"path": "cases/render.jsonl", "digest": None}
    if prompt_items is not None:
        promptset_path = root / "promptsets" / "prompts.jsonl"
        promptset_path.write_text("".join(json.dumps(row) + "\n" for row in prompt_items), encoding="utf-8")
        inputs["promptset"] = {"path": "promptsets/prompts.jsonl", "digest": None}
    if shared_checkpoint is not None:
        inputs["checkpoint"] = shared_checkpoint
    run = {
        "schema_version": 1,
        "id": "render-1",
        "type": "render",
        "inputs": inputs,
        "generator": {"name": "comfyui", "endpoint": ""},
        "executor": {"name": "local"},
        "workflow_patches": patches if patches is not None else PATCHES,
        "render": {"output_dir": "samples/images", "timeout_sec": 5, "default_seed": None},
    }
    (run_dir / "run.yaml").write_text(yaml.safe_dump(run, sort_keys=False), encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"state": "draft"}), encoding="utf-8")
    return run_dir


def _write_checkpoint(root: Path, checkpoint: dict[str, Any], payload: bytes) -> None:
    path = root / checkpoint["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _resolved_cases(run_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (run_dir / "resolved" / "cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class RenderCasesCompileTests(unittest.TestCase):
    def test_legacy_image_freeze_uses_exact_source_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            patches = {
                "control_image": {"node": "19", "field": "inputs.image", "type": "image"},
            }
            run_dir = _workspace(
                root,
                cases=None,
                prompt_items=[
                    {"id": "front", "control_image": "front.png"},
                    {"id": "front_seed", "control_image": "front_seed.png"},
                ],
                patches=patches,
            )
            run_path = run_dir / "run.yaml"
            run = yaml.safe_load(run_path.read_text(encoding="utf-8"))
            run["render"]["workflow_fixed"] = ["prompt", "negative_prompt", "seed"]
            run_path.write_text(yaml.safe_dump(run, sort_keys=False), encoding="utf-8")
            (root / "promptsets" / "front.png").write_bytes(b"FRONT")
            (root / "promptsets" / "front_seed.png").write_bytes(b"FRONT-SEED")

            compile_render(root, run_dir)

            frozen_dir = run_dir / "resolved" / "images" / "control_image"
            self.assertEqual((frozen_dir / "front.png").read_bytes(), b"FRONT")
            self.assertEqual((frozen_dir / "front_seed.png").read_bytes(), b"FRONT-SEED")
            cases = _resolved_cases(run_dir)
            self.assertEqual([row["source_id"] for row in cases], ["front", "front_seed"])
            self.assertEqual(
                [row["values"]["control_image"] for row in cases],
                [
                    "resolved/images/control_image/front.png",
                    "resolved/images/control_image/front_seed.png",
                ],
            )
            used = [
                json.loads(line)
                for line in (run_dir / "resolved" / "promptset_used.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [item["control_image"] for item in used],
                [
                    "resolved/images/control_image/front.png",
                    "resolved/images/control_image/front_seed.png",
                ],
            )
            manifest = yaml.safe_load((run_dir / "resolved" / "manifest.lock.yaml").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["promptset_images"]), 2)

    def test_checkpoint_id_must_identify_one_artifact_across_cases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                _case("a", step=200, prompt="first", seed=1, strength=1.0, cfg=4.0),
                _case("b", step=400, prompt="second", seed=2, strength=1.0, cfg=4.0),
            ]
            rows[1]["checkpoint"]["id"] = rows[0]["checkpoint"]["id"]
            run_dir = _workspace(root, cases=rows)
            _write_checkpoint(root, rows[0]["checkpoint"], b"first")
            _write_checkpoint(root, rows[1]["checkpoint"], b"second")

            with self.assertRaisesRegex(ValueError, "one checkpoint id must map to one path and hash"):
                compile_render(root, run_dir)

    def test_compile_freezes_closed_cases_and_checkpoint_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authored_cases = [
                _case("step0200-reconstruction", step=200, prompt="learned outfit", seed=42, strength=1.0, cfg=4.0),
                _case("step0400-transfer", step=400, prompt="everyday clothes", seed=43, strength=0.8, cfg=6.0),
            ]
            run_dir = _workspace(root, cases=authored_cases)
            _write_checkpoint(root, authored_cases[0]["checkpoint"], b"weights-200")
            _write_checkpoint(root, authored_cases[1]["checkpoint"], b"weights-400")
            authored_run = (run_dir / "run.yaml").read_bytes()
            authored_jsonl = (root / "cases" / "render.jsonl").read_bytes()

            compile_render(root, run_dir)

            self.assertEqual((run_dir / "run.yaml").read_bytes(), authored_run)
            self.assertEqual((root / "cases" / "render.jsonl").read_bytes(), authored_jsonl)
            cases = _resolved_cases(run_dir)
            self.assertEqual([row["index"] for row in cases], [1, 2])
            self.assertEqual([row["id"] for row in cases], ["step0200-reconstruction", "step0400-transfer"])
            self.assertEqual(cases[0]["values"], authored_cases[0]["values"])
            self.assertEqual(cases[0]["meta"], authored_cases[0]["meta"])
            self.assertEqual(cases[0]["checkpoint"]["path"], authored_cases[0]["checkpoint"]["path"])
            self.assertEqual(cases[0]["checkpoint"]["hash"], _digest(b"weights-200"))
            self.assertEqual(cases[1]["checkpoint"]["hash"], _digest(b"weights-400"))
            manifest = yaml.safe_load((run_dir / "resolved" / "manifest.lock.yaml").read_text(encoding="utf-8"))
            self.assertEqual(manifest["inputs"]["cases"]["digest"], _digest(authored_jsonl))

    def test_cases_can_use_a_shared_checkpoint_or_be_checkpointless(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                _case("a", step=None, prompt="first", seed=1, strength=1.0, cfg=4.0),
                _case("b", step=None, prompt="second", seed=2, strength=0.8, cfg=5.0),
            ]
            shared = _checkpoint(1800)
            run_dir = _workspace(root, cases=rows, shared_checkpoint=shared)
            _write_checkpoint(root, shared, b"shared")
            compile_render(root, run_dir)
            resolved = _resolved_cases(run_dir)
            self.assertEqual([row["checkpoint"]["path"] for row in resolved], [shared["path"], shared["path"]])
            self.assertEqual({row["checkpoint"]["hash"] for row in resolved}, {_digest(b"shared")})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _workspace(root, cases=rows, shared_checkpoint=None, patches={key: value for key, value in PATCHES.items() if key != "lora"})
            compile_render(root, run_dir)
            self.assertTrue(all("checkpoint" not in row for row in _resolved_cases(run_dir)))

    def test_sidecar_lora_insert_allows_baseline_and_lora_cases_in_one_queue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = _case("baseline", step=None, prompt="same", seed=42, strength=1.0, cfg=4.0)
            lora = _case("lora", step=1800, prompt="same", seed=42, strength=1.0, cfg=4.0)
            patches = {key: value for key, value in PATCHES.items() if key != "lora"}
            run_dir = _workspace(root, cases=[baseline, lora], patches=patches)
            (root / "workflows" / "wf.kura.yaml").write_text(
                "lora_insert:\n  kind: model_only\n  model_node: '12'\n",
                encoding="utf-8",
            )
            _write_checkpoint(root, lora["checkpoint"], b"lora")

            compile_render(root, run_dir)

            resolved = _resolved_cases(run_dir)
            self.assertNotIn("checkpoint", resolved[0])
            self.assertEqual(resolved[1]["checkpoint"]["hash"], _digest(b"lora"))

    def test_cases_and_promptset_are_mutually_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _workspace(
                root,
                cases=[_case("a", step=None, prompt="first", seed=1, strength=1.0, cfg=4.0)],
                prompt_items=[{"id": "legacy", "prompt": "legacy", "seeds": [1], "strength": 1.0, "cfg": 4.0}],
            )
            with self.assertRaises(ValueError) as caught:
                compile_render(root, run_dir)
            self.assertIn("cases", str(caught.exception))
            self.assertIn("promptset", str(caught.exception))
            self.assertFalse((run_dir / "resolved" / "cases.jsonl").exists())

    def test_compile_requires_one_case_source_and_a_closed_cases_descriptor(self) -> None:
        malformed = (
            (None, "exactly one"),
            ("cases/render.jsonl", "mapping"),
            ({"path": None, "digest": None}, "path"),
            ({"path": 7, "digest": None}, "path"),
            ({"path": "", "digest": None}, "path"),
            ({"path": "   ", "digest": None}, "path"),
            ({"path": "cases/render.jsonl", "digest": None, "ignored": True}, "ignored"),
            ({"path": "cases/render.jsonl", "digest": 7}, "digest"),
        )
        for descriptor, expected in malformed:
            with self.subTest(descriptor=descriptor), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run_dir = _workspace(
                    root,
                    cases=[_case("a", step=None, prompt="first", seed=1, strength=1.0, cfg=4.0)],
                    patches={key: value for key, value in PATCHES.items() if key != "lora"},
                )
                run_path = run_dir / "run.yaml"
                run = yaml.safe_load(run_path.read_text(encoding="utf-8"))
                if descriptor is None:
                    run["inputs"].pop("cases")
                else:
                    run["inputs"]["cases"] = descriptor
                run_path.write_text(yaml.safe_dump(run, sort_keys=False), encoding="utf-8")
                with self.assertRaises(ValueError) as caught:
                    compile_render(root, run_dir)
                self.assertIn(expected, str(caught.exception).lower())
                self.assertFalse((run_dir / "resolved" / "cases.jsonl").exists())

    def test_case_checkpoint_and_nonempty_shared_checkpoint_are_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared = _checkpoint(1800)
            row = _case("a", step=200, prompt="first", seed=1, strength=1.0, cfg=4.0)
            run_dir = _workspace(root, cases=[row], shared_checkpoint=shared)
            with self.assertRaises(ValueError) as caught:
                compile_render(root, run_dir)
            self.assertIn("checkpoint", str(caught.exception).lower())
            self.assertIn("case", str(caught.exception).lower())

    def test_empty_shared_checkpoint_does_not_conflict_with_case_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = _case("a", step=200, prompt="first", seed=1, strength=1.0, cfg=4.0)
            run_dir = _workspace(root, cases=[row], shared_checkpoint={})
            _write_checkpoint(root, row["checkpoint"], b"weights")
            compile_render(root, run_dir)
            self.assertEqual(_resolved_cases(run_dir)[0]["checkpoint"]["hash"], _digest(b"weights"))

    def test_compile_rejects_malformed_case_rows_before_freezing(self) -> None:
        good_values = {"prompt": "x", "negative_prompt": "", "seed": 1, "strength": 1.0, "cfg": 4.0}
        malformed = (
            ({"id": "a", "values": good_values, "ignored": True}, "ignored"),
            ({"values": good_values}, "id"),
            ({"id": "../a", "values": good_values}, "safe"),
            ({"id": "a"}, "values"),
            ({"id": "a", "values": []}, "mapping"),
            ({"id": "a", "values": good_values, "meta": []}, "meta"),
            ({"id": "a", "values": good_values, "checkpoint": "model.safetensors"}, "checkpoint"),
            ({"id": "a", "values": good_values, "checkpoint": {"path": "x", "hash": None, "ignored": True}}, "ignored"),
        )
        for row, expected in malformed:
            with self.subTest(row=row), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run_dir = _workspace(root, cases=[row])
                with self.assertRaises(ValueError) as caught:
                    compile_render(root, run_dir)
                self.assertIn(expected, str(caught.exception).lower())
                self.assertFalse((run_dir / "resolved" / "cases.jsonl").exists())

    def test_authored_case_id_errors_name_the_cases_surface(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text(json.dumps({"values": {}}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"cases:1: id is required"):
                authored_cases(path)

            path.write_text(json.dumps({"id": "../bad", "values": {}}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"cases:1: id must be a single safe file name"):
                authored_cases(path)

    def test_compile_rejects_duplicate_ids_and_unknown_values(self) -> None:
        scenarios = (
            ([
                _case("same", step=None, prompt="x", seed=1, strength=1.0, cfg=4.0),
                _case("same", step=None, prompt="y", seed=2, strength=1.0, cfg=4.0),
            ], PATCHES, "duplicate"),
            ([{"id": "a", "values": {"prompt": "x", "seed": 1, "strength": 1.0, "cfg": 4.0, "unknown": 7}}], PATCHES, "unknown"),
        )
        for rows, patches, expected in scenarios:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run_dir = _workspace(root, cases=rows, patches=patches)
                with self.assertRaises(ValueError) as caught:
                    compile_render(root, run_dir)
                self.assertIn(expected, str(caught.exception).lower())

    def test_compile_requires_every_authored_value_to_have_a_workflow_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = _case("a", step=None, prompt="x", seed=1, strength=1.0, cfg=4.0)
            patches = {key: value for key, value in PATCHES.items() if key != "cfg"}
            run_dir = _workspace(root, cases=[row], patches=patches)
            with self.assertRaises(ValueError) as caught:
                compile_render(root, run_dir)
            self.assertIn("cfg", str(caught.exception))
            self.assertFalse((run_dir / "resolved" / "cases.jsonl").exists())

    def test_compile_rejects_unknown_workflow_fixed_names_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _workspace(
                root,
                cases=[_case("a", step=None, prompt="x", seed=1, strength=1.0, cfg=4.0)],
            )
            run_path = run_dir / "run.yaml"
            run = yaml.safe_load(run_path.read_text(encoding="utf-8"))
            run["render"]["workflow_fixed"] = ["width"]
            run_path.write_text(yaml.safe_dump(run, sort_keys=False), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                r"workflow_fixed only accepts .*remove: width",
            ):
                compile_render(root, run_dir)

    def test_compile_normalizes_null_workflow_patches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _workspace(root, cases=[{"id": "fixed", "values": {}}], patches={})
            run_path = run_dir / "run.yaml"
            run = yaml.safe_load(run_path.read_text(encoding="utf-8"))
            run["workflow_patches"] = None
            run["render"]["workflow_fixed"] = ["prompt", "negative_prompt", "seed"]
            run_path.write_text(yaml.safe_dump(run, sort_keys=False), encoding="utf-8")

            compile_render(root, run_dir)

            manifest = yaml.safe_load(
                (run_dir / "resolved" / "manifest.lock.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["workflow_patches"], {})

    def test_compile_rejects_non_mapping_falsy_workflow_patches(self) -> None:
        for patches in ([], False, 0, ""):
            with self.subTest(patches=patches), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run_dir = _workspace(root, cases=[{"id": "fixed", "values": {}}], patches={})
                run_path = run_dir / "run.yaml"
                run = yaml.safe_load(run_path.read_text(encoding="utf-8"))
                run["workflow_patches"] = patches
                run["render"]["workflow_fixed"] = ["prompt", "negative_prompt", "seed"]
                run_path.write_text(yaml.safe_dump(run, sort_keys=False), encoding="utf-8")

                with self.assertRaisesRegex(
                    ValueError,
                    r"workflow_patches must be a mapping",
                ):
                    compile_render(root, run_dir)
                self.assertFalse((run_dir / "resolved" / "cases.jsonl").exists())


class RenderCasesLaunchTests(unittest.TestCase):
    def test_launch_uses_resolved_cases_and_records_case_and_applied_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authored = [
                _case("step0200-reconstruction", step=200, prompt="learned outfit", seed=42, strength=1.0, cfg=4.0),
                _case("step0400-transfer", step=400, prompt="everyday clothes", seed=43, strength=0.8, cfg=6.0),
            ]
            run_dir = _workspace(root, cases=authored)
            for index, row in enumerate(authored):
                _write_checkpoint(root, row["checkpoint"], f"weights-{index}".encode())
            compile_render(root, run_dir)
            frozen = _resolved_cases(run_dir)
            (root / "cases" / "render.jsonl").write_text(
                json.dumps(_case("changed", step=None, prompt="must not render", seed=999, strength=9.0, cfg=9.0)) + "\n",
                encoding="utf-8",
            )
            queued: list[dict[str, Any]] = []
            progress_seen_before_second: list[dict[str, Any]] = []
            lora_dir = root / "comfyui" / "models" / "loras"

            class FakeClient:
                def __init__(self, endpoint: str, timeout: int) -> None:
                    pass

                def lora_names(self) -> set[str]:
                    return {f"Kura_tmp/{path.name}" for path in (lora_dir / "Kura_tmp").glob("*")}

                def queue(self, workflow: dict[str, Any]) -> str:
                    if queued:
                        progress_seen_before_second.append(json.loads((run_dir / "status.json").read_text(encoding="utf-8")))
                    queued.append(workflow)
                    return f"prompt-{len(queued)}"

                def wait(self, prompt_id: str) -> list[dict[str, Any]]:
                    if prompt_id == "prompt-1":
                        return [
                            {"filename": "first.png", "subfolder": "", "type": "output"},
                            {"filename": "second.png", "subfolder": "", "type": "output"},
                        ]
                    return [{"filename": "only.png", "subfolder": "", "type": "output"}]

                def download(self, image: dict[str, Any]) -> bytes:
                    return b"png"

            with patch("kura.render.ComfyUIClient", FakeClient):
                self.assertEqual(launch_render(root, run_dir, endpoint_override="http://127.0.0.1:8188"), 0)

            self.assertEqual(len(queued), 2)
            self.assertEqual([workflow["6"]["inputs"]["text"] for workflow in queued], ["learned outfit", "everyday clothes"])
            self.assertEqual([workflow["3"]["inputs"]["seed"] for workflow in queued], [42, 43])
            self.assertEqual([workflow["17"]["inputs"]["strength"] for workflow in queued], [1.0, 0.8])
            self.assertEqual([workflow["18"]["inputs"]["cfg"] for workflow in queued], [4.0, 6.0])
            self.assertEqual(progress_seen_before_second[0]["last_step"], 1)
            self.assertEqual(progress_seen_before_second[0]["total_steps"], 2)
            self.assertEqual(progress_seen_before_second[0]["current_case_id"], "step0400-transfer")

            records = [json.loads(line) for line in (run_dir / "samples" / "images.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 3)
            self.assertEqual([record["case"]["id"] for record in records], ["step0200-reconstruction", "step0200-reconstruction", "step0400-transfer"])
            self.assertEqual(records[0]["case"], frozen[0])
            self.assertEqual(records[1]["case"], frozen[0])
            self.assertEqual(records[2]["case"], frozen[1])
            self.assertEqual(records[0]["case"]["values"], authored[0]["values"])
            self.assertEqual(
                {key: records[0]["applied_values"][key] for key in ("prompt", "negative_prompt", "seed", "strength", "cfg")},
                authored[0]["values"],
            )
            self.assertTrue(records[0]["applied_values"]["lora"].startswith("Kura_tmp/"))
            final_status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual((final_status["last_step"], final_status["total_steps"]), (2, 2))
            self.assertIsNone(final_status["current_case_id"])
            realization = json.loads((run_dir / final_status["last_realization"]).read_text(encoding="utf-8"))
            self.assertEqual(realization["completed_case_count"], 2)
            self.assertEqual(len(realization["checkpoints"]), 2)
            self.assertEqual(
                [item["hash"] for item in realization["checkpoints"]],
                [_digest(b"weights-0"), _digest(b"weights-1")],
            )
            self.assertTrue(all(item["comfyui_lora_name"].startswith("Kura_tmp/") for item in realization["checkpoints"]))
            self.assertTrue(all(item["lora_stage"] for item in realization["checkpoints"]))

    def test_applied_values_distinguish_frozen_and_staged_image_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = _case("controlled", step=None, prompt="first", seed=1, strength=1.0, cfg=4.0)
            row["values"]["control_image"] = "a/control.png"
            patches = {
                **{key: value for key, value in PATCHES.items() if key != "lora"},
                "control_image": {"node": "19", "field": "inputs.image", "type": "image"},
            }
            run_dir = _workspace(root, cases=[row], patches=patches)
            source = root / "cases" / "a" / "control.png"
            source.parent.mkdir()
            source.write_bytes(b"control")
            compile_render(root, run_dir)
            frozen = _resolved_cases(run_dir)[0]
            self.assertEqual(frozen["values"]["control_image"], "resolved/images/control_image/controlled.png")
            input_dir = root / "comfyui" / "input"
            staged_during_queue: list[bool] = []

            class FakeClient:
                def __init__(self, endpoint: str, timeout: int) -> None:
                    pass

                def queue(self, workflow: dict[str, Any]) -> str:
                    staged_name = workflow["19"]["inputs"]["image"]
                    staged_during_queue.append((input_dir / staged_name).is_file())
                    return "prompt-1"

                def wait(self, prompt_id: str) -> list[dict[str, Any]]:
                    return [{"filename": "out.png", "subfolder": "", "type": "output"}]

                def download(self, image: dict[str, Any]) -> bytes:
                    return b"png"

            with patch("kura.render.ComfyUIClient", FakeClient):
                self.assertEqual(launch_render(root, run_dir, endpoint_override="http://127.0.0.1:8188"), 0)
            record = json.loads((run_dir / "samples" / "images.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(record["case"], frozen)
            self.assertEqual(record["case"]["values"]["control_image"], "resolved/images/control_image/controlled.png")
            self.assertTrue(record["applied_values"]["control_image"].startswith("Kura_tmp/"))
            self.assertEqual(staged_during_queue, [True])
            self.assertEqual(list((input_dir / "Kura_tmp").glob("*")), [])

    def test_runpod_uses_remote_image_names_without_local_staging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = _case("controlled", step=None, prompt="first", seed=1, strength=1.0, cfg=4.0)
            row["values"]["control_image"] = "a/control.png"
            patches = {
                **{key: value for key, value in PATCHES.items() if key != "lora"},
                "control_image": {"node": "19", "field": "inputs.image", "type": "image"},
            }
            run_dir = _workspace(root, cases=[row], patches=patches)
            run_config = yaml.safe_load((run_dir / "run.yaml").read_text(encoding="utf-8"))
            run_config["executor"] = {"name": "runpod"}
            (run_dir / "run.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
            source = root / "cases" / "a" / "control.png"
            source.parent.mkdir()
            source.write_bytes(b"control")
            compile_render(root, run_dir)
            frozen_name = "resolved/images/control_image/controlled.png"
            remote_name = "Kura_tmp/render-1-control.png"
            queued: list[dict[str, Any]] = []

            class FakeClient:
                def __init__(self, endpoint: str, timeout: int) -> None:
                    pass

                def queue(self, workflow: dict[str, Any]) -> str:
                    queued.append(workflow)
                    return "prompt-1"

                def wait(self, prompt_id: str) -> list[dict[str, Any]]:
                    return [{"filename": "out.png", "subfolder": "", "type": "output"}]

                def download(self, image: dict[str, Any]) -> bytes:
                    return b"png"

            with patch("kura.render.ComfyUIClient", FakeClient):
                self.assertEqual(
                    launch_render(
                        root,
                        run_dir,
                        endpoint_override="http://127.0.0.1:8188",
                        executor_name="runpod",
                        image_name_overrides={frozen_name: remote_name},
                    ),
                    0,
                )
            self.assertEqual(queued[0]["19"]["inputs"]["image"], remote_name)
            record = json.loads((run_dir / "samples" / "images.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(record["patch_inputs"]["control_image"], frozen_name)
            self.assertEqual(record["applied_values"]["control_image"], remote_name)

    def test_runpod_rejects_incomplete_image_mapping_before_running(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = _case("controlled", step=None, prompt="first", seed=1, strength=1.0, cfg=4.0)
            row["values"]["control_image"] = "a/control.png"
            patches = {
                **{key: value for key, value in PATCHES.items() if key != "lora"},
                "control_image": {"node": "19", "field": "inputs.image", "type": "image"},
            }
            run_dir = _workspace(root, cases=[row], patches=patches)
            run_config = yaml.safe_load((run_dir / "run.yaml").read_text(encoding="utf-8"))
            run_config["executor"] = {"name": "runpod"}
            (run_dir / "run.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
            source = root / "cases" / "a" / "control.png"
            source.parent.mkdir()
            source.write_bytes(b"control")
            compile_render(root, run_dir)

            with self.assertRaisesRegex(ValueError, "remote image mapping"):
                launch_render(
                    root,
                    run_dir,
                    endpoint_override="http://127.0.0.1:8188",
                    executor_name="runpod",
                    image_name_overrides={},
                )
            state = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(state["state"], "compiled")

    def test_failure_preserves_logical_progress_after_completed_cases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                _case("a", step=None, prompt="first", seed=1, strength=1.0, cfg=4.0),
                _case("b", step=None, prompt="second", seed=2, strength=1.0, cfg=4.0),
            ]
            patches = {key: value for key, value in PATCHES.items() if key != "lora"}
            run_dir = _workspace(root, cases=rows, patches=patches)
            compile_render(root, run_dir)

            class FailingClient:
                def __init__(self, endpoint: str, timeout: int) -> None:
                    pass

                def queue(self, workflow: dict[str, Any]) -> str:
                    return "first" if workflow["6"]["inputs"]["text"] == "first" else "second"

                def wait(self, prompt_id: str) -> list[dict[str, Any]]:
                    if prompt_id == "second":
                        raise RuntimeError("case failed")
                    return [
                        {"filename": "one.png", "subfolder": "", "type": "output"},
                        {"filename": "two.png", "subfolder": "", "type": "output"},
                    ]

                def download(self, image: dict[str, Any]) -> bytes:
                    return b"png"

            with patch("kura.render.ComfyUIClient", FailingClient):
                self.assertEqual(launch_render(root, run_dir, endpoint_override="http://127.0.0.1:8188"), 1)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "failed")
            self.assertEqual((status["last_step"], status["total_steps"], status["current_case_id"]), (1, 2, "b"))
            realization = json.loads((run_dir / status["last_realization"]).read_text(encoding="utf-8"))
            self.assertEqual(realization["completed_case_count"], 1)
            self.assertEqual(realization["failed_case_id"], "b")
            self.assertEqual(realization["generated_image_count"], 2)

    def test_failure_falls_back_when_normal_failed_status_update_breaks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = _case("broken", step=None, prompt="first", seed=1, strength=1.0, cfg=4.0)
            patches = {key: value for key, value in PATCHES.items() if key != "lora"}
            run_dir = _workspace(root, cases=[row], patches=patches)
            compile_render(root, run_dir)

            class FailingClient:
                def __init__(self, endpoint: str, timeout: int) -> None:
                    pass

                def queue(self, workflow: dict[str, Any]) -> str:
                    raise RuntimeError("queue failed")

            real_status = render_module.status

            def status_with_broken_failed_update(path: Path, **changes: Any) -> None:
                if changes.get("state") == "failed":
                    raise json.JSONDecodeError("corrupt status", "", 0)
                real_status(path, **changes)

            with (
                patch("kura.render.ComfyUIClient", FailingClient),
                patch("kura.render.status", side_effect=status_with_broken_failed_update),
            ):
                self.assertEqual(launch_render(root, run_dir, endpoint_override="http://127.0.0.1:8188"), 1)

            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "failed")
            self.assertEqual((status["last_step"], status["total_steps"]), (0, 1))
            realization = json.loads((run_dir / status["last_realization"]).read_text(encoding="utf-8"))
            self.assertEqual(realization["failed_case_id"], "broken")
            self.assertEqual(realization["completed_case_count"], 0)

    def test_failure_logs_when_failed_realization_cannot_be_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = _case("broken", step=None, prompt="first", seed=1, strength=1.0, cfg=4.0)
            patches = {key: value for key, value in PATCHES.items() if key != "lora"}
            run_dir = _workspace(root, cases=[row], patches=patches)
            compile_render(root, run_dir)

            class FailingClient:
                def __init__(self, endpoint: str, timeout: int) -> None:
                    pass

                def queue(self, workflow: dict[str, Any]) -> str:
                    raise RuntimeError("queue failed")

            with (
                patch("kura.render.ComfyUIClient", FailingClient),
                patch("kura.render.write_realization", side_effect=RuntimeError("realization broke")),
            ):
                self.assertEqual(launch_render(root, run_dir, endpoint_override="http://127.0.0.1:8188"), 1)

            stdout = (run_dir / "logs" / "stdout.log").read_text(encoding="utf-8")
            self.assertIn(
                "warning: failed to persist render failure realization: RuntimeError: realization broke",
                stdout,
            )

    def test_legacy_checkpoint_and_promptset_compile_to_cases_and_keep_output_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = _checkpoint(1800)
            run_dir = _workspace(
                root,
                cases=None,
                prompt_items=[{"id": "legacy", "prompt": "hello", "negative_prompt": "", "seeds": [42, 43], "strength": 1.0, "cfg": 4.0}],
                shared_checkpoint=checkpoint,
            )
            _write_checkpoint(root, checkpoint, b"legacy")
            compile_render(root, run_dir)
            frozen = _resolved_cases(run_dir)
            self.assertEqual([row["index"] for row in frozen], [1, 2])
            self.assertEqual([row["values"]["seed"] for row in frozen], [42, 43])
            self.assertEqual({row["checkpoint"]["hash"] for row in frozen}, {_digest(b"legacy")})

            class FakeClient:
                def __init__(self, endpoint: str, timeout: int) -> None:
                    pass

                def queue(self, workflow: dict[str, Any]) -> str:
                    return f"prompt-{workflow['3']['inputs']['seed']}"

                def wait(self, prompt_id: str) -> list[dict[str, Any]]:
                    return [{"filename": "out.png", "subfolder": "", "type": "output"}]

                def download(self, image: dict[str, Any]) -> bytes:
                    return b"png"

            with patch("kura.render.ComfyUIClient", FakeClient):
                self.assertEqual(
                    launch_render(root, run_dir, endpoint_override="http://127.0.0.1:8188", manage_lora_stage=False),
                    0,
                )
            records = [json.loads(line) for line in (run_dir / "samples" / "images.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([record["file"] for record in records], ["samples/images/legacy_seed42_0.png", "samples/images/legacy_seed43_0.png"])
            self.assertEqual([record["case"] for record in records], frozen)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            realization = json.loads((run_dir / status["last_realization"]).read_text(encoding="utf-8"))
            self.assertEqual(realization["checkpoint_hash"], _digest(b"legacy"))
            self.assertIn("comfyui_lora_name", realization)
            self.assertIn("lora_stage", realization)
            self.assertEqual(realization["promptset_digest"], realization["cases_digest"])
            stdout = (run_dir / "logs" / "stdout.log").read_text(encoding="utf-8")
            self.assertEqual(stdout.count("queued legacy seed="), 2)
            self.assertNotIn("queued case=", stdout)


class RenderCasesMonitorTests(unittest.TestCase):
    def test_monitor_shows_one_render_run_with_logical_case_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "render-1"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "run.yaml").write_text(
                yaml.safe_dump({"id": "render-1", "type": "render", "executor": {"name": "local"}}),
                encoding="utf-8",
            )
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(
                yaml.safe_dump({"id": "render-1", "type": "render", "executor": {"name": "local"}}),
                encoding="utf-8",
            )
            (run_dir / "resolved" / "cases.jsonl").write_text(
                "".join(json.dumps({"id": f"case-{index}", "index": index, "values": {"prompt": str(index)}}) + "\n" for index in range(3)),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(
                json.dumps({"state": "running", "last_step": 1, "total_steps": 3, "current_case_id": "case-1"}),
                encoding="utf-8",
            )

            summaries = collect_run_summaries(root)
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0].id, "render-1")
            self.assertEqual((summaries[0].progress.step, summaries[0].progress.total), (1, 3))
            self.assertEqual(summaries[0].progress.current_case_id, "case-1")
            console = Console(file=io.StringIO(), record=True, width=180, color_system=None)
            console.print(render_monitor(root))
            rendered = console.export_text()
            self.assertEqual(rendered.count("render-1"), 1)
            self.assertIn("1/3", rendered)
            self.assertIn("case-1", rendered)


if __name__ == "__main__":
    unittest.main()

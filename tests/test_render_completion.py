from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from kura.run_commands.render_completion import format_render_completion


class RenderCompletionTests(unittest.TestCase):
    def test_completion_surfaces_provenance_without_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_dir = root / "runs" / "train-1"
            render_dir = root / "runs" / "render-1"
            (render_dir / "resolved").mkdir(parents=True)
            (render_dir / "samples").mkdir()
            train_dir.mkdir(parents=True)
            (train_dir / "status.json").write_text(
                json.dumps({
                    "state": "completed",
                    "last_step": 1,
                    "outputs": ["outputs/final.safetensors"],
                }),
                encoding="utf-8",
            )
            (render_dir / "status.json").write_text(
                json.dumps({
                    "state": "completed",
                    "exit_code": 0,
                    "started": "2026-08-15T11:00:00+09:00",
                    "ended": "2026-08-15T11:00:40+09:00",
                }),
                encoding="utf-8",
            )
            (render_dir / "resolved" / "manifest.lock.yaml").write_text(
                yaml.safe_dump({
                    "inputs": {
                        "train_run": "train-1",
                        "checkpoint": {"path": "runs/train-1/outputs/final.safetensors", "hash": "sha256:secret"},
                        "workflow": {"path": "workflows/sd15_api.json", "digest": "sha256:workflow"},
                    },
                    "render": {"output_dir": "samples/images"},
                }),
                encoding="utf-8",
            )
            record = {
                "file": "samples/images/case-1_seed42_0.png",
                "prompt_id": "case-1",
                "prompt": "a tiny synthetic smoke image",
                "negative_prompt": "text, watermark",
                "seed": 42,
                "checkpoint_hash": "sha256:checkpoint",
                "workflow_digest": "sha256:workflow",
                "checkpoint_application": {
                    "kind": "lora_insert",
                    "class_type": "LoraLoader",
                    "strength_model": 0.8,
                    "strength_clip": 0.8,
                },
            }
            (render_dir / "samples" / "images.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

            output = format_render_completion(root, render_dir, exit_code=0)

            self.assertIn("completed  exit 0  40s", output)
            self.assertIn("rendered   1 image  samples/images", output)
            self.assertIn("artifact   train-1 · final.safetensors · final step 1", output)
            self.assertIn("workflow   sd15_api.json", output)
            self.assertIn("applied    LoRA model+CLIP · strength model 0.8 / clip 0.8", output)
            self.assertIn('prompt    "a tiny synthetic smoke image"', output)
            self.assertIn('negative  "text, watermark"', output)
            self.assertNotIn("sha256", output)
            self.assertNotIn("digest", output)

    def test_multiple_cases_report_variation_and_provenance_without_sampling_cases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            render_dir = root / "runs" / "render-1"
            (render_dir / "resolved").mkdir(parents=True)
            (render_dir / "samples").mkdir()
            (render_dir / "status.json").write_text(
                json.dumps({"state": "completed", "exit_code": 0}),
                encoding="utf-8",
            )
            (render_dir / "resolved" / "manifest.lock.yaml").write_text(
                yaml.safe_dump({
                    "inputs": {
                        "workflow": {"path": "workflows/sd15_api.json"},
                        "promptset": {"path": "promptsets/evaluation.jsonl"},
                    },
                    "render": {"output_dir": "samples/images"},
                }),
                encoding="utf-8",
            )
            records = [
                {"file": "samples/images/a.png", "prompt_id": "a", "prompt": "first arbitrary prompt", "negative_prompt": "text", "seed": 1},
                {"file": "samples/images/b.png", "prompt_id": "b", "prompt": "second arbitrary prompt", "negative_prompt": "text", "seed": 2},
            ]
            (render_dir / "samples" / "images.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            output = format_render_completion(root, render_dir, exit_code=0)

            self.assertIn("inputs     2 cases", output)
            self.assertIn("prompts   vary by case · evaluation.jsonl", output)
            self.assertIn('negative  "text"', output)
            self.assertIn("seeds     vary by case", output)
            self.assertIn("provenance samples/images.jsonl", output)
            self.assertNotIn("first arbitrary prompt", output)
            self.assertNotIn("second arbitrary prompt", output)


if __name__ == "__main__":
    unittest.main()

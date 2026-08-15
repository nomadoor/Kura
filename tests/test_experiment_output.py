"""Agent-facing experiment context and completion output tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from kura.run_commands.experiment import experiment_context, format_experiment_context, format_run_completion
from kura.run_commands.plan import format_run_plan


class ExperimentOutputTests(unittest.TestCase):
    @staticmethod
    def _train_run(
        root: Path,
        run_id: str,
        *,
        learning_rate: float,
        steps: int,
        state: str = "completed",
        note: str = "# Notes\n",
    ) -> Path:
        run_dir = root / "runs" / run_id
        (run_dir / "resolved").mkdir(parents=True)
        (run_dir / "run.yaml").write_text(
            yaml.safe_dump(
                {
                    "id": run_id,
                    "type": "train",
                    "experiment": "anima-control",
                    "created": f"2026-08-{1 if run_id == 'older' else 2:02d}T00:00:00+00:00",
                    "intent": "Compare the control recipe.",
                    "model": {"base": "anima"},
                    "backend": {"name": "sd-scripts", "config": {"architecture": "anima"}},
                    "recipe": {"steps": steps},
                    "compute": {"executor": "docker"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (run_dir / "resolved" / "backend-display.lock.json").write_text(
            json.dumps({"architecture": "anima", "learning_rate": learning_rate, "rank": 8}),
            encoding="utf-8",
        )
        (run_dir / "status.json").write_text(json.dumps({"state": state}), encoding="utf-8")
        (run_dir / "notes.md").write_text(note, encoding="utf-8")
        return run_dir

    def test_context_shows_varying_facts_note_excerpt_and_completed_renders(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._train_run(
                root,
                "older",
                learning_rate=1e-4,
                steps=100,
                note="# Notes\n\n## Review\n\n- Structure is sound but control is weak\n  on diagonal lines.\n",
            )
            self._train_run(root, "current", learning_rate=7e-5, steps=200)
            render_dir = root / "runs" / "render-current"
            render_dir.mkdir(parents=True)
            (render_dir / "run.yaml").write_text(
                yaml.safe_dump({"id": "render-current", "type": "render", "inputs": {"train_run": "current"}}),
                encoding="utf-8",
            )
            (render_dir / "status.json").write_text(json.dumps({"state": "completed"}), encoding="utf-8")

            context = experiment_context(root, "current")

        assert context is not None
        self.assertEqual(context["name"], "anima-control")
        self.assertEqual(context["varying_facts"], ["lr", "steps"])
        older, current = context["runs"]
        self.assertEqual(older["note"], "Structure is sound but control is weak on diagonal lines.")
        self.assertIsNone(current["note"])
        self.assertEqual(current["completed_render_runs"], 1)
        output = format_experiment_context(context)
        self.assertIn("Experiment anima-control", output)
        self.assertIn("lr 0.0001", output)
        self.assertIn("steps 200", output)
        self.assertIn("| —  <- this run", output)
        self.assertIn("completed render runs for this run  1", output)

    def test_completion_compresses_checkpoint_paths_and_includes_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._train_run(root, "current", learning_rate=7e-5, steps=200)
            status = {
                "state": "completed",
                "exit_code": 0,
                "started": "2026-08-15T00:00:00.000000000Z",
                "ended": "2026-08-15T05:21:03.123456789Z",
                "outputs": [
                    "outputs/control-step00000100.safetensors",
                    "outputs/control-step00000200.safetensors",
                    "outputs/control.safetensors",
                ],
            }

            output = format_run_completion(root, run_dir, status)

        self.assertIn("completed  exit 0  5h 21m", output)
        self.assertIn("intent     Compare the control recipe.", output)
        self.assertIn("produced   3 checkpoints  step 100-200", output)
        self.assertIn("final  control.safetensors", output)
        self.assertNotIn("outputs/control-step", output)
        self.assertIn("Experiment anima-control", output)

    def test_plan_text_prints_normalized_resources_and_keeps_experiment_near_top(self) -> None:
        payload = {
            "id": "current",
            "type": "train",
            "source": "runs/current/run.yaml",
            "intent_source": "runs/current/run.yaml",
            "compiled": False,
            "backend": {"name": "sd-scripts", "config": {"learning_rate": 7e-5}},
            "model": {"base": "anima"},
            "compute": {"executor": "docker"},
            "datasets": [],
            "recipe": {"steps": 1},
            "resources": {
                "hardware": {},
                "executor": {},
                "model": {},
                "training": {"learning_rate": 7e-5, "rank": 8},
                "memory": {"fp8_base": False, "blocks_to_swap": 0},
            },
            "experiment": {
                "name": "anima-control",
                "current_run": "current",
                "varying_facts": [],
                "runs": [{"id": "current", "state": "compiled", "current": True, "differences": {}, "note": None, "completed_render_runs": 0}],
            },
        }

        output = format_run_plan(payload)

        self.assertLess(output.index("Experiment anima-control"), output.index("Backend\n"))
        self.assertNotIn("Backend config", output)
        self.assertEqual(output.count("learning_rate"), 1)
        self.assertIn("rank         8", output)
        self.assertIn("fp8_base     False", output)
        self.assertIn("blocks_to_swap 0", output)


if __name__ == "__main__":
    unittest.main()

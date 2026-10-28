"""A training run's results are found in one way: the render runs that name it, and its folder under the workspace's comparisons/."""

from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from kura import run_results


def _run(root: Path, run_id: str, run: dict, *, state: str, manifest: dict | None = None) -> Path:
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.yaml").write_text(yaml.safe_dump({"id": run_id, **run}), encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"state": state}), encoding="utf-8")
    if manifest is not None:
        (run_dir / "resolved").mkdir()
        (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump({"id": run_id, **manifest}), encoding="utf-8")
    return run_dir


def _workspace(root: Path) -> Path:
    (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
    train = _run(root, "20261027-0900_train_aaaa", {"type": "train", "experiment": "e"}, state="completed")
    _run(root, "20261027-1000_steps_bbbb", {"type": "render", "inputs": {"train_run": train.name}}, state="completed")
    _run(root, "20261027-1100_sheet_cccc", {"type": "render", "inputs": {"train_run": train.name}}, state="running",
         manifest={"type": "render", "render": {"output_dir": "samples/grid"}})
    _run(root, "20261027-1200_other_dddd", {"type": "render", "inputs": {"train_run": "20261027-0800_else_eeee"}}, state="completed")
    _run(root, "20261027-1300_plain_ffff", {"type": "render", "inputs": {"train_run": None}}, state="completed")
    return train


class TrainRunResultsTests(unittest.TestCase):
    def test_a_training_run_lists_the_renders_that_name_it_and_its_comparisons(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            train = _workspace(Path(directory))
            results = run_results.train_run_results(train)
            self.assertEqual(results, {
                "renders": [
                    {"id": "20261027-1000_steps_bbbb", "state": "completed", "images": "runs/20261027-1000_steps_bbbb/samples/images"},
                    {"id": "20261027-1100_sheet_cccc", "state": "running", "images": "runs/20261027-1100_sheet_cccc/samples/grid"},
                ],
                "comparisons": None,
            })
            # A sheet is not part of the run's record: it lives outside the run, under the workspace.
            (train / "comparisons").mkdir()
            self.assertIsNone(run_results.train_run_results(train)["comparisons"])
            (Path(directory) / "comparisons" / train.name).mkdir(parents=True)
            self.assertEqual(run_results.train_run_results(train)["comparisons"], f"comparisons/{train.name}")

    def test_a_render_run_has_no_such_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _workspace(root)
            self.assertIsNone(run_results.train_run_results(root / "runs" / "20261027-1000_steps_bbbb"))

    def test_status_shows_them_for_a_training_run(self) -> None:
        from kura.cli import cmd_run_status

        with tempfile.TemporaryDirectory() as directory:
            train = _workspace(Path(directory))
            stdout = io.StringIO()
            with patch("kura.cli._run_path", return_value=train), patch("sys.stdout", stdout):
                cmd_run_status(argparse.Namespace(run_id=train.name))
            summary = json.loads(stdout.getvalue())["summary"]
            self.assertEqual([render["id"] for render in summary["renders"]], ["20261027-1000_steps_bbbb", "20261027-1100_sheet_cccc"])
            self.assertIsNone(summary["comparisons"])
            (Path(directory) / "comparisons" / train.name).mkdir(parents=True)
            stdout = io.StringIO()
            with patch("kura.cli._run_path", return_value=train), patch("sys.stdout", stdout):
                cmd_run_status(argparse.Namespace(run_id=train.name))
            self.assertEqual(json.loads(stdout.getvalue())["summary"]["comparisons"], f"comparisons/{train.name}")

    def test_status_open_and_the_monitor_find_the_sheets_through_one_owner(self) -> None:
        from kura.monitor import collect_run_summary
        from kura.tui import _path_targets

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = _workspace(root)
            with patch.object(run_results, "comparisons_dir", wraps=run_results.comparisons_dir) as owner:
                run_results.train_run_results(train)
                run_results.results_folder(train)
                _path_targets(collect_run_summary(root, train.name))
            self.assertEqual(owner.call_count, 3)

    def test_the_experiment_counts_completed_renders_from_the_same_list(self) -> None:
        from kura.run_commands import experiment

        with tempfile.TemporaryDirectory() as directory:
            train = _workspace(Path(directory))
            with patch.object(experiment, "render_runs_by_training_run", wraps=run_results.render_runs_by_training_run) as owner:
                context = experiment.experiment_context(Path(directory), train.name)
            owner.assert_called()
            self.assertEqual(context["runs"][0]["completed_render_runs"], 1)

    def test_the_monitor_offers_them_as_paths_to_open(self) -> None:
        from kura.monitor import collect_run_summary
        from kura.tui import _path_targets

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = _workspace(root)
            (root / "comparisons" / train.name).mkdir(parents=True)
            labels = {target.label: target.path for target in _path_targets(collect_run_summary(root, train.name))}
            self.assertEqual(labels["comparisons"], root / "comparisons" / train.name)
            self.assertEqual(labels["render 20261027-1000_steps_bbbb"], root / "runs" / "20261027-1000_steps_bbbb" / "samples" / "images")
            self.assertEqual(labels["render 20261027-1100_sheet_cccc"], root / "runs" / "20261027-1100_sheet_cccc" / "samples" / "grid")

    def test_a_render_writes_to_the_folder_the_list_names(self) -> None:
        from kura import render

        self.assertIs(render.render_output_dir, run_results.render_output_dir)


if __name__ == "__main__":
    unittest.main()

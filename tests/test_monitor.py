"""Tests for read-only run monitoring projections."""

from __future__ import annotations

import json
import io
import subprocess
import tempfile
import unittest
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from kura.monitor import RunSummary, _format_time_cell, _split_for_monitor, collect_run_summaries, loss_sparkline, render_monitor


class MonitorProjectionTests(unittest.TestCase):
    def test_active_time_shows_total_elapsed_and_update_age(self) -> None:
        now = datetime.now().astimezone()
        summary = RunSummary(
            id="local",
            experiment=None,
            type="train",
            executor="docker",
            state="running",
            started=now - timedelta(minutes=10),
            last_updated=now - timedelta(seconds=5),
        )

        rendered = _format_time_cell(summary)

        self.assertIn("10m0s elapsed", rendered)
        self.assertIn("ago", rendered)

    def test_legacy_run_is_isolated_as_unreadable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "runs" / "legacy"
            current = root / "runs" / "current"
            legacy.mkdir(parents=True)
            current.mkdir(parents=True)
            (legacy / "run.yaml").write_text("id: legacy\ntype: train\nparams: {steps: 1}\n", encoding="utf-8")
            (current / "run.yaml").write_text("id: current\ntype: train\nrecipe: {steps: 1, seed: 1}\n", encoding="utf-8")
            summaries = {item.id: item for item in collect_run_summaries(root)}
        self.assertEqual(summaries["legacy"].state, "unreadable")
        self.assertNotEqual(summaries["current"].state, "unreadable")

    def test_ai_toolkit_display_projection_is_not_read_as_musubi(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "ai"
            run_dir.mkdir(parents=True)
            run = {"id": "ai", "type": "train", "recipe": {"steps": 10, "seed": 1}, "backend": {"name": "ai-toolkit", "config": {"config": {"network": {"linear": 8, "linear_alpha": 4}, "train": {"lr": 0.0001, "batch_size": 2}}}}}
            (run_dir / "run.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
            summary = collect_run_summaries(root)[0]
        self.assertEqual(summary.key_config["rank"], 8)
        self.assertEqual(summary.key_config["alpha"], 4)
        self.assertEqual(summary.key_config["lr"], 0.0001)
        self.assertEqual(summary.key_config["batch_size"], 2)
    def test_sparkline_tracks_increasing_values(self) -> None:
        line = loss_sparkline([1, 2, 3, 4], width=4)
        self.assertEqual(line, "▁▃▆█")

    def test_collect_run_summaries_tolerates_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "missing-pieces"
            run_dir.mkdir(parents=True)
            (run_dir / "run.yaml").write_text("id: missing-pieces\ntype: train\n", encoding="utf-8")

            summaries = collect_run_summaries(root)

            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0].id, "missing-pieces")
            self.assertIsNone(summaries[0].state)

    def test_collect_run_summaries_reads_config_status_and_loss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "train-1"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "metrics").mkdir()
            (run_dir / "logs").mkdir()
            (root / "index.jsonl").write_text(json.dumps({"id": "train-1"}) + "\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text(
                "\n".join(
                    [
                        "id: train-1",
                        "type: train",
                        "experiment: exp",
                        "created: '2026-06-21T10:00:00+09:00'",
                        "datasets: [{id: tiny}]",
                        "recipe: {steps: 3}",
                        "backend: {name: musubi-tuner, config: {network_dim: 4, learning_rate: 0.0001}}",
                        "compute: {executor: docker}",
                    ]
                ),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(
                json.dumps({"state": "running", "started": "2026-06-21T10:01:00+09:00", "last_step": 0, "total_steps": 3, "exit_code": 0}),
                encoding="utf-8",
            )
            (run_dir / "metrics" / "metrics.jsonl").write_text(
                "\n".join(json.dumps({"loss": value}) for value in (0.9, 0.7, 0.8)) + "\n",
                encoding="utf-8",
            )

            summary = collect_run_summaries(root, loss_tail=2)[0]

            self.assertEqual(summary.id, "train-1")
            self.assertEqual(summary.experiment, "exp")
            self.assertEqual(summary.executor, "docker")
            self.assertEqual(summary.state, "running")
            self.assertEqual(summary.key_config["rank"], 4)
            self.assertEqual(summary.progress.step, 0)
            self.assertEqual(summary.progress.total, 3)
            self.assertEqual(summary.exit_code, 0)
            self.assertEqual(summary.losses, (0.7, 0.8))
            self.assertEqual(summary.latest_loss, 0.8)
            self.assertEqual(summary.best_loss, 0.7)

    def test_monitor_sort_tolerates_mixed_timezone_datetimes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aware = root / "runs" / "aware"
            naive = root / "runs" / "naive"
            aware.mkdir(parents=True)
            naive.mkdir(parents=True)
            (aware / "run.yaml").write_text("id: aware\ntype: train\ncreated: '2026-06-21T10:00:00+09:00'\n", encoding="utf-8")
            (aware / "status.json").write_text(json.dumps({"state": "completed"}), encoding="utf-8")
            (naive / "run.yaml").write_text("id: naive\ntype: train\ncreated: '2026-06-21T10:00:00'\n", encoding="utf-8")
            (naive / "status.json").write_text(json.dumps({"state": "completed"}), encoding="utf-8")

            summaries = collect_run_summaries(root)
            active, history = _split_for_monitor(summaries, limit=10)

            self.assertEqual(active, [])
            self.assertEqual({summary.id for summary in history}, {"aware", "naive"})

    def test_render_monitor_distinguishes_hidden_drafts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            draft = root / "runs" / "draft-run"
            complete = root / "runs" / "complete-run"
            draft.mkdir(parents=True)
            complete.mkdir(parents=True)
            (draft / "run.yaml").write_text("id: draft-run\ntype: train\n", encoding="utf-8")
            (draft / "status.json").write_text(json.dumps({"state": "draft"}), encoding="utf-8")
            (complete / "run.yaml").write_text("id: complete-run\ntype: train\n", encoding="utf-8")
            (complete / "status.json").write_text(json.dumps({"state": "completed"}), encoding="utf-8")

            console = Console(file=io.StringIO(), record=True, width=120, color_system=None)
            console.print(render_monitor(root))
            hidden_text = console.export_text()

            self.assertIn("complete-run", hidden_text)
            self.assertNotIn("draft-run", hidden_text)
            self.assertIn("1 draft run(s) hidden (--all to show)", hidden_text)

            console = Console(file=io.StringIO(), record=True, width=120, color_system=None)
            console.print(render_monitor(root, include_drafts=True))
            all_text = console.export_text()

            self.assertIn("draft-run", all_text)
            self.assertNotIn("draft run(s) hidden", all_text)

    def test_render_monitor_omits_watch_hint_when_only_drafts_are_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            draft = root / "runs" / "draft-run"
            draft.mkdir(parents=True)
            (draft / "run.yaml").write_text("id: draft-run\ntype: train\n", encoding="utf-8")
            (draft / "status.json").write_text(json.dumps({"state": "draft"}), encoding="utf-8")

            console = Console(file=io.StringIO(), record=True, width=120, color_system=None)
            console.print(render_monitor(root))
            text = console.export_text()

            self.assertIn("1 draft run(s) hidden (--all to show)", text)
            self.assertNotIn("watch: uv run kura run watch", text)

    def test_render_samples_images_are_reported_as_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "render-1"
            (run_dir / "samples" / "images").mkdir(parents=True)
            (run_dir / "run.yaml").write_text("id: render-1\ntype: render\ncreated: '2026-06-21T10:00:00+09:00'\n", encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "completed"}), encoding="utf-8")
            (run_dir / "samples" / "images" / "image.png").write_bytes(b"png")

            summaries = collect_run_summaries(root)

            self.assertEqual(summaries[0].outputs_path, run_dir / "samples" / "images")

    def test_collect_run_summaries_overlays_finished_local_docker_state_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "docker-finished"
            (run_dir / "realizations").mkdir(parents=True)
            (root / "index.jsonl").write_text(json.dumps({"id": "docker-finished"}) + "\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text(
                "\n".join(
                    [
                        "id: docker-finished",
                        "type: train",
                        "backend: {name: ai-toolkit}",
                        "compute: {executor: docker}",
                    ]
                ),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(
                json.dumps({"state": "running", "container_id": "container-1", "last_realization": "realizations/r1.json"}),
                encoding="utf-8",
            )
            (run_dir / "realizations" / "r1.json").write_text(
                json.dumps({"id": "r1", "executor": "docker", "state": "running", "container": {"id": "container-1"}}),
                encoding="utf-8",
            )
            result = subprocess.CompletedProcess([], 0, '{"Running": false, "ExitCode": 0, "FinishedAt": "2026-06-29T01:02:03Z"}', "")

            with patch("kura.monitor.shutil.which", return_value="/usr/bin/docker"), patch("kura.monitor.subprocess.run", return_value=result) as run:
                summary = collect_run_summaries(root)[0]

            run.assert_called_once_with(["/usr/bin/docker", "inspect", "--format", "{{json .State}}", "container-1"], text=True, capture_output=True, check=False, timeout=2)
            self.assertEqual(summary.state, "completed")
            self.assertEqual(summary.exit_code, 0)
            self.assertIsNotNone(summary.ended)
            self.assertEqual(json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["state"], "running")

    def test_collect_run_summaries_ignores_timed_out_docker_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "docker-timeout"
            (run_dir / "realizations").mkdir(parents=True)
            (root / "index.jsonl").write_text(json.dumps({"id": "docker-timeout"}) + "\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text("id: docker-timeout\ntype: train\ncompute: {executor: docker}\n", encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "running", "container_id": "container-1"}), encoding="utf-8")

            with patch("kura.monitor.shutil.which", return_value="/usr/bin/docker"), patch("kura.monitor.subprocess.run", side_effect=subprocess.TimeoutExpired(["docker"], 2)):
                summary = collect_run_summaries(root)[0]

            self.assertEqual(summary.state, "running")

    def test_collect_run_summaries_falls_back_to_ai_toolkit_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "stdout-train"
            (run_dir / "metrics").mkdir(parents=True)
            (run_dir / "logs").mkdir()
            (run_dir / "run.yaml").write_text(
                "\n".join(
                    [
                        "id: stdout-train",
                        "type: train",
                        "recipe: {steps: 30}",
                        "compute: {executor: docker}",
                    ]
                ),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(json.dumps({"state": "completed", "last_step": 0, "exit_code": 0}), encoding="utf-8")
            (run_dir / "metrics" / "metrics.jsonl").write_text("", encoding="utf-8")
            (run_dir / "logs" / "stdout.log").write_text(
                "\rstdout-train:  3%|▎| 1/30 [00:11<05:36, lr: 1.0e-04 loss: 3.825e-01]"
                "\rstdout-train: 97%|█| 29/30 [01:34<00:03, lr: 1.0e-04 loss: 8.186e-01]\n",
                encoding="utf-8",
            )

            summary = collect_run_summaries(root)[0]

            self.assertEqual(summary.progress.step, 30)
            self.assertEqual(summary.progress.total, 30)
            self.assertEqual(summary.losses, (0.3825, 0.8186))
            self.assertEqual(summary.best_loss, 0.3825)

    def test_collect_run_summaries_falls_back_to_musubi_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "musubi-train"
            (run_dir / "metrics").mkdir(parents=True)
            (run_dir / "logs").mkdir()
            (run_dir / "run.yaml").write_text(
                "\n".join(
                    [
                        "id: musubi-train",
                        "type: train",
                        "backend: {name: musubi-tuner}",
                        "recipe: {steps: 100}",
                        "compute: {executor: docker}",
                    ]
                ),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(json.dumps({"state": "completed", "exit_code": 0}), encoding="utf-8")
            (run_dir / "metrics" / "metrics.jsonl").write_text("", encoding="utf-8")
            (run_dir / "logs" / "stdout.log").write_text(
                "\rsteps:  99%|█████████▉| 99/100 [04:13<00:02,  2.56s/it, avr_loss=0.316]\n"
                "\rsteps: 100%|██████████| 100/100 [04:16<00:00,  2.56s/it, avr_loss=0.321]\n",
                encoding="utf-8",
            )

            summary = collect_run_summaries(root)[0]

            self.assertEqual(summary.progress.step, 100)
            self.assertEqual(summary.progress.total, 100)
            self.assertEqual(summary.progress.seconds_per_iter, 2.56)
            self.assertEqual(summary.losses, (0.316, 0.321))
            self.assertEqual(summary.latest_loss, 0.321)
            self.assertEqual(summary.best_loss, 0.316)

    def test_collect_run_summaries_reads_model_download_activity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "downloading"
            (run_dir / "logs").mkdir(parents=True)
            (run_dir / "run.yaml").write_text(
                "\n".join(
                    [
                        "id: downloading",
                        "type: train",
                        "backend: {name: musubi-tuner}",
                        "recipe: {steps: 20}",
                        "compute: {executor: docker}",
                    ]
                ),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(json.dumps({"state": "running", "last_step": 0}), encoding="utf-8")
            (run_dir / "logs" / "stdout.log").write_text(
                "[kura] musubi step 1/6: hf_hub_download\n"
                "[kura] hf download progress dit:raw.safetensors files=40 bytes=2147483648\n",
                encoding="utf-8",
            )

            summary = collect_run_summaries(root)[0]

            self.assertEqual(summary.activity, "downloading dit raw.safetensors · 2.0GB")

    def test_collect_run_summaries_reads_downloaded_stdout_for_remote_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "remote-train"
            downloaded = run_dir / "downloads" / "remote-train"
            (run_dir / "metrics").mkdir(parents=True)
            (run_dir / "logs").mkdir()
            (downloaded / "logs").mkdir(parents=True)
            (run_dir / "run.yaml").write_text(
                "\n".join(
                    [
                        "id: remote-train",
                        "type: train",
                        "backend: {name: musubi-tuner}",
                        "recipe: {steps: 1}",
                        "compute: {executor: runpod}",
                    ]
                ),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(
                json.dumps(
                    {
                        "state": "completed",
                        "exit_code": 0,
                        "downloaded_run": "downloads/remote-train",
                        "outputs": ["downloads/remote-train/outputs/result.safetensors"],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "metrics" / "metrics.jsonl").write_text("", encoding="utf-8")
            (run_dir / "logs" / "stdout.log").write_text("", encoding="utf-8")
            (downloaded / "logs" / "stdout.log").write_text(
                "\rsteps: 100%|██████████| 1/1 [00:00<00:00,  4.33it/s, avr_loss=0.379]\n",
                encoding="utf-8",
            )

            summary = collect_run_summaries(root)[0]

            self.assertEqual(summary.progress.step, 1)
            self.assertEqual(summary.progress.total, 1)
            self.assertEqual(summary.losses, (0.379,))
            self.assertEqual(summary.latest_loss, 0.379)

    def test_collect_run_summaries_reads_dataset_array_roles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "paired"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "metrics").mkdir()
            (run_dir / "logs").mkdir()
            (root / "datasets" / "cond").mkdir(parents=True)
            (root / "datasets" / "target").mkdir(parents=True)
            (run_dir / "run.yaml").write_text("id: paired\ntype: train\n", encoding="utf-8")
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(
                "\n".join(
                    [
                        "id: paired",
                        "type: train",
                        "datasets:",
                        "  - {id: cond, digest: sha256:aaa, role: cond}",
                        "  - {id: target, digest: sha256:bbb, role: target}",
                        "recipe: {steps: 10}",
                        "backend: {name: musubi-tuner, config: {network_dim: 8, learning_rate: 0.0001}}",
                        "compute: {executor: runpod}",
                    ]
                ),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")

            summary = collect_run_summaries(root)[0]

            self.assertEqual([dataset.id for dataset in summary.datasets], ["cond", "target"])
            self.assertEqual([dataset.role for dataset in summary.datasets], ["cond", "target"])
            self.assertEqual(summary.datasets[0].path, root / "datasets" / "cond")
            self.assertEqual(summary.key_config["dataset"], "cond+target")

    def test_collect_run_summaries_estimates_runpod_cost_from_observation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "remote"
            (run_dir / "realizations").mkdir(parents=True)
            (run_dir / "run.yaml").write_text(
                "\n".join(
                    [
                        "id: remote",
                        "type: train",
                        "compute: {executor: runpod}",
                    ]
                ),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(
                json.dumps(
                    {
                        "state": "interrupted",
                        "started": "2026-06-22T10:00:00+00:00",
                        "ended": "2026-06-22T10:30:00+00:00",
                        "pod_id": "pod1",
                        "pulled_outputs": [
                            {"name": "model-step00000250.safetensors", "step": 250},
                            {"name": "model-step00000500.safetensors", "step": 500},
                        ],
                        "checkpoint_sync_error": "temporary transfer failure",
                        "last_realization": "realizations/launch.json",
                        "last_observation": "realizations/launch.observed-1.json",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "realizations" / "launch.json").write_text(
                json.dumps(
                    {
                        "id": "launch",
                        "executor": "runpod",
                        "launched_at": "2026-06-22T10:00:00+00:00",
                        "pod": {"id": "pod1", "desired_status": "RUNNING"},
                        "request": {"gpuTypeIds": ["NVIDIA A40"]},
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "realizations" / "launch.observed-1.json").write_text(
                json.dumps(
                    {
                        "observed_at": "2026-06-22T10:10:00+00:00",
                        "state": "running",
                        "pod_id": "pod1",
                        "desired_status": "RUNNING",
                        "last_started_at": "2026-06-22T10:00:00+00:00",
                        "cost_per_h": 0.44,
                        "machine": {"gpu_display_name": "NVIDIA A40"},
                    }
                ),
                encoding="utf-8",
            )

            summary = collect_run_summaries(root)[0]

            self.assertEqual(summary.executor_info.kind, "remote")
            self.assertEqual(summary.executor_info.gpu, "NVIDIA A40")
            self.assertIsNotNone(summary.executor_info.pod)
            assert summary.executor_info.pod is not None
            self.assertEqual(summary.executor_info.pod.cost_per_h, 0.44)
            self.assertAlmostEqual(summary.executor_info.pod.cost_used or 0.0, 0.22)
            self.assertEqual(summary.executor_info.mirrored_checkpoint_step, 500)
            self.assertEqual(summary.executor_info.checkpoint_sync_error, "temporary transfer failure")

    def test_collect_run_summaries_estimates_runpod_cost_from_launch_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "remote"
            (run_dir / "realizations").mkdir(parents=True)
            (run_dir / "run.yaml").write_text(
                "\n".join(
                    [
                        "id: remote",
                        "type: train",
                        "compute: {executor: runpod}",
                    ]
                ),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(
                json.dumps(
                    {
                        "state": "completed",
                        "started": "2026-06-22T19:12:00+09:00",
                        "ended": "2026-06-22T10:20:00+00:00",
                        "pod_stopped_at": "2026-06-22T10:17:00+00:00",
                        "pod_id": "pod1",
                        "last_realization": "realizations/launch.json",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "realizations" / "launch.json").write_text(
                json.dumps(
                    {
                        "id": "launch",
                        "executor": "runpod",
                        "launched_at": "2026-06-22T19:12:00+09:00",
                        "pod": {
                            "id": "pod1",
                            "desired_status": "RUNNING",
                            "last_started_at": "2026-06-22 10:14:00.000 +0000 UTC",
                            "cost_per_h": 0.60,
                        },
                        "request": {"gpuTypeIds": ["NVIDIA A40"]},
                    }
                ),
                encoding="utf-8",
            )

            summary = collect_run_summaries(root)[0]

            self.assertIsNotNone(summary.executor_info.pod)
            assert summary.executor_info.pod is not None
            self.assertEqual(summary.executor_info.pod.cost_per_h, 0.60)
            self.assertAlmostEqual(summary.executor_info.pod.cost_used or 0.0, 0.03)

    def test_collect_run_summaries_reads_batch_and_accumulation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "musubi-batch"
            run_dir.mkdir(parents=True)
            (run_dir / "run.yaml").write_text(
                "\n".join(
                    [
                        "id: musubi-batch",
                        "type: train",
                        "backend: {name: musubi-tuner}",
                        "recipe: {steps: 100}",
                        "backend:",
                        "  name: musubi-tuner",
                        "  config:",
                        "    extra_args:",
                        "      - --gradient_accumulation_steps",
                        "      - '2'",
                        "    dataset_config:",
                        "      general: {batch_size: 1}",
                    ]
                ),
                encoding="utf-8",
            )

            summary = collect_run_summaries(root)[0]

            self.assertEqual(summary.key_config["batch_size"], 1)
            self.assertEqual(summary.key_config["gradient_accumulation_steps"], 2)
            self.assertEqual(summary.key_config["effective_batch_size"], 2)


if __name__ == "__main__":
    unittest.main()

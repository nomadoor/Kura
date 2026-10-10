"""Tests for run monitoring projections."""

from __future__ import annotations

import json
import inspect
import subprocess
import tempfile
import unittest
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from kura.monitor import RunSummary, collect_run_summaries, loss_sparkline
from kura.container_scripts import hf_download
from kura.tui import KuraMonitorApp, MonitorScreen, _status_bar


class MonitorProjectionTests(unittest.TestCase):
    def test_the_fallback_realization_is_never_an_observation_or_publication(self) -> None:
        from kura.monitor import _latest_realization

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            realizations = run_dir / "realizations"
            realizations.mkdir()
            rid = "20261004-010203-000001"
            (realizations / f"{rid}.json").write_text(json.dumps({"id": rid, "executor": "docker"}), encoding="utf-8")
            for suffix in ("observed-20261004-020000-000000", "publication", "remote-exit-observed-x"):
                (realizations / f"{rid}.{suffix}.json").write_text(json.dumps({"kind": suffix}), encoding="utf-8")
            (realizations / "stage-20261004-030000-000000.json").write_text(json.dumps({"stage": True}), encoding="utf-8")
            (realizations / "remote-exit-20261004-040000.json").write_text(json.dumps({"event": "remote_exit"}), encoding="utf-8")
            (realizations / "stage.json").write_text(json.dumps({"stage": True}), encoding="utf-8")
            self.assertEqual(_latest_realization(run_dir, {})["id"], rid)

    def test_resume_progress_and_comparison_use_the_logical_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "derived"
            run_dir.mkdir(parents=True)
            run = {
                "id": "derived",
                "type": "train",
                "parent_run": "source",
                "recipe": {"steps": 1000, "seed": 1},
                # A valid Resume +500 from 1000: the monitor reads the total through run_envelope.final_step.
                "continuation": {
                    "mode": "resume",
                    "source": {"artifact_id": "ts-source", "manifest_sha256": "a" * 64, "observed_step": 1000, "recipe_sha256": "b" * 64},
                    "additional_steps": 500,
                    "target_step": 1500,
                    "restoration_contract": {"level": "best_effort_resume", "restored": ["model", "optimizer", "scheduler", "rng"], "not_restored": ["exact_dataloader_position"]},
                },
            }
            (run_dir / "run.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "running", "last_step": 1001}), encoding="utf-8")

            summary = collect_run_summaries(root)[0]

            self.assertEqual(summary.progress.total, 1500)
            self.assertEqual(summary.key_config["steps"], 1500)

    def test_summary_projects_resume_lineage_and_latest_recoverable_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "derived"
            run_dir.mkdir(parents=True)
            (run_dir / "run.yaml").write_text(
                yaml.safe_dump(
                    {
                        "id": "derived",
                        "type": "train",
                        "parent_run": "source",
                        "recipe": {"steps": 100, "seed": 1},
                        "continuation": {
                            "mode": "resume",
                            "source": {"artifact_id": "state-1"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(
                json.dumps(
                    {
                        "state": "failed",
                        "recoverable_training_states": [
                            {"artifact_id": "state-2", "observed_step": 20, "restoration_level": "best_effort_resume"}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            summary = collect_run_summaries(root)[0]
            self.assertEqual(summary.resume_source_run, "source")
            self.assertEqual(summary.resume_artifact_id, "state-1")
            self.assertEqual(summary.recoverable_state_step, 20)
            self.assertEqual(summary.recoverable_state_level, "best_effort_resume")

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
            run = {"id": "ai", "type": "train", "recipe": {"steps": 10, "seed": 1}, "backend": {"name": "ai-toolkit", "config": {"network_dim": 8, "network_alpha": 4, "learning_rate": 0.0001, "batch_size": 2}}}
            (run_dir / "run.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
            summary = collect_run_summaries(root)[0]
        self.assertEqual(summary.key_config["rank"], 8)
        self.assertEqual(summary.key_config["alpha"], 4)
        self.assertEqual(summary.key_config["lr"], 0.0001)
        self.assertEqual(summary.key_config["batch_size"], 2)

    def test_monitor_isolates_a_run_config_the_adapter_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rejected = root / "runs" / "rejected"
            current = root / "runs" / "current"
            rejected.mkdir(parents=True)
            current.mkdir(parents=True)
            run = {"id": "rejected", "type": "train", "recipe": {"steps": 1, "seed": 1}, "backend": {"name": "musubi-tuner", "config": {"architecture": "hunyuan_video", "extra_args": ["--blocks_to_swap", "36"]}}}
            (rejected / "run.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
            (current / "run.yaml").write_text("id: current\ntype: train\nrecipe: {steps: 1, seed: 1}\n", encoding="utf-8")
            summaries = {item.id: item for item in KuraMonitorApp(root).collect_summaries_cached()}
        self.assertEqual(summaries["rejected"].state, "unreadable")
        self.assertIn("--blocks_to_swap", summaries["rejected"].activity)
        self.assertNotEqual(summaries["current"].state, "unreadable")

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

    def test_queued_capacity_wait_becomes_stale_after_missed_polls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "queued-run"
            run_dir.mkdir(parents=True)
            (run_dir / "run.yaml").write_text("id: queued-run\ntype: train\ncompute: {executor: runpod}\n", encoding="utf-8")
            old = (datetime.now().astimezone() - timedelta(minutes=5)).isoformat()
            (run_dir / "status.json").write_text(
                json.dumps({"state": "queued", "capacity_wait": {"last_attempt_at": old, "poll_interval_sec": 30}}),
                encoding="utf-8",
            )

            summary = collect_run_summaries(root, stale_after=90)[0]

            self.assertTrue(summary.is_stale)
            self.assertIsNotNone(summary.capacity_wait)
            self.assertIn("GPU wait stopped", summary.activity or "")

    def test_active_capacity_wait_is_rendered_as_waiting_with_probe_details(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "waiting-run"
            (run_dir / "realizations").mkdir(parents=True)
            (run_dir / "run.yaml").write_text("id: waiting-run\ntype: train\ncompute: {executor: runpod, gpu: NVIDIA A40}\n", encoding="utf-8")
            now = datetime.now().astimezone().isoformat()
            (run_dir / "status.json").write_text(
                json.dumps(
                    {
                        "state": "queued",
                        "capacity_wait": {
                            "started_at": now,
                            "last_attempt_at": now,
                            "attempts": 3,
                            "remaining_sec": 21_540,
                            "poll_interval_sec": 30,
                            "gpu_type_ids": ["NVIDIA A40"],
                            "cloud_types": ["SECURE"],
                            "last_result": "capacity",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "realizations" / "stage.json").write_text(json.dumps({"timestamp": now, "state": "staged"}), encoding="utf-8")

            summary = collect_run_summaries(root)[0]
            rendered = _status_bar([summary], width=180).plain

            self.assertFalse(summary.is_stale)
            self.assertIsNone(summary.ended)
            self.assertEqual(summary.capacity_wait.attempts if summary.capacity_wait else None, 3)
            self.assertEqual(summary.executor_info.gpu, "NVIDIA A40")
            self.assertEqual(summary.activity, "waiting for GPU · NVIDIA A40 · probe 3 · 5h59m left")
            self.assertIn("1 waiting", rendered)
            self.assertIn("0 queued", rendered)

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

            screen = MonitorScreen()
            screen.summaries = collect_run_summaries(root)

            self.assertEqual(screen.active_runs, [])
            self.assertEqual({summary.id for summary in screen.history_pool}, {"aware", "naive"})

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

    def test_collect_run_summaries_reads_without_observing_the_container(self) -> None:
        # Viewers never reconcile (run-records ADR, decision 7); they show what was recorded.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "docker-running"
            (run_dir / "realizations").mkdir(parents=True)
            (root / "index.jsonl").write_text(json.dumps({"id": "docker-running"}) + "\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text("id: docker-running\ntype: train\ncompute: {executor: docker}\n", encoding="utf-8")
            status = {"state": "running", "container_id": "container-1", "last_realization": "realizations/r1.json"}
            (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
            (run_dir / "realizations" / "r1.json").write_text(
                json.dumps({"id": "r1", "executor": "docker", "state": "running", "container": {"id": "container-1"}}),
                encoding="utf-8",
            )

            with patch("kura.executors.docker.subprocess.run") as run:
                summary = collect_run_summaries(root)[0]

            run.assert_not_called()
            self.assertEqual(summary.state, "running")
            self.assertEqual(json.loads((run_dir / "status.json").read_text(encoding="utf-8")), status)

    def test_collect_run_summaries_uses_materialized_ai_toolkit_progress_and_stdout_losses(self) -> None:
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
            (run_dir / "status.json").write_text(json.dumps({"state": "completed", "last_step": 30, "total_steps": 30, "exit_code": 0}), encoding="utf-8")
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

    def test_collect_run_summaries_uses_materialized_musubi_progress_and_stdout_losses(self) -> None:
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
            (run_dir / "status.json").write_text(json.dumps({"state": "completed", "last_step": 100, "total_steps": 100, "seconds_per_iter": 2.56, "exit_code": 0}), encoding="utf-8")
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

    def test_training_stdout_ignores_date_like_progress_before_loss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "train"
            (run_dir / "logs").mkdir(parents=True)
            (run_dir / "run.yaml").write_text("id: train\ntype: train\nrecipe: {steps: 100}\n", encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "running", "last_step": 10, "total_steps": 100}), encoding="utf-8")
            (run_dir / "logs" / "stdout.log").write_text(
                "steps: 10/100 [00:10<01:30, 1.0s/it, avr_loss=0.5]\n"
                "2026/07/13 12:00:00 INFO epoch loss: 0.999\n",
                encoding="utf-8",
            )

            summary = collect_run_summaries(root)[0]

            self.assertEqual((summary.progress.step, summary.progress.total), (10, 100))
            self.assertEqual(summary.losses, (0.5,))

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
                "[kura] hf download shared activity dit:raw.safetensors repo_files_delta=40 repo_bytes_delta=2147483648 expected_size_bytes=4294967296\n",
                encoding="utf-8",
            )

            summary = collect_run_summaries(root)[0]

            self.assertEqual(summary.activity, "downloading dit raw.safetensors · 2.0GB")

    def test_sd_scripts_label_uses_generic_step_activity_parser(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "sd"
            (run_dir / "logs").mkdir(parents=True)
            (run_dir / "run.yaml").write_text("id: sd\ntype: train\nbackend: {name: sd-scripts}\n", encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
            (run_dir / "logs" / "stdout.log").write_text("[kura] sd-scripts step 3/5: launch\n", encoding="utf-8")

            summary = collect_run_summaries(root)[0]

            self.assertEqual(summary.activity, "launch · step 3/5")

    def test_hf_download_activity_contract_matches_container_producer(self) -> None:
        source = inspect.getsource(hf_download)
        self.assertIn("hf download shared activity", source)
        self.assertIn("repo_bytes_delta=", source)
        self.assertIn("hf download shared idle", source)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "downloading"
            (run_dir / "logs").mkdir(parents=True)
            (run_dir / "run.yaml").write_text("id: downloading\ntype: train\n", encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
            (run_dir / "logs" / "stdout.log").write_text(
                "[kura] hf download shared activity dit:model.safetensors repo_files_delta=2 repo_bytes_delta=1048576 expected_size_bytes=2097152\n"
                "[kura] hf download shared idle dit:model.safetensors idle=15s expected_size_bytes=2097152\n",
                encoding="utf-8",
            )

            summary = collect_run_summaries(root)[0]

            self.assertEqual(summary.activity, "download idle 15s · dit model.safetensors")

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
                        "last_step": 1,
                        "total_steps": 1,
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
                        "mirrored_outputs": [
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

    def test_collect_run_summaries_counts_scheduled_checkpoints_in_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "train"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "outputs").mkdir()
            (run_dir / "run.yaml").write_text("id: train\ntype: train\nrecipe: {steps: 1000}\n", encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
            (run_dir / "resolved" / "backend-display.lock.json").write_text(json.dumps({"checkpoint": {"save_every_n_steps": 250}}), encoding="utf-8")
            for step in (250, 500, 750):
                (run_dir / "outputs" / f"train-step{step:08d}.safetensors").touch()
            (run_dir / "outputs" / "train.safetensors").touch()

            summary = collect_run_summaries(root)[0]

            self.assertEqual(summary.checkpoint_count, 3)
            self.assertEqual(summary.checkpoint_expected, 4)
            self.assertEqual(summary.outputs_path, run_dir / "outputs")

    def test_collect_run_summaries_counts_final_weight_without_scheduled_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "train"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "outputs").mkdir()
            (run_dir / "run.yaml").write_text("id: train\ntype: train\nrecipe: {steps: 1}\n", encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "completed"}), encoding="utf-8")
            (run_dir / "resolved" / "backend-display.lock.json").write_text(
                json.dumps({"checkpoint": {"save_every_n_steps": 1}}),
                encoding="utf-8",
            )
            (run_dir / "outputs" / "train.safetensors").touch()

            summary = collect_run_summaries(root)[0]

            self.assertEqual(summary.checkpoint_count, 1)
            self.assertEqual(summary.checkpoint_expected, 1)

    def test_collect_run_summaries_ignores_retained_ai_toolkit_legacy_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "train"
            legacy_dir = run_dir / "outputs" / "train"
            (run_dir / "resolved").mkdir(parents=True)
            legacy_dir.mkdir(parents=True)
            (run_dir / "run.yaml").write_text("id: train\ntype: train\nrecipe: {steps: 1000}\n", encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "completed"}), encoding="utf-8")
            (run_dir / "resolved" / "backend-display.lock.json").write_text(
                json.dumps({"checkpoint": {"save_every_n_steps": 500}}),
                encoding="utf-8",
            )
            for step in (500, 1000):
                name = f"train-step{step:08d}.safetensors"
                (run_dir / "outputs" / name).touch()
                (legacy_dir / name).touch()
            (legacy_dir / "keep.txt").touch()

            summary = collect_run_summaries(root)[0]

            self.assertEqual(summary.checkpoint_count, 2)
            self.assertEqual(summary.checkpoint_expected, 2)

    def test_collect_run_summaries_counts_checkpoints_in_legacy_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "train"
            downloaded_outputs = run_dir / "downloads" / "train" / "outputs"
            (run_dir / "resolved").mkdir(parents=True)
            downloaded_outputs.mkdir(parents=True)
            (run_dir / "run.yaml").write_text("id: train\ntype: train\nrecipe: {steps: 1000}\n", encoding="utf-8")
            (run_dir / "status.json").write_text(
                json.dumps({"state": "completed", "downloaded_run": "downloads/train"}),
                encoding="utf-8",
            )
            (run_dir / "resolved" / "backend-display.lock.json").write_text(
                json.dumps({"checkpoint": {"save_every_n_steps": 250}}),
                encoding="utf-8",
            )
            for step in (250, 500, 750):
                (downloaded_outputs / f"train-step{step:08d}.safetensors").touch()

            summary = collect_run_summaries(root)[0]

            self.assertEqual(summary.outputs_path, downloaded_outputs)
            self.assertEqual(summary.checkpoint_count, 3)
            self.assertEqual(summary.checkpoint_expected, 4)

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
                        "    gradient_accumulation_steps: 2",
                        "    batch_size: 1",
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

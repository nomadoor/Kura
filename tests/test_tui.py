"""Tests for monitor TUI read-only formatting helpers."""

from __future__ import annotations

import json
import io
import tempfile
import unittest
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from kura.monitor import ExecutorInfo, PodInfo, RunSummary
from kura.tui import HostMetrics, KuraMonitorApp, MonitorScreen, PathRow, RunRow, SegmentButton, WatchScreen, _aware_datetime, _batch, _events_table, _filter_run_history, _open_path, _open_url, _parse_nvidia_smi_csv, _parse_remote_metrics_output, _remote_execution_phase, _resolve_run_selection, _retained_run_history, _runpod_pod_url


class TuiMetricsTests(unittest.TestCase):
    def test_history_filter_separates_training_and_render_runs(self) -> None:
        train = RunSummary(id="train", experiment=None, type="train", executor="docker", state="completed")
        render = RunSummary(id="render", experiment=None, type="render", executor="local", state="completed")
        history = [render, train]
        self.assertEqual([item.id for item in _filter_run_history(history, "all")], ["render", "train"])
        self.assertEqual([item.id for item in _filter_run_history(history, "train")], ["train"])
        self.assertEqual([item.id for item in _filter_run_history(history, "render")], ["render"])

    def test_retained_history_is_bounded_but_covers_each_filter(self) -> None:
        history = [RunSummary(id=f"train-{index}", experiment=None, type="train", executor="docker", state="completed") for index in range(200)]
        history += [RunSummary(id=f"render-{index}", experiment=None, type="render", executor="local", state="completed") for index in range(200)]

        retained = _retained_run_history(history, limit=30)

        self.assertLessEqual(len(retained), 90)
        self.assertEqual(len([item for item in retained if item.type == "render"]), 30)

    def test_run_row_uses_type_and_environment_emoji(self) -> None:
        train = RunRow(RunSummary(id="train", experiment=None, type="train", executor="docker", state="completed"), lane="history")
        render = RunRow(RunSummary(id="render", experiment=None, type="render", executor="local", state="completed"), lane="history")
        self.assertTrue(train.render_row().plain.startswith("T "))
        self.assertTrue(render.render_row().plain.startswith("R "))
        self.assertNotIn("⌂", train.render_row().plain)
        self.assertNotIn("LOC", train.render_row().plain)
        local_active = RunRow(RunSummary(id="local", experiment=None, type="train", executor="docker", state="running"), lane="active")
        self.assertIn("LOC", local_active.render_row().plain)
        remote = RunRow(RunSummary(id="remote", experiment=None, type="train", executor="runpod", state="running", executor_info=ExecutorInfo(kind="remote")), lane="active")
        self.assertIn("POD", remote.render_row().plain)

    def test_click_focus_does_not_recolor_an_entire_pane(self) -> None:
        self.assertNotIn(".pane:focus", KuraMonitorApp.CSS)

    def test_empty_output_path_is_rendered_as_disabled(self) -> None:
        row = PathRow("out", path=Path("/tmp/missing"), enabled=False)
        with patch.object(row, "post_message") as post:
            row.on_click(None)  # type: ignore[arg-type]
        post.assert_not_called()
        self.assertTrue(all("underline" not in str(span.style) for span in row.render().spans))

    def test_remote_metrics_never_block_screen_switching(self) -> None:
        app = KuraMonitorApp(Path("."))
        summary = RunSummary(
            id="remote",
            experiment=None,
            type="train",
            executor="runpod",
            state="running",
            executor_info=ExecutorInfo(kind="remote", provider="runpod", pod=PodInfo(id="pod-1")),
        )

        def slow_metrics(*_: object) -> HostMetrics:
            time.sleep(0.2)
            return HostMetrics(gpu_name="NVIDIA A40")

        with patch("kura.tui._runpod_host_metrics", side_effect=slow_metrics):
            started = time.monotonic()
            initial = app.metrics_for(summary)
            elapsed = time.monotonic() - started
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and app.metrics_for(summary).gpu_name is None:
                time.sleep(0.01)
            loaded = app.metrics_for(summary)

        self.assertLess(elapsed, 0.05)
        self.assertIsNone(initial.gpu_name)
        self.assertEqual(loaded.gpu_name, "NVIDIA A40")

    def test_remote_phase_distinguishes_job_from_pod_state(self) -> None:
        summary = RunSummary(
            id="remote",
            experiment=None,
            type="train",
            executor="runpod",
            state="running",
            executor_info=ExecutorInfo(
                kind="remote",
                provider="runpod",
                pod=PodInfo(id="pod-1", state="RUNNING"),
                remote_state="completed",
                recovery_required=True,
            ),
        )
        self.assertEqual(_remote_execution_phase(summary), "● job complete · awaiting download and pod stop")

    def test_parse_nvidia_smi_csv(self) -> None:
        metrics = _parse_nvidia_smi_csv("NVIDIA A40, 91, 43000, 46068, 70, 220.5, 300.0\n")

        self.assertEqual(metrics.gpu_name, "NVIDIA A40")
        self.assertEqual(metrics.gpu_util, 91)
        self.assertEqual(metrics.vram_used_mb, 43000)
        self.assertEqual(metrics.vram_total_mb, 46068)
        self.assertEqual(metrics.gpu_temp_c, 70)
        self.assertEqual(metrics.power_draw_w, 220.5)
        self.assertEqual(metrics.power_limit_w, 300.0)

    def test_parse_remote_metrics_output_includes_ram(self) -> None:
        metrics = _parse_remote_metrics_output(
            "NVIDIA A40, 74, 32000, 46068, 65, 181.0, 300.0\n"
            "__KURA_MEM__ 64000,16000\n"
        )

        self.assertEqual(metrics.gpu_name, "NVIDIA A40")
        self.assertEqual(metrics.gpu_util, 74)
        self.assertEqual(metrics.ram_used_bytes, 48000 * 1024)
        self.assertEqual(metrics.ram_total_bytes, 64000 * 1024)

    def test_runpod_pod_url(self) -> None:
        self.assertEqual(_runpod_pod_url("6oebwlls62iz1z"), "https://console.runpod.io/pods?id=6oebwlls62iz1z")
        self.assertIsNone(_runpod_pod_url(None))

    def test_open_url_uses_windows_browser_bridge_on_wsl(self) -> None:
        with patch("kura.tui._is_wsl", return_value=True), patch("kura.tui._windows_command", side_effect=lambda name: name if name == "cmd.exe" else None), patch("kura.tui.subprocess.Popen") as popen:
            opened = _open_url("https://console.runpod.io/pods?id=pod-1")

        self.assertTrue(opened)
        popen.assert_called_once()
        self.assertEqual(popen.call_args.args[0][:4], ["cmd.exe", "/c", "start", ""])

    def test_open_path_missing_wsl_bridge_returns_false(self) -> None:
        with patch("kura.tui._is_wsl", return_value=True), patch("kura.tui._windows_command", return_value=None), patch("kura.tui.subprocess.run") as run, patch("kura.tui.subprocess.Popen") as popen:
            run.return_value.returncode = 0
            run.return_value.stdout = r"\\wsl.localhost\\Ubuntu\\tmp" + "\n"
            opened = _open_path(Path("/tmp"))

        self.assertFalse(opened)
        popen.assert_not_called()

    def test_aware_datetime_normalizes_naive_values(self) -> None:
        naive = datetime(2026, 6, 21, 10, 0, 0)
        aware = datetime(2026, 6, 21, 10, 0, 0, tzinfo=timezone.utc)

        self.assertIsNotNone(_aware_datetime(naive).tzinfo)
        self.assertEqual(_aware_datetime(aware), aware)


    def test_active_selection_does_not_fall_through_to_history(self) -> None:
        selected, lane = _resolve_run_selection("run-1", "active", [], {"run-1"})

        self.assertIsNone(selected)
        self.assertEqual(lane, "active")

    def test_active_selection_moves_to_next_active_run(self) -> None:
        active = [RunSummary(id="run-2", experiment=None, type="train", executor="runpod", state="running")]

        selected, lane = _resolve_run_selection("run-1", "active", active, {"run-1", "run-2"})

        self.assertEqual(selected, "run-2")
        self.assertEqual(lane, "active")

    def test_history_selection_is_preserved(self) -> None:
        active = [RunSummary(id="run-2", experiment=None, type="train", executor="runpod", state="running")]

        selected, lane = _resolve_run_selection("run-1", "history", active, {"run-1", "run-2"})

        self.assertEqual(selected, "run-1")
        self.assertEqual(lane, "history")

    def test_batch_label_shows_accumulation(self) -> None:
        summary = RunSummary(
            id="run-1",
            experiment=None,
            type="train",
            executor="runpod",
            state="running",
            key_config={"batch_size": 1, "gradient_accumulation_steps": 4, "effective_batch_size": 4},
        )

        self.assertEqual(_batch(summary), "1 × accum 4 = effective 4")

    def test_events_render_oldest_to_newest_with_scoped_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "logs").mkdir()
            events = [
                {"event": "run_started", "timestamp": "2026-07-13T09:00:00+09:00", "executor": "runpod"},
                {"event": "run_outputs_pulled", "timestamp": "2026-07-13T09:01:00+09:00", "count": 1, "outputs": [{"step": 20}]},
                {"event": "remote_exit_observed", "observed_at": "2026-07-13T09:02:00+09:00", "remote_state": "completed", "exit_code": 0},
                {"event": "run_terminated", "timestamp": "2026-07-13T09:02:30+09:00", "pod_id": "legacy-pod"},
                {"event": "runpod_pod_stopped", "timestamp": "2026-07-13T09:03:00+09:00", "pod_id": "pod-1"},
            ]
            (run_dir / "logs" / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
            summary = RunSummary(id="run", experiment=None, type="train", executor="runpod", state="completed", run_dir=run_dir)

            table = _events_table(summary, lines=10)
            output = io.StringIO()
            Console(file=output, width=100, color_system=None).print(table)
            rendered = output.getvalue()

            positions = [rendered.index(label) for label in ("started", "weight pulled", "job exited", "pod stopped")]
            self.assertEqual(positions, sorted(positions))
            self.assertEqual(rendered.count("pod stopped"), 2)

    def test_deferred_refresh_ignores_unmounted_screens(self) -> None:
        for screen in (MonitorScreen(), WatchScreen("run")):
            with patch.object(screen, "refresh_data") as refresh:
                screen._start_refresh_loop()
            refresh.assert_not_called()


class TuiHistoryFilterTests(unittest.IsolatedAsyncioTestCase):
    async def test_history_filter_reuses_rows_instead_of_remounting(self) -> None:
        train = RunSummary(id="train", experiment=None, type="train", executor="docker", state="completed")
        render = RunSummary(id="render", experiment=None, type="render", executor="local", state="completed")
        app = KuraMonitorApp(Path("."), interval=60)

        with patch.object(app, "collect_summaries_cached", return_value=[train, render]):
            async with app.run_test() as pilot:
                await pilot.pause()
                screen = app.screen
                self.assertIsInstance(screen, MonitorScreen)
                rows_before = {row.summary.id: row for row in screen.query("#nav-history RunRow") if row.summary}

                with patch.object(rows_before["render"], "render_row", wraps=rows_before["render"].render_row) as hidden_render:
                    screen.on_segment_button_selected(SegmentButton.Selected("train"))
                    await pilot.pause()
                    hidden_render.assert_not_called()
                rows_after = {row.summary.id: row for row in screen.query("#nav-history RunRow") if row.summary}

                self.assertEqual(set(rows_after), {"train", "render"})
                self.assertIs(rows_after["train"], rows_before["train"])
                self.assertIs(rows_after["render"], rows_before["render"])
                self.assertTrue(rows_after["train"].display)
                self.assertFalse(rows_after["render"].display)

                screen.on_segment_button_selected(SegmentButton.Selected("render"))
                await pilot.pause()
                self.assertFalse(rows_after["train"].display)
                self.assertTrue(rows_after["render"].display)

                screen.on_segment_button_selected(SegmentButton.Selected("all"))
                await pilot.pause()
                self.assertTrue(rows_after["train"].display)
                self.assertTrue(rows_after["render"].display)


if __name__ == "__main__":
    unittest.main()

"""Tests for monitor TUI read-only formatting helpers."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from kura.monitor import RunSummary
from kura.tui import _aware_datetime, _batch, _open_path, _open_url, _parse_nvidia_smi_csv, _parse_remote_metrics_output, _resolve_run_selection, _runpod_pod_url


class TuiMetricsTests(unittest.TestCase):
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
        with patch("kura.tui._is_wsl", return_value=True), patch("kura.tui.shutil.which", side_effect=lambda name: name == "cmd.exe"), patch("kura.tui.subprocess.Popen") as popen:
            opened = _open_url("https://console.runpod.io/pods?id=pod-1")

        self.assertTrue(opened)
        popen.assert_called_once()
        self.assertEqual(popen.call_args.args[0][:4], ["cmd.exe", "/c", "start", ""])

    def test_open_path_missing_wsl_bridge_returns_false(self) -> None:
        with patch("kura.tui._is_wsl", return_value=True), patch("kura.tui.shutil.which", return_value=None), patch("kura.tui.subprocess.run") as run, patch("kura.tui.subprocess.Popen") as popen:
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


if __name__ == "__main__":
    unittest.main()

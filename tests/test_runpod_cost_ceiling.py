"""The most a RunPod Pod can bill before its lease deletes it, decided once for the plan, the
billed launch confirmation of training and render runs, and a lease change."""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from kura.executors.runpod import confirm_runpod_billing, format_cost_ceiling, pod_cost_ceiling, runpod_cost_ceiling
from kura.run_commands.plan import cmd_run_plan


def _measurement(*prices: tuple[str, str, float | None], gpu_count: int = 1) -> dict[str, object]:
    candidates: dict[str, dict[str, object]] = {}
    for gpu, cloud, price in prices:
        candidate = candidates.setdefault(gpu, {"gpu_type_id": gpu, "display_name": gpu, "memory_gb": 48, "clouds": []})
        stock = "Low" if price is not None else "None"
        candidate["clouds"].append({"cloud_type": cloud, "stock_status": stock, "available": price is not None, "price_per_hour": price})
    return {"status": "ok", "checked_at": "2026-10-11T12:00:00+09:00", "gpu_count": gpu_count, "candidates": list(candidates.values())}


class CostCeilingTests(unittest.TestCase):
    def test_highest_price_among_the_requested_gpus_for_the_whole_lease(self) -> None:
        measurement = _measurement(("A40", "COMMUNITY", 0.4), ("A40", "SECURE", 0.44), ("H100", "SECURE", 2.99))
        ceiling = runpod_cost_ceiling(measurement, ["A40"], max_lease_sec=12 * 3600)
        self.assertEqual(ceiling["max_lease"], "12h")
        self.assertEqual(ceiling["hourly_price"], 0.44)
        self.assertAlmostEqual(ceiling["max_cost"], 5.28)
        self.assertEqual(format_cost_ceiling(ceiling), "at most about $5.28 (12h at $0.440/hr, the highest current quote); disk storage is billed separately")

    def test_the_price_for_the_configured_gpu_count_is_used_as_returned(self) -> None:
        # RunPod's lowestPrice for gpuCount N is already the price of all N GPUs.
        ceiling = runpod_cost_ceiling(_measurement(("A40", "SECURE", 1.18), gpu_count=2), ["A40"], max_lease_sec=3 * 3600)
        self.assertAlmostEqual(ceiling["max_cost"], 3.54)
        self.assertEqual(format_cost_ceiling(ceiling), "at most about $3.54 (3h at $1.180/hr, the highest current quote); disk storage is billed separately")

    def test_unknown_prices_make_the_ceiling_unknown_or_partial(self) -> None:
        unknown = runpod_cost_ceiling({"status": "unavailable", "reason": "RUNPOD_API_KEY is not set", "candidates": []}, ["A40"], max_lease_sec=3600)
        self.assertIsNone(unknown["max_cost"])
        self.assertEqual(format_cost_ceiling(unknown), "unknown: RUNPOD_API_KEY is not set")
        none_requested = runpod_cost_ceiling(_measurement(("A40", "COMMUNITY", 0.4)), [], max_lease_sec=3600)
        self.assertEqual(format_cost_ceiling(none_requested), "unknown: no RunPod GPU type is requested")
        no_stock = runpod_cost_ceiling(_measurement(("A40", "COMMUNITY", None)), ["A40"], max_lease_sec=3600)
        self.assertEqual(format_cost_ceiling(no_stock), "unknown: A40 COMMUNITY has no current price (no stock)")
        partial = runpod_cost_ceiling(_measurement(("A40", "COMMUNITY", 0.4), ("A5000", "COMMUNITY", None)), ["A40", "A5000"], max_lease_sec=3600)
        self.assertEqual(
            format_cost_ceiling(partial),
            "at most about $0.40 for the GPUs with a current price (1h at $0.400/hr, the highest current quote); "
            "A5000 COMMUNITY has no current price (no stock), and landing there is not covered; disk storage is billed separately",
        )

    def test_a_running_pods_ceiling_uses_its_own_price(self) -> None:
        self.assertEqual(format_cost_ceiling(pod_cost_ceiling(0.58, lease_sec=6 * 3600)), "at most about $3.48 (6h at $0.580/hr, the Pod's price); disk storage is billed separately")
        self.assertEqual(format_cost_ceiling(pod_cost_ceiling(None, lease_sec=3600)), "unknown: the Pod's hourly price was not recorded")


class ConfirmationShowsCeilingTests(unittest.TestCase):
    def test_billed_confirmation_shows_the_ceiling(self) -> None:
        stderr = io.StringIO()
        with (
            patch("sys.stdin", io.StringIO()),
            patch("sys.stderr", stderr),
            patch("kura.executors.runpod.runpod_gpu_availability", return_value=_measurement(("NVIDIA A40", "COMMUNITY", 0.4))),
        ):
            confirm_runpod_billing({"gpu_type_ids": ["NVIDIA A40"], "cloud_types": ["COMMUNITY"]}, "registry/image:tag", yes=True, max_lease_sec=6 * 3600)
        text = stderr.getvalue()
        self.assertIn("  Maximum lease: 6h\n", text)
        self.assertIn("  Cost ceiling: at most about $2.40 (6h at $0.400/hr, the highest current quote); disk storage is billed separately\n", text)

    def test_billed_confirmation_says_when_the_ceiling_is_unknown(self) -> None:
        stderr = io.StringIO()
        with (
            patch("sys.stdin", io.StringIO()),
            patch("sys.stderr", stderr),
            patch("kura.executors.runpod.runpod_gpu_availability", return_value={"status": "unavailable", "reason": "offline", "candidates": []}),
        ):
            confirm_runpod_billing({"gpu_type_ids": ["NVIDIA A40"]}, "registry/image:tag", yes=True, max_lease_sec=12 * 3600)
        self.assertIn("  Cost ceiling: unknown: offline\n", stderr.getvalue())


class LeaseChangeShowsCeilingTests(unittest.TestCase):
    def test_a_lease_change_shows_the_new_ceiling_from_the_pods_price(self) -> None:
        from kura.run_commands import runpod_ssh

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            (run_dir / "realizations").mkdir(parents=True)
            (run_dir / "realizations" / "r1.json").write_text(json.dumps({
                "executor": "runpod", "purpose": "training", "pod": {"id": "pod-1", "cost_per_h": 0.58},
            }), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "running", "pod_id": "pod-1", "last_realization": "realizations/r1.json"}), encoding="utf-8")
            now, current = 1_800_000_000, 1_800_003_600
            replies = [
                subprocess.CompletedProcess([], 0, f"{now}\n{current}\n", ""),
                subprocess.CompletedProcess([], 0, f"{now + 6 * 3600}\n", ""),
            ]
            stderr = io.StringIO()
            with (
                patch.object(runpod_ssh, "_runpod_ssh_details", return_value={}),
                patch.object(runpod_ssh, "_ssh_base", return_value=["ssh"]),
                patch.object(runpod_ssh.subprocess, "run", side_effect=replies),
                patch.object(runpod_ssh, "record_lease_deadline"),
                patch.object(runpod_ssh, "_training_left_sec", return_value=None),
                patch("sys.stderr", stderr),
            ):
                self.assertEqual(runpod_ssh.change_runpod_lease(run_dir, 6 * 3600, yes=True), 0)
        self.assertIn("Cost ceiling from now: at most about $3.48 (6h at $0.580/hr, the Pod's price)", stderr.getvalue())


class PlanShowsCeilingAndCapacityChoicesTests(unittest.TestCase):
    def _plan(self, capacity: dict[str, str] | None) -> str:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "ceiling-plan"
            run_dir.mkdir(parents=True)
            (root / "workspace.yaml").write_text(
                "runpod:\n  gpu_type_ids: [NVIDIA RTX A5000, NVIDIA A40]\n  cloud_types: [COMMUNITY]\n", encoding="utf-8",
            )
            compute: dict[str, object] = {"executor": "runpod", "gpu": "NVIDIA A40"}
            if capacity is not None:
                compute["capacity"] = capacity
            (run_dir / "run.yaml").write_text(yaml.safe_dump({
                "id": "ceiling-plan", "type": "train", "model": {"base": "custom"}, "compute": compute,
                "datasets": [], "recipe": {"steps": 10}, "backend": {"name": "ai-toolkit", "config": {}},
            }, sort_keys=False), encoding="utf-8")
            measurement = _measurement(("NVIDIA A40", "COMMUNITY", 0.4), ("NVIDIA RTX A5000", "COMMUNITY", 0.27))
            previous = Path.cwd()
            os.chdir(root)
            try:
                with (
                    patch("kura.run_commands.plan.runpod_gpu_availability", return_value=measurement),
                    patch("kura.run_commands.plan._hf_file_size_probe", return_value={"status": "missing_metadata", "size_bytes": None}),
                    patch("sys.stdout", new_callable=io.StringIO) as stdout,
                ):
                    self.assertEqual(cmd_run_plan(argparse.Namespace(run_id="ceiling-plan", json=False)), 0)
            finally:
                os.chdir(previous)
        return stdout.getvalue()

    def test_plan_shows_max_lease_and_ceiling_of_the_requested_gpu(self) -> None:
        output = self._plan(None)
        self.assertRegex(output, r"max_lease +12h \(the default; `kura run execute --max-lease` sets it\)")
        # The alternative RTX A5000 is cheaper and not requested; the ceiling is the A40's.
        self.assertRegex(output, r"cost_ceiling +at most about \$4\.80 \(12h at \$0\.400/hr, the highest current quote\); disk storage is billed separately")

    def test_plan_offers_waiting_only_when_the_run_does_not_wait(self) -> None:
        waiting = self._plan(None)
        self.assertNotIn("set compute.capacity.mode=wait", waiting)
        self.assertIn("wait for the selected GPU (compute.capacity.mode is wait)", waiting)
        immediate = self._plan({"mode": "immediate"})
        self.assertIn("wait for the selected GPU: set compute.capacity.mode=wait before compile", immediate)


if __name__ == "__main__":
    unittest.main()

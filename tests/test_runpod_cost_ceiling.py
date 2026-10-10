"""The most a RunPod Pod can bill before its maximum lease deletes it, decided once for the plan
and the billed launch confirmation of training and render runs."""

from __future__ import annotations

import argparse
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from kura.executors.runpod import confirm_runpod_billing, format_cost_ceiling, runpod_cost_ceiling
from kura.run_commands.plan import cmd_run_plan


def _measurement(*prices: tuple[str, str, float | None], gpu_count: int = 1) -> dict[str, object]:
    candidates: dict[str, dict[str, object]] = {}
    for gpu, cloud, price in prices:
        candidate = candidates.setdefault(gpu, {"gpu_type_id": gpu, "display_name": gpu, "memory_gb": 48, "clouds": []})
        candidate["clouds"].append({"cloud_type": cloud, "stock_status": "Low", "available": price is not None, "price_per_hour": price})
    return {"status": "ok", "checked_at": "2026-10-11T12:00:00+09:00", "gpu_count": gpu_count, "candidates": list(candidates.values())}


class CostCeilingTests(unittest.TestCase):
    def test_highest_price_among_the_requested_gpus_for_the_whole_lease(self) -> None:
        measurement = _measurement(("A40", "COMMUNITY", 0.4), ("A40", "SECURE", 0.44), ("H100", "SECURE", 2.99))
        ceiling = runpod_cost_ceiling(measurement, ["A40"], max_lease_sec=12 * 3600)
        self.assertEqual(ceiling["max_lease"], "12h")
        self.assertEqual(ceiling["hourly_price"], 0.44)
        self.assertAlmostEqual(ceiling["max_cost"], 5.28)
        self.assertEqual(format_cost_ceiling(ceiling), "at most about $5.28 (12h at $0.440/hr)")

    def test_gpu_count_multiplies_the_ceiling(self) -> None:
        ceiling = runpod_cost_ceiling(_measurement(("A40", "SECURE", 0.5), gpu_count=2), ["A40"], max_lease_sec=3 * 3600)
        self.assertAlmostEqual(ceiling["max_cost"], 3.0)
        self.assertEqual(format_cost_ceiling(ceiling), "at most about $3.00 (3h at $0.500/hr × 2 GPUs)")

    def test_unknown_prices_make_the_ceiling_unknown_or_partial(self) -> None:
        unknown = runpod_cost_ceiling({"status": "unavailable", "reason": "no key", "candidates": []}, ["A40"], max_lease_sec=3600)
        self.assertIsNone(unknown["max_cost"])
        self.assertTrue(format_cost_ceiling(unknown).startswith("unknown"))
        partial = runpod_cost_ceiling(_measurement(("A40", "COMMUNITY", 0.4), ("A5000", "COMMUNITY", None)), ["A40", "A5000"], max_lease_sec=3600)
        self.assertEqual(
            format_cost_ceiling(partial),
            "at most about $0.40 (1h at $0.400/hr) where a price is known; unknown for A5000 COMMUNITY, which has no current price",
        )


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
        self.assertIn("  Cost ceiling: at most about $2.40 (6h at $0.400/hr)\n", text)

    def test_billed_confirmation_says_when_the_ceiling_is_unknown(self) -> None:
        stderr = io.StringIO()
        with (
            patch("sys.stdin", io.StringIO()),
            patch("sys.stderr", stderr),
            patch("kura.executors.runpod.runpod_gpu_availability", return_value={"status": "unavailable", "reason": "offline", "candidates": []}),
        ):
            confirm_runpod_billing({"gpu_type_ids": ["NVIDIA A40"]}, "registry/image:tag", yes=True, max_lease_sec=12 * 3600)
        self.assertIn("  Cost ceiling: unknown", stderr.getvalue())


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
        self.assertRegex(output, r"cost_ceiling +at most about \$4\.80 \(12h at \$0\.400/hr\)")

    def test_plan_offers_waiting_only_when_the_run_does_not_wait(self) -> None:
        waiting = self._plan(None)
        self.assertNotIn("set compute.capacity.mode=wait", waiting)
        self.assertIn("wait for the selected GPU (compute.capacity.mode is wait)", waiting)
        immediate = self._plan({"mode": "immediate"})
        self.assertIn("wait for the selected GPU: set compute.capacity.mode=wait before compile", immediate)


if __name__ == "__main__":
    unittest.main()

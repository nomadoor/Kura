"""One parser reads the step from a checkpoint name for every backend, wherever Kura needs it."""

from __future__ import annotations

import unittest

NAMES = {
    # sd-scripts and Musubi Tuner
    "vivi-step00000100.safetensors": 100,
    "vivi-step00000100-state": 100,
    # AI-Toolkit
    "vivi_000000500.safetensors": 500,
    "my_lora_v2_000001000.safetensors": 1000,
    # AI-Toolkit names checkpoints after the run ID, which ends in 4 hex digits
    "20261008-2356_smoke-test-gc_7868_000000025.safetensors": 25,
    # final weights carry no step, even when the run ID ends in digits
    "20261008-2356_smoke-test-gc_7868.safetensors": None,
    "vivi.safetensors": None,
    "my_lora_v2.safetensors": None,
}


class CheckpointNameTests(unittest.TestCase):
    def test_every_reader_takes_the_step_from_the_same_parser(self) -> None:
        from kura import monitor
        from kura.run_commands import experiment, render_completion, runpod_ssh
        from kura.training_artifacts import checkpoint_step

        for name, step in NAMES.items():
            with self.subTest(name=name):
                self.assertEqual(checkpoint_step(name), step)
        for module in (monitor, experiment, render_completion, runpod_ssh):
            with self.subTest(module=module.__name__):
                self.assertIs(module.checkpoint_step, checkpoint_step)


if __name__ == "__main__":
    unittest.main()

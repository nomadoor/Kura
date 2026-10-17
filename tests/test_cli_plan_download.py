"""The plan says how a trainer-resolved model is downloaded, as it differs by executor."""

from __future__ import annotations

import unittest
from pathlib import Path

from kura.paths import AI_TOOLKIT_MODELS_PATH
from kura.run_commands.plan import _model_download_preflight_report


class BackendResolvedModelTests(unittest.TestCase):
    def test_runpod_downloads_on_every_run_and_local_reuses_the_cache(self) -> None:
        run = {"type": "train", "backend": {"name": "ai-toolkit"}, "model": {"base": "stabilityai/sdxl"}}
        cache = Path("/mnt/e/hf-cache")
        local = _model_download_preflight_report(run, {}, executor="docker", local_cache=cache)[0]["fact"]
        remote = _model_download_preflight_report(run, {}, executor="runpod", local_cache=cache)[0]["fact"]
        self.assertIn("stabilityai/sdxl", local)
        self.assertIn("reusing files an earlier run left there", local)
        self.assertIn("on every run, while the Pod bills", remote)

    def test_the_acquisition_line_names_where_the_trainer_downloads(self) -> None:
        run = {"type": "train", "backend": {"name": "ai-toolkit"}, "model": {"base": "org/model"}}
        cache = Path("/mnt/e/hf-cache")
        local = _model_download_preflight_report(run, {}, executor="docker", local_cache=cache)[0]["fact"]
        remote = _model_download_preflight_report(run, {}, executor="runpod", local_cache=cache)[0]["fact"]
        self.assertIn(f"into the Hugging Face cache at {cache / 'ai-toolkit-models'},", local)
        self.assertIn(f"into the Pod's Hugging Face cache ({AI_TOOLKIT_MODELS_PATH})", remote)
        for fact in (local, remote):
            self.assertIn("may write tens of GB there", fact)


if __name__ == "__main__":
    unittest.main()

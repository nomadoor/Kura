"""The plan says how a trainer-resolved model is downloaded, as it differs by executor."""

from __future__ import annotations

import unittest

from kura.run_commands.plan import _model_download_preflight_report


class BackendResolvedModelTests(unittest.TestCase):
    def test_runpod_downloads_on_every_run_and_local_reuses_the_cache(self) -> None:
        run = {"type": "train", "backend": {"name": "ai-toolkit"}, "model": {"base": "stabilityai/sdxl"}}
        local = _model_download_preflight_report(run, {}, executor="docker")[0]["fact"]
        remote = _model_download_preflight_report(run, {}, executor="runpod")[0]["fact"]
        self.assertIn("stabilityai/sdxl", local)
        self.assertIn("cache already holds it", local)
        self.assertIn("on every run, while the Pod bills", remote)


if __name__ == "__main__":
    unittest.main()

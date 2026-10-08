"""Every run state Kura names comes from one vocabulary, and each decision over states uses one shared set."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "kura"


class RunStateVocabularyTests(unittest.TestCase):
    def test_the_decisions_over_states_use_the_shared_sets(self) -> None:
        from kura import runner
        from kura.executors import common
        from kura.run_commands import launch, plan

        self.assertIs(launch.RELAUNCHABLE_STATES, common.RELAUNCHABLE_STATES)
        self.assertIs(plan.RELAUNCHABLE_STATES, common.RELAUNCHABLE_STATES)
        self.assertIs(runner.EXIT_FOR_STATE, common.EXIT_CODE_FOR_STATE)
        for name in ("ACTIVE_STATES", "OBSERVABLE_STATES", "TERMINAL_STATES", "UNFINISHED_STATES", "RELAUNCHABLE_STATES", "CLEANUP_ELIGIBLE_STATES", "UNSUCCESSFUL_STATES"):
            with self.subTest(name=name):
                self.assertLessEqual(set(getattr(common, name)), common.RUN_STATES)
        self.assertLessEqual(set(common.EXIT_CODE_FOR_STATE), common.RUN_STATES)

    def test_no_module_keeps_its_own_list_of_run_states(self) -> None:
        # A literal set or tuple of two or more run states outside executors/common.py is a second copy.
        from kura.executors.common import RUN_STATES

        # Decisions made in exactly one place, about that place's own step.
        local = {
            "render.py": ["compiled", "running"],  # a RunPod render starts after its session marks the run running
            "executors/docker.py": ["completed", "failed"],  # the outcomes a container's exit code gives
        }
        literal = re.compile(r"[{(]\s*((?:\"[a-z_]+\"\s*,\s*)+\"[a-z_]+\")\s*,?\s*[})]")
        offenders = []
        for path in SRC.rglob("*.py"):
            if path.name == "common.py" and path.parent.name == "executors":
                continue
            for match in literal.finditer(path.read_text(encoding="utf-8")):
                names = re.findall(r"\"([a-z_]+)\"", match.group(1))
                if len(names) >= 2 and set(names) <= RUN_STATES and len(set(names) & {"completed", "failed", "interrupted", "compiled", "running", "launch_failed"}) >= 2:
                    if local.get(str(path.relative_to(SRC))) != names:
                        offenders.append(f"{path.relative_to(SRC)}: {names}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()

"""Platform scope for the test suite.

Kura supports Linux and macOS directly and Windows through WSL2 (see
docs/adr/windows-execution-model.md). On native Windows, Kura deliberately
refuses to read datasets because it requires no-follow, directory-relative
opens there, and several lifecycle paths rely on POSIX files, users, and
paths. The Windows CI job runs everything else, so a new failure there is a
real platform-neutral regression rather than this known refusal.
"""

from __future__ import annotations

import os
import unittest

NATIVE_WINDOWS = os.name == "nt"


def posix_only(reason: str):
    """Skip on native Windows, naming the POSIX behavior the test depends on."""
    return unittest.skipIf(NATIVE_WINDOWS, f"native Windows is unsupported (use WSL2): {reason}")


DATASET_IO = "dataset inputs require no-follow directory-relative opens"
POSIX_PATHS = "records and container paths use POSIX semantics"


def small_local_reserve() -> dict[str, float]:
    """A workspace's `docker` section whose local reserve (docker.min_free_gb, 20 GiB by default) is
    about 10 MiB, for tests that pass through a RunPod stage or pull without being about the reserve,
    so they pass on a host with little free space. Tests about the reserve set their own value and
    fake the free space."""
    return {"min_free_gb": 0.01}

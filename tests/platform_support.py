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

"""Keep a test module away from the user's real Kura configuration directory.

Every RunPod launch and SSH lookup reads Kura's own SSH key, which lives beside
the user secrets file and is created there on first use. A module that launches
RunPod calls `isolate_user_config()` from `setUpModule`, so the key (and any user
secrets file) lives in a temporary directory for that module instead.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def isolate_user_config() -> Path:
    """Point the user secrets file, and so Kura's RunPod SSH key, at a temporary directory
    until the calling module's tests end; the directory it returns."""
    directory = tempfile.TemporaryDirectory()
    unittest.addModuleCleanup(directory.cleanup)
    root = Path(directory.name)
    patcher = patch("kura.secrets.user_secrets_path", return_value=root / "secrets.env")
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)
    return root

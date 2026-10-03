"""Content Kura ships for usage sessions: skills, knowledge, and workflow samples.

This directory is the single source. `kura init` writes it into a workspace,
and the repository's skill mirrors are generated from it. Read it through
`shipped_root()`, never through a repository-relative path.
"""

from __future__ import annotations

from importlib.resources import files
from importlib.resources.abc import Traversable

SHIPPED_SKILLS = (
    "comfyui-render-workflow",
    "dataset-prep",
    "local-disk-safety",
    "lora-evaluation",
    "publishing-huggingface-modelscope",
    "runpod-lifecycle",
    "training-parameter-planning",
)


def shipped_root() -> Traversable:
    return files(__name__)

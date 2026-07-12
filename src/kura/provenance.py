"""Small provenance observations shared by compile and executors."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def artifact_pinning(identity: dict[str, Any], *, observable: bool) -> dict[str, Any]:
    if isinstance(identity.get("sha256"), str):
        return {"strength": "content-hash", "observation": "observed"}
    revision = identity.get("revision")
    if isinstance(revision, str) and len(revision) >= 40 and all(char in "0123456789abcdefABCDEF" for char in revision):
        return {"strength": "immutable-revision", "observation": "observed"}
    if revision:
        return {"strength": "mutable-reference", "observation": "observed", "detail": "revision is not proven immutable"}
    if identity.get("kind") == "path":
        return {"strength": "external-unobserved", "observation": "not-observed" if observable else "not-observable", "detail": "Kura did not hash the external model path during compile"}
    return {"strength": "mutable-reference", "observation": "not-observed" if observable else "not-observable", "detail": "no immutable revision or content hash was observed"}


def adapter_source_identity(backend_name: str) -> dict[str, str]:
    root = Path(__file__).resolve().parent / "backends"
    container_root = Path(__file__).resolve().parent / "container_scripts"
    if backend_name == "ai-toolkit":
        paths = [root / "common.py", root / "ai_toolkit.py", root / "registry.py"]
    elif backend_name == "musubi-tuner":
        paths = [
            root / "common.py",
            root / "registry.py",
            *sorted(root.glob("musubi_*.py")),
            *(container_root / name for name in (
                "hf_download.py",
                "musubi_dataset_assert.py",
                "prune_checkpoints.py",
                "safetensors_validator.py",
            )),
        ]
    else:
        raise ValueError(f"unsupported backend for source identity: {backend_name}")
    hasher = hashlib.sha256()
    for path in paths:
        hasher.update(path.relative_to(Path(__file__).resolve().parent).as_posix().encode("utf-8") + b"\0")
        hasher.update(path.read_bytes() + b"\0")
    return {"kind": "source-tree-sha256", "value": hasher.hexdigest(), "backend": backend_name}


def image_reference_identity(reference: str, observed_id: str | None = None) -> dict[str, Any]:
    if observed_id and observed_id.startswith("sha256:"):
        return {"reference": reference, "pinning": {"strength": "content-hash", "observation": "observed", "value": observed_id}}
    if "@sha256:" in reference:
        return {"reference": reference, "pinning": {"strength": "content-hash", "observation": "observed", "value": reference.split("@", 1)[1]}}
    return {
        "reference": reference,
        "pinning": {
            "strength": "mutable-reference",
            "observation": "not-observed",
            "detail": "runtime image digest was not observed",
        },
    }

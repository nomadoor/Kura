"""Small provenance observations shared by compile and executors."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def adapter_source_identity(backend_name: str) -> dict[str, str]:
    root = Path(__file__).resolve().parent / "backends"
    hasher = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        hasher.update(path.name.encode("utf-8") + b"\0")
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

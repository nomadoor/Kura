"""Freeze real manifest-v2 handoffs for compiler tests.

Compiler tests must read locks produced by ``freeze_dataset_handoff``; they
never write a projection lock directly. This helper builds the smallest real
dataset for a run and freezes it through the selected adapter's projector.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kura.backends import get_backend
from kura.dataset_handoff import freeze_dataset_handoff


def freeze_fixture(
    run: dict[str, Any],
    resolved: Path,
    *,
    target: str = "sample.png",
    controls: tuple[str, ...] = (),
    caption: str | None = "caption",
) -> dict[str, Any]:
    """Freeze one single-sample dataset for ``run`` and return the projection report."""
    backend = get_backend(run["backend"]["name"])
    dataset_id = str(run["datasets"][0]["id"])
    workspace = resolved / "fixture-workspace"
    dataset = workspace / "datasets" / dataset_id
    dataset.mkdir(parents=True, exist_ok=True)
    files = [{"type": "file", "role": "target", "path": target}]
    files.extend({"type": "file", "role": "control", "path": path} for path in controls)
    for item in files:
        (dataset / item["path"]).write_bytes(f"payload:{item['path']}".encode("utf-8"))
    (dataset / "dataset.yaml").write_text(
        f"id: {dataset_id}\nitems_schema_version: 2\n", encoding="utf-8",
    )
    (dataset / "items.jsonl").write_text(json.dumps({
        "id": "sample",
        "files": files,
        "caption": {"text": caption} if caption is not None else None,
    }) + "\n", encoding="utf-8")
    freeze_dataset_handoff(
        run,
        workspace,
        resolved,
        backend=backend.name,
        project=lambda selection: backend.project_dataset(run, selection),
    )
    return json.loads((resolved / "dataset-projection.lock.json").read_text(encoding="utf-8"))

"""Publish a verified inventory of the outputs promised by a training run."""

from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

from kura.fsio import atomic_write_json
from kura.training_artifacts import _validate_safetensors_file, is_training_state_output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_contract(run_dir: Path) -> dict[str, Any] | None:
    """Return the frozen contract, or None for a genuine legacy command lock."""

    path = run_dir / "resolved" / "backend-command.lock.json"
    if not path.is_file():
        if (run_dir / "resolved" / "manifest.lock.yaml").is_file():
            raise ValueError("frozen backend command is missing; publication cannot be verified")
        return None
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("frozen backend command is not valid JSON") from exc
    if not isinstance(spec, dict):
        raise ValueError("frozen backend command is not a mapping")
    contract = spec.get("output_contract")
    if contract is None:
        return None
    if contract != {"required": [{"role": "trained-adapter", "suffix": ".safetensors", "minimum": 1}]}:
        raise ValueError("frozen backend output contract is unsupported")
    return contract


def existing_output_snapshot(run_dir: Path) -> dict[str, dict[str, int]]:
    """Freeze which output files already existed before a local realization."""

    output_dir = run_dir / "outputs"
    if output_dir.is_symlink():
        raise ValueError("output directory is a symlink")
    snapshot: dict[str, dict[str, int]] = {}
    if output_dir.is_dir():
        for path in sorted(output_dir.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"output contains a symlink: {path.relative_to(run_dir)}")
            if path.is_file() and not is_training_state_output(path.relative_to(output_dir)):
                stat = path.stat()
                snapshot[path.relative_to(run_dir).as_posix()] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return snapshot


def record_publication_failure(run_dir: Path, realization_id: str, error: str) -> str:
    """Append a durable failure fact; status may point to the latest attempt."""

    timestamp = datetime.now().astimezone()
    compact = timestamp.strftime("%Y%m%d-%H%M%S-%f")
    path = run_dir / "realizations" / f"{realization_id}.publication-attempt-{compact}-{secrets.token_hex(3)}.json"
    atomic_write_json(path, {
        "realization_id": realization_id,
        "observed_at": timestamp.isoformat(),
        "result": "blocked",
        "error": error,
    })
    return path.relative_to(run_dir).as_posix()


def publish_outputs(
    run_dir: Path,
    realization_id: str,
    contract: dict[str, Any],
    *,
    baseline: dict[str, dict[str, int]] | None = None,
    candidate_paths: list[str] | None = None,
) -> tuple[str, list[str]]:
    """Validate required local payloads and atomically publish their inventory."""

    if contract != {"required": [{"role": "trained-adapter", "suffix": ".safetensors", "minimum": 1}]}:
        raise ValueError("frozen backend output contract is unsupported")
    output_dir = run_dir / "outputs"
    if output_dir.is_symlink() or (output_dir.exists() and not output_dir.is_dir()):
        raise ValueError("output directory is not a Kura-owned directory")
    files: list[dict[str, Any]] = []
    adapter_count = 0
    candidates = set(candidate_paths) if candidate_paths is not None else None
    if output_dir.is_dir():
        for path in sorted(output_dir.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"output contains a symlink: {path.relative_to(run_dir)}")
            if not path.is_file() or is_training_state_output(path.relative_to(output_dir)):
                continue
            relative = path.relative_to(run_dir).as_posix()
            before = path.stat()
            if candidates is not None and relative not in candidates:
                continue
            if baseline is not None and baseline.get(relative) == {"size": before.st_size, "mtime_ns": before.st_mtime_ns}:
                continue
            if path.suffix == ".safetensors":
                try:
                    _validate_safetensors_file(path)
                except ValueError as exc:
                    raise ValueError(f"invalid safetensors output: {path.relative_to(run_dir)}: {exc}") from exc
                adapter_count += 1
            digest = _sha256(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise ValueError(f"output changed during publication: {path.relative_to(run_dir)}")
            files.append({"path": relative, "size": after.st_size, "sha256": digest})
    if adapter_count < 1:
        raise ValueError("required trained-adapter safetensors output is missing")
    manifest_path = run_dir / "realizations" / f"{realization_id}.publication.json"
    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "realization_id": realization_id,
        "contract": contract,
        "files": files,
    }
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior != manifest:
            raise ValueError("output inventory changed after publication; the prior manifest is immutable")
    else:
        atomic_write_json(manifest_path, manifest)
    return manifest_path.relative_to(run_dir).as_posix(), [item["path"] for item in files]

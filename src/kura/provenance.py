"""Small provenance observations shared by compile and executors."""

from __future__ import annotations

import ast
import hashlib
import json
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


def _hash_source_parts(parts: list[tuple[str, bytes]], backend_name: str, *, scope: str) -> dict[str, str]:
    hasher = hashlib.sha256()
    for label, payload in parts:
        hasher.update(label.encode("utf-8") + b"\0")
        hasher.update(payload + b"\0")
    return {"kind": "source-tree-sha256", "value": hasher.hexdigest(), "backend": backend_name, "scope": scope}


def _source_symbol(path: Path, symbol: str) -> bytes:
    payload = path.read_bytes()
    text = payload.decode("utf-8")
    tree = ast.parse(text, filename=str(path))
    definitions: dict[str, list[ast.AST]] = {}
    for node in tree.body:
        names: list[str] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.extend(target.id for target in targets if isinstance(target, ast.Name))
        for name in names:
            definitions.setdefault(name, []).append(node)
    if len(definitions.get(symbol, [])) != 1:
        raise ValueError(f"source identity symbol {symbol!r} was not found exactly once in {path.name}")
    pending = [symbol]
    closure: dict[str, ast.AST] = {}
    while pending:
        name = pending.pop()
        if name in closure:
            continue
        matches = definitions.get(name, [])
        if len(matches) != 1:
            raise ValueError(f"source identity dependency {name!r} was not found exactly once in {path.name}")
        node = matches[0]
        closure[name] = node
        pending.extend(
            child.id for child in ast.walk(node)
            if isinstance(child, ast.Name)
            and isinstance(child.ctx, ast.Load)
            and child.id in definitions
        )
    parts: list[bytes] = []
    for name, node in sorted(closure.items()):
        source = ast.get_source_segment(text, node)
        if source is None:
            raise ValueError(f"source identity symbol {name!r} has no source segment in {path.name}")
        parts.append(name.encode("utf-8") + b"\0" + source.encode("utf-8"))
    return b"\0".join(parts)


def legacy_adapter_source_identity(backend_name: str) -> dict[str, str]:
    """Calculate the old whole-file algorithm on the current tree.

    Historical migration values are reproducible only from their pre-migration
    source trees; this helper does not recreate those trees or their hashes.
    """
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
    parts: list[tuple[str, bytes]] = []
    for path in paths:
        parts.append((path.relative_to(Path(__file__).resolve().parent).as_posix(), path.read_bytes()))
    return _hash_source_parts(parts, backend_name, scope="legacy-whole-files")


def adapter_source_identity(backend_name: str) -> dict[str, str]:
    """Hash only the selected adapter and the helpers it actually consumes."""
    package_root = Path(__file__).resolve().parent
    backend_root = package_root / "backends"
    container_root = package_root / "container_scripts"
    shared = backend_root / "shared.py"
    registry = backend_root / "registry.py"
    run_envelope = package_root / "run_envelope.py"
    if backend_name == "ai-toolkit":
        paths = [
            backend_root / "ai_toolkit.py",
            backend_root / "dataset_profiles.py",
        ]
        symbols = [
            *((shared, name) for name in ("_datasets", "_script_command")),
            *((run_envelope, name) for name in (
                "backend_config", "resume_intent", "run_executor",
                "training_state_policy", "validated_recipe",
            )),
        ]
        runtime_paths = [
            container_root / "ai_toolkit_state.py",
            container_root / "ai_toolkit_video_assert.py",
            container_root / "training_state_verify.py",
        ]
    elif backend_name == "musubi-tuner":
        paths = [
            backend_root / "common.py",
            backend_root / "dataset_profiles.py",
            *sorted(backend_root.glob("musubi_*.py")),
        ]
        symbols = [
            (shared, name)
            for name in (
                "_datasets", "_toml_scalar", "_script_command", "_truthy",
                "_extra_args", "_reject_owned_extra_args", "_append_flag", "_int_or_none",
            )
        ]
        symbols.extend((run_envelope, name) for name in (
            "backend_config", "resume_intent", "training_state_policy", "validated_recipe",
        ))
        runtime_paths = [
            container_root / name
            for name in ("hf_download.py", "musubi_dataset_assert.py", "prune_checkpoints.py", "safetensors_validator.py", "training_state_verify.py")
        ]
    elif backend_name == "sd-scripts":
        paths = [
            backend_root / "dataset_profiles.py",
            *sorted(backend_root.glob("sd_scripts*.py")),
        ]
        symbols = [
            (shared, name)
            for name in (
                "_datasets", "_toml_scalar", "_script_command", "_truthy",
                "_extra_args", "_reject_owned_extra_args", "_int_or_none", "_append_flag",
            )
        ]
        symbols.extend((run_envelope, name) for name in (
            "backend_config", "resume_intent", "training_state_policy", "validated_recipe",
        ))
        runtime_paths = [
            container_root / name
            for name in (
                "hf_download.py",
                "sd_scripts_probe.py",
                "sd_scripts_publish_anima.py",
                "sd_scripts_state.py",
                "sd_scripts_validate.py",
                "training_state_verify.py",
            )
        ]
    else:
        raise ValueError(f"unsupported backend for source identity: {backend_name}")
    paths.extend([package_root / "dataset_handoff.py", package_root / "dataset_manifest.py"])
    missing = [path for path in [*paths, *runtime_paths] if not path.is_file()]
    if missing:
        raise ValueError("source identity input is missing: " + ", ".join(path.name for path in missing))
    parts = [(path.relative_to(package_root).as_posix(), path.read_bytes()) for path in [*paths, *runtime_paths]]
    parts.extend((f"{path.relative_to(package_root).as_posix()}:{symbol}", _source_symbol(path, symbol)) for path, symbol in symbols)
    # Surface membership changes which authored intent reaches the adapter and
    # therefore belongs to adapter identity even though registry dispatch itself
    # remains outside the per-adapter source hash.
    from kura.backends.registry import _GENERAL_ML_ALIASES, _GENERAL_UNAVAILABLE, get_backend

    surface = get_backend(backend_name).surface
    parts.append(("backends/registry.py:validate_backend_config", _source_symbol(registry, "validate_backend_config")))
    parts.append((
        "backend-surface-contract.json",
        json.dumps(
            {
                "fields": sorted(surface.fields),
                "escape_hatches": sorted(surface.escape_hatches),
                "conditions": [
                    {
                        "field": item.field,
                        "when_any": [
                            {selector: list(allowed) for selector, allowed in clause}
                            for clause in item.when_any
                        ],
                    }
                    for item in surface.conditions
                ],
                "selector_defaults": dict(surface.selector_defaults),
                "nested_config_fields": surface.nested_config_fields or {},
                **({"config_value_choices": {field: list(values) for field, values in surface.config_value_choices}}
                   if surface.config_value_choices else {}),
                "aliases": {
                    key: value for key, value in _GENERAL_ML_ALIASES.items()
                    if value in surface.fields | surface.escape_hatches
                },
                "unavailable": {**_GENERAL_UNAVAILABLE, **dict(surface.unavailable)},
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
    ))
    return _hash_source_parts(parts, backend_name, scope="selected-adapter-v2")


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


def training_runtime_contract(
    adapter_source: dict[str, Any],
    local_image_identity: dict[str, Any],
    remote_image_identity: dict[str, Any],
) -> str:
    """Identify the declared local/remote runtime pair for portability checks."""

    payload = {
        "schema": "training-runtime-pair-v1",
        "adapter_source": adapter_source,
        "local_image_identity": local_image_identity,
        "remote_image_identity": remote_image_identity,
    }
    return "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

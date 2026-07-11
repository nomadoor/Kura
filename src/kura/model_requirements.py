"""Backend-independent projection of model requirements.

This module describes ownership and runtime references. It does not download
models or change backend-native configuration.
"""

from __future__ import annotations

from typing import Any


def _backend_name(run: dict[str, Any]) -> str | None:
    backend = run.get("backend")
    name = backend.get("name") if isinstance(backend, dict) else None
    return name if isinstance(name, str) and name else None


def _backend_override(run: dict[str, Any], backend_name: str) -> dict[str, Any]:
    overrides = run.get("backend_overrides")
    if not isinstance(overrides, dict):
        return {}
    value = overrides.get(backend_name)
    return value if isinstance(value, dict) else {}


def _ai_toolkit_requirements(run: dict[str, Any]) -> list[dict[str, Any]]:
    model = run.get("model") if isinstance(run.get("model"), dict) else {}
    base = model.get("base")
    if not isinstance(base, str) or not base:
        return []
    revision = model.get("revision")
    acquisition = "backend"
    if base.startswith(("/", "./", "../", "~")):
        acquisition = "local-path"
        identity: dict[str, Any] = {"kind": "path", "path": base}
    else:
        identity = {"kind": "huggingface-repository", "repo_id": base}
        if isinstance(revision, str) and revision:
            identity["revision"] = revision
    return [
        {
            "role": "base_model",
            "acquisition": acquisition,
            "identity": identity,
            "runtime_reference": base,
            "expected_format": "backend-native-repository",
            "measurement": {"scope": "backend-runtime", "status": "not-measured-by-kura"},
        }
    ]


def _musubi_requirements(run: dict[str, Any], download_estimate: dict[str, Any]) -> list[dict[str, Any]]:
    requirements: list[dict[str, Any]] = []
    items = download_estimate.get("items")
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            identity = {
                "kind": "huggingface-file",
                "repo_id": item.get("repo_id"),
                "filename": item.get("filename"),
            }
            if item.get("revision"):
                identity["revision"] = item["revision"]
            measurement = {
                "scope": "controller",
                "status": item.get("size_status") or "unknown",
                "size_bytes": item.get("size_bytes"),
                "cached": bool(item.get("cached")),
            }
            if item.get("size_detail"):
                measurement["detail"] = item["size_detail"]
            requirements.append(
                {
                    "role": item.get("key") or "model",
                    "acquisition": "kura",
                    "identity": identity,
                    "runtime_reference": item.get("runtime_reference"),
                    "expected_format": "backend-role-file",
                    "measurement": measurement,
                }
            )

    override = _backend_override(run, "musubi-tuner")
    model_paths = override.get("model_paths")
    if isinstance(model_paths, dict):
        for role, path in sorted(model_paths.items()):
            if not isinstance(role, str) or not isinstance(path, str) or not path:
                continue
            requirements.append(
                {
                    "role": role,
                    "acquisition": "local-path",
                    "identity": {"kind": "path", "path": path},
                    "runtime_reference": path,
                    "expected_format": "backend-role-file",
                    "measurement": {"scope": "compile", "status": "declared"},
                }
            )
    return requirements


def model_requirements(run: dict[str, Any], download_estimate: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return a normalized projection without changing acquisition behavior."""

    backend_name = _backend_name(run)
    if backend_name == "ai-toolkit":
        return _ai_toolkit_requirements(run)
    if backend_name == "musubi-tuner":
        return _musubi_requirements(run, download_estimate or {})
    return []

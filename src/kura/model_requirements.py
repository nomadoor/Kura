"""Backend-independent projection of model requirements.

This module describes ownership and runtime references. It does not download
models or change backend-native configuration.
"""

from __future__ import annotations

from typing import Any

from kura.run_envelope import backend_config, backend_name


def _pinning(identity: dict[str, Any], *, observable: bool) -> dict[str, Any]:
    if isinstance(identity.get("sha256"), str):
        return {"strength": "content-hash", "observation": "observed"}
    revision = identity.get("revision")
    if isinstance(revision, str) and len(revision) >= 40 and all(char in "0123456789abcdefABCDEF" for char in revision):
        return {"strength": "immutable-revision", "observation": "observed"}
    if revision:
        return {"strength": "mutable-reference", "observation": "observed", "detail": "revision is not proven immutable"}
    if identity.get("kind") == "path":
        return {
            "strength": "external-unobserved",
            "observation": "not-observed" if observable else "not-observable",
            "detail": "Kura did not hash the external model path during compile",
        }
    return {
        "strength": "mutable-reference",
        "observation": "not-observed" if observable else "not-observable",
        "detail": "no immutable revision or content hash was observed",
    }


def _backend_name(run: dict[str, Any]) -> str | None:
    return backend_name(run)


def _backend_override(run: dict[str, Any], backend_name: str) -> dict[str, Any]:
    return backend_config(run, backend_name)


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
        expected_format = "backend-native-path"
    else:
        identity = {"kind": "huggingface-repository", "repo_id": base}
        expected_format = "backend-native-repository"
        if isinstance(revision, str) and revision:
            identity["revision"] = revision
    return [
        {
            "role": "base_model",
            "acquisition": acquisition,
            "identity": identity,
            "runtime_reference": base,
            "expected_format": expected_format,
            "measurement": {"scope": "backend-runtime", "status": "not-measured-by-kura"},
            "pinning": _pinning(identity, observable=False),
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
                "scope": item.get("measurement_scope") or "controller",
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
                    "pinning": _pinning(identity, observable=True),
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
                    "pinning": _pinning({"kind": "path", "path": path}, observable=True),
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


def declared_model_requirements(run: dict[str, Any]) -> list[dict[str, Any]]:
    """Project compile-time requirements without network measurement."""

    backend_name = _backend_name(run)
    if backend_name != "musubi-tuner":
        return model_requirements(run)

    override = _backend_override(run, "musubi-tuner")
    model_paths = override.get("model_paths")
    existing_paths = {
        role: path
        for role, path in model_paths.items()
        if isinstance(role, str) and isinstance(path, str) and path
    } if isinstance(model_paths, dict) else {}

    from kura.backends.musubi_models import musubi_model_download_specs

    specs, _ = musubi_model_download_specs(run, existing_paths=existing_paths)
    estimate = {
        "items": [
            {
                "key": item.get("key"),
                "repo_id": item.get("repo_id"),
                "filename": item.get("filename"),
                "revision": item.get("revision"),
                "runtime_reference": item.get("link_path"),
                "size_status": "not-measured",
                "measurement_scope": "compile",
                "size_bytes": None,
                "cached": False,
            }
            for item in specs
        ]
    }
    return _musubi_requirements(run, estimate)

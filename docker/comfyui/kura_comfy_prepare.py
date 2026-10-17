#!/usr/bin/env python3
"""Download and link the ComfyUI models a RunPod render needs, as Kura resolved them at compile."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import shutil
import socket
import urllib.parse
import urllib.request
from typing import Any

# The fields of one model Kura resolved at compile (kura.comfyui_models.resolve_model_specs),
# written to resolved/comfyui_models.json. This script only downloads and links them: Kura decides
# which models a workflow needs and where each comes from.
SPEC_KEYS = frozenset({"node", "class_type", "input", "type", "name", "filename", "target_dir", "target_name", "repo", "url", "revision", "subfolder"})


def _safe_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    root_resolved = root.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"refusing unsafe model target: {relative}") from exc
    return candidate


def _checked_specs(specs: Any) -> list[dict[str, str]]:
    if not isinstance(specs, list):
        raise ValueError("comfyui_models.json must hold a list of resolved model specs")
    checked: list[dict[str, str]] = []
    for spec in specs:
        if not isinstance(spec, dict) or not all(isinstance(value, str) and value for value in spec.values()):
            raise ValueError(f"resolved model spec must map names to non-empty strings: {spec!r}; compile the render again")
        unknown = sorted(set(spec) - SPEC_KEYS)
        if unknown:
            raise ValueError(f"resolved model spec has unsupported key(s): {', '.join(unknown)}; compile the render again")
        if ("repo" in spec) == ("url" in spec) or not all(key in spec for key in ("name", "filename", "target_dir", "target_name")):
            raise ValueError(f"resolved model spec needs exactly one of repo or url, and name, filename, target_dir, target_name: {spec!r}; compile the render again")
        checked.append(spec)
    return checked


def _validate_direct_download_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("direct model downloads require an absolute https:// URL")
    port = parsed.port or 443
    try:
        addresses = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ValueError(f"direct model download host could not be resolved: {parsed.hostname}") from exc
    checked: set[str] = set()
    for item in addresses:
        address = item[4][0]
        if address in checked:
            continue
        checked.add(address)
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError(f"direct model download host resolves to a non-public address: {parsed.hostname}")
    return url


def _download_model(spec: dict[str, str], cache_dir: Path) -> Path:
    if spec.get("url"):
        url = _validate_direct_download_url(spec["url"])
        root = cache_dir
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        target = root / "direct" / digest / Path(spec["filename"]).name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            return target
        temporary = target.with_suffix(target.suffix + ".tmp")
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        temporary.replace(target)
        return target
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        repo_id=spec["repo"],
        filename=spec["filename"],
        subfolder=spec.get("subfolder"),
        revision=spec.get("revision"),
        cache_dir=str(cache_dir),
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None,
    ))


def _require_workspace_cache_dir(cache_dir: Path) -> None:
    workspace = Path(os.environ.get("KURA_WORKSPACE", "/workspace")).resolve()
    resolved = cache_dir.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"ComfyUI model prepare cache_dir must be under {workspace}: {cache_dir}") from exc


def prepare(specs: Any, *, comfyui_root: Path, cache_dir: Path | None) -> list[dict[str, str]]:
    models_root = comfyui_root / "models"
    checked = _checked_specs(specs)
    if checked and cache_dir is None:
        raise ValueError("ComfyUI model prepare requires HF_HUB_CACHE or --cache-dir before downloading models")
    if checked and cache_dir is not None:
        _require_workspace_cache_dir(cache_dir)
    for spec in checked:
        downloaded = _download_model(spec, cache_dir)
        target = _safe_child(models_root, f"{spec['target_dir']}/{spec['target_name']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if target.is_symlink() and target.resolve() == downloaded.resolve():
                continue
            raise ValueError(f"refusing to replace existing ComfyUI model target: {target}")
        os.symlink(downloaded, target)
        print(json.dumps({"event": "model_ready", "model": spec["name"], "target": str(target), "source": str(downloaded)}, ensure_ascii=False), flush=True)
    return checked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("models_json", help="resolved/comfyui_models.json, the models Kura resolved at compile")
    parser.add_argument("--comfyui-root", default=os.environ.get("COMFYUI_ROOT", "/opt/ComfyUI"))
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HUB_CACHE"))
    args = parser.parse_args()
    specs = json.loads(Path(args.models_json).read_text(encoding="utf-8"))
    prepared = prepare(specs, comfyui_root=Path(args.comfyui_root), cache_dir=Path(args.cache_dir) if args.cache_dir else None)
    print(json.dumps({"event": "models_prepared", "count": len(prepared), "models": prepared}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

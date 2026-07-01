#!/usr/bin/env python3
"""Download ComfyUI models declared by Kura into ComfyUI's model folders."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


def _safe_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    root_resolved = root.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"refusing unsafe model target: {relative}") from exc
    return candidate


def prepare(specs: list[dict[str, Any]], *, comfyui_root: Path, cache_dir: Path | None) -> None:
    models_root = comfyui_root / "models"
    for spec in specs:
        repo = spec.get("repo")
        filename = spec.get("filename")
        target_dir = spec.get("target_dir")
        target_name = spec.get("target_name") or filename
        if not all(isinstance(value, str) and value for value in (repo, filename, target_dir, target_name)):
            raise ValueError(f"invalid model spec: {spec}")
        downloaded = hf_hub_download(
            repo_id=repo,
            filename=filename,
            subfolder=spec.get("subfolder") if isinstance(spec.get("subfolder"), str) else None,
            revision=spec.get("revision") if isinstance(spec.get("revision"), str) else None,
            cache_dir=str(cache_dir) if cache_dir else None,
            token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None,
        )
        target = _safe_child(models_root, f"{target_dir}/{target_name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if target.is_symlink() and target.resolve() == Path(downloaded).resolve():
                continue
            target.unlink()
        os.symlink(downloaded, target)
        print(json.dumps({"model": spec.get("name"), "target": str(target), "source": downloaded}, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("specs_json")
    parser.add_argument("--comfyui-root", default=os.environ.get("COMFYUI_ROOT", "/opt/ComfyUI"))
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    args = parser.parse_args()
    data = json.loads(Path(args.specs_json).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("specs_json must contain a list")
    prepare(data, comfyui_root=Path(args.comfyui_root), cache_dir=Path(args.cache_dir) if args.cache_dir else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

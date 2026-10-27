"""`kura init`: create a workspace in the current directory."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from kura.doctor import readiness_gaps
from kura.managed import created_once, is_kura_checkout, plan_restore, record_created_once, sync
from kura.fsio import atomic_write_text
from kura.workspace import WORKSPACE_SCHEMA_VERSION, require_workspace
from kura.yamlio import dump_yaml

DIRECTORIES = (
    "datasets", "runs", "artifacts/training-state", "workflows", "promptsets",
    "cache/huggingface", "cache/models", "knowledge/model-families",
)

# The user's own knowledge layer. Kura writes these once and never again.
USER_KNOWLEDGE = {
    "knowledge/regrets.md": "# Regrets\n\nOne `trigger -> reminder` entry per real regret, with a `source:` line.\n",
    "knowledge/user-preferences.md": "# Preferences\n\nYour standing choices for training and rendering, which agents apply before Kura's defaults.\n",
}

# The ComfyUI folders a local render stages files into. They have no value until the user names
# one, and an empty value is refused, so a new workspace.yaml shows them as commented lines.
COMFYUI_FOLDER_HINTS = (
    "  # lora_dir: /path/to/ComfyUI/models/loras                  # where a render puts the LoRA it tests\n"
    "  # model_patches_dir: /path/to/ComfyUI/models/model_patches # for workflows that load a model patch\n"
    "  # input_dir: /path/to/ComfyUI/input                        # for workflows that read an input image\n"
)


def _new_workspace_yaml(name: str) -> str:
    """The workspace.yaml `kura init` writes: the settings below, with the ComfyUI folder hints
    under `comfyui:`."""
    text = dump_yaml({
        "schema_version": WORKSPACE_SCHEMA_VERSION, "name": name,
        # Off until the user lets agents open dataset and sample images (docs/adr/agents-and-user-images.md).
        "agents": {"view_images": False},
        "comfyui": {"endpoint": "http://127.0.0.1:8188", "lora_stage_subdir": "Kura_tmp", "lora_stage_mode": "auto", "lora_stage_cleanup": "remove_after_render", "model_patch_stage_subdir": "Kura_tmp", "model_patch_stage_mode": "auto", "model_patch_stage_cleanup": "remove_after_render", "model_registry": {}, "runpod": {"gpu_type_ids": ["NVIDIA RTX A5000", "NVIDIA A40"], "container_disk_gb": 80, "ports": ["22/tcp"]}},
        "runpod": {"storage_mode": "upload", "gpu_type_ids": ["NVIDIA RTX A5000", "NVIDIA A40"], "gpu_count": 1, "container_disk_gb": 150, "workspace_path": "/workspace", "backend_ports": {"comfyui": ["22/tcp"]}, "gpu_type_priority": "custom", "interruptible": False},
    })
    return text.replace("\ncomfyui:\n", "\ncomfyui:\n" + COMFYUI_FOLDER_HINTS, 1)


def _enclosing_workspace(root: Path) -> Path | None:
    for parent in root.resolve().parents:
        if (parent / "workspace.yaml").is_file():
            return parent
    return None


def _write_once(path: Path, text: str) -> None:
    """Create `path` with `text` unless it exists."""
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    except FileExistsError:
        return
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)


def cmd_init(args: argparse.Namespace) -> int:
    root = Path.cwd()
    enclosing = _enclosing_workspace(root)
    if enclosing is not None:
        print(
            f"cannot initialize a workspace inside another workspace: {enclosing} "
            "(move or delete its workspace.yaml if that is not a Kura workspace)",
            file=sys.stderr,
        )
        return 1
    if (root / "workspace.yaml").is_file():
        try:
            require_workspace()
        except ValueError as exc:
            print(f"cannot initialize workspace: {exc}", file=sys.stderr)
            return 1
    for relative in DIRECTORIES:
        (root / relative).mkdir(parents=True, exist_ok=True)
    workspace = root / "workspace.yaml"
    if not workspace.exists():
        atomic_write_text(workspace, _new_workspace_yaml(root.name))
    # Written once each; a file the user deleted after that stays deleted.
    already = created_once(root)
    once = list(USER_KNOWLEDGE)
    for relative in once:
        if relative not in already:
            _write_once(root / relative, USER_KNOWLEDGE[relative])
    try:
        record_created_once(root, once)
    except (OSError, ValueError) as exc:
        print(f"cannot record the files kura init wrote: {exc}", file=sys.stderr)
        return 1
    (root / "index.jsonl").touch(exist_ok=True)
    if is_kura_checkout(root):
        print("this is a Kura source checkout: its agent files are maintained by hand, so Kura does not manage them here")
    else:
        restore = bool(getattr(args, "restore", False))
        if restore:
            targets = plan_restore(root)
            if targets:
                print("kura init --restore will replace or recreate these managed files:")
                for path in targets:
                    print(f"  {path}")
                if not getattr(args, "yes", False):
                    if not sys.stdin.isatty():
                        print("re-run with --yes to restore them")
                        return 1
                    if input("restore them? [y/N] ").strip().lower() not in {"y", "yes"}:
                        print("nothing was changed")
                        return 1
            else:
                print("every managed file already matches the shipped version")
        try:
            report = sync(root, restore=restore)
        except (OSError, ValueError) as exc:
            print(f"cannot write Kura's agent files: {exc}", file=sys.stderr)
            return 1
        for line in report.lines():
            print(line)
    print(f"initialized workspace: {root}")
    gaps = readiness_gaps(root)
    if gaps:
        print("not ready yet, for the kind of run each line names:")
        for gap in gaps:
            print(f"  - {gap}")
    else:
        print("basic checks passed; `kura doctor docker` and `kura doctor runpod` check everything a run needs")
    print("next:")
    print("  1. Put a dataset under datasets/<id>/ with dataset.yaml and items.jsonl.")
    print("  2. Check it with: kura dataset validate datasets/<id>")
    print("  3. Tell your AI agent what LoRA or render run you want and which model to use.")
    print("  4. Watch progress with: kura monitor")
    return 0

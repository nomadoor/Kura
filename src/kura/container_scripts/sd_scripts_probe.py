# This script runs inside the sd-scripts container.

import importlib
import json
import os
import pathlib
import subprocess
import sys


SCRIPTS = (
    "train_network.py",
    "sdxl_train_network.py",
    "flux_train_network.py",
    "anima_train_network.py",
    "anima_train_control_net_lllite.py",
    "networks/convert_anima_lora_to_comfy.py",
)
REQUIRED_OPTIONS = {
    "anima_train_network.py": ("--attn_mode",),
    "anima_train_control_net_lllite.py": ("--attn_mode",),
}


def symlink_compatibility(root):
    files = {
        "sd_checkpoint_symlink_safe": (root / "library" / "model_io.py", "os.path.realpath(name_or_path)"),
        "sdxl_checkpoint_symlink_safe": (root / "library" / "sdxl_train_util.py", "os.readlink(name_or_path)"),
    }
    result = {}
    for name, (path, unsafe_call) in files.items():
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        result[name] = bool(text) and unsafe_call not in text
    return result


def main():
    root = pathlib.Path(os.environ.get("SD_SCRIPTS_ROOT", "/opt/sd-scripts"))
    result = {
        "root": str(root),
        "scripts": {},
        "imports": {},
        "torch": {},
        "compatibility": symlink_compatibility(root),
    }
    for name in SCRIPTS:
        path = root / name
        item = {"exists": path.is_file()}
        if path.is_file():
            try:
                process = subprocess.run([sys.executable, str(path), "--help"], cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
            except subprocess.TimeoutExpired:
                item["help_exit_code"] = 1
                item["error"] = "timed out after 120 seconds"
            else:
                item["help_exit_code"] = process.returncode
                required = REQUIRED_OPTIONS.get(name, ())
                item["required_options"] = {option: option in process.stdout for option in required}
                if process.returncode:
                    item["error"] = process.stderr[-1000:]
        result["scripts"][name] = item
    for name in ("torch", "accelerate", "bitsandbytes", "safetensors"):
        try:
            module = importlib.import_module(name)
            result["imports"][name] = getattr(module, "__version__", "present")
        except Exception as exc:
            result["imports"][name] = {"error": str(exc)}
    try:
        import torch

        result["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        result["torch"] = {"error": str(exc)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    failed = [name for name, item in result["scripts"].items() if not item.get("exists") or item.get("help_exit_code") != 0]
    failed += [name for name, item in result["scripts"].items() if any(value is not True for value in item.get("required_options", {}).values())]
    failed += [name for name, value in result["imports"].items() if isinstance(value, dict)]
    failed += [name for name, value in result["compatibility"].items() if value is not True]
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

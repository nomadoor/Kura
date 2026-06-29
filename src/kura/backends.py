"""Backend adapters compile intent; they never execute commands."""

from __future__ import annotations

import json
import shlex
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _datasets(run: dict[str, Any]) -> list[dict[str, Any]]:
    datasets = run.get("datasets")
    if isinstance(datasets, list):
        return [item for item in datasets if isinstance(item, dict)]
    dataset = run.get("dataset")
    if isinstance(dataset, dict):
        return [dataset]
    return []


def _ai_toolkit_datasets(datasets: list[dict[str, Any]], override_folder: Any, resolution: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for index, dataset in enumerate(datasets):
        dataset_id = dataset.get("id", "")
        folder = override_folder if index == 0 and isinstance(override_folder, str) and override_folder else f"/workspace/datasets/{dataset_id}/images"
        entries.append({"folder_path": folder, "caption_ext": ".txt", "cache_latents_to_disk": True, "resolution": resolution})
    return entries


def compile_ai_toolkit(run: dict[str, Any], destination: Path) -> None:
    """Write AI-Toolkit native YAML for configured training runs."""
    override = run.get("backend_overrides", {}).get("ai-toolkit", {})
    params = run.get("params", {})
    model = run.get("model", {})
    datasets = _datasets(run)
    native = override.get("config")
    config = {
        "job": "extension",
        "config": {
            "name": run["id"],
            "process": [{
                "type": "sd_trainer",
                "training_folder": f"/workspace/runs/{run['id']}/outputs",
                "device": "cuda:0",
                "network": {"type": "lora", "linear": params.get("rank"), "linear_alpha": params.get("alpha")},
                "save": {"dtype": "bf16", "save_every": 1, "max_step_saves_to_keep": 1},
                "datasets": _ai_toolkit_datasets(datasets, override.get("dataset_folder"), params.get("resolution")),
                "train": {"batch_size": params.get("batch_size"), "steps": params.get("steps"), "gradient_accumulation_steps": 1, "train_unet": True, "train_text_encoder": False, "gradient_checkpointing": False, "noise_scheduler": "flowmatch", "optimizer": "adamw8bit", "lr": params.get("lr"), "dtype": "bf16", "disable_sampling": True},
                "model": {"name_or_path": model.get("base"), "arch": override.get("model_arch"), "quantize": False, "quantize_te": False, "low_vram": False},
            }],
        },
    }
    process = config["config"]["process"][0]
    if "config" in override and not isinstance(native, dict):
        raise ValueError("backend_overrides.ai-toolkit.config must be a mapping.")
    if isinstance(native, dict):
        for section, values in native.items():
            if section in process and isinstance(process[section], dict) and isinstance(values, dict):
                process[section].update(deepcopy(values))
            else:
                process[section] = deepcopy(values)
    destination.with_suffix(".yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")


def command_ai_toolkit(run: dict[str, Any]) -> dict[str, Any]:
    """Return a container-native command spec, without executing it."""
    command = run.get("backend_overrides", {}).get("ai-toolkit", {}).get("command")
    if command is None:
        return {"cwd": "/opt/ai-toolkit", "argv": ["python", "run.py", f"/workspace/runs/{run['id']}/resolved/ai-toolkit.yaml"], "env": {}}
    if not isinstance(command, dict):
        raise ValueError(
            "AI-Toolkit command is not configured. "
            "Set backend_overrides.ai-toolkit.command."
        )
    cwd, argv, env = command.get("cwd"), command.get("argv"), command.get("env", {})
    if not isinstance(cwd, str) or not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
        raise ValueError("AI-Toolkit command must provide string cwd and argv values.")
    if not isinstance(env, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in env.items()):
        raise ValueError("AI-Toolkit command env must be a string-to-string mapping.")
    if any(any(part in key.upper() for part in ("TOKEN", "SECRET", "PASSWORD", "API_KEY")) for key in env):
        raise ValueError("AI-Toolkit command env must not contain secrets; use the process environment instead.")
    return {"cwd": cwd, "argv": argv, "env": env}


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_scalar(item) for item in value) + "]"
    return json.dumps("" if value is None else str(value))


def _write_musubi_dataset_config(run: dict[str, Any], destination: Path) -> None:
    params = run.get("params", {})
    override = run.get("backend_overrides", {}).get("musubi-tuner", {})
    datasets = _datasets(run)
    if not datasets:
        raise ValueError("Musubi Tuner requires datasets[]")
    general = {
        "resolution": params.get("resolution") or [960, 544],
        "caption_extension": ".txt",
        "batch_size": params.get("batch_size") or 1,
        "enable_bucket": True,
        "bucket_no_upscale": False,
    }
    dataset_config = override.get("dataset_config")
    if isinstance(dataset_config, dict):
        general.update({key: value for key, value in dataset_config.get("general", {}).items() if isinstance(key, str)})
    lines = ["# Generated by Kura for Musubi Tuner.", "[general]"]
    for key, value in general.items():
        if value is not None:
            lines.append(f"{key} = {_toml_scalar(value)}")
    items = _musubi_dataset_items(run, destination, datasets, dataset_config)
    for item in items:
        lines.extend(["", "[[datasets]]"])
        for key, value in item.items():
            if value is not None:
                lines.append(f"{key} = {_toml_scalar(value)}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    referenced_jsonl = {Path(value).name for item in items for key, value in item.items() if key == "image_jsonl_file" and isinstance(value, str)}
    for path in destination.parent.glob("*.jsonl"):
        if path.name not in referenced_jsonl:
            path.unlink()


def _musubi_dataset_items(run: dict[str, Any], destination: Path, datasets: list[dict[str, Any]], dataset_config: Any) -> list[dict[str, Any]]:
    raw_items: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, dataset in enumerate(datasets):
        dataset_id = dataset.get("id", "")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError("Musubi Tuner datasets[].id must name a dataset directory")
        override_item: dict[str, Any] = {}
        if isinstance(dataset_config, dict):
            overrides = dataset_config.get("datasets")
            if isinstance(overrides, list) and index < len(overrides) and isinstance(overrides[index], dict):
                override_item = {key: value for key, value in overrides[index].items() if isinstance(key, str)}
        uses_jsonl = "image_jsonl_file" in override_item or "paired_jsonl" in override_item
        item = {
            "cache_directory": f"/workspace/runs/{run['id']}/cache/musubi/{dataset_id}",
            "num_repeats": dataset.get("num_repeats") or dataset.get("repeats") or 1,
        }
        if not uses_jsonl:
            item["image_directory"] = f"/workspace/datasets/{dataset_id}/images"
        if "paired_jsonl" in override_item:
            paired = override_item.pop("paired_jsonl")
            if isinstance(paired, dict):
                item["_kura_paired_key"] = _paired_dataset_key(dataset_id, paired)
            item["image_jsonl_file"] = _write_musubi_paired_jsonl(run, destination, dataset_id, paired)
        item.update(override_item)
        raw_items.append((dataset, item))
    return _collapse_duplicate_musubi_bucket_items(raw_items)


def _collapse_duplicate_musubi_bucket_items(raw_items: list[tuple[dict[str, Any], dict[str, Any]]]) -> list[dict[str, Any]]:
    """Reject ambiguous duplicate Musubi bucket definitions.

    Musubi's `resolution` is a dataset-block maximum, not a list of choices.
    Defining the same paired dataset twice at 768 and 1024 would either double
    the sample pool or, if silently collapsed, erase the lower-resolution block.
    Kura must not choose either behavior implicitly.  If a run wants mixed
    resolution blocks for the same dataset, the paired JSONL specs must select
    disjoint subsets so the blocks are unambiguous.
    """

    collapsed: list[dict[str, Any]] = []
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for dataset, item in raw_items:
        key = (
            dataset.get("id"),
            item.get("_kura_paired_key") or item.get("image_jsonl_file") or item.get("image_directory"),
            item.get("control_directory"),
            item.get("num_repeats"),
            item.get("batch_size"),
        )
        existing = by_key.get(key)
        if existing is None:
            clean = dict(item)
            by_key[key] = clean
            collapsed.append(clean)
            continue
        raise ValueError(
            "refusing ambiguous Musubi duplicate dataset blocks for "
            f"{dataset.get('id')!r}; split the paired_jsonl inputs into disjoint subsets "
            "instead of repeating the same images at multiple resolutions"
        )
    for item in collapsed:
        item.pop("_kura_paired_key", None)
    return collapsed


def _paired_dataset_key(dataset_id: str, spec: dict[str, Any]) -> tuple[Any, ...]:
    return (
        dataset_id,
        str(_relative_dataset_path(spec.get("target_dir") or spec.get("image_dir") or "target")),
        str(_relative_dataset_path(spec.get("control_dir") or "cond")),
        str(_relative_dataset_path(spec.get("caption_dir") or "caption")),
        _selection_key(spec.get("select")),
    )


def _selection_key(value: Any) -> tuple[Any, ...] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("paired_jsonl select must be a mapping")
    if "modulo" in value or "remainder" in value:
        modulo = _int_or_none(value.get("modulo"))
        remainder = _int_or_none(value.get("remainder"))
        if modulo is None or modulo <= 0:
            raise ValueError("paired_jsonl select.modulo must be a positive integer")
        if remainder is None or remainder < 0 or remainder >= modulo:
            raise ValueError("paired_jsonl select.remainder must be between 0 and modulo-1")
        return ("modulo", modulo, remainder)
    raise ValueError("unsupported paired_jsonl select; supported: {modulo, remainder}")


def _write_musubi_paired_jsonl(run: dict[str, Any], destination: Path, dataset_id: str, spec: Any) -> str:
    if not isinstance(spec, dict):
        raise ValueError("Musubi paired_jsonl must be a mapping")
    target_dir = _relative_dataset_path(spec.get("target_dir") or spec.get("image_dir") or "target")
    control_dir = _relative_dataset_path(spec.get("control_dir") or "cond")
    caption_dir = _relative_dataset_path(spec.get("caption_dir") or "caption")
    filename = str(spec.get("filename") or f"{dataset_id}-{target_dir.name}.jsonl")
    if not filename.endswith(".jsonl") or "/" in filename or "\\" in filename or filename in (".jsonl",):
        raise ValueError(f"invalid paired_jsonl filename: {filename!r}")
    output_dir = destination.parent
    workspace = output_dir.parents[3]
    dataset_root = workspace / "datasets" / dataset_id
    host_target = dataset_root / target_dir
    host_control = dataset_root / control_dir
    host_caption = dataset_root / caption_dir
    for role, path in (("target_dir", host_target), ("control_dir", host_control), ("caption_dir", host_caption)):
        if not path.is_dir():
            raise ValueError(f"paired_jsonl {role} does not exist: {path}")
    target_files = sorted(path for path in host_target.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    selector = _selection_key(spec.get("select"))
    if selector is not None:
        _, modulo, remainder = selector
        target_files = [path for index, path in enumerate(target_files) if index % modulo == remainder]
    if not target_files:
        raise ValueError(f"paired_jsonl target_dir contains no images: {host_target}")
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / filename
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for target in target_files:
            stem = target.stem
            control = _paired_existing_file(host_control, stem, target.suffix)
            caption = host_caption / f"{stem}.txt"
            if control is None:
                raise ValueError(f"paired_jsonl missing control image for {stem!r}")
            if not caption.is_file():
                raise ValueError(f"paired_jsonl missing caption for {stem!r}")
            payload = {
                "image_path": _container_dataset_path(dataset_id, target_dir / target.name),
                "control_path": _container_dataset_path(dataset_id, control_dir / control.name),
                "caption": caption.read_text(encoding="utf-8", errors="replace").strip(),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return f"/workspace/runs/{run['id']}/resolved/musubi/{filename}"


def _relative_dataset_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("paired_jsonl paths must be non-empty relative strings")
    path = Path(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"paired_jsonl path must stay inside the dataset: {value!r}")
    return path


def _paired_existing_file(directory: Path, stem: str, preferred_suffix: str) -> Path | None:
    preferred = directory / f"{stem}{preferred_suffix}"
    if preferred.is_file():
        return preferred
    for suffix in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _container_dataset_path(dataset_id: str, relative: Path) -> str:
    return "/workspace/datasets/" + dataset_id + "/" + "/".join(relative.parts)


def compile_musubi_tuner(run: dict[str, Any], destination: Path) -> None:
    """Write Musubi Tuner native dataset TOML and a readable command manifest."""
    destination.mkdir(parents=True, exist_ok=True)
    _write_musubi_dataset_config(run, destination / "dataset.toml")
    command = command_musubi_tuner(run)
    (destination / "command.json").write_text(json.dumps(command, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (destination / "model-bundle.lock.yaml").write_text(
        yaml.safe_dump(_musubi_model_lock(run), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _musubi_model_paths(run: dict[str, Any]) -> dict[str, str]:
    override = run.get("backend_overrides", {}).get("musubi-tuner", {})
    clean = _musubi_explicit_model_paths(override)
    downloads = _musubi_model_downloads(run, existing_paths=clean)
    clean.update(downloads[1])
    if not clean:
        raise ValueError("Musubi Tuner requires model_paths, model_downloads, or a known model.base bundle")
    return clean


def _musubi_explicit_model_paths(override: dict[str, Any]) -> dict[str, str]:
    paths = override.get("model_paths")
    clean: dict[str, str] = {}
    if isinstance(paths, dict):
        for key, value in paths.items():
            if isinstance(key, str) and isinstance(value, str) and value:
                clean[key] = value
    return clean


def _safe_download_filename(filename: str) -> str:
    if filename.startswith("/") or any(part in ("", ".", "..") for part in filename.split("/")):
        raise ValueError(f"invalid Hugging Face filename for Musubi Tuner: {filename}")
    return filename


def _safe_cache_component(value: str) -> str:
    component = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "-" for ch in value.strip())
    component = component.strip(".-")
    if not component or component in (".", ".."):
        raise ValueError(f"invalid Hugging Face cache component for Musubi Tuner: {value}")
    return component


def _musubi_model_cache_path(repo_id: str, key: str, filename: str) -> str:
    repo_component = _safe_cache_component(repo_id.replace("/", "--"))
    key_component = _safe_cache_component(key)
    return f"/workspace/cache/models/musubi/{repo_component}/{key_component}/{filename}"


def _flux2_klein_bundle(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    model = run.get("model", {})
    base = str(model.get("base") or "").lower().replace("_", "-")
    override = run.get("backend_overrides", {}).get("musubi-tuner", {})
    architecture = _musubi_architecture(run)
    if architecture not in ("flux2", "flux_2"):
        return {}
    model_version = str(override.get("model_version") or "").lower().replace("_", "-")
    bundle = str(override.get("model_bundle") or "auto").lower().replace("_", "-")
    if bundle in ("none", "off", "false"):
        return {}
    is_base_4b = (
        bundle in ("flux2-klein-base-4b", "flux2-klein-base-4b-comfy", "comfy-flux2-klein-base-4b")
        or model_version == "klein-base-4b"
        or base in {"black-forest-labs/flux.2-klein-base-4b", "flux.2-klein-base-4b", "flux2-klein-base-4b"}
    )
    is_distilled_4b = (
        bundle in ("flux2-klein-4b", "flux2-klein-4b-comfy", "comfy-flux2-klein-4b")
        or model_version == "klein-4b"
        or base in {"black-forest-labs/flux.2-klein-4b", "flux.2-klein-4b", "flux2-klein-4b"}
    )
    is_base_9b = (
        bundle in ("flux2-klein-base-9b", "bfl-flux2-klein-base-9b")
        or model_version == "klein-base-9b"
        or base in {"black-forest-labs/flux.2-klein-base-9b", "flux.2-klein-base-9b", "flux2-klein-base-9b"}
    )
    is_distilled_9b = (
        bundle in ("flux2-klein-9b", "bfl-flux2-klein-9b")
        or model_version == "klein-9b"
        or base in {"black-forest-labs/flux.2-klein-9b", "flux.2-klein-9b", "flux2-klein-9b"}
    )
    if is_base_4b or is_distilled_4b:
        repo = "Comfy-Org/vae-text-encorder-for-flux-klein-4b"
        dit_name = "flux-2-klein-4b.safetensors" if is_distilled_4b else "flux-2-klein-base-4b.safetensors"
        return {
            "dit": {"repo": repo, "filename": f"split_files/diffusion_models/{dit_name}"},
            "vae": {"repo": repo, "filename": "split_files/vae/flux2-vae.safetensors"},
            "text_encoder": {"repo": repo, "filename": "split_files/text_encoders/qwen_3_4b.safetensors"},
        }
    if is_base_9b or is_distilled_9b:
        model_repo = "black-forest-labs/FLUX.2-klein-base-9B" if is_base_9b else "black-forest-labs/FLUX.2-klein-9B"
        dit_name = "flux-2-klein-base-9b.safetensors" if is_base_9b else "flux-2-klein-9b.safetensors"
        text_files = [f"text_encoder/model-0000{index}-of-00004.safetensors" for index in range(1, 5)]
        return {
            "dit": {"repo": model_repo, "filename": dit_name},
            "vae": {"repo": model_repo, "filename": "vae/diffusion_pytorch_model.safetensors"},
            "text_encoder": {"repo": model_repo, "filename": text_files[0], "filenames": [*text_files, "text_encoder/model.safetensors.index.json"]},
        }
    return {}


def _krea2_bundle(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    override = run.get("backend_overrides", {}).get("musubi-tuner", {})
    architecture = _musubi_architecture(run)
    if architecture not in ("krea2", "krea_2"):
        return {}
    bundle = str(override.get("model_bundle") or "auto").lower().replace("_", "-")
    if bundle in ("none", "off", "false"):
        return {}
    downloads: dict[str, dict[str, Any]] = {
        "dit": {"repo": "krea/Krea-2-Raw", "filename": "raw.safetensors"},
        "vae": {"repo": "Comfy-Org/Qwen-Image_ComfyUI", "filename": "split_files/vae/qwen_image_vae.safetensors"},
        "text_encoder": {"repo": "Comfy-Org/Qwen3-VL", "filename": "text_encoders/qwen3vl_4b_bf16.safetensors"},
    }
    if _truthy(override.get("include_turbo_dit")):
        downloads["turbo_dit"] = {"repo": "krea/Krea-2-Turbo", "filename": "turbo.safetensors"}
    return downloads


def _known_musubi_bundle(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    downloads = _flux2_klein_bundle(run)
    downloads.update(_krea2_bundle(run))
    return downloads


def _musubi_model_downloads(run: dict[str, Any], existing_paths: dict[str, str] | None = None) -> tuple[list[list[str]], dict[str, str]]:
    override = run.get("backend_overrides", {}).get("musubi-tuner", {})
    existing_paths = existing_paths or {}
    resolved_downloads: dict[str, Any] = _known_musubi_bundle(run)
    downloads = override.get("model_downloads")
    if isinstance(downloads, dict):
        resolved_downloads.update(downloads)
    if not resolved_downloads:
        return [], {}
    download_specs: list[dict[str, str]] = []
    paths: dict[str, str] = {}
    for key, value in resolved_downloads.items():
        if key in existing_paths:
            continue
        if not isinstance(key, str) or not isinstance(value, dict):
            raise ValueError("Musubi Tuner model_downloads must map model keys to download mappings")
        repo_id = value.get("repo_id") or value.get("repo")
        filenames_value = value.get("filenames")
        filenames = [item for item in filenames_value if isinstance(item, str) and item] if isinstance(filenames_value, list) else []
        filename = value.get("filename") or value.get("file") or (filenames[0] if filenames else None)
        if not isinstance(repo_id, str) or not repo_id or not isinstance(filename, str) or not filename:
            raise ValueError(f"Musubi Tuner model_downloads.{key} requires repo_id and filename")
        filename = _safe_download_filename(filename)
        filenames = [_safe_download_filename(item) for item in (filenames or [filename])]
        if value.get("local_dir"):
            raise ValueError("Musubi Tuner model_downloads.local_dir is not supported; use HF_HOME cache or explicit model_paths")
        for item_filename in filenames:
            item = {
                "key": key,
                "repo_id": repo_id,
                "filename": item_filename,
                "link_path": _musubi_model_cache_path(repo_id, key, item_filename),
            }
            revision = value.get("revision")
            if isinstance(revision, str) and revision:
                item["revision"] = revision
            repo_type = value.get("repo_type")
            if isinstance(repo_type, str) and repo_type:
                item["repo_type"] = repo_type
            download_specs.append(item)
        paths[key] = _musubi_model_cache_path(repo_id, key, filename)
    if not download_specs:
        return [], {}
    code = r'''
import json
import os
import subprocess
import sys
import time


def env_int(name, default):
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


ATTEMPTS = env_int("KURA_HF_DOWNLOAD_ATTEMPTS", 4)
POLL_SEC = env_int("KURA_HF_DOWNLOAD_POLL_SEC", 15)
NO_PROGRESS_SEC = env_int("KURA_HF_DOWNLOAD_NO_PROGRESS_SEC", 180)


CHILD = r"""
import json
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import hf_hub_download

item = json.loads(sys.argv[1])
cache_dir = os.environ.get("HF_HOME") or "/root/.cache/huggingface"
kwargs = dict(repo_id=item["repo_id"], filename=item["filename"], cache_dir=cache_dir)
if item.get("revision"):
    kwargs["revision"] = item["revision"]
if item.get("repo_type"):
    kwargs["repo_type"] = item["repo_type"]
path = hf_hub_download(**kwargs)
print(path, flush=True)
"""


def tree_snapshot(directory):
    total = 0
    newest = 0.0
    count = 0
    for root, _, files in os.walk(directory):
        for name in files:
            path = os.path.join(root, name)
            try:
                stat = os.stat(path)
            except OSError:
                continue
            total += stat.st_size
            newest = max(newest, stat.st_mtime)
            count += 1
    return total, newest, count


def repo_cache_dirs(cache_dir, item):
    repo_type = item.get("repo_type") or "model"
    prefix = {"model": "models", "dataset": "datasets", "space": "spaces"}.get(repo_type, f"{repo_type}s")
    repo_dir = f"{prefix}--{item['repo_id'].replace('/', '--')}"
    hub_dir = os.path.join(cache_dir, "hub")
    return [
        os.path.join(hub_dir, repo_dir),
        os.path.join(hub_dir, ".locks", repo_dir),
    ]


def remove_incomplete_files(directories):
    removed = 0
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        for root, _, files in os.walk(directory):
            for name in files:
                if ".incomplete" not in name:
                    continue
                path = os.path.join(root, name)
                try:
                    os.remove(path)
                except OSError:
                    continue
                removed += 1
    return removed


def run_one(item):
    link_path = item["link_path"]
    link_dir = os.path.dirname(link_path)
    os.makedirs(link_dir, exist_ok=True)
    cache_dir = os.environ.get("HF_HOME") or "/root/.cache/huggingface"
    os.makedirs(cache_dir, exist_ok=True)
    label = f"{item['key']}:{item['filename']}"
    last_total, last_mtime, _ = tree_snapshot(cache_dir)
    for attempt in range(1, ATTEMPTS + 1):
        print(f"[kura] hf download start {label} attempt {attempt}/{ATTEMPTS}", flush=True)
        process = subprocess.Popen([sys.executable, "-c", CHILD, json.dumps(item, ensure_ascii=False)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        last_progress = time.monotonic()
        while process.poll() is None:
            time.sleep(POLL_SEC)
            total, newest, count = tree_snapshot(cache_dir)
            if total != last_total or newest != last_mtime:
                last_total, last_mtime = total, newest
                last_progress = time.monotonic()
                print(f"[kura] hf download progress {label} files={count} bytes={total}", flush=True)
                continue
            idle = int(time.monotonic() - last_progress)
            print(f"[kura] hf download idle {label} idle={idle}s bytes={total}", flush=True)
            if idle >= NO_PROGRESS_SEC:
                process.kill()
                process.wait(timeout=30)
                removed = remove_incomplete_files(repo_cache_dirs(cache_dir, item))
                print(f"[kura] hf download stalled {label}; removed {removed} incomplete file(s); retrying", flush=True)
                last_total, last_mtime, _ = tree_snapshot(cache_dir)
                break
        output = ""
        if process.stdout is not None:
            output = process.stdout.read() or ""
        if process.returncode == 0:
            path = output.strip().splitlines()[-1] if output.strip() else ""
            if not path:
                raise SystemExit(f"[kura] hf download did not return a cache path: {label}")
            if os.path.lexists(link_path):
                if os.path.islink(link_path):
                    os.unlink(link_path)
                elif os.path.realpath(link_path) != os.path.realpath(path):
                    raise SystemExit(f"[kura] cannot replace non-symlink model cache path: {link_path}")
            if not os.path.lexists(link_path):
                os.symlink(path, link_path)
            print(f"[kura] downloaded {item['key']} -> {path}", flush=True)
            print(f"[kura] linked {item['key']} -> {link_path}", flush=True)
            return
        if output.strip():
            print(output.strip(), flush=True)
        if attempt < ATTEMPTS:
            time.sleep(min(30, POLL_SEC))
    raise SystemExit(f"[kura] hf download failed after {ATTEMPTS} attempts: {label}")


for item in json.loads(sys.argv[1]):
    run_one(item)
'''.strip()
    return [["python", "-c", code, json.dumps(download_specs, ensure_ascii=False)]], paths


def _musubi_architecture(run: dict[str, Any]) -> str:
    override = run.get("backend_overrides", {}).get("musubi-tuner", {})
    return str(override.get("architecture") or override.get("model_arch") or "flux2").lower().replace("-", "_")


def _unsupported_musubi_adapter_error(architecture: str) -> ValueError:
    return ValueError(
        "unsupported Kura built-in Musubi adapter: "
        f"{architecture}. Musubi Tuner may support this architecture upstream, "
        "but Kura does not generate its command automatically yet. "
        "Use backend_overrides.musubi-tuner.command for an explicit command, "
        "or add a Kura adapter."
    )


def _musubi_flux2_model_version(run: dict[str, Any]) -> str:
    override = run.get("backend_overrides", {}).get("musubi-tuner", {})
    model_version = str(override.get("model_version") or "").lower().replace("_", "-")
    if model_version:
        return model_version

    model = run.get("model", {})
    base = str(model.get("base") or "").lower().replace("_", "-")
    bundle = str(override.get("model_bundle") or "").lower().replace("_", "-")
    candidates = {base, bundle}
    if candidates & {"black-forest-labs/flux.2-klein-base-9b", "flux.2-klein-base-9b", "flux2-klein-base-9b", "bfl-flux2-klein-base-9b"}:
        return "klein-base-9b"
    if candidates & {"black-forest-labs/flux.2-klein-9b", "flux.2-klein-9b", "flux2-klein-9b", "bfl-flux2-klein-9b"}:
        return "klein-9b"
    if candidates & {"black-forest-labs/flux.2-klein-base-4b", "flux.2-klein-base-4b", "flux2-klein-base-4b", "flux2-klein-base-4b-comfy", "comfy-flux2-klein-base-4b"}:
        return "klein-base-4b"
    if candidates & {"black-forest-labs/flux.2-klein-4b", "flux.2-klein-4b", "flux2-klein-4b", "flux2-klein-4b-comfy", "comfy-flux2-klein-4b"}:
        return "klein-4b"
    raise ValueError("Musubi FLUX.2 requires backend_overrides.musubi-tuner.model_version or a recognized model.base/model_bundle; refusing to default to 4B")


def _musubi_model_expectations(run: dict[str, Any]) -> dict[str, str]:
    architecture = _musubi_architecture(run)
    override = run.get("backend_overrides", {}).get("musubi-tuner", {})
    model_version = str(override.get("model_version") or "").lower().replace("_", "-")
    model_base = str(run.get("model", {}).get("base") or "").lower().replace("_", "-")
    text_encoder_format = "qwen3_8b_text_encoder" if "9b" in model_version or "9b" in model_base else "qwen3_4b_text_encoder"
    defaults: dict[str, dict[str, str]] = {
        "flux2": {
            "dit": "flux2_dit",
            "vae": "flux2_vae",
            "text_encoder": text_encoder_format,
        },
        "flux_2": {
            "dit": "flux2_dit",
            "vae": "flux2_vae",
            "text_encoder": text_encoder_format,
        },
        "wan": {
            "dit": "safetensors",
            "vae": "safetensors",
            "t5": "safetensors",
            "clip": "safetensors",
        },
        "krea2": {
            "dit": "safetensors",
            "vae": "safetensors",
            "text_encoder": "safetensors",
            "turbo_dit": "safetensors",
        },
        "krea_2": {
            "dit": "safetensors",
            "vae": "safetensors",
            "text_encoder": "safetensors",
            "turbo_dit": "safetensors",
        },
    }
    expectations = dict(defaults.get(architecture, {}))
    user_expectations = override.get("model_expectations")
    if isinstance(user_expectations, dict):
        for key, value in user_expectations.items():
            if isinstance(key, str) and isinstance(value, str) and value:
                expectations[key] = value
    return expectations


def _musubi_model_sources(run: dict[str, Any], paths: dict[str, str]) -> dict[str, dict[str, str]]:
    override = run.get("backend_overrides", {}).get("musubi-tuner", {})
    explicit_paths = _musubi_explicit_model_paths(override)
    downloads = _known_musubi_bundle(run)
    user_downloads = override.get("model_downloads")
    if isinstance(user_downloads, dict):
        downloads.update({key: value for key, value in user_downloads.items() if isinstance(key, str) and isinstance(value, dict)})
    sources: dict[str, dict[str, str]] = {}
    for role, path in paths.items():
        source: dict[str, str] = {"path": path}
        if role in explicit_paths:
            sources[role] = {**source, "source": "model_paths"}
            continue
        download = downloads.get(role)
        if isinstance(download, dict):
            repo = download.get("repo_id") or download.get("repo")
            filename = download.get("filename") or download.get("file")
            if isinstance(repo, str):
                source["repo"] = repo
            if isinstance(filename, str):
                source["filename"] = filename
            if isinstance(download.get("revision"), str):
                source["revision"] = download["revision"]
        else:
            source["source"] = "model_paths"
        sources[role] = source
    return sources


def _musubi_model_lock(run: dict[str, Any]) -> dict[str, Any]:
    paths = _musubi_model_paths(run)
    expectations = _musubi_model_expectations(run)
    sources = _musubi_model_sources(run, paths)
    return {
        "schema_version": 1,
        "backend": "musubi-tuner",
        "architecture": _musubi_architecture(run),
        "models": [
            {
                "role": role,
                "path": path,
                "expected_format": expectations.get(role, "safetensors"),
                **{key: value for key, value in sources.get(role, {}).items() if key != "path"},
            }
            for role, path in sorted(paths.items())
        ],
        "output": _musubi_output_compatibility(run),
    }


def _musubi_output_compatibility(run: dict[str, Any]) -> dict[str, str]:
    override = run.get("backend_overrides", {}).get("musubi-tuner", {})
    value = override.get("output_compatibility") or override.get("output_format") or "comfyui"
    return {"lora_format": str(value)}


def _safetensors_validator_code() -> str:
    return r'''
import glob
import json
import os
import struct
import sys


def die(message):
    raise SystemExit("[kura] validation failed: " + message)


def read_header(path):
    if not os.path.isfile(path):
        die(f"missing safetensors file: {path}")
    with open(path, "rb") as handle:
        size_raw = handle.read(8)
        if len(size_raw) != 8:
            die(f"not a safetensors file: {path}")
        size = struct.unpack("<Q", size_raw)[0]
        if size <= 0 or size > 100 * 1024 * 1024:
            die(f"invalid safetensors header size in {path}: {size}")
        try:
            header = json.loads(handle.read(size))
        except Exception as exc:
            die(f"invalid safetensors header JSON in {path}: {exc}")
    keys = [key for key in header if key != "__metadata__"]
    metadata = header.get("__metadata__") or {}
    if not keys:
        die(f"safetensors file has no tensors: {path}")
    return keys, metadata


def has_key(keys, name):
    return name in keys


def has_prefix(keys, prefix):
    return any(key.startswith(prefix) for key in keys)


def has_fragment(keys, fragment):
    return any(fragment in key for key in keys)


def validate_model(role, path, expected):
    keys, metadata = read_header(path)
    base = os.path.basename(path).lower()
    if expected == "safetensors":
        return
    if expected in ("flux2_vae", "flux2_ae"):
        if base == "ae.safetensors":
            if expected != "flux2_ae":
                die(f"{role} uses ae.safetensors; this filename is only accepted for explicit FLUX.2 AE bundles")
        diffusers = has_prefix(keys, "encoder.down_blocks.") and has_prefix(keys, "decoder.up_blocks.") and has_prefix(keys, "quant_conv.")
        native = (
            has_prefix(keys, "encoder.down.")
            and has_prefix(keys, "decoder.up.")
            and (has_prefix(keys, "quant_conv.") or has_prefix(keys, "post_quant_conv.") or has_prefix(keys, "decoder.post_quant_conv."))
        )
        if not (diffusers or native):
            die(f"{role} is not recognized as a FLUX.2 VAE/AE: {path}")
        return
    if expected in ("qwen3_4b_text_encoder", "qwen3_8b_text_encoder"):
        if has_prefix(keys, "model.language_model."):
            die(f"{role} looks like a Qwen/VL wrapper checkpoint, not the Qwen3 4B text encoder Musubi expects: {path}")
        if not has_key(keys, "model.embed_tokens.weight"):
            die(f"{role} is missing model.embed_tokens.weight; expected Qwen3 text encoder layout: {path}")
        return
    if expected == "flux2_dit":
        if not (has_prefix(keys, "double_blocks.") or has_prefix(keys, "single_blocks.") or has_prefix(keys, "transformer_blocks.")):
            die(f"{role} is not recognized as a FLUX.2 diffusion transformer: {path}")
        return
    die(f"unknown model expected_format {expected!r} for {role}")


def validate_lora(pattern, architecture, compatibility):
    paths = sorted(glob.glob(pattern))
    if not paths:
        die(f"no LoRA safetensors matched {pattern}")
    for path in paths:
        keys, metadata = read_header(path)
        has_down = any(key.endswith(".lora_down.weight") for key in keys)
        has_up = any(key.endswith(".lora_up.weight") for key in keys)
        has_lora = any(key.startswith("lora_") for key in keys)
        if not (has_lora and has_down and has_up):
            die(f"output is not a recognized LoRA safetensors file: {path}")
        if architecture in ("flux2", "flux_2"):
            module = str(metadata.get("ss_network_module") or "")
            model_spec = str(metadata.get("modelspec.architecture") or "")
            if module and module != "networks.lora_flux_2":
                die(f"FLUX.2 LoRA has unexpected ss_network_module={module!r}: {path}")
            if not any(key.startswith("lora_unet_") for key in keys):
                die(f"FLUX.2 LoRA is missing lora_unet_* keys expected by Musubi/Kohya-style loaders: {path}")
            if compatibility == "comfyui" and model_spec and "Flux.2" not in model_spec and "flux" not in model_spec.lower():
                die(f"LoRA metadata does not identify a FLUX architecture for ComfyUI compatibility target: {path}")


spec = json.loads(sys.argv[1])
for item in spec.get("models", []):
    validate_model(item["role"], item["path"], item.get("expected_format") or "safetensors")
if spec.get("lora"):
    lora = spec["lora"]
    validate_lora(lora["pattern"], spec.get("architecture", ""), lora.get("compatibility", "comfyui"))
print("[kura] validation ok", flush=True)
'''


def _musubi_model_validation_command(run: dict[str, Any], paths: dict[str, str]) -> list[str]:
    expectations = _musubi_model_expectations(run)
    spec = {
        "architecture": _musubi_architecture(run),
        "models": [
            {"role": role, "path": path, "expected_format": expectations.get(role, "safetensors")}
            for role, path in sorted(paths.items())
            if role in expectations or path.endswith(".safetensors")
        ],
    }
    return ["python", "-c", _safetensors_validator_code(), json.dumps(spec, ensure_ascii=False)]


def _musubi_lora_validation_command(run: dict[str, Any], output_dir: str, output_name: str) -> list[str]:
    compatibility = _musubi_output_compatibility(run)["lora_format"].lower()
    spec = {
        "architecture": _musubi_architecture(run),
        "lora": {
            "pattern": f"{output_dir.rstrip('/')}/{output_name}*.safetensors",
            "compatibility": compatibility,
        },
    }
    return ["python", "-c", _safetensors_validator_code(), json.dumps(spec, ensure_ascii=False)]


def _musubi_prune_checkpoints_command(output_dir: str, output_name: str, before_step: Any) -> list[str] | None:
    if before_step in (None, False):
        return None
    try:
        threshold = int(before_step)
    except (TypeError, ValueError) as exc:
        raise ValueError("Musubi Tuner prune_checkpoints_before_step must be an integer") from exc
    if threshold <= 0:
        return None
    code = r'''
import glob
import os
import re
import sys

output_dir, output_name, threshold_raw = sys.argv[1], sys.argv[2], sys.argv[3]
threshold = int(threshold_raw)
pattern = os.path.join(output_dir.rstrip("/"), output_name + "-step*.safetensors")
removed = []
for path in sorted(glob.glob(pattern)):
    match = re.search(r"-step(\d+)\.safetensors$", os.path.basename(path))
    if not match:
        continue
    step = int(match.group(1))
    if step < threshold:
        os.remove(path)
        removed.append(os.path.basename(path))
print(f"[kura] pruned {len(removed)} checkpoints before step {threshold}", flush=True)
'''
    return ["python", "-c", code, output_dir, output_name, str(threshold)]


def _require_paths(paths: dict[str, str], names: tuple[str, ...]) -> list[str]:
    missing = [name for name in names if not paths.get(name)]
    if missing:
        raise ValueError("Musubi Tuner model_paths missing: " + ", ".join(missing))
    return [paths[name] for name in names]


def _script_command(commands: list[list[str]]) -> list[str]:
    lines = [
        "set -euo pipefail",
        'export PATH="/opt/conda/bin:/usr/local/bin:$PATH"',
    ]
    for index, command in enumerate(commands, 1):
        label = "hf_hub_download" if command[:2] == ["python", "-c"] and len(command) > 2 and "hf_hub_download" in command[2] else (command[1] if len(command) > 1 else command[0])
        lines.append(f"echo '[kura] musubi step {index}/{len(commands)}: {shlex.quote(label)}'")
        lines.append(shlex.join(command))
    return ["bash", "-lc", "\n".join(lines)]


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _extra_args(override: dict[str, Any]) -> list[str]:
    extra_args = override.get("extra_args")
    if extra_args is None:
        return []
    if not isinstance(extra_args, list) or not all(isinstance(arg, str) for arg in extra_args):
        raise ValueError("Musubi Tuner extra_args must be a list of strings")
    return list(extra_args)


def _musubi_save_precision(override: dict[str, Any]) -> str:
    value = override.get("save_precision", "bf16")
    if not isinstance(value, str) or value not in {"float", "fp32", "fp16", "bf16"}:
        raise ValueError("Musubi Tuner save_precision must be one of: float, fp32, fp16, bf16")
    return value


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _musubi_micro_batch(run: dict[str, Any], override: dict[str, Any]) -> int | None:
    dataset_config = override.get("dataset_config")
    if isinstance(dataset_config, dict):
        general = dataset_config.get("general")
        if isinstance(general, dict):
            batch_size = _int_or_none(general.get("batch_size"))
            if batch_size is not None:
                return batch_size
        datasets = dataset_config.get("datasets")
        if isinstance(datasets, list):
            for item in datasets:
                if not isinstance(item, dict):
                    continue
                batch_size = _int_or_none(item.get("batch_size"))
                if batch_size is not None:
                    return batch_size
    return _int_or_none(run.get("params", {}).get("batch_size"))


def _musubi_max_resolution(run: dict[str, Any], override: dict[str, Any]) -> int | None:
    values: list[int] = []
    dataset_config = override.get("dataset_config")
    if isinstance(dataset_config, dict):
        general = dataset_config.get("general")
        if isinstance(general, dict):
            resolution = general.get("resolution")
            if isinstance(resolution, list):
                values.extend(value for item in resolution if (value := _int_or_none(item)) is not None)
        datasets = dataset_config.get("datasets")
        if isinstance(datasets, list):
            for item in datasets:
                if not isinstance(item, dict):
                    continue
                for key in ("resolution", "control_resolution"):
                    resolution = item.get(key)
                    if isinstance(resolution, list):
                        values.extend(value for part in resolution if (value := _int_or_none(part)) is not None)
    params_resolution = run.get("params", {}).get("resolution")
    if isinstance(params_resolution, list):
        values.extend(value for item in params_resolution if (value := _int_or_none(item)) is not None)
    return max(values) if values else None


def _validate_musubi_resource_flags(run: dict[str, Any], override: dict[str, Any], architecture: str) -> None:
    extra_args = _extra_args(override)
    if "--block_swap_h2d_only" in extra_args and not (_truthy(override.get("gradient_checkpointing")) or "--gradient_checkpointing" in extra_args):
        raise ValueError("Musubi H2D-only block swap requires explicit gradient_checkpointing")
    if architecture not in ("flux2", "flux_2"):
        return
    model_version = str(override.get("model_version") or "").lower()
    if "9b" not in model_version:
        return
    gpu = str(run.get("compute", {}).get("gpu") or "")
    if "A40" not in gpu.upper():
        return
    micro_batch = _musubi_micro_batch(run, override)
    if micro_batch is None or micro_batch <= 1:
        max_resolution = _musubi_max_resolution(run, override)
        rank = _int_or_none(run.get("params", {}).get("rank")) or _int_or_none(override.get("network_dim"))
        checkpointing = _truthy(override.get("gradient_checkpointing")) or "--gradient_checkpointing" in extra_args
        if max_resolution is not None and max_resolution >= 1024 and (rank is None or rank >= 32) and not checkpointing:
            if _truthy(override.get("allow_a40_uncheckpointed_9b")):
                return
            raise ValueError(
                "Musubi FLUX.2 9B on NVIDIA A40 at 1024-class resolution/rank32 has been observed to OOM "
                "even with batch_size=1. Set backend_overrides.musubi-tuner.gradient_checkpointing: true, "
                "or set allow_a40_uncheckpointed_9b: true to accept the risk."
            )
        return
    if not _truthy(override.get("allow_a40_large_micro_batch")):
        raise ValueError(
            "Musubi FLUX.2 9B on NVIDIA A40 treats batch_size as GPU micro-batch; "
            f"batch_size={micro_batch} has been observed to OOM before step 1. "
            "Use batch_size: 1 and keep the intended effective batch with explicit "
            "--gradient_accumulation_steps, or set backend_overrides.musubi-tuner."
            "allow_a40_large_micro_batch: true to accept the risk."
        )


def _musubi_uses_sample_prompts(override: dict[str, Any], extra_args: list[str]) -> bool:
    if override.get("sample_prompts") or override.get("sample_every_n_steps") or override.get("sample_every_n_epochs"):
        return True
    return any(arg.startswith("--sample_") for arg in extra_args)


def _validate_krea2_dataset_shape(override: dict[str, Any]) -> None:
    dataset_config = override.get("dataset_config")
    if not isinstance(dataset_config, dict):
        return
    datasets = dataset_config.get("datasets")
    if not isinstance(datasets, list):
        return
    forbidden = {"paired_jsonl", "control_directory", "control_resolution", "conditioning_data_dir"}
    for item in datasets:
        if isinstance(item, dict) and any(key in item for key in forbidden):
            raise ValueError("Musubi Krea2 supports plain image/caption datasets only; remove paired/control dataset fields")


def _backend_env(backend_name: str, override: dict[str, Any]) -> dict[str, str]:
    env = override.get("env", {})
    if env is None:
        return {}
    if not isinstance(env, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in env.items()):
        raise ValueError(f"{backend_name} command env must be a string-to-string mapping")
    if any(any(part in key.upper() for part in ("TOKEN", "SECRET", "PASSWORD", "API_KEY")) for key in env):
        raise ValueError(f"{backend_name} command env must not contain secrets; use the process environment instead")
    return dict(env)


def command_musubi_tuner(run: dict[str, Any]) -> dict[str, Any]:
    """Return a Musubi Tuner command spec without executing it."""
    override = run.get("backend_overrides", {}).get("musubi-tuner", {})
    explicit = override.get("command")
    if explicit is not None:
        if not isinstance(explicit, dict):
            raise ValueError("Musubi Tuner command must be a mapping")
        cwd, argv, env = explicit.get("cwd"), explicit.get("argv"), explicit.get("env", {})
        if not isinstance(cwd, str) or not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
            raise ValueError("Musubi Tuner command must provide string cwd and argv values")
        if not isinstance(env, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in env.items()):
            raise ValueError("Musubi Tuner command env must be a string-to-string mapping")
        if any(any(part in key.upper() for part in ("TOKEN", "SECRET", "PASSWORD", "API_KEY")) for key in env):
            raise ValueError("Musubi Tuner command env must not contain secrets; use the process environment instead")
        return {"cwd": cwd, "argv": argv, "env": env}

    architecture = str(override.get("architecture") or override.get("model_arch") or "flux2").lower().replace("-", "_")
    _validate_musubi_resource_flags(run, override, architecture)
    params = run.get("params", {})
    explicit_paths = _musubi_explicit_model_paths(override)
    paths = _musubi_model_paths(run)
    download_commands, _ = _musubi_model_downloads(run, existing_paths=explicit_paths)
    dataset_config = f"/workspace/runs/{run['id']}/resolved/musubi/dataset.toml"
    output_dir = f"/workspace/runs/{run['id']}/outputs"
    output_name = run["id"]
    common = [
        "accelerate", "launch", "--num_cpu_threads_per_process", "1", "--mixed_precision", "bf16",
    ]
    precache = bool(override.get("precache", True))
    if architecture in ("flux2", "flux_2"):
        dit, vae, text_encoder = _require_paths(paths, ("dit", "vae", "text_encoder"))
        model_version = _musubi_flux2_model_version(run)
        train_argv = [
            *common, "src/musubi_tuner/flux_2_train_network.py",
            "--model_version", model_version,
            "--dit", dit, "--vae", vae, "--text_encoder", text_encoder,
            "--dataset_config", dataset_config,
            "--sdpa", "--mixed_precision", "bf16",
            "--timestep_sampling", str(override.get("timestep_sampling") or "flux2_shift"),
            "--weighting_scheme", str(override.get("weighting_scheme") or "none"),
            "--optimizer_type", str(override.get("optimizer_type") or "adamw8bit"),
            "--learning_rate", str(params.get("lr") or override.get("learning_rate") or "1e-4"),
            "--max_data_loader_n_workers", str(override.get("max_data_loader_n_workers") or 2),
            "--persistent_data_loader_workers",
            "--network_module", "networks.lora_flux_2",
            "--network_dim", str(params.get("rank") or override.get("network_dim") or 32),
            "--max_train_steps", str(params.get("steps") or override.get("max_train_steps") or 1600),
            "--save_every_n_steps", str(override.get("save_every_n_steps") or params.get("steps") or 1600),
            "--save_precision", _musubi_save_precision(override),
            "--seed", str(params.get("seed") or override.get("seed") or 42),
            "--output_dir", output_dir, "--output_name", output_name,
        ]
        if _truthy(override.get("gradient_checkpointing")):
            train_argv.append("--gradient_checkpointing")
        if params.get("alpha") is not None:
            train_argv.extend(["--network_alpha", str(params["alpha"])])
        train_argv.extend(_extra_args(override))
        commands = [*download_commands]
        if override.get("validate_models", True):
            commands.append(_musubi_model_validation_command(run, paths))
        if precache:
            latent_argv = [
                "python", "src/musubi_tuner/flux_2_cache_latents.py",
                "--dataset_config", dataset_config,
                "--vae", vae,
                "--model_version", model_version,
                "--skip_existing",
            ]
            if override.get("vae_dtype"):
                latent_argv.extend(["--vae_dtype", str(override["vae_dtype"])])
            text_argv = [
                "python", "src/musubi_tuner/flux_2_cache_text_encoder_outputs.py",
                "--dataset_config", dataset_config,
                "--text_encoder", text_encoder,
                "--batch_size", str(override.get("text_encoder_batch_size") or 1),
                "--model_version", model_version,
                "--skip_existing",
            ]
            if override.get("fp8_text_encoder"):
                text_argv.append("--fp8_text_encoder")
            commands.extend([latent_argv, text_argv])
        commands.append(train_argv)
        prune_command = _musubi_prune_checkpoints_command(output_dir, output_name, override.get("prune_checkpoints_before_step"))
        if prune_command is not None:
            commands.append(prune_command)
        if str(_musubi_output_compatibility(run)["lora_format"]).lower() not in ("none", "off", "false"):
            commands.append(_musubi_lora_validation_command(run, output_dir, output_name))
        argv = _script_command(commands)
    elif architecture == "wan":
        dit, vae, t5 = _require_paths(paths, ("dit", "vae", "t5"))
        task = str(override.get("task") or "t2v-1.3B")
        train_argv = [
            *common, "src/musubi_tuner/wan_train_network.py",
            "--task", task,
            "--dit", dit, "--vae", vae, "--t5", t5,
            "--dataset_config", dataset_config,
            "--sdpa", "--mixed_precision", "bf16",
            "--optimizer_type", str(override.get("optimizer_type") or "adamw8bit"),
            "--learning_rate", str(params.get("lr") or override.get("learning_rate") or "2e-4"),
            "--max_data_loader_n_workers", str(override.get("max_data_loader_n_workers") or 2),
            "--persistent_data_loader_workers",
            "--network_module", "networks.lora_wan",
            "--network_dim", str(params.get("rank") or override.get("network_dim") or 32),
            "--timestep_sampling", str(override.get("timestep_sampling") or "shift"),
            "--discrete_flow_shift", str(override.get("discrete_flow_shift") or "3.0"),
            "--max_train_steps", str(params.get("steps") or override.get("max_train_steps") or 1600),
            "--save_every_n_steps", str(override.get("save_every_n_steps") or params.get("steps") or 1600),
            "--save_precision", _musubi_save_precision(override),
            "--seed", str(params.get("seed") or override.get("seed") or 42),
            "--output_dir", output_dir, "--output_name", output_name,
        ]
        if _truthy(override.get("fp8_base")):
            train_argv.append("--fp8_base")
        if _truthy(override.get("gradient_checkpointing")):
            train_argv.append("--gradient_checkpointing")
        if params.get("alpha") is not None:
            train_argv.extend(["--network_alpha", str(params["alpha"])])
        train_argv.extend(_extra_args(override))
        commands = [*download_commands]
        if override.get("validate_models", True):
            commands.append(_musubi_model_validation_command(run, paths))
        if precache:
            latent_argv = [
                "python", "src/musubi_tuner/wan_cache_latents.py",
                "--dataset_config", dataset_config,
                "--vae", vae,
                "--skip_existing",
            ]
            text_argv = [
                "python", "src/musubi_tuner/wan_cache_text_encoder_outputs.py",
                "--dataset_config", dataset_config,
                "--t5", t5,
                "--batch_size", str(override.get("text_encoder_batch_size") or 1),
                "--skip_existing",
            ]
            if "i2v" in task:
                latent_argv.append("--i2v")
            if paths.get("clip"):
                latent_argv.extend(["--clip", paths["clip"]])
            if override.get("fp8_t5"):
                text_argv.append("--fp8_t5")
            commands.extend([latent_argv, text_argv])
        commands.append(train_argv)
        prune_command = _musubi_prune_checkpoints_command(output_dir, output_name, override.get("prune_checkpoints_before_step"))
        if prune_command is not None:
            commands.append(prune_command)
        if str(_musubi_output_compatibility(run)["lora_format"]).lower() not in ("none", "off", "false"):
            commands.append(_musubi_lora_validation_command(run, output_dir, output_name))
        argv = _script_command(commands)
    elif architecture in ("krea2", "krea_2"):
        _validate_krea2_dataset_shape(override)
        dit, vae, text_encoder = _require_paths(paths, ("dit", "vae", "text_encoder"))
        extra_args = _extra_args(override)
        train_argv = [
            *common, "src/musubi_tuner/krea2_train_network.py",
            "--dit", dit, "--vae", vae,
            "--dataset_config", dataset_config,
            "--sdpa", "--mixed_precision", "bf16",
            "--timestep_sampling", str(override.get("timestep_sampling") or "krea2_shift"),
            "--weighting_scheme", str(override.get("weighting_scheme") or "none"),
            "--optimizer_type", str(override.get("optimizer_type") or "adamw8bit"),
            "--learning_rate", str(params.get("lr") or override.get("learning_rate") or "1e-4"),
            "--max_data_loader_n_workers", str(override.get("max_data_loader_n_workers") or 2),
            "--persistent_data_loader_workers",
            "--network_module", "networks.lora_krea2",
            "--network_dim", str(params.get("rank") or override.get("network_dim") or 32),
            "--network_alpha", str(params.get("alpha") or override.get("network_alpha") or params.get("rank") or override.get("network_dim") or 32),
            "--max_train_steps", str(params.get("steps") or override.get("max_train_steps") or 1600),
            "--save_every_n_steps", str(override.get("save_every_n_steps") or params.get("steps") or 1600),
            "--save_precision", _musubi_save_precision(override),
            "--seed", str(params.get("seed") or override.get("seed") or 42),
            "--output_dir", output_dir, "--output_name", output_name,
        ]
        if _truthy(override.get("gradient_checkpointing")):
            train_argv.append("--gradient_checkpointing")
        if _truthy(override.get("fp8_base")):
            train_argv.extend(["--fp8_base", "--fp8_scaled"])
        elif _truthy(override.get("fp8_scaled")):
            train_argv.append("--fp8_scaled")
        if _musubi_uses_sample_prompts(override, extra_args):
            train_argv.extend(["--text_encoder", text_encoder])
            if paths.get("turbo_dit"):
                train_argv.extend(["--turbo_dit", paths["turbo_dit"]])
        train_argv.extend(extra_args)
        commands = [*download_commands]
        if override.get("validate_models", True):
            commands.append(_musubi_model_validation_command(run, paths))
        if precache:
            commands.extend([
                [
                    "python", "src/musubi_tuner/krea2_cache_latents.py",
                    "--dataset_config", dataset_config,
                    "--vae", vae,
                    "--skip_existing",
                ],
                [
                    "python", "src/musubi_tuner/krea2_cache_text_encoder_outputs.py",
                    "--dataset_config", dataset_config,
                    "--text_encoder", text_encoder,
                    "--batch_size", str(override.get("text_encoder_batch_size") or 1),
                    "--skip_existing",
                ],
            ])
        commands.append(train_argv)
        prune_command = _musubi_prune_checkpoints_command(output_dir, output_name, override.get("prune_checkpoints_before_step"))
        if prune_command is not None:
            commands.append(prune_command)
        if str(_musubi_output_compatibility(run)["lora_format"]).lower() not in ("none", "off", "false"):
            commands.append(_musubi_lora_validation_command(run, output_dir, output_name))
        argv = _script_command(commands)
    else:
        raise _unsupported_musubi_adapter_error(architecture)

    return {"cwd": "/opt/musubi-tuner", "argv": argv, "env": _backend_env("Musubi Tuner", override)}

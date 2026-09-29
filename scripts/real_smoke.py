#!/usr/bin/env python3
"""Prepare and verify real one-step trainer smokes through the normal Kura CLI.

This is a developer acceptance harness, not a release gate. It never launches a
run: `prepare` creates, compiles, and plans runs with `kura run new/compile/plan`
so the user can approve them; the approved launch is an ordinary
`uv run kura run execute <run-id>`; `verify` then checks the recorded result.

Smoke datasets are manifest-v2 datasets under `datasets/real-smoke-*`. They
are created only when absent and are never rewritten: an existing directory is
validated with `kura dataset validate` and used as is.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


IMAGE_DATASET = "real-smoke-image"
CONTROL_DATASET = "real-smoke-image-control"
VIDEO_DATASET = "real-smoke-video"
MUSUBI_IMAGE = "nomadoor/kura-musubi-tuner:dev"
A40 = "NVIDIA A40"


@dataclass(frozen=True)
class Smoke:
    """One real smoke: a backend selector and the run fields that exercise it."""

    backend: str
    architecture: str
    model_base: str
    dataset: str
    config: dict[str, Any]
    executor: str = "runpod"
    gpu: str = A40
    expected_script: str | None = None
    note: str = ""
    dataset_options: dict[str, Any] = field(default_factory=dict)


def _download(repo: str, filename: str, revision: str | None = None) -> dict[str, str]:
    item = {"repo": repo, "filename": filename}
    if revision:
        item["revision"] = revision
    return item


_MUSUBI_COMMON: dict[str, Any] = {
    "resolution": [256, 256],
    "batch_size": 1,
    "learning_rate": 1e-6,
    "network_dim": 1,
    "network_alpha": 1,
    "gradient_checkpointing": True,
    "save_every_n_steps": 1,
}

_VIDEO_ONE_FRAME = {"target_frames": [1], "frame_extraction": "head", "source_fps": 24.0}

_AITK_COMMON: dict[str, Any] = {
    "resolution": [256, 256],
    "batch_size": 1,
    "learning_rate": 1e-6,
    "network_dim": 1,
    "network_alpha": 1,
    "gradient_checkpointing": True,
    "save_every_n_steps": 1,
}


def _sd_scripts_dataset(dataset_id: str) -> dict[str, Any]:
    return {
        "general": {"resolution": [512, 512], "caption_extension": ".txt", "enable_bucket": True, "min_bucket_reso": 256, "max_bucket_reso": 512},
        "datasets": [{"batch_size": 1, "subsets": [{"dataset_id": dataset_id, "num_repeats": 1}]}],
    }


def _musubi(architecture: str, model_base: str, dataset: str, script: str, **config: Any) -> Smoke:
    options = config.pop("dataset_options", {})
    return Smoke("musubi-tuner", architecture, model_base, dataset, {"architecture": architecture, **_MUSUBI_COMMON, **config}, expected_script=script, dataset_options=options)


def _aitk(name: str, model_arch: str, model_base: str, dataset: str = IMAGE_DATASET, **config: Any) -> Smoke:
    # Model sources and memory defaults follow the pinned AI-Toolkit UI entry
    # for each architecture (extensions_built_in/diffusion_models/ui.tsx).
    return Smoke("ai-toolkit", name, model_base, dataset, {"model_arch": model_arch, **_AITK_COMMON, **config}, expected_script="run.py")


SMOKES: dict[str, Smoke] = {
    # sd-scripts: the pre-contract optimizer smokes are historical only.
    "sd-scripts-sdxl": Smoke(
        "sd-scripts", "sdxl", "stabilityai/stable-diffusion-xl-base-1.0", IMAGE_DATASET,
        {
            "architecture": "sdxl", "mode": "lora",
            "model_downloads": {"base": _download("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors", "462165984030d82259a11f4367a4eed129e94a7b")},
            "dataset_config": _sd_scripts_dataset(IMAGE_DATASET),
            "learning_rate": 1e-4, "optimizer_type": "AdamW8bit", "mixed_precision": "bf16", "gradient_checkpointing": True,
            "network_dim": 4, "network_alpha": 1, "network_train_unet_only": True,
            "cache_latents_to_disk": True, "cache_text_encoder_outputs_to_disk": True, "disk_cache_estimate_gb": 1,
        },
        executor="docker", gpu="gpu", expected_script="sdxl_train_network.py",
    ),
    "sd-scripts-flux1": Smoke(
        "sd-scripts", "flux1", "Comfy-Org/flux1-dev", IMAGE_DATASET,
        {
            "architecture": "flux1", "mode": "lora",
            "model_downloads": {
                "dit": _download("Comfy-Org/flux1-dev", "flux1-dev-fp8.safetensors", "0f6b956e6e2e041fb73d079b72ec0e761506f601"),
                "clip_l": _download("comfyanonymous/flux_text_encoders", "clip_l.safetensors", "6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5"),
                "t5xxl": _download("comfyanonymous/flux_text_encoders", "t5xxl_fp16.safetensors", "6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5"),
                "ae": _download("black-forest-labs/FLUX.1-schnell", "ae.safetensors", "741f7c3ce8b383c54771c7003378a50191e9efe9"),
            },
            "dataset_config": _sd_scripts_dataset(IMAGE_DATASET),
            "learning_rate": 1e-4, "optimizer_type": "AdamW8bit", "mixed_precision": "bf16", "gradient_checkpointing": True,
            "network_dim": 4, "network_alpha": 1, "network_train_unet_only": True, "fp8_base": True,
            "cache_latents_to_disk": True, "cache_text_encoder_outputs_to_disk": True, "disk_cache_estimate_gb": 1,
            "timestep_sampling": "flux_shift", "guidance_scale": 1.0, "model_prediction_type": "raw",
        },
        expected_script="flux_train_network.py",
    ),
    "sd-scripts-anima-lora": Smoke(
        "sd-scripts", "anima", "circlestone-labs/Anima", IMAGE_DATASET,
        {
            "architecture": "anima", "mode": "lora",
            "model_downloads": {
                "dit": _download("circlestone-labs/Anima", "split_files/diffusion_models/anima-base-v1.0.safetensors", "f7382c4bf9d7ffe4ceea593a0adbb470c56dd79b"),
                "qwen3": _download("circlestone-labs/Anima", "split_files/text_encoders/qwen_3_06b_base.safetensors", "f7382c4bf9d7ffe4ceea593a0adbb470c56dd79b"),
                "vae": _download("circlestone-labs/Anima", "split_files/vae/qwen_image_vae.safetensors", "f7382c4bf9d7ffe4ceea593a0adbb470c56dd79b"),
            },
            "dataset_config": _sd_scripts_dataset(IMAGE_DATASET),
            "learning_rate": 1e-4, "optimizer_type": "AdamW8bit", "mixed_precision": "bf16", "gradient_checkpointing": True,
            "network_dim": 8, "network_alpha": 1, "network_train_unet_only": True,
            "timestep_sampling": "sigmoid", "discrete_flow_shift": 1.0, "qwen_image_vae_2d": True,
            "cache_latents_to_disk": True, "cache_text_encoder_outputs_to_disk": True, "disk_cache_estimate_gb": 1,
        },
        executor="docker", gpu="gpu", expected_script="anima_train_network.py",
    ),
    "sd-scripts-anima-lllite": Smoke(
        "sd-scripts", "anima", "circlestone-labs/Anima", CONTROL_DATASET,
        {
            "architecture": "anima", "mode": "controlnet_lllite",
            "model_downloads": {
                "dit": _download("circlestone-labs/Anima", "split_files/diffusion_models/anima-base-v1.0.safetensors", "f7382c4bf9d7ffe4ceea593a0adbb470c56dd79b"),
                "qwen3": _download("circlestone-labs/Anima", "split_files/text_encoders/qwen_3_06b_base.safetensors", "f7382c4bf9d7ffe4ceea593a0adbb470c56dd79b"),
                "vae": _download("circlestone-labs/Anima", "split_files/vae/qwen_image_vae.safetensors", "f7382c4bf9d7ffe4ceea593a0adbb470c56dd79b"),
            },
            "dataset_config": _sd_scripts_dataset(CONTROL_DATASET),
            "learning_rate": 5e-5, "optimizer_type": "AdamW8bit", "mixed_precision": "bf16", "gradient_checkpointing": True,
            "timestep_sampling": "shift", "discrete_flow_shift": 3.0, "attn_mode": "sdpa", "qwen_image_vae_2d": True,
            "cond_emb_dim": 32, "lllite_cond_dim": 64, "lllite_cond_resblocks": 1, "lllite_mlp_dim": 64, "lllite_target_layers": "self_attn_q",
            "cache_latents_to_disk": True, "cache_text_encoder_outputs_to_disk": True, "disk_cache_estimate_gb": 1,
        },
        executor="docker", gpu="gpu", expected_script="anima_train_control_net_lllite.py",
    ),
    # Musubi Tuner image families.
    "musubi-flux2-klein-4b": _musubi(
        "flux2", "black-forest-labs/FLUX.2-klein-base-4B", IMAGE_DATASET, "flux_2_train_network.py",
        model_version="klein-base-4b", fp8_base=True, fp8_scaled=True, blocks_to_swap=8,
    ),
    "musubi-krea2": _musubi(
        "krea2", "krea/Krea-2-Raw", IMAGE_DATASET, "krea2_train_network.py",
        fp8_base=True, fp8_scaled=True, blocks_to_swap=26, prune_checkpoints_before_step=1,
    ),
    "musubi-qwen-image": _musubi(
        "qwen_image", "Comfy-Org/Qwen-Image_ComfyUI", IMAGE_DATASET, "qwen_image_train_network.py",
        model_version="original", fp8_base=True, fp8_scaled=True, fp8_vl=True, blocks_to_swap=45,
        model_downloads={
            "dit": _download("Comfy-Org/Qwen-Image_ComfyUI", "split_files/diffusion_models/qwen_image_bf16.safetensors"),
            "text_encoder": _download("Comfy-Org/Qwen-Image_ComfyUI", "split_files/text_encoders/qwen_2.5_vl_7b.safetensors"),
            "vae": _download("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors"),
        },
    ),
    "musubi-zimage": _musubi(
        "zimage", "Comfy-Org/z_image", IMAGE_DATASET, "zimage_train_network.py",
        fp8_base=True, fp8_scaled=True, fp8_llm=True, blocks_to_swap=24,
        model_downloads={
            "dit": _download("Comfy-Org/z_image", "split_files/diffusion_models/z_image_bf16.safetensors"),
            "vae": _download("Comfy-Org/z_image", "split_files/vae/ae.safetensors"),
            "text_encoder": _download("Comfy-Org/z_image", "split_files/text_encoders/qwen_3_4b.safetensors"),
        },
    ),
    "musubi-flux-kontext": _musubi(
        "flux_kontext", "black-forest-labs/FLUX.1-Kontext-dev", CONTROL_DATASET, "flux_kontext_train_network.py",
        fp8_base=True, fp8_scaled=True, blocks_to_swap=24,
        model_downloads={
            "dit": _download("black-forest-labs/FLUX.1-Kontext-dev", "flux1-kontext-dev.safetensors"),
            "vae": _download("black-forest-labs/FLUX.1-Kontext-dev", "ae.safetensors"),
            "text_encoder1": _download("comfyanonymous/flux_text_encoders", "t5xxl_fp8_e4m3fn.safetensors"),
            "text_encoder2": _download("comfyanonymous/flux_text_encoders", "clip_l.safetensors"),
        },
    ),
    "musubi-ideogram4": _musubi(
        "ideogram4", "Comfy-Org/Ideogram-4", IMAGE_DATASET, "ideogram4_train_network.py",
        dit_dtype="bfloat16", blocks_to_swap=24,
        model_downloads={
            "dit": _download("Comfy-Org/Ideogram-4", "diffusion_models/ideogram4_fp8_scaled.safetensors"),
            "vae": _download("Comfy-Org/Ideogram-4", "vae/flux2-vae.safetensors"),
            "text_encoder": _download("Comfy-Org/Ideogram-4", "text_encoders/qwen3vl_8b_fp8_scaled.safetensors"),
        },
    ),
    "musubi-hidream-o1": _musubi(
        "hidream_o1", "Comfy-Org/HiDream-O1-Image", IMAGE_DATASET, "hidream_o1_train_network.py",
        model_type="dev", task="t2i", blocks_to_swap=24, noise_scale_start=7.5, noise_scale_end=7.5, noise_clip_std=2.5,
        model_downloads={"dit": _download("Comfy-Org/HiDream-O1-Image", "checkpoints/hidream_o1_image_dev_bf16.safetensors")},
    ),
    # Musubi Tuner video families.
    "musubi-hunyuan-video": _musubi(
        "hunyuan_video", "hunyuanvideo-community/HunyuanVideo", VIDEO_DATASET, "hv_train_network.py",
        fp8_base=True, blocks_to_swap=36,
        dataset_options={VIDEO_DATASET: dict(_VIDEO_ONE_FRAME)},
        model_downloads={
            "dit": _download("kohya-ss/HunyuanVideo-fp8_e4m3fn-unofficial", "mp_rank_00_model_states_fp8.safetensors"),
            "vae": _download("tencent/HunyuanVideo", "hunyuan-video-t2v-720p/vae/pytorch_model.pt"),
            "text_encoder1": _download("Comfy-Org/HunyuanVideo_repackaged", "split_files/text_encoders/llava_llama3_fp16.safetensors"),
            "text_encoder2": _download("Comfy-Org/HunyuanVideo_repackaged", "split_files/text_encoders/clip_l.safetensors"),
        },
    ),
    "musubi-hunyuan-video-1-5": _musubi(
        "hunyuan_video_1_5", "Comfy-Org/HunyuanVideo_1.5_repackaged", VIDEO_DATASET, "hv_1_5_train_network.py",
        task="t2v", fp8_base=True, fp8_scaled=True, fp8_vl=True, blocks_to_swap=51,
        dataset_options={VIDEO_DATASET: dict(_VIDEO_ONE_FRAME)},
        model_downloads={
            "dit": _download("Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/diffusion_models/hunyuanvideo1.5_720p_t2v_fp16.safetensors"),
            "vae": _download("Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/vae/hunyuanvideo15_vae_fp16.safetensors"),
            "text_encoder": _download("Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/text_encoders/qwen_2.5_vl_7b.safetensors"),
            "byt5": _download("Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors"),
        },
    ),
    "musubi-framepack": _musubi(
        "framepack", "Kijai/HunyuanVideo_comfy", VIDEO_DATASET, "fpack_train_network.py",
        fp8_base=True, fp8_scaled=True, fp8_llm=True, blocks_to_swap=36,
        dataset_options={VIDEO_DATASET: {"target_frames": [37], "frame_extraction": "head", "source_fps": 24.0}},
        model_downloads={
            "dit": _download("Kijai/HunyuanVideo_comfy", "FramePackI2V_HY_bf16.safetensors"),
            "vae": _download("tencent/HunyuanVideo", "hunyuan-video-t2v-720p/vae/pytorch_model.pt"),
            "text_encoder1": _download("Comfy-Org/HunyuanVideo_repackaged", "split_files/text_encoders/llava_llama3_fp16.safetensors"),
            "text_encoder2": _download("Comfy-Org/HunyuanVideo_repackaged", "split_files/text_encoders/clip_l.safetensors"),
            "image_encoder": _download("Comfy-Org/sigclip_vision_384", "sigclip_vision_patch14_384.safetensors"),
        },
    ),
    "musubi-kandinsky5-lite": _musubi(
        "kandinsky5", "kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s", VIDEO_DATASET, "kandinsky5_train_network.py",
        task="k5-lite-t2v-5s-sd", fp8_base=True, fp8_scaled=True, blocks_to_swap=16,
        dataset_options={VIDEO_DATASET: dict(_VIDEO_ONE_FRAME)},
        model_paths={"text_encoder_qwen": "Qwen/Qwen2.5-VL-7B-Instruct", "text_encoder_clip": "openai/clip-vit-large-patch14"},
        model_downloads={
            "dit": _download("kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s", "model/kandinsky5lite_t2v_sft_5s.safetensors"),
            "vae": _download("hunyuanvideo-community/HunyuanVideo", "vae/diffusion_pytorch_model.safetensors"),
        },
    ),
    # AI-Toolkit image families (first-class manifest projections).
    "ai-toolkit-flux": _aitk("flux", "flux", "black-forest-labs/FLUX.1-dev", quantize=True, quantize_te=True),
    "ai-toolkit-flux-kontext": _aitk("flux_kontext", "flux_kontext", "black-forest-labs/FLUX.1-Kontext-dev", CONTROL_DATASET, quantize=True, quantize_te=True),
    "ai-toolkit-flex2": _aitk("flex2", "flex2", "ostris/Flex.2-preview", CONTROL_DATASET, quantize=True, quantize_te=True, bypass_guidance_embedding=True),
    "ai-toolkit-chroma": _aitk("chroma", "chroma", "lodestones/Chroma1-Base", quantize=True, quantize_te=True),
    "ai-toolkit-qwen-image": _aitk("qwen_image", "qwen_image", "Qwen/Qwen-Image", quantize=True, quantize_te=True, low_vram=True),
    "ai-toolkit-qwen-image-edit": _aitk("qwen_image_edit", "qwen_image_edit", "Qwen/Qwen-Image-Edit", CONTROL_DATASET, quantize=True, quantize_te=True, low_vram=True),
    "ai-toolkit-qwen-image-2": _aitk("qwen_image_2", "qwen_image_2", "Comfy-Org/Qwen-Image-2.1", quantize=True, quantize_te=True, low_vram=True),
    "ai-toolkit-anima": _aitk("anima", "anima", "circlestone-labs/Anima-Base-v1.0-Diffusers"),
    "ai-toolkit-mageflow": _aitk("mageflow", "mageflow", "microsoft/Mage-Flow-Base", quantize=True, quantize_te=True, low_vram=True),
    "ai-toolkit-hidream": _aitk("hidream", "hidream", "HiDream-ai/HiDream-I1-Full", quantize=True, quantize_te=True),
    "ai-toolkit-flux2-klein-4b": _aitk("flux2_klein_4b", "flux2_klein_4b", "black-forest-labs/FLUX.2-klein-base-4B", quantize=True, quantize_te=True, low_vram=True),
    "ai-toolkit-krea2": _aitk("krea2", "krea2", "krea/Krea-2-Raw", quantize=True, quantize_te=True, low_vram=True),
    "ai-toolkit-zimage": _aitk("zimage", "zimage", "Tongyi-MAI/Z-Image", quantize=True, quantize_te=True, low_vram=True),
}


def _png(width: int = 256, height: int = 256, *, variant: int = 0) -> bytes:
    rows = b"".join(
        b"\x00" + bytes(channel for x in range(width) for channel in ((x + variant * 97) % 256, y % 256, (x + y) % 256))
        for y in range(height)
    )

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def _write_manifest(root: Path, dataset_id: str, rows: list[dict[str, Any]]) -> None:
    (root / "dataset.yaml").write_text(
        yaml.safe_dump({"id": dataset_id, "items_schema_version": 2, "description": "Generated one-item real-smoke dataset; never rewritten by the harness."}, sort_keys=False),
        encoding="utf-8",
    )
    (root / "items.jsonl").write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def _caption(path: str) -> dict[str, Any]:
    return {"file": {"type": "file", "path": path}}


def _create_image_dataset(root: Path) -> None:
    (root / "0001.png").write_bytes(_png())
    (root / "0001.txt").write_text("a tiny synthetic smoke-test image\n", encoding="utf-8")
    _write_manifest(root, root.name, [{"id": "0001", "files": [{"type": "file", "role": "target", "path": "0001.png"}], "caption": _caption("0001.txt")}])


def _create_control_dataset(root: Path) -> None:
    (root / "target").mkdir()
    (root / "control").mkdir()
    (root / "target" / "0001.png").write_bytes(_png())
    (root / "control" / "0001.png").write_bytes(_png(variant=1))
    (root / "target" / "0001.txt").write_text("a tiny synthetic smoke-test image\n", encoding="utf-8")
    _write_manifest(root, root.name, [{
        "id": "0001",
        "files": [
            {"type": "file", "role": "target", "path": "target/0001.png"},
            {"type": "file", "role": "control", "path": "control/0001.png"},
        ],
        "caption": _caption("target/0001.txt"),
    }])


_VIDEO_GENERATOR = r'''
import cv2, numpy as np, sys
writer = cv2.VideoWriter(sys.argv[1], cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (256, 256))
if not writer.isOpened():
    raise SystemExit("cannot open VideoWriter")
for i in range(37):
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    frame[:, :, 0] = np.arange(256, dtype=np.uint8)[None, :]
    frame[:, :, 1] = np.arange(256, dtype=np.uint8)[:, None]
    frame[220:236, i % 64:64 + i % 64] = (0, 180, 255)
    writer.write(frame)
writer.release()
'''


def _create_video_dataset(root: Path) -> None:
    docker = shutil.which("docker")
    if docker is None:
        raise SystemExit(f"creating {VIDEO_DATASET} needs Docker to encode the MP4 inside {MUSUBI_IMAGE}")
    user = ["--user", f"{os.getuid()}:{os.getgid()}"] if hasattr(os, "getuid") else []
    result = subprocess.run(
        [docker, "run", "--rm", *user, "-v", f"{root}:/out", "--entrypoint", "python", MUSUBI_IMAGE, "-c", _VIDEO_GENERATOR, "/out/0001.mp4"],
        text=True, capture_output=True, check=False, timeout=300,
    )
    if result.returncode:
        raise SystemExit(result.stderr or result.stdout or "video generation failed")
    (root / "0001.txt").write_text("a tiny synthetic smoke-test video\n", encoding="utf-8")
    _write_manifest(root, root.name, [{"id": "0001", "files": [{"type": "file", "role": "target", "path": "0001.mp4"}], "caption": _caption("0001.txt")}])


_CREATORS = {IMAGE_DATASET: _create_image_dataset, CONTROL_DATASET: _create_control_dataset, VIDEO_DATASET: _create_video_dataset}


def ensure_dataset(workspace: Path, dataset_id: str) -> str:
    """Create a smoke dataset once; an existing one is validated, never rewritten."""
    root = workspace / "datasets" / dataset_id
    if root.exists():
        result = _kura(workspace, "dataset", "validate", dataset_id)
        if result.returncode:
            raise SystemExit(f"existing {dataset_id} is not a valid manifest; fix or remove it yourself:\n{result.stdout}{result.stderr}")
        return "existing"
    staging = root.with_name(f".{dataset_id}.creating")
    if staging.exists():
        raise SystemExit(f"{staging} is left from an interrupted creation; remove it and retry")
    staging.mkdir(parents=True)
    _CREATORS[dataset_id](staging)
    staging.rename(root)
    result = _kura(workspace, "dataset", "validate", dataset_id)
    if result.returncode:
        raise SystemExit(f"created {dataset_id} does not validate:\n{result.stdout}{result.stderr}")
    return "created"


def _kura(workspace: Path, *args: str, timeout: float = 600) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "KURA_NOTIFY": "none"}
    return subprocess.run(["uv", "run", "kura", *args], cwd=workspace, text=True, capture_output=True, check=False, env=env, timeout=timeout)


def build_run_fields(smoke_id: str, smoke: Smoke) -> dict[str, Any]:
    """The run.yaml fields a smoke owns; everything else comes from `kura run new`."""
    config = json.loads(json.dumps(smoke.config))
    if smoke.dataset_options:
        config["dataset_options"] = json.loads(json.dumps(smoke.dataset_options))
    return {
        "intent": f"Real one-step {smoke.backend} smoke of {smoke.architecture} through the manifest-v2 handoff ({smoke_id}). Not a quality run.",
        "backend": {"name": smoke.backend, "version": None, "adapter_version": 1, "config": config},
        "model": {"base": smoke.model_base, "revision": None},
        "datasets": [{"id": smoke.dataset, "digest": None, "role": None}],
        "recipe": {"steps": 1, "seed": 1},
        "compute": {
            "executor": smoke.executor,
            "gpu": smoke.gpu,
            # Wait for the recorded GPU instead of failing or silently switching it.
            **({"capacity": {"mode": "wait", "timeout": "30m", "poll_interval": "30s"}} if smoke.executor == "runpod" else {}),
        },
        "safety": {"allow_large_model_downloads": True},
    }


def prepare(workspace: Path, smoke_id: str) -> str:
    smoke = SMOKES[smoke_id]
    ensure_dataset(workspace, smoke.dataset)
    created = _kura(workspace, "run", "new", "--experiment", "real-smoke", "--slug", smoke_id, "--backend", smoke.backend, "--executor", smoke.executor, "--gpu", smoke.gpu)
    match = re.search(r"([0-9]{8}-[0-9]{4}_[a-z0-9-]+_[0-9a-f]{4})", created.stdout + created.stderr)
    if created.returncode or match is None:
        raise SystemExit(f"kura run new failed for {smoke_id}:\n{created.stdout}{created.stderr}")
    run_id = match.group(1)
    run_path = workspace / "runs" / run_id / "run.yaml"
    run = yaml.safe_load(run_path.read_text(encoding="utf-8"))
    run.update(build_run_fields(smoke_id, smoke))
    run_path.write_text(yaml.safe_dump(run, allow_unicode=True, sort_keys=False), encoding="utf-8")
    compiled = _kura(workspace, "run", "compile", run_id)
    if compiled.returncode:
        raise SystemExit(f"{smoke_id} ({run_id}) does not compile:\n{compiled.stdout}{compiled.stderr}")
    return run_id


_LOSS = re.compile(r"(?:\bloss:\s*|\bavr_loss=|\bloss=)([+-]?(?:nan|inf(?:inity)?|(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?))", re.IGNORECASE)


def verify(workspace: Path, run_id: str) -> dict[str, Any]:
    run_dir = workspace / "runs" / run_id
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    command_path = run_dir / "resolved" / "backend-command.lock.json"
    command_text = command_path.read_text(encoding="utf-8", errors="replace") if command_path.is_file() else ""
    stdout_path = run_dir / "logs" / "stdout.log"
    log = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.is_file() else ""
    losses = [float(value) for value in _LOSS.findall(log)]
    outputs = sorted((run_dir / "outputs").glob("*.safetensors"))
    postflight = status.get("dataset_input_postflight") if isinstance(status.get("dataset_input_postflight"), dict) else {}
    slug = run_id.split("_", 1)[1].rsplit("_", 1)[0]
    smoke = SMOKES.get(slug)
    remote = status.get("host") == "runpod"
    checks = {
        "completed": status.get("state") == "completed" and status.get("exit_code") == 0,
        "one_step": status.get("last_step") == 1 and status.get("total_steps") == 1,
        "entrypoint": smoke is None or smoke.expected_script is None or smoke.expected_script in command_text,
        "output": bool(outputs),
        "finite_loss": bool(losses) and all(math.isfinite(loss) for loss in losses),
        "published": status.get("publication_state") == "completed",
        "input_postflight_matched": postflight.get("status") == "matched",
        "pod_stopped": not remote or isinstance(status.get("pod_stopped_at"), str),
    }
    return {"run_id": run_id, "smoke": slug, "checks": checks, "ok": all(checks.values()), "outputs": [path.name for path in outputs]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List the smokes this harness knows")
    prepare_parser = commands.add_parser("prepare", help="Create and compile runs; never launches")
    prepare_parser.add_argument("smoke", nargs="+", choices=sorted(SMOKES))
    verify_parser = commands.add_parser("verify", help="Check a finished run")
    verify_parser.add_argument("run_id", nargs="+")
    args = parser.parse_args()
    workspace = Path.cwd()
    if not (workspace / "workspace.yaml").is_file():
        raise SystemExit("run from a Kura workspace root")
    if args.command == "list":
        for smoke_id, smoke in sorted(SMOKES.items()):
            print(f"{smoke_id:30} {smoke.backend:13} {smoke.architecture:18} {smoke.executor:7} {smoke.gpu:12} {smoke.model_base}")
        return 0
    if args.command == "prepare":
        for smoke_id in args.smoke:
            print(json.dumps({"smoke": smoke_id, "run_id": prepare(workspace, smoke_id)}))
        return 0
    results = [verify(workspace, run_id) for run_id in args.run_id]
    print(json.dumps(results, indent=2))
    return 0 if all(result["ok"] for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())

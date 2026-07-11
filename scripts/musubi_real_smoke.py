#!/usr/bin/env python3
"""Run explicit, real one-step Musubi adapter smoke tests.

This is a developer acceptance test, not a release gate. It downloads or uses
real model files, launches Kura through its normal Docker or RunPod executor,
and requires one actual optimizer step to complete.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SmokeSpec:
    architecture: str
    model_base: str
    dataset_id: str
    model_paths: dict[str, str] | None
    model_downloads: dict[str, dict[str, Any]] | None
    extra_override: dict[str, Any]
    params: dict[str, Any]
    expected_script: str
    expected_outputs: int = 1


SPECS: dict[str, SmokeSpec] = {
    "hunyuan_video": SmokeSpec(
        architecture="hunyuan_video",
        model_base="hunyuanvideo-community/HunyuanVideo",
        dataset_id="musubi-video-smoke",
        model_paths=None,
        model_downloads={
            "dit": {"repo": "kohya-ss/HunyuanVideo-fp8_e4m3fn-unofficial", "filename": "mp_rank_00_model_states_fp8.safetensors"},
            "vae": {"repo": "tencent/HunyuanVideo", "filename": "hunyuan-video-t2v-720p/vae/pytorch_model.pt"},
            "text_encoder1": {"repo": "Comfy-Org/HunyuanVideo_repackaged", "filename": "split_files/text_encoders/llava_llama3_fp16.safetensors"},
            "text_encoder2": {"repo": "Comfy-Org/HunyuanVideo_repackaged", "filename": "split_files/text_encoders/clip_l.safetensors"},
        },
        extra_override={
            "dataset_config": {
                "general": {"resolution": [256, 256], "batch_size": 1},
                "datasets": [
                    {
                        "image_directory": None,
                        "video_directory": "/workspace/datasets/musubi-video-smoke/videos",
                        "target_frames": [1],
                        "frame_extraction": "head",
                        "source_fps": 24.0,
                    }
                ],
            },
            "gradient_checkpointing": True,
            "fp8_base": True,
            "fp8_llm": True,
            "text_encoder_batch_size": 1,
            "vae_chunk_size": 8,
            "vae_tiling": True,
            "extra_args": ["--blocks_to_swap", "36"],
            "save_every_n_steps": 1,
        },
        params={
            "rank": 1,
            "alpha": 1,
            "lr": "1e-6",
            "scheduler": None,
            "steps": 1,
            "batch_size": 1,
            "resolution": [256, 256],
            "seed": 1,
        },
        expected_script="hv_train_network.py",
        expected_outputs=2,
    ),
    "hunyuan_video_1_5": SmokeSpec(
        architecture="hunyuan_video_1_5",
        model_base="Comfy-Org/HunyuanVideo_1.5_repackaged",
        dataset_id="musubi-video-smoke",
        model_paths=None,
        model_downloads={
            "dit": {"repo": "Comfy-Org/HunyuanVideo_1.5_repackaged", "filename": "split_files/diffusion_models/hunyuanvideo1.5_720p_t2v_fp16.safetensors"},
            "vae": {"repo": "Comfy-Org/HunyuanVideo_1.5_repackaged", "filename": "split_files/vae/hunyuanvideo15_vae_fp16.safetensors"},
            "text_encoder": {"repo": "Comfy-Org/HunyuanVideo_1.5_repackaged", "filename": "split_files/text_encoders/qwen_2.5_vl_7b.safetensors"},
            "byt5": {"repo": "Comfy-Org/HunyuanVideo_1.5_repackaged", "filename": "split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors"},
        },
        extra_override={
            "task": "t2v",
            "dataset_config": {
                "general": {"resolution": [256, 256], "batch_size": 1},
                "datasets": [
                    {
                        "image_directory": None,
                        "video_directory": "/workspace/datasets/musubi-video-smoke/videos",
                        "target_frames": [1],
                        "frame_extraction": "head",
                        "source_fps": 24.0,
                    }
                ],
            },
            "gradient_checkpointing": True,
            "fp8_base": True,
            "fp8_scaled": True,
            "fp8_vl": True,
            "text_encoder_batch_size": 1,
            "extra_args": ["--blocks_to_swap", "51", "--vae_sample_size", "128", "--vae_enable_patch_conv"],
            "save_every_n_steps": 1,
        },
        params={
            "rank": 1,
            "alpha": 1,
            "lr": "1e-6",
            "scheduler": None,
            "steps": 1,
            "batch_size": 1,
            "resolution": [256, 256],
            "seed": 1,
        },
        expected_script="hv_1_5_train_network.py",
        expected_outputs=2,
    ),
    "framepack": SmokeSpec(
        architecture="framepack",
        model_base="Kijai/HunyuanVideo_comfy",
        dataset_id="musubi-video-smoke",
        model_paths=None,
        model_downloads={
            "dit": {"repo": "Kijai/HunyuanVideo_comfy", "filename": "FramePackI2V_HY_bf16.safetensors"},
            "vae": {"repo": "tencent/HunyuanVideo", "filename": "hunyuan-video-t2v-720p/vae/pytorch_model.pt"},
            "text_encoder1": {"repo": "Comfy-Org/HunyuanVideo_repackaged", "filename": "split_files/text_encoders/llava_llama3_fp16.safetensors"},
            "text_encoder2": {"repo": "Comfy-Org/HunyuanVideo_repackaged", "filename": "split_files/text_encoders/clip_l.safetensors"},
            "image_encoder": {"repo": "Comfy-Org/sigclip_vision_384", "filename": "sigclip_vision_patch14_384.safetensors"},
        },
        extra_override={
            "dataset_config": {
                "general": {"resolution": [256, 256], "batch_size": 1},
                "datasets": [
                    {
                        "image_directory": None,
                        "video_directory": "/workspace/datasets/musubi-video-smoke/videos",
                        "frame_extraction": "full",
                        "max_frames": 37,
                        "source_fps": 24.0,
                        "fp_latent_window_size": 9,
                    }
                ],
            },
            "gradient_checkpointing": True,
            "fp8_base": True,
            "fp8_scaled": True,
            "fp8_llm": True,
            "text_encoder_batch_size": 1,
            "vae_chunk_size": 8,
            "extra_args": ["--blocks_to_swap", "36", "--latent_window_size", "9"],
            "save_every_n_steps": 1,
        },
        params={
            "rank": 1,
            "alpha": 1,
            "lr": "1e-6",
            "scheduler": None,
            "steps": 1,
            "batch_size": 1,
            "resolution": [256, 256],
            "seed": 1,
        },
        expected_script="fpack_train_network.py",
        expected_outputs=2,
    ),
    "kandinsky5": SmokeSpec(
        architecture="kandinsky5",
        model_base="kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s",
        dataset_id="musubi-video-smoke",
        model_paths={
            "text_encoder_qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
            "text_encoder_clip": "openai/clip-vit-large-patch14",
        },
        model_downloads={
            "dit": {"repo": "kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s", "filename": "model/kandinsky5lite_t2v_sft_5s.safetensors"},
            "vae": {"repo": "hunyuanvideo-community/HunyuanVideo", "filename": "vae/diffusion_pytorch_model.safetensors"},
        },
        extra_override={
            "task": "k5-lite-t2v-5s-sd",
            "dataset_config": {
                "general": {"resolution": [256, 256], "batch_size": 1},
                "datasets": [
                    {
                        "image_directory": None,
                        "video_directory": "/workspace/datasets/musubi-video-smoke/videos",
                        "target_frames": [1],
                        "frame_extraction": "head",
                        "source_fps": 24.0,
                    }
                ],
            },
            "gradient_checkpointing": True,
            "fp8_base": True,
            "fp8_scaled": True,
            "quantized_qwen": True,
            "text_encoder_batch_size": 1,
            "extra_args": ["--blocks_to_swap", "16"],
            "save_every_n_steps": 1,
        },
        params={
            "rank": 1,
            "alpha": 1,
            "lr": "1e-6",
            "scheduler": None,
            "steps": 1,
            "batch_size": 1,
            "resolution": [256, 256],
            "seed": 1,
        },
        expected_script="kandinsky5_train_network.py",
        expected_outputs=2,
    ),
    "flux_kontext": SmokeSpec(
        architecture="flux_kontext",
        model_base="black-forest-labs/FLUX.1-Kontext-dev",
        dataset_id="flux-kontext-smoke",
        model_paths=None,
        model_downloads={
            "dit": {"repo": "black-forest-labs/FLUX.1-Kontext-dev", "filename": "flux1-kontext-dev.safetensors"},
            "vae": {"repo": "black-forest-labs/FLUX.1-Kontext-dev", "filename": "ae.safetensors"},
            "text_encoder1": {"repo": "comfyanonymous/flux_text_encoders", "filename": "t5xxl_fp8_e4m3fn.safetensors"},
            "text_encoder2": {"repo": "comfyanonymous/flux_text_encoders", "filename": "clip_l.safetensors"},
        },
        extra_override={
            "dataset_config": {
                "general": {"resolution": [256, 256], "batch_size": 1},
                "datasets": [
                    {
                        "image_directory": "/workspace/datasets/flux-kontext-smoke/pose/target",
                        "control_directory": "/workspace/datasets/flux-kontext-smoke/pose/cond",
                    }
                ],
            },
            "gradient_checkpointing": True,
            "fp8_base": True,
            "fp8_scaled": True,
            "fp8_t5": True,
            "text_encoder_batch_size": 1,
            "extra_args": ["--timestep_sampling", "flux_shift", "--weighting_scheme", "none", "--blocks_to_swap", "24"],
            "save_every_n_steps": 1,
        },
        params={
            "rank": 1,
            "alpha": 1,
            "lr": "1e-6",
            "scheduler": None,
            "steps": 1,
            "batch_size": 1,
            "resolution": [256, 256],
            "seed": 1,
        },
        expected_script="flux_kontext_train_network.py",
        expected_outputs=2,
    ),
    "hidream_o1": SmokeSpec(
        architecture="hidream_o1",
        model_base="Comfy-Org/HiDream-O1-Image",
        dataset_id="flux2-klein-tiny",
        model_paths=None,
        model_downloads={
            "dit": {"repo": "Comfy-Org/HiDream-O1-Image", "filename": "checkpoints/hidream_o1_image_dev_bf16.safetensors"},
        },
        extra_override={
            "model_type": "dev",
            "task": "t2i",
            "dataset_config": {"general": {"resolution": [256, 256], "batch_size": 1}},
            "gradient_checkpointing": True,
            "skip_t2i_visual_dummy": True,
            "extra_args": ["--blocks_to_swap", "24"],
            "noise_scale_start": 7.5,
            "noise_scale_end": 7.5,
            "noise_clip_std": 2.5,
            "save_every_n_steps": 1,
        },
        params={
            "rank": 1,
            "alpha": 1,
            "lr": "1e-6",
            "scheduler": None,
            "steps": 1,
            "batch_size": 1,
            "resolution": [256, 256],
            "seed": 1,
        },
        expected_script="hidream_o1_train_network.py",
        expected_outputs=2,
    ),
    "ideogram4": SmokeSpec(
        architecture="ideogram4",
        model_base="Comfy-Org/Ideogram-4",
        dataset_id="flux2-klein-tiny",
        model_paths=None,
        model_downloads={
            "dit": {"repo": "Comfy-Org/Ideogram-4", "filename": "diffusion_models/ideogram4_fp8_scaled.safetensors"},
            "vae": {"repo": "Comfy-Org/Ideogram-4", "filename": "vae/flux2-vae.safetensors"},
            "text_encoder": {"repo": "Comfy-Org/Ideogram-4", "filename": "text_encoders/qwen3vl_8b_fp8_scaled.safetensors"},
        },
        extra_override={
            "dataset_config": {"general": {"resolution": [256, 256], "batch_size": 1}},
            "gradient_checkpointing": True,
            "dit_dtype": "bfloat16",
            "vae_dtype": "bfloat16",
            "text_encoder_batch_size": 1,
            "extra_args": ["--blocks_to_swap", "24"],
            "save_every_n_steps": 1,
        },
        params={
            "rank": 1,
            "alpha": 1,
            "lr": "1e-6",
            "scheduler": None,
            "steps": 1,
            "batch_size": 1,
            "resolution": [256, 256],
            "seed": 1,
        },
        expected_script="ideogram4_train_network.py",
        expected_outputs=2,
    ),
    "zimage": SmokeSpec(
        architecture="zimage",
        model_base="Comfy-Org/z_image",
        dataset_id="flux2-klein-tiny",
        model_paths=None,
        model_downloads={
            "dit": {"repo": "Comfy-Org/z_image", "filename": "split_files/diffusion_models/z_image_bf16.safetensors"},
            "vae": {"repo": "Comfy-Org/z_image", "filename": "split_files/vae/ae.safetensors"},
            "text_encoder": {"repo": "Comfy-Org/z_image", "filename": "split_files/text_encoders/qwen_3_4b.safetensors"},
        },
        extra_override={
            "dataset_config": {"general": {"resolution": [256, 256], "batch_size": 1}},
            "gradient_checkpointing": True,
            "fp8_base": True,
            "fp8_scaled": True,
            "fp8_llm": True,
            "extra_args": ["--blocks_to_swap", "24"],
            "save_every_n_steps": 1,
        },
        params={
            "rank": 1,
            "alpha": 1,
            "lr": "1e-6",
            "scheduler": None,
            "steps": 1,
            "batch_size": 1,
            "resolution": [256, 256],
            "seed": 1,
        },
        expected_script="zimage_train_network.py",
        expected_outputs=2,
    ),
    "krea2": SmokeSpec(
        architecture="krea2",
        model_base="krea/Krea-2-Raw",
        dataset_id="flux2-klein-tiny",
        model_paths=None,
        model_downloads=None,
        extra_override={
            "dataset_config": {"general": {"resolution": [256, 256], "batch_size": 1}},
            "gradient_checkpointing": True,
            "fp8_base": True,
            "fp8_scaled": True,
            "extra_args": ["--blocks_to_swap", "26"],
            "save_every_n_steps": 1,
            "prune_checkpoints_before_step": 1,
        },
        params={
            "rank": 1,
            "alpha": 1,
            "lr": "1e-6",
            "scheduler": None,
            "steps": 1,
            "batch_size": 1,
            "resolution": [256, 256],
            "seed": 1,
        },
        expected_script="krea2_train_network.py",
        expected_outputs=2,
    ),
    "qwen_image": SmokeSpec(
        architecture="qwen_image",
        model_base="Comfy-Org/Qwen-Image_ComfyUI",
        dataset_id="flux2-klein-tiny",
        model_paths=None,
        model_downloads={
            "dit": {"repo": "Comfy-Org/Qwen-Image_ComfyUI", "filename": "split_files/diffusion_models/qwen_image_bf16.safetensors"},
            "text_encoder": {"repo": "Comfy-Org/Qwen-Image_ComfyUI", "filename": "split_files/text_encoders/qwen_2.5_vl_7b.safetensors"},
            "vae": {"repo": "Comfy-Org/Qwen-Image_ComfyUI", "filename": "split_files/vae/qwen_image_vae.safetensors"},
        },
        extra_override={
            "model_version": "original",
            "dataset_config": {"general": {"resolution": [256, 256], "batch_size": 1}},
            "gradient_checkpointing": True,
            "fp8_base": True,
            "fp8_scaled": True,
            "fp8_vl": True,
            "extra_args": ["--blocks_to_swap", "45"],
            "save_every_n_steps": 1,
        },
        params={
            "rank": 1,
            "alpha": 1,
            "lr": "1e-6",
            "scheduler": None,
            "steps": 1,
            "batch_size": 1,
            "resolution": [256, 256],
            "seed": 1,
        },
        expected_script="qwen_image_train_network.py",
        expected_outputs=2,
    ),
    "wan": SmokeSpec(
        architecture="wan",
        model_base="Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        dataset_id="flux2-klein-tiny",
        model_paths=None,
        model_downloads={
            "dit": {"repo": "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "filename": "split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors"},
            "vae": {"repo": "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "filename": "split_files/vae/wan_2.1_vae.safetensors"},
            "t5": {"repo": "Wan-AI/Wan2.1-I2V-14B-720P", "filename": "models_t5_umt5-xxl-enc-bf16.pth"},
        },
        extra_override={
            "task": "t2v-1.3B",
            "dataset_config": {"general": {"resolution": [256, 256], "batch_size": 1}},
            "gradient_checkpointing": True,
            "fp8_base": True,
            "save_every_n_steps": 1,
        },
        params={
            "rank": 1,
            "alpha": 1,
            "lr": "1e-6",
            "scheduler": None,
            "steps": 1,
            "batch_size": 1,
            "resolution": [256, 256],
            "seed": 1,
        },
        expected_script="wan_train_network.py",
        expected_outputs=2,
    ),
}

DEFAULT_MUSUBI_IMAGE = "nomadoor/kura-musubi-tuner:dev"


def ensure_generated_dataset(root: Path, dataset_id: str, *, image: str = DEFAULT_MUSUBI_IMAGE) -> None:
    if dataset_id == "musubi-video-smoke":
        dataset_root = root / "datasets" / dataset_id
        video_dir = dataset_root / "videos"
        video_path = video_dir / "0001.mp4"
        caption_path = video_dir / "0001.txt"
        video_dir.mkdir(parents=True, exist_ok=True)
        caption_path.write_text("a tiny synthetic smoke-test video\n", encoding="utf-8")
        if not video_path.is_file():
            image = root / "datasets" / "flux2-klein-tiny" / "images" / "00001.png"
            if not image.is_file():
                raise SystemExit("musubi-video-smoke requires datasets/flux2-klein-tiny/images/00001.png")
            generator = r'''
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

src = Path("/workspace/datasets/flux2-klein-tiny/images/00001.png")
dst = Path("/workspace/datasets/musubi-video-smoke/videos/0001.mp4")
base = Image.open(src).convert("RGB").resize((256, 256))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(dst), fourcc, 24.0, (256, 256))
if not writer.isOpened():
    raise SystemExit("failed to open cv2 VideoWriter for smoke mp4")
for i in range(37):
    frame = base.copy()
    draw = ImageDraw.Draw(frame)
    draw.rectangle((i % 64, 220, 64 + i % 64, 236), fill=(255, 180, 0))
    writer.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
writer.release()
'''
            docker = shutil.which("docker")
            if docker is None:
                raise SystemExit("docker is required to generate the musubi video smoke dataset")
            result = run(
                [
                    docker,
                    "run",
                    "--rm",
                    "--user",
                    f"{os.getuid()}:{os.getgid()}",
                    "-v",
                    f"{root}:/workspace",
                    "-w",
                    "/workspace",
                    "--entrypoint",
                    "python",
                    image,
                    "-c",
                    generator,
                ],
                timeout=120,
            )
            if result.returncode != 0:
                raise SystemExit(result.stderr or result.stdout or "failed to generate musubi video smoke dataset")
        (dataset_root / "dataset.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "id": dataset_id,
                    "modality": "video",
                    "description": "Generated one-item smoke dataset for Musubi video adapter verification.",
                    "source": [],
                    "caption": {"strategy": "manual", "version": 1},
                    "stats": {"count": 1},
                    "layout": {"root": "videos", "video_dir": "videos"},
                    "digest": {"raw": None, "dataset": None},
                },
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        item = {"id": "0001", "path": "videos/0001.mp4", "caption": caption_path.read_text(encoding="utf-8").strip(), "role": "target"}
        (dataset_root / "items.jsonl").write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")
        return
    if dataset_id != "flux-kontext-smoke":
        return
    source_image = root / "datasets" / "flux2-klein-tiny" / "images" / "00001.png"
    source_caption = root / "datasets" / "flux2-klein-tiny" / "images" / "00001.txt"
    if not source_image.is_file() or not source_caption.is_file():
        raise SystemExit("flux-kontext-smoke requires datasets/flux2-klein-tiny/images/00001.png and 00001.txt")
    dataset_root = root / "datasets" / dataset_id
    target_dir = dataset_root / "pose" / "target"
    control_dir = dataset_root / "pose" / "cond"
    caption_dir = dataset_root / "pose" / "caption"
    for directory in (target_dir, control_dir, caption_dir):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_image, target_dir / "0001.png")
    shutil.copy2(source_image, control_dir / "0001.png")
    shutil.copy2(source_caption, target_dir / "0001.txt")
    shutil.copy2(source_caption, caption_dir / "0001.txt")
    (dataset_root / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": dataset_id,
                "modality": "image-pair",
                "description": "Generated one-item smoke dataset for FLUX Kontext adapter verification.",
                "source": [],
                "caption": {"strategy": "manual", "version": 1},
                "stats": {"count": 1},
                "layout": {"root": "pose", "target_dir": "pose/target", "control_dir": "pose/cond", "caption_dir": "pose/caption"},
                "digest": {"raw": None, "dataset": None},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    item = {
        "id": "0001",
        "path": "pose/target/0001.png",
        "caption": source_caption.read_text(encoding="utf-8").strip(),
        "role": "target",
        "control_path": "pose/cond/0001.png",
        "caption_path": "pose/caption/0001.txt",
    }
    (dataset_root / "items.jsonl").write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")


def run(command: list[str], *, env: dict[str, str] | None = None, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False, env=env, timeout=timeout)


def run_disk_safety_gate(*, env: dict[str, str], allow_warnings: bool) -> int:
    result = run(["uv", "run", "kura", "doctor", "disk"], env=env, timeout=180)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode == 0:
        return 0
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        payload = {}
    warnings = payload.get("warnings") if isinstance(payload, dict) else None
    if allow_warnings and isinstance(warnings, list) and warnings:
        print(
            "warning: continuing despite `kura doctor disk` warnings because --allow-disk-warnings was passed",
            file=sys.stderr,
        )
        return 0
    print(
        "real smoke aborted because `kura doctor disk` reported warnings; "
        "fix disk/cache/permission issues first, or pass --allow-disk-warnings after explicit review",
        file=sys.stderr,
    )
    return result.returncode or 1


def workspace_root() -> Path:
    root = Path.cwd()
    if not (root / "workspace.yaml").is_file():
        raise SystemExit("workspace.yaml was not found; run from a Kura workspace root")
    return root


def write_run(root: Path, spec: SmokeSpec, *, executor: str, gpu: str, image: str = DEFAULT_MUSUBI_IMAGE) -> str:
    ensure_generated_dataset(root, spec.dataset_id, image=image)
    run_id = f"{datetime.now():%Y%m%d-%H%M}_musubi-real-smoke-{spec.architecture}_{secrets.token_hex(2)}"
    run_dir = root / "runs" / run_id
    for relative in ("resolved", "logs", "metrics", "samples", "checkpoints", "outputs"):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    override: dict[str, Any] = {
        "architecture": spec.architecture,
        **spec.extra_override,
    }
    if spec.model_paths is not None:
        override["model_paths"] = dict(spec.model_paths)
    if spec.model_downloads is not None:
        override["model_downloads"] = {key: dict(value) for key, value in spec.model_downloads.items()}
    params = dict(spec.params)
    for source, target in (("lr", "learning_rate"), ("rank", "network_dim"), ("alpha", "network_alpha")):
        if params.get(source) is not None:
            override.setdefault(target, params[source])
    general = {}
    if params.get("resolution") is not None:
        general["resolution"] = params["resolution"]
    if params.get("batch_size") is not None:
        general["batch_size"] = params["batch_size"]
    if general:
        dataset_config = override.setdefault("dataset_config", {})
        dataset_config.setdefault("general", {}).update({key: value for key, value in general.items() if key not in dataset_config.get("general", {})})
    run_yaml = {
        "schema_version": 1,
        "id": run_id,
        "type": "train",
        "experiment": "musubi-real-smoke",
        "created": datetime.now().astimezone().isoformat(),
        "created_by": "agent",
        "parent_run": None,
        "intent": "real one-step Musubi adapter smoke with actual model files",
        "model": {"base": spec.model_base, "revision": None},
        "datasets": [{"id": spec.dataset_id, "digest": None, "role": None}],
        "recipe": {key: params[key] for key in ("steps", "seed") if params.get(key) is not None},
        "backend": {"name": "musubi-tuner", "version": None, "adapter_version": 1, "config": override},
        "compute": {"executor": executor, "gpu": gpu},
        "safety": {"allow_large_model_downloads": True},
        "sampling": {"prompts": [], "cadence_steps": None},
    }
    (run_dir / "run.yaml").write_text(yaml.safe_dump(run_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps({"state": "draft", "started": None, "ended": None, "last_step": 0, "total_steps": None, "exit_code": None, "host": None, "outputs": []}, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "plan.md").write_text("# Real smoke plan\n\n", encoding="utf-8")
    (run_dir / "notes.md").write_text("# Notes\n\n", encoding="utf-8")
    for relative in ("logs/events.jsonl", "metrics/metrics.jsonl", "samples/samples.jsonl"):
        (run_dir / relative).touch()
    return run_id


def validate_result(root: Path, run_id: str, spec: SmokeSpec) -> dict[str, Any]:
    run_dir = root / "runs" / run_id
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    stdout_log = run_dir / "logs" / "stdout.log"
    log = stdout_log.read_text(encoding="utf-8", errors="replace") if stdout_log.is_file() else ""
    command_path = run_dir / "resolved" / "musubi" / "command.json"
    command_text = command_path.read_text(encoding="utf-8", errors="replace") if command_path.exists() else ""
    outputs = list((run_dir / "outputs").glob("*.safetensors"))
    outputs.extend((run_dir / "downloads").glob("**/outputs/*.safetensors"))
    checks = {
        "completed": status.get("state") == "completed",
        "exit_code_zero": status.get("exit_code") == 0,
        "one_step": status.get("last_step") == 1 and status.get("total_steps") == 1,
        "script_seen": spec.expected_script in log or spec.expected_script in command_text,
        "loss_seen": "avr_loss=" in log,
        "outputs_seen": len(outputs) >= spec.expected_outputs,
    }
    return {
        "run_id": run_id,
        "architecture": spec.architecture,
        "status": status,
        "outputs": [str(path.relative_to(run_dir)) for path in outputs],
        "checks": checks,
        "ok": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a real one-step Musubi adapter smoke through Kura Docker or RunPod.")
    parser.add_argument("architecture", choices=sorted(SPECS))
    parser.add_argument("--executor", choices=("docker", "runpod"), default="docker")
    parser.add_argument("--gpu", default="gpu", help="GPU selector recorded in run.yaml; RunPod accepts a GPU type such as 'NVIDIA A40'")
    parser.add_argument("--no-launch", action="store_true", help="Create and compile the run, but do not launch it")
    parser.add_argument("--timeout", type=float, default=1800.0, help="Launch timeout in seconds")
    parser.add_argument("--hold-for", default="0", help="RunPod review hold after successful download")
    parser.add_argument("--max-lease", default="4h", help="RunPod max lease safety fuse")
    parser.add_argument("--image", help="Override the Docker/RunPod image for this smoke run")
    parser.add_argument("--allow-disk-warnings", action="store_true", help="Continue after reviewing non-zero `kura doctor disk` warnings")
    args = parser.parse_args()

    root = workspace_root()
    spec = SPECS[args.architecture]
    env = dict(os.environ)
    env["KURA_NOTIFY"] = "none"
    disk_gate = run_disk_safety_gate(env=env, allow_warnings=args.allow_disk_warnings)
    if disk_gate != 0:
        return disk_gate
    run_id = write_run(root, spec, executor=args.executor, gpu=args.gpu, image=args.image or DEFAULT_MUSUBI_IMAGE)
    for command in (
        ["uv", "run", "kura", "run", "compile", run_id],
        ["uv", "run", "kura", "run", "plan", run_id],
    ):
        result = run(command, env=env, timeout=120)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        if result.returncode != 0:
            return result.returncode
    if args.no_launch:
        print(json.dumps({"run_id": run_id, "launched": False}, indent=2))
        return 0
    if args.executor == "docker":
        command = ["uv", "run", "kura", "run", "launch", run_id, "--executor", "docker", "--wait"]
        if args.image:
            command.extend(["--image", args.image])
    else:
        command = [
            "uv", "run", "kura", "run", "remote", run_id,
            "--hold-for", args.hold_for,
            "--max-lease", args.max_lease,
            "--job-timeout", str(int(args.timeout)),
            "--download-attempts", "20",
            "--download-interval", "30",
            "--notify", "none",
        ]
        if args.image:
            command.extend(["--image", args.image])
    result = run(command, env=env, timeout=args.timeout + 600)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    report = validate_result(root, run_id, spec)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if result.returncode == 0 and report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

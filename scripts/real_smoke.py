#!/usr/bin/env python3
"""Prepare and verify real one-step trainer smokes through the normal Kura CLI.

This is a developer acceptance harness, not a release gate. `prepare` creates,
compiles, and plans runs with `kura run new/compile/plan` so the user can
approve them; the approved launch is an ordinary
`uv run kura run execute <run-id>`; `verify` then checks the recorded result.

`conformance` runs the fixed scenario of `docs/adr/promises-and-verification.md`
on each backend in a separate workspace and prints one pass/fail table per
promise. Without `--yes` it only prints its plan; it is the one mode that
launches runs.

Smoke datasets are manifest-v2 datasets under `datasets/real-smoke-*`. They
are created only when absent and are never rewritten: an existing directory is
validated with `kura dataset validate` and used as is.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
import struct
import pickle
import subprocess
import sys
import tempfile
import time
import zipfile
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kura.paths import local_hf_cache  # noqa: E402
from kura.provenance import image_reference_identity  # noqa: E402
from kura.training_artifacts import checkpoint_files, checkpoint_step  # noqa: E402


IMAGE_DATASET = "real-smoke-image"
BUCKET_DATASET = "real-smoke-buckets"
# Three items in three aspect-ratio buckets: at batch size 1 an epoch is three steps.
BUCKET_SIZES = ((256, 256), (320, 192), (192, 320))
CONTROL_DATASET = "real-smoke-image-control"
VIDEO_DATASET = "real-smoke-video"
# FramePack trains at 30 fps. The pinned loader can only drop frames, so a
# 24 fps source never yields a full 37-frame latent window after conversion;
# this video is encoded at 30 fps.
FPS30_VIDEO_DATASET = "real-smoke-video-30fps"
VIDEO_BUCKET_DATASET = "real-smoke-video-buckets"
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


def _sd_scripts_dataset(dataset_id: str, resolution: int = 512) -> dict[str, Any]:
    return {
        "general": {"resolution": [resolution, resolution], "caption_extension": ".txt", "enable_bucket": True, "min_bucket_reso": resolution // 2, "max_bucket_reso": resolution},
        "datasets": [{"batch_size": 1, "subsets": [{"dataset_id": dataset_id, "num_repeats": 1}]}],
    }


def _musubi(architecture: str, model_base: str, dataset: str, script: str, **config: Any) -> Smoke:
    options = config.pop("dataset_options", {})
    return Smoke("musubi-tuner", architecture, model_base, dataset, {"architecture": architecture, **_MUSUBI_COMMON, **config}, expected_script=script, dataset_options=options)


def _aitk(name: str, model_arch: str, model_base: str, dataset: str = IMAGE_DATASET, **config: Any) -> Smoke:
    # Model sources follow the pinned AI-Toolkit UI entry for each
    # architecture. Training defaults (scheduler, precision, quantization,
    # caching) are not set here: the adapter fills them from the pinned UI
    # baseline, so a smoke exercises what a user gets without native overrides.
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
        "framepack", "Kijai/HunyuanVideo_comfy", FPS30_VIDEO_DATASET, "fpack_train_network.py",
        fp8_base=True, fp8_scaled=True, fp8_llm=True, blocks_to_swap=36,
        dataset_options={FPS30_VIDEO_DATASET: {"target_frames": [37], "frame_extraction": "head", "source_fps": 30.0}},
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
    "ai-toolkit-sd1": Smoke(
        "ai-toolkit", "sd1", "hf-internal-testing/tiny-stable-diffusion-pipe", IMAGE_DATASET,
        {"model_arch": "sd1", **_AITK_COMMON}, executor="docker", gpu="gpu", expected_script="run.py",
    ),
    "ai-toolkit-sdxl": Smoke(
        "ai-toolkit", "sdxl", "stabilityai/stable-diffusion-xl-base-1.0", IMAGE_DATASET,
        {"model_arch": "sdxl", **_AITK_COMMON}, executor="docker", gpu="gpu", expected_script="run.py",
    ),
    "ai-toolkit-flux": _aitk("flux", "flux", "black-forest-labs/FLUX.1-dev"),
    "ai-toolkit-flux-kontext": _aitk("flux_kontext", "flux_kontext", "black-forest-labs/FLUX.1-Kontext-dev", CONTROL_DATASET),
    "ai-toolkit-flex2": _aitk("flex2", "flex2", "ostris/Flex.2-preview", CONTROL_DATASET, bypass_guidance_embedding=True),
    "ai-toolkit-chroma": _aitk("chroma", "chroma", "lodestones/Chroma1-Base"),
    "ai-toolkit-qwen-image": _aitk("qwen_image", "qwen_image", "Qwen/Qwen-Image"),
    "ai-toolkit-qwen-image-edit": _aitk("qwen_image_edit", "qwen_image_edit", "Qwen/Qwen-Image-Edit", CONTROL_DATASET),
    "ai-toolkit-qwen-image-2": _aitk("qwen_image_2", "qwen_image_2", "Comfy-Org/Qwen-Image-2.1"),
    "ai-toolkit-anima": _aitk("anima", "anima", "circlestone-labs/Anima-Base-v1.0-Diffusers"),
    "ai-toolkit-hidream": _aitk("hidream", "hidream", "HiDream-ai/HiDream-I1-Full"),
    "ai-toolkit-flux2-klein-4b": _aitk("flux2_klein_4b", "flux2_klein_4b", "black-forest-labs/FLUX.2-klein-base-4B"),
    "ai-toolkit-krea2": _aitk("krea2", "krea2", "krea/Krea-2-Raw"),
    "ai-toolkit-zimage": _aitk("zimage", "zimage", "Tongyi-MAI/Z-Image"),
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
        yaml.safe_dump({"id": dataset_id, "items_schema_version": 2, "description": "Generated real-smoke dataset; never rewritten by the harness."}, sort_keys=False),
        encoding="utf-8",
    )
    (root / "items.jsonl").write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def _caption(path: str) -> dict[str, Any]:
    return {"file": {"type": "file", "path": path}}


def _create_image_dataset(root: Path, dataset_id: str) -> None:
    (root / "0001.png").write_bytes(_png())
    (root / "0001.txt").write_text("a tiny synthetic smoke-test image\n", encoding="utf-8")
    _write_manifest(root, dataset_id, [{"id": "0001", "files": [{"type": "file", "role": "target", "path": "0001.png"}], "caption": _caption("0001.txt")}])


def _create_bucket_dataset(root: Path, dataset_id: str) -> None:
    rows = []
    for index, (width, height) in enumerate(BUCKET_SIZES, start=1):
        name = f"{index:04d}"
        (root / f"{name}.png").write_bytes(_png(width, height, variant=index))
        (root / f"{name}.txt").write_text(f"a tiny synthetic smoke-test image {index}\n", encoding="utf-8")
        rows.append({"id": name, "files": [{"type": "file", "role": "target", "path": f"{name}.png"}], "caption": _caption(f"{name}.txt")})
    _write_manifest(root, dataset_id, rows)


def _create_control_dataset(root: Path, dataset_id: str) -> None:
    (root / "target").mkdir()
    (root / "control").mkdir()
    (root / "target" / "0001.png").write_bytes(_png())
    (root / "control" / "0001.png").write_bytes(_png(variant=1))
    (root / "target" / "0001.txt").write_text("a tiny synthetic smoke-test image\n", encoding="utf-8")
    _write_manifest(root, dataset_id, [{
        "id": "0001",
        "files": [
            {"type": "file", "role": "target", "path": "target/0001.png"},
            {"type": "file", "role": "control", "path": "control/0001.png"},
        ],
        "caption": _caption("target/0001.txt"),
    }])


_VIDEO_GENERATOR = r'''
import cv2, numpy as np, sys
width, height = int(sys.argv[4]), int(sys.argv[5])
writer = cv2.VideoWriter(sys.argv[1], cv2.VideoWriter_fourcc(*"mp4v"), float(sys.argv[3]), (width, height))
if not writer.isOpened():
    raise SystemExit("cannot open VideoWriter")
for i in range(int(sys.argv[2])):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = (np.arange(width) % 256).astype(np.uint8)[None, :]
    frame[:, :, 1] = (np.arange(height) % 256).astype(np.uint8)[:, None]
    frame[height - 36:height - 20, i % 64:64 + i % 64] = (0, 180, 255)
    writer.write(frame)
writer.release()
'''


def _create_video_dataset(root: Path, dataset_id: str, frames: int = 37, fps: float = 24.0, sizes: tuple[tuple[int, int], ...] = ((256, 256),)) -> None:
    docker = shutil.which("docker")
    if docker is None:
        raise SystemExit(f"creating {dataset_id} needs Docker to encode the MP4 inside {MUSUBI_IMAGE}")
    user = ["--user", f"{os.getuid()}:{os.getgid()}"] if hasattr(os, "getuid") else []
    rows = []
    for index, (width, height) in enumerate(sizes, start=1):
        name = f"{index:04d}"
        result = subprocess.run(
            [docker, "run", "--rm", *user, "-v", f"{root}:/out", "--entrypoint", "python", MUSUBI_IMAGE, "-c", _VIDEO_GENERATOR, f"/out/{name}.mp4", str(frames), str(fps), str(width), str(height)],
            text=True, capture_output=True, check=False, timeout=300,
        )
        if result.returncode:
            raise SystemExit(result.stderr or result.stdout or "video generation failed")
        (root / f"{name}.txt").write_text("a tiny synthetic smoke-test video\n", encoding="utf-8")
        rows.append({"id": name, "files": [{"type": "file", "role": "target", "path": f"{name}.mp4"}], "caption": _caption(f"{name}.txt")})
    _write_manifest(root, dataset_id, rows)


_CREATORS = {
    IMAGE_DATASET: _create_image_dataset,
    BUCKET_DATASET: _create_bucket_dataset,
    CONTROL_DATASET: _create_control_dataset,
    VIDEO_DATASET: _create_video_dataset,
    FPS30_VIDEO_DATASET: lambda root, dataset_id: _create_video_dataset(root, dataset_id, frames=45, fps=30.0),
    VIDEO_BUCKET_DATASET: lambda root, dataset_id: _create_video_dataset(root, dataset_id, sizes=BUCKET_SIZES),
}


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
    _CREATORS[dataset_id](staging, dataset_id)
    staging.rename(root)
    result = _kura(workspace, "dataset", "validate", dataset_id)
    if result.returncode:
        raise SystemExit(f"created {dataset_id} does not validate:\n{result.stdout}{result.stderr}")
    return "created"


def _kura_argv(*args: str) -> list[str]:
    # --project runs this checkout's Kura from any workspace.
    return ["uv", "run", "--project", str(REPO), "kura", *args]


def _kura(workspace: Path, *args: str, timeout: float = 600) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "KURA_NOTIFY": "none"}
    return subprocess.run(_kura_argv(*args), cwd=workspace, text=True, capture_output=True, check=False, env=env, timeout=timeout)


def build_run_fields(smoke_id: str, smoke: Smoke, *, gpu: str | None = None, steps: int = 1, intent: str | None = None) -> dict[str, Any]:
    """The run.yaml fields a smoke owns; everything else comes from `kura run new`."""
    config = json.loads(json.dumps(smoke.config))
    if smoke.dataset_options:
        config["dataset_options"] = json.loads(json.dumps(smoke.dataset_options))
    return {
        "intent": intent or f"Real one-step {smoke.backend} smoke of {smoke.architecture} through the manifest-v2 handoff ({smoke_id}). Not a quality run.",
        "backend": {"name": smoke.backend, "adapter_version": 1, "config": config},
        "model": {"base": smoke.model_base, "revision": None},
        "datasets": [{"id": smoke.dataset}],
        "recipe": {"steps": steps, "seed": 1},
        "compute": {
            "executor": smoke.executor,
            "gpu": gpu or smoke.gpu,
            # Wait for the recorded GPU instead of failing or silently switching it.
            **({"capacity": {"mode": "wait", "timeout": "30m", "poll_interval": "30s"}} if smoke.executor == "runpod" else {}),
        },
        "safety": {"allow_large_model_downloads": True},
    }


def prepare(workspace: Path, smoke_id: str, *, gpu: str | None = None) -> str:
    smoke = SMOKES[smoke_id]
    if gpu and smoke.executor != "runpod":
        raise SystemExit(f"{smoke_id} runs on {smoke.executor}; --gpu selects a RunPod GPU type")
    ensure_dataset(workspace, smoke.dataset)
    return _create_run(workspace, "real-smoke", smoke_id, replace(smoke, gpu=gpu or smoke.gpu), build_run_fields(smoke_id, smoke, gpu=gpu))


def _create_run(workspace: Path, experiment: str, slug: str, smoke: Smoke, fields: dict[str, Any]) -> str:
    """Create a run with `kura run new`, set the fields the harness owns, and compile it."""
    created = _kura(workspace, "run", "new", "--experiment", experiment, "--slug", slug, "--backend", smoke.backend, "--executor", smoke.executor, "--gpu", smoke.gpu)
    match = re.search(r"([0-9]{8}-[0-9]{4}_[a-z0-9-]+_[0-9a-f]{4})", created.stdout + created.stderr)
    if created.returncode or match is None:
        raise SystemExit(f"kura run new failed for {slug}:\n{created.stdout}{created.stderr}")
    run_id = match.group(1)
    run_path = workspace / "runs" / run_id / "run.yaml"
    run = yaml.safe_load(run_path.read_text(encoding="utf-8"))
    run.update(fields)
    run_path.write_text(yaml.safe_dump(run, allow_unicode=True, sort_keys=False), encoding="utf-8")
    _compile(workspace, run_id)
    return run_id


def _compile(workspace: Path, run_id: str) -> None:
    compiled = _kura(workspace, "run", "compile", run_id)
    if compiled.returncode:
        raise SystemExit(f"{run_id} does not compile:\n{compiled.stdout}{compiled.stderr}")


_LOSS = re.compile(r"(?:\bloss:\s*|\bavr_loss=|\bloss=)([+-]?(?:nan|inf(?:inity)?|(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?))", re.IGNORECASE)


def verify(workspace: Path, run_id: str) -> dict[str, Any]:
    run_dir = workspace / "runs" / run_id
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    command_path = run_dir / "resolved" / "backend-command.lock.json"
    command_text = command_path.read_text(encoding="utf-8", errors="replace") if command_path.is_file() else ""
    stdout_path = run_dir / "logs" / "stdout.log"
    log = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.is_file() else ""
    losses = [float(value) for value in _LOSS.findall(log)]
    # Published outputs as Kura recorded them; local AI-Toolkit nests them
    # under outputs/<run-id>/.
    outputs = sorted(
        run_dir / str(item) for item in status.get("outputs") or []
        if str(item).endswith(".safetensors") and (run_dir / str(item)).is_file()
    )
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
    return {"run_id": run_id, "smoke": slug, "checks": checks, "ok": all(checks.values()), "outputs": [path.relative_to(run_dir).as_posix() for path in outputs]}


def evidence(workspace: Path, run_id: str, *, artifact: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the backend-smoke-evidence record and the campaign summary for a verified run."""
    result = verify(workspace, run_id)
    if not result["ok"]:
        raise SystemExit(f"{run_id} did not pass verify: {result['checks']}")
    run_dir = workspace / "runs" / run_id
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    realization = json.loads((run_dir / status["last_realization"]).read_text(encoding="utf-8"))
    run = yaml.safe_load((run_dir / "run.yaml").read_text(encoding="utf-8"))
    smoke = SMOKES[result["smoke"]]
    config = run["backend"]["config"]
    image = realization["image_identity"]["pinning"]["value"]
    native: dict[str, Any] = {"architecture": smoke.architecture}
    for key in ("mode", "model_arch", "model_version", "task", "model_type"):
        if key in config:
            native[key] = config[key]
    native.update({"executor": realization["executor"] if isinstance(realization.get("executor"), str) else smoke.executor, "dataset": smoke.dataset})
    record: dict[str, Any] = {
        # Date and launch minute keep several smokes of one selector apart.
        "id": f"{result['smoke']}-{run_id[:4]}-{run_id[4:6]}-{run_id[6:8]}-{run_id[9:13]}",
        "backend": smoke.backend,
        "adapter_source": {"kind": realization["adapter_source"]["kind"], "value": realization["adapter_source"]["value"]},
    }
    if smoke.executor == "runpod":
        from kura.provenance import executor_source_identity

        native["transfer"] = "selected-files"
        record["executor_source"] = {"kind": "source-tree-sha256", "value": executor_source_identity("runpod")["value"]}
    native["optimizer_steps"] = 1
    record.update({
        "runtime_image": {"kind": "docker-image-digest", "value": image},
        "native_path": native,
        "evidence_kind": "real-optimizer-step",
        "outcome": "passed",
        "observed_at": f"{run_id[:4]}-{run_id[4:6]}-{run_id[6:8]}",
        "artifact": artifact,
    })
    log = (run_dir / "logs" / "stdout.log").read_text(encoding="utf-8", errors="replace")
    pod = realization.get("pod") if isinstance(realization.get("pod"), dict) else {}
    summary = {
        "run_id": run_id,
        "model": smoke.model_base,
        "dataset": smoke.dataset,
        "image": realization["image_identity"]["reference"],
        "losses": [float(value) for value in _LOSS.findall(log)],
        "outputs": result["outputs"],
        "input_postflight": status["dataset_input_postflight"]["status"],
        "publication": status["publication_state"],
    }
    if pod:
        machine = pod.get("machine") if isinstance(pod.get("machine"), dict) else {}
        summary.update({"gpu": machine.get("gpu_display_name"), "cost_per_h": pod.get("cost_per_h"), "launched_at": realization.get("launched_at"), "pod_stopped_at": status.get("pod_stopped_at")})
    return record, summary


# Conformance: the fixed scenario of docs/adr/promises-and-verification.md, section 2.
#
# Every step count below is chosen so the checks need no trainer arithmetic beyond
# `expected_saves`: each Resume starts on a multiple of the cadence, so a trainer that
# counts its Resume from zero saves on the same logical steps as one that continues the
# logical count; the bucket dataset has three items, so at batch size 1 an epoch is three
# steps, its split run passes the first epoch boundary, and its target is not a multiple
# of three.
CADENCE = 2
KEEP_GENERATIONS = 2  # Kura's default `recovery.training_state.keep_generations`; the scenario leaves it unset.
PROMISES = ("P1", "P2", "P3", "P4", "P6", "P7", "P8")


@dataclass(frozen=True)
class ScenarioRun:
    key: str
    dataset: str  # "one" item or "multi" items in buckets; `conformance_datasets` names them per backend
    steps: int  # a fresh run's recipe steps, or a Resume's target step
    resume_of: str | None = None


SCENARIO = (
    ScenarioRun("one-control", "one", 4),
    ScenarioRun("one-control-2", "one", 4),
    ScenarioRun("one-split", "one", 2),
    ScenarioRun("one-resume", "one", 4, resume_of="one-split"),
    ScenarioRun("multi-fresh", "multi", 7),
    ScenarioRun("multi-split", "multi", 4),
    ScenarioRun("multi-resume", "multi", 7, resume_of="multi-split"),
)
# A Resume and the uninterrupted runs it is checked against (P3, `resume_problems`): two
# controls on one item, whose learned state the Resume must equal when they equal each
# other; one on several, whose step count and scheduler it must equal.
COMPARISONS = {"one-resume": ("one-control", "one-control-2"), "multi-resume": ("multi-fresh",)}
# Compares weights and optimizers by value inside a run's trainer image; see its docstring.
COMPARE_VALUES = Path(__file__).resolve().with_name("compare_state_values.py")
LEARNED_FILES = ("model.safetensors", "optimizer.bin", "optimizer.pt")

SD15 = _download("Comfy-Org/stable-diffusion-v1-5-archive", "v1-5-pruned-emaonly-fp16.safetensors", "ce8e3e3657a9b767b33fc0171ff91fb04aed1b2f")
# Wan 2.1 T2V 1.3B is Musubi Tuner's smallest supported model (DiT 2.8 GB, VAE 0.25 GB, T5 11 GB).
WAN_1_3B = {
    "dit": _download("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors", "123acf1cc74bccbb9bfff8ac1ee72edc08c2341d"),
    "vae": _download("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/vae/wan_2.1_vae.safetensors", "123acf1cc74bccbb9bfff8ac1ee72edc08c2341d"),
    "t5": _download("Wan-AI/Wan2.1-I2V-14B-720P", "models_t5_umt5-xxl-enc-bf16.pth", "8823af45fcc58a8aa999a54b04be9abc7d2aac98"),
}


def conformance_datasets(backend: str) -> dict[str, str]:
    # Kura projects Musubi Tuner's Wan text-to-video task from videos only; one frame of each is trained.
    if backend == "musubi-tuner":
        return {"one": VIDEO_DATASET, "multi": VIDEO_BUCKET_DATASET}
    return {"one": IMAGE_DATASET, "multi": BUCKET_DATASET}


def conformance_smoke(backend: str, kind: str) -> Smoke:
    """The one small model per backend, training its `kind` ("one" or "multi") dataset with a save cadence."""
    dataset = conformance_datasets(backend)[kind]
    if backend == "sd-scripts":
        config = {
            "architecture": "sd15", "mode": "lora", "model_downloads": {"base": SD15},
            "dataset_config": _sd_scripts_dataset(dataset, resolution=256),
            "learning_rate": 1e-4, "optimizer_type": "AdamW8bit", "mixed_precision": "bf16",
            "network_dim": 4, "network_alpha": 1, "save_every_n_steps": CADENCE,
        }
        return Smoke(backend, "sd15", SD15["repo"], dataset, config, executor="docker", gpu="gpu", expected_script="train_network.py")
    if backend == "ai-toolkit":
        smoke = SMOKES["ai-toolkit-sd1"]
        return replace(smoke, dataset=dataset, config={**smoke.config, "save_every_n_steps": CADENCE})
    if backend == "musubi-tuner":
        config = {"architecture": "wan", "task": "t2v-1.3B", **_MUSUBI_COMMON, "save_every_n_steps": CADENCE, "fp8_t5": True, "model_downloads": WAN_1_3B}
        return Smoke(
            backend, "wan", "Comfy-Org/Wan_2.1_ComfyUI_repackaged", dataset, config, executor="docker", gpu="gpu",
            expected_script="wan_train_network.py", dataset_options={dataset: dict(_VIDEO_ONE_FRAME)},
        )
    raise ValueError(f"no conformance model for {backend}")


CONFORMANCE_BACKENDS = ("sd-scripts", "ai-toolkit", "musubi-tuner")


def expected_saves(start: int, end: int, cadence: int) -> list[tuple[int, int | None]]:
    """The weight files a run from logical step `start` to `end` saves at `cadence`: (the
    optimizer steps the file holds, the step its name carries, None for the final weights).

    Trainer facts, read in the pinned images: sd-scripts (train_network.py) and Musubi Tuner
    (training/trainer_base.py) add 1 to `global_step` after each update and save
    `<name>-step%08d` when `global_step % save_every_n_steps == 0`; Kura's AI-Toolkit patch
    (docker/ai-toolkit/patches/0001-save-by-completed-updates.patch) saves and names
    `<name>_%09d` by the completed updates the same way. Every trainer writes its final
    weights, unnamed, at the end.
    """
    saves: list[tuple[int, int | None]] = [(step, step) for step in range(start + 1, end + 1) if step % cadence == 0]
    return [*saves, (end, None)]


def expected_state_steps(saves: list[tuple[int, int | None]], keep: int = KEEP_GENERATIONS) -> list[int]:
    """The training-state steps left published: the newest `keep` steps the run saved at."""
    return sorted({step for step, _ in saves})[-keep:]


class _PlainUnpickler(pickle.Unpickler):
    """Reads a pickle of plain Python values only; anything else is refused, never imported."""

    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"not a plain value: {module}.{name}")

    def persistent_load(self, pid: Any) -> Any:
        raise pickle.UnpicklingError("holds a tensor")


def _archive_members(path: Path) -> dict[str, bytes]:
    """A torch.save archive's members by name below its top folder, without its per-save ID."""
    with zipfile.ZipFile(path) as archive:
        return {name.split("/", 1)[-1]: archive.read(name) for name in archive.namelist() if not name.endswith("serialization_id")}


def _plain_archive(path: Path) -> Any:
    members = _archive_members(path)
    return _PlainUnpickler(io.BytesIO(members["data.pkl"])).load()


def state_counters(payload: Path) -> dict[str, Any]:
    """The optimizer-step counts a training-state payload records: the trainer's state file,
    Kura's step marker, and the scheduler's step count; a counter it cannot read is a message."""
    counters: dict[str, Any] = {}
    for name, key in (("train_state.json", "current_step"), ("kura-state-info.json", "logical_step"), ("state-info.json", "logical_step")):
        if (payload / name).is_file():
            counters[f"{name}:{key}"] = json.loads((payload / name).read_text(encoding="utf-8")).get(key)
    if (payload / "scheduler.bin").is_file():
        try:
            counters["scheduler.bin:last_epoch"] = _plain_archive(payload / "scheduler.bin").get("last_epoch")
        except (KeyError, zipfile.BadZipFile, pickle.UnpicklingError, AttributeError) as exc:
            counters["scheduler.bin:last_epoch"] = f"unreadable ({exc})"
    return counters


def compare_values(image: str, first: Path, second: Path, names: list[str]) -> dict[str, list[int]]:
    """Each named file's [differing, total] value leaves in two payloads, compared by
    `compare_state_values.py` in `image`, the run's pinned trainer image (it has torch),
    with no network and everything mounted read-only."""
    argv = [
        "docker", "run", "--rm", "--network", "none", "--entrypoint", "python",
        "-v", f"{first.resolve()}:/a:ro", "-v", f"{second.resolve()}:/b:ro", "-v", f"{COMPARE_VALUES.parent}:/c:ro",
        image, "-I", f"/c/{COMPARE_VALUES.name}", "/a", "/b", *names,
    ]
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip()[-500:] or f"exit code {result.returncode}")
    return json.loads(result.stdout.strip().splitlines()[-1])


CANNOT_COMPARE = "cannot compare values in the trainer image"


def compare_states(first: Path, second: Path, *, image: str | None = None) -> list[str]:
    """How two final training states differ: their step counts and scheduler, and with
    `image` (the run's pinned trainer image) the values of their weights and optimizer
    too. RNG state is not compared: Kura does not claim its exact position."""
    counters = state_counters(second)
    differences = [f"{name}: {value} vs {counters.get(name)}" for name, value in state_counters(first).items() if value != counters.get(name)]
    by_value = []
    for name in ["scheduler.bin", *(LEARNED_FILES if image else ())]:
        left, right = first / name, second / name
        if not left.is_file() and not right.is_file():
            continue
        if not (left.is_file() and right.is_file()):
            differences.append(f"{name}: present in only one state")
        elif name != "scheduler.bin":
            by_value.append(name)
        else:
            # A scheduler holds plain values, so its pickle is the same when they are.
            a, b = _archive_members(left), _archive_members(right)
            changed = sorted(key for key in a.keys() | b.keys() if a.get(key) != b.get(key))
            if changed:
                differences.append(f"{name}: {len(changed)} of {len(a.keys() | b.keys())} entries differ")
    if by_value and image:
        try:
            counts = compare_values(image, first, second, by_value)
        except (OSError, RuntimeError, ValueError) as exc:
            return [*differences, f"{CANNOT_COMPARE}: {exc}"]
        differences += [f"{name}: {changed} of {total} entries differ" for name, (changed, total) in counts.items() if changed]
    return differences


def resume_problems(resumed: Path, controls: list[Path], image: str) -> tuple[list[str], list[str]]:
    """P3 for a Resume's final state against its uninterrupted controls' final states:
    (failures, informational notes). The Resume must equal the first control in step
    counts and scheduler; in learned state too when there are two controls and they equal
    each other. Controls that differ (a nondeterministic trainer) leave exactness uncheckable;
    a value comparison that could not run fails."""
    if len(controls) < 2:
        return compare_states(resumed, controls[0]), []
    between_controls = compare_states(controls[1], controls[0], image=image)
    broken = [difference for difference in between_controls if difference.startswith(CANNOT_COMPARE)]
    if broken:
        # A comparison that did not run says nothing about the trainer; it is a failure.
        return broken, []
    if between_controls:
        return compare_states(resumed, controls[0]), ["controls differ: exact Resume not checkable"]
    return compare_states(resumed, controls[0], image=image), []


def _run_states(workspace: Path, run_id: str) -> dict[int, tuple[dict[str, Any], Path]]:
    """The run's published training states by step, with their payload directories."""
    states: dict[int, tuple[dict[str, Any], Path]] = {}
    for path in sorted((workspace / "artifacts" / "training-state").glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("source_run") == run_id:
            states[manifest["observed_step"]] = (manifest, path.parent / "payload")
    return states


def _weights(run_dir: Path, outputs: list[Any]) -> list[Path]:
    """A run's recorded weight files relative to its outputs directory, as Kura counts checkpoints."""
    paths = [Path(str(item)) for item in outputs if str(item).endswith(".safetensors")]
    found = checkpoint_files([path.relative_to("outputs") for path in paths if path.parts[:1] == ("outputs",)], run_dir.name)
    return [*found.stepped, *found.final]


def recorded_weight_step(path: Path) -> int | None:
    """The update count a trainer wrote into a weight file's safetensors metadata
    (`training_info.step`, which AI-Toolkit writes), or None when it records none."""
    with path.open("rb") as handle:
        size = int.from_bytes(handle.read(8), "little")
        metadata = json.loads(handle.read(size)).get("__metadata__") or {}
    try:
        step = json.loads(metadata.get("training_info") or "{}").get("step")
    except (json.JSONDecodeError, AttributeError):
        return None
    return step if isinstance(step, int) and not isinstance(step, bool) else None


def _env_lock(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "resolved" / "env.lock"
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.is_file() else {}


def _image_digest(reference: Any) -> str | None:
    """The content digest an image reference names: `name@sha256:<d>`, or a bare image ID."""
    if not isinstance(reference, str):
        return None
    if reference.startswith("sha256:"):
        return reference
    return image_reference_identity(reference)["pinning"].get("value")


def check_run(workspace: Path, run_id: str, start: int, end: int, cadence: int = CADENCE) -> dict[str, list[str]]:
    """Check one finished run against P1, P2, P4, P6, and P7; each promise maps to its failures."""
    run_dir = workspace / "runs" / run_id
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    problems: dict[str, list[str]] = {"P1": [], "P2": [], "P4": [], "P7": []}
    states = _run_states(workspace, run_id)
    saves = expected_saves(start, end, cadence)

    if status.get("state") != "completed" or status.get("exit_code") != 0:
        problems["P1"].append(f"ended {status.get('state')} with exit code {status.get('exit_code')}")
    if status.get("last_step") != end:
        problems["P1"].append(f"status last_step {status.get('last_step')}, expected {end}")
    if end not in states:
        problems["P1"].append(f"no training state published at step {end}")

    weights = _weights(run_dir, status.get("outputs") or [])
    names = sorted((checkpoint_step(path.name) for path in weights), key=lambda step: -1 if step is None else step)
    for path in weights:
        # The trainer's own count, where it records one, must equal the step the file holds.
        held = checkpoint_step(path.name) or end
        recorded = recorded_weight_step(run_dir / "outputs" / path) if (run_dir / "outputs" / path).is_file() else None
        if recorded is not None and recorded != held:
            problems["P2"].append(f"{path.name} records training_info.step {recorded}, holds step {held}")
    wanted = sorted((name for _, name in saves), key=lambda step: -1 if step is None else step)
    if names != wanted:
        problems["P2"].append(f"weight file steps {names}, expected {wanted} (None is the final weights)")
    if sorted(states) != expected_state_steps(saves):
        problems["P2"].append(f"published state steps {sorted(states)}, expected {expected_state_steps(saves)}")
    for step, (manifest, payload) in sorted(states.items()):
        if not str(manifest.get("id")).startswith(f"state-step-{step:08d}-"):
            problems["P2"].append(f"state {manifest.get('id')} is named for another step than {step}")
        for counter, value in state_counters(payload).items():
            if value != step:
                problems["P2"].append(f"state at step {step} records {counter} = {value}")

    postflight = status.get("dataset_input_postflight") if isinstance(status.get("dataset_input_postflight"), dict) else {}
    if postflight.get("status") != "matched":
        problems["P4"].append(f"dataset input postflight {postflight.get('status')}")
    env_lock = _env_lock(run_dir)
    realization_path = run_dir / str(status.get("last_realization"))
    realization = json.loads(realization_path.read_text(encoding="utf-8")) if realization_path.is_file() else {}
    used = (realization.get("image_identity") or {}).get("reference")
    # The compiled image is its reference's digest, or the image ID compile observed for it,
    # which a local Resume launches by.
    observed = (env_lock.get("selected_image_identity") or {}).get("pinning") or {}
    compiled = {_image_digest(env_lock.get("selected_image")), observed.get("value") if observed.get("strength") == "content-hash" else None} - {None}
    if _image_digest(used) not in compiled:
        problems["P4"].append(f"ran image {used}, compiled for {env_lock.get('selected_image')}")

    if status.get("publication_state") != "completed":
        problems["P7"].append(f"publication {status.get('publication_state')}")
    missing = [str(item) for item in status.get("outputs") or [] if not (run_dir / str(item)).exists()]
    if missing or not status.get("outputs"):
        problems["P7"].append(f"recorded outputs missing: {missing or 'none recorded'}")
    if status.get("host") == "runpod":
        problems["P6"] = [] if isinstance(status.get("pod_stopped_at"), str) else ["no pod_stopped_at recorded"]
    return problems


def disk_problems(peak: dict[str, int], estimate: dict[str, Any]) -> list[str]:
    """P8: the checkpoints on disk at once never exceed the launch estimate."""
    problems = []
    if peak["count"] > estimate.get("count", 0):
        problems.append(f"{peak['count']} checkpoint files at once, estimated {estimate.get('count')}")
    if peak["bytes"] > estimate.get("bytes", 0):
        problems.append(f"{peak['bytes']} checkpoint bytes at once, estimated {estimate.get('bytes')}")
    return problems


def _checkpoint_estimate(run_dir: Path) -> dict[str, Any]:
    from kura.run_commands.plan import _estimate_checkpoint_write_bytes

    return _estimate_checkpoint_write_bytes(yaml.safe_load((run_dir / "resolved" / "manifest.lock.yaml").read_text(encoding="utf-8")))


def _sample_peak(run_dir: Path, peak: dict[str, int]) -> None:
    outputs = run_dir / "outputs"
    sizes = 0
    # A hidden directory is a training state being staged, not a checkpoint.
    paths = [path.relative_to(outputs) for path in outputs.rglob("*.safetensors")] if outputs.is_dir() else []
    weights = checkpoint_files([path for path in paths if not any(part.startswith(".") for part in path.parts)], run_dir.name)
    files = [*weights.stepped, *weights.final]
    for path in files:
        try:
            sizes += (outputs / path).stat().st_size
        except OSError:
            pass
    peak["count"] = max(peak["count"], len(files))
    peak["bytes"] = max(peak["bytes"], sizes)


def _execute(workspace: Path, run_id: str, *, runpod: bool) -> tuple[int, dict[str, int], str]:
    """`kura run execute`, sampling the checkpoints on local disk while it runs."""
    run_dir = workspace / "runs" / run_id
    peak = {"count": 0, "bytes": 0}
    env = {**os.environ, "KURA_NOTIFY": "none"}
    with tempfile.TemporaryFile("w+", encoding="utf-8") as log:
        process = subprocess.Popen(_kura_argv("run", "execute", run_id, *(["--yes"] if runpod else [])), cwd=workspace, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
        while process.poll() is None:
            _sample_peak(run_dir, peak)
            time.sleep(0.5)
        _sample_peak(run_dir, peak)
        log.seek(0)
        return process.returncode, peak, log.read()[-3000:]


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def _cache_state(hf_cache: Path, smoke: Smoke) -> str:
    """Whether the backend's model files are already in the workspace's Hugging Face cache."""
    downloads = smoke.config.get("model_downloads") or {}
    if not downloads:
        repo = hf_cache / "hub" / f"models--{smoke.model_base.replace('/', '--')}"
        return "cached" if repo.is_dir() else "not cached (downloaded at launch)"
    parts = []
    for role, item in sorted(downloads.items()):
        path = hf_cache / "hub" / f"models--{item['repo'].replace('/', '--')}" / "snapshots" / item["revision"] / item["filename"]
        parts.append(f"{role} {path.stat().st_size / 1e9:.2f} GB cached" if path.is_file() else f"{role} NOT cached (downloaded at launch)")
    return ", ".join(parts)


def conformance(workspace: Path, backends: list[str], *, runpod: bool, gpu: str, yes: bool) -> int:
    if workspace.resolve() == REPO.resolve():
        raise SystemExit("conformance creates runs; give it a separate workspace (`kura init` one), not this checkout")
    if not (workspace / "workspace.yaml").is_file():
        raise SystemExit(f"{workspace} is not a Kura workspace; create it with `kura init`")
    if runpod and len(backends) != 1:
        raise SystemExit("the RunPod pass runs one backend; name it with --backend")
    config = yaml.safe_load((workspace / "workspace.yaml").read_text(encoding="utf-8")) or {}
    hf_cache = local_hf_cache(workspace, config)
    print(f"Conformance plan: workspace {workspace}, executor {'runpod (' + gpu + ')' if runpod else 'local Docker'}, Hugging Face cache {hf_cache}")
    for backend in backends:
        smoke = conformance_smoke(backend, "one")
        print(f"  {backend:13} {smoke.architecture:5} {smoke.model_base}: {_cache_state(hf_cache, smoke)}")
    datasets = sorted({dataset for backend in backends for dataset in conformance_datasets(backend).values()})
    print("  runs per backend, in order: " + "; ".join(f"{run.key} {'Resume to ' if run.resume_of else ''}{run.steps} steps" for run in SCENARIO))
    print(f"  datasets {', '.join(datasets)} (1 item, or 3 items in 3 buckets): created when absent, never rewritten")
    if not runpod and not yes:
        print("Nothing launched. Add --yes to run it.")
        return 0

    digests = {}
    for dataset in datasets:
        ensure_dataset(workspace, dataset)
        digests[dataset] = _tree_digest(workspace / "datasets" / dataset)
    table: dict[str, dict[str, list[str] | None]] = {}
    notes: list[str] = []
    for backend in backends:
        results: dict[str, list[str] | None] = {promise: None for promise in PROMISES}
        def fail(promise: str, messages: list[str], key: str) -> None:
            results[promise] = (results[promise] or []) + [f"{key}: {message}" for message in messages]
        ids: dict[str, str] = {}
        for run in SCENARIO:
            smoke = conformance_smoke(backend, run.dataset)
            if runpod:
                smoke = replace(smoke, executor="runpod", gpu=gpu)
            slug = f"conf-{backend.split('-')[0]}-{run.key}"
            if run.resume_of is None:
                intent = f"Kura conformance ({run.key}) of {backend}; checks promises, not quality."
                ids[run.key] = _create_run(workspace, "conformance", slug, smoke, build_run_fields(slug, smoke, steps=run.steps, intent=intent))
            elif run.resume_of in ids:
                created = _kura(workspace, "run", "resume", ids[run.resume_of], "--to-step", str(run.steps), "--slug", slug)
                if created.returncode:
                    fail("P3", [f"cannot create the Resume: {created.stderr.strip()}"], run.key)
                    continue
                ids[run.key] = created.stdout.strip().splitlines()[-1]
                _compile(workspace, ids[run.key])
            else:
                fail("P3", ["its source run did not complete"], run.key)
                continue
            run_id = ids[run.key]
            if runpod and len(ids) == 1:
                plan = _kura(workspace, "run", "plan", run_id).stdout
                print("\n".join(line for line in plan.splitlines() if "cost_ceiling" in line or "max_lease" in line))
                print(f"  this pass starts {len(SCENARIO)} Pods one after another, each bounded by that ceiling")
                if not yes:
                    print(f"Nothing launched. Add --yes to run it, or remove the compiled run with `kura run discard {run_id} --yes`.")
                    return 0
            print(f"{backend} {run.key}: executing {run_id}", flush=True)
            code, peak, log = _execute(workspace, run_id, runpod=runpod)
            if code:
                print(log)
                fail("P1", [f"kura run execute exited {code}"], run.key)
                del ids[run.key]
                continue
            source = next((item for item in SCENARIO if item.key == run.resume_of), None)
            for promise, messages in check_run(workspace, run_id, source.steps if source else 0, run.steps).items():
                fail(promise, messages, run.key)
            if not runpod:
                fail("P8", disk_problems(peak, _checkpoint_estimate(workspace / "runs" / run_id)), run.key)
        for resumed, controls in COMPARISONS.items():
            if all(key in ids for key in (resumed, *controls)):
                steps = next(item.steps for item in SCENARIO if item.key == resumed)
                states = [_run_states(workspace, ids[key]).get(steps) for key in (resumed, *controls)]
                if None in states:
                    fail("P3", ["final state missing"], resumed)
                    continue
                image = str(_env_lock(workspace / "runs" / ids[resumed]).get("selected_image"))
                failures, informational = resume_problems(states[0][1], [state[1] for state in states[1:]], image)
                fail("P3", failures, resumed)
                notes += [f"{backend} P3 {resumed}: {note} (informational)" for note in informational]
        fail("P7", [f"dataset {dataset} changed" for dataset, digest in digests.items() if _tree_digest(workspace / "datasets" / dataset) != digest], "datasets")
        table[backend] = results

    print("\n" + f"{'promise':8}" + "".join(f"{backend:14}" for backend in backends))
    for promise in PROMISES:
        print(f"{promise:8}" + "".join(f"{('-' if table[b][promise] is None else 'FAIL' if table[b][promise] else 'pass'):14}" for b in backends))
    for backend in backends:
        for promise in PROMISES:
            for message in table[backend][promise] or []:
                print(f"{backend} {promise} {message}")
    for note in notes:
        print(note)
    return 1 if any(table[b][p] for b in backends for p in PROMISES) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List the smokes this harness knows")
    prepare_parser = commands.add_parser("prepare", help="Create and compile runs; never launches")
    prepare_parser.add_argument("smoke", nargs="+", choices=sorted(SMOKES))
    prepare_parser.add_argument("--gpu", help="RunPod GPU type overriding the smoke default, for example when host RAM is the limit")
    verify_parser = commands.add_parser("verify", help="Check a finished run")
    verify_parser.add_argument("run_id", nargs="+")
    evidence_parser = commands.add_parser("evidence", help="Print evidence records for verified runs")
    evidence_parser.add_argument("--artifact", required=True, help="Path under docs/ of the campaign smoke-evidence file")
    evidence_parser.add_argument("run_id", nargs="+")
    conformance_parser = commands.add_parser("conformance", help="Run the promise conformance scenario in a separate workspace; prints its plan unless --yes")
    conformance_parser.add_argument("--workspace", required=True, type=Path, help="A Kura workspace for the conformance runs, never this checkout")
    conformance_parser.add_argument("--backend", action="append", choices=CONFORMANCE_BACKENDS, help="Only this backend (repeatable); default all")
    conformance_parser.add_argument("--runpod", action="store_true", help="The billed RunPod pass of one backend; shows the cost ceiling first")
    conformance_parser.add_argument("--gpu", default=A40, help="RunPod GPU type for --runpod")
    conformance_parser.add_argument("--yes", action="store_true", help="Launch the runs after showing the plan")
    args = parser.parse_args()
    if args.command == "conformance":
        return conformance(args.workspace.resolve(), args.backend or list(CONFORMANCE_BACKENDS), runpod=args.runpod, gpu=args.gpu, yes=args.yes)
    workspace = Path.cwd()
    if not (workspace / "workspace.yaml").is_file():
        raise SystemExit("run from a Kura workspace root")
    if args.command == "list":
        for smoke_id, smoke in sorted(SMOKES.items()):
            print(f"{smoke_id:30} {smoke.backend:13} {smoke.architecture:18} {smoke.executor:7} {smoke.gpu:12} {smoke.model_base}")
        return 0
    if args.command == "prepare":
        for smoke_id in args.smoke:
            print(json.dumps({"smoke": smoke_id, "run_id": prepare(workspace, smoke_id, gpu=args.gpu)}))
        return 0
    if args.command == "evidence":
        pairs = [evidence(workspace, run_id, artifact=args.artifact) for run_id in args.run_id]
        print(yaml.safe_dump({"records": [record for record, _ in pairs], "runs": {record["id"]: summary for record, summary in pairs}}, sort_keys=False, allow_unicode=True))
        return 0
    results = [verify(workspace, run_id) for run_id in args.run_id]
    print(json.dumps(results, indent=2))
    return 0 if all(result["ok"] for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())

# This script runs inside training containers with stdlib only.
# Do not import kura here; it is delivered as `python -c` source text.

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
    if expected == "hf_model_id_or_path":
        if os.path.exists(path):
            return
        first_part = path.split("/", 1)[0]
        if (
            os.path.isabs(path)
            or path.startswith("./")
            or path.startswith("../")
            or path.startswith("~")
            or first_part in ("models", "weights", "checkpoints", "cache", "runs", "datasets", "data")
            or path.endswith((".safetensors", ".pt", ".pth", ".bin"))
        ):
            die(f"{role} path does not exist: {path}")
        if "/" in path:
            return
        die(f"{role} is neither a local path nor a Hugging Face model id: {path}")
    if expected == "file":
        if os.path.isfile(path):
            return
        die(f"{role} is not a readable model file: {path}")
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


def main():
    spec = json.loads(sys.argv[1])
    for item in spec.get("models", []):
        validate_model(item["role"], item["path"], item.get("expected_format") or "safetensors")
    if spec.get("lora"):
        lora = spec["lora"]
        validate_lora(lora["pattern"], spec.get("architecture", ""), lora.get("compatibility", "comfyui"))
    print("[kura] validation ok", flush=True)


if __name__ == "__main__":
    main()

# This script runs inside training containers with stdlib only.
# Do not import kura here; it is delivered as `python -c` source text.

import glob
import json
import os
import struct
import sys


def fail(message):
    raise SystemExit("[kura] sd-scripts validation failed: " + message)


def header(path):
    if not os.path.isfile(path):
        fail(f"missing file: {path}")
    with open(path, "rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            fail(f"not a safetensors file: {path}")
        size = struct.unpack("<Q", raw)[0]
        if size <= 0 or size > 100 * 1024 * 1024:
            fail(f"invalid safetensors header size: {path}")
        try:
            value = json.loads(handle.read(size))
        except Exception as exc:
            fail(f"invalid safetensors header: {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"safetensors header must be an object: {path}")
    keys = [key for key in value if key != "__metadata__"]
    if not keys:
        fail(f"safetensors has no tensors: {path}")
    return keys, value.get("__metadata__") or {}


def validate_models(items):
    for item in items:
        path = item["path"]
        role = item["role"]
        if role == "t5_tokenizer" and os.path.exists(path):
            continue
        if role == "qwen3" and os.path.isdir(path):
            continue
        header(path)


def validate_output(spec):
    paths = sorted(glob.glob(spec["pattern"]))
    if len(paths) != 1:
        fail(f"expected exactly one output matching {spec['pattern']}, found {len(paths)}")
    path = paths[0]
    keys, metadata = header(path)
    kind = spec["kind"]
    if kind == "lora":
        down = any(key.endswith((".lora_down.weight", ".lora_A.weight")) for key in keys)
        up = any(key.endswith((".lora_up.weight", ".lora_B.weight")) for key in keys)
        if not (down and up):
            fail(f"output is not a recognized LoRA: {path}")
    elif kind == "anima-lllite":
        if str(metadata.get("lllite.version", "")) != "2":
            fail(f"Anima LLLite requires lllite.version=2 metadata: {path}")
        if "lllite_conditioning1.conv1.weight" not in keys:
            fail(f"Anima LLLite is missing the ComfyUI core loader key lllite_conditioning1.conv1.weight: {path}")
    else:
        fail(f"unknown output kind: {kind}")
    print(json.dumps({"path": path, "kind": kind, "tensor_count": len(keys), "metadata": metadata}, ensure_ascii=False), flush=True)


def main():
    spec = json.loads(sys.argv[1])
    validate_models(spec.get("models", []))
    if spec.get("output"):
        validate_output(spec["output"])


if __name__ == "__main__":
    main()

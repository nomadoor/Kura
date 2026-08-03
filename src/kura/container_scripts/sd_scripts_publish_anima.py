# This script runs inside training containers with stdlib only.
# Do not import kura here; it is delivered as `python -c` source text.

import hashlib
import json
import os
import pathlib
import re
import shutil
import signal
import struct
import subprocess
import sys
import time


def fail(message):
    raise RuntimeError("[kura] sd-scripts Anima publication failed: " + message)


def lora_header(path):
    with path.open("rb") as handle:
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
    keys = [key for key in value if key != "__metadata__"]
    offsets = []
    for key in keys:
        descriptor = value.get(key)
        data_offsets = descriptor.get("data_offsets") if isinstance(descriptor, dict) else None
        if (
            not isinstance(data_offsets, list)
            or len(data_offsets) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in data_offsets)
            or data_offsets[0] < 0
            or data_offsets[1] < data_offsets[0]
        ):
            fail(f"invalid tensor data offsets for {key}: {path}")
        offsets.append(data_offsets[1])
    expected_size = 8 + size + max(offsets, default=0)
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        fail(f"incomplete safetensors data: {path}: expected {expected_size} bytes, found {actual_size}")
    down = any(key.endswith((".lora_down.weight", ".lora_A.weight")) for key in keys)
    up = any(key.endswith((".lora_up.weight", ".lora_B.weight")) for key in keys)
    if not keys or not (down and up):
        fail(f"not a recognized LoRA: {path}")
    return len(keys)


def tensor_fingerprint(path):
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
        tensors = {key: value for key, value in header.items() if key != "__metadata__"}
        data_hash = hashlib.sha256()
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            data_hash.update(chunk)
    descriptor = json.dumps(tensors, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(descriptor).hexdigest(), data_hash.hexdigest()


def native_files(native_dir, output_name, *, require_final):
    final_name = output_name + ".safetensors"
    checkpoint = re.compile(re.escape(output_name) + r"-step[0-9]{8}\.safetensors\Z")
    if not native_dir.is_dir():
        if require_final:
            fail(f"missing native output directory: {native_dir}")
        return []
    matches = sorted(
        path
        for path in native_dir.iterdir()
        if path.is_file() and (path.name == final_name or checkpoint.fullmatch(path.name))
    )
    if require_final and not any(path.name == final_name for path in matches):
        fail(f"missing final native LoRA: {native_dir / final_name}")
    return matches


def recover(files, recovery_dir):
    recovery_dir.mkdir(parents=True, exist_ok=True)
    for source in files:
        destination = recovery_dir / source.name
        if not destination.exists():
            shutil.copy2(source, destination)


def publish(files, spec, published):
    staging_dir = pathlib.Path(spec["staging_dir"])
    output_dir = pathlib.Path(spec["output_dir"])
    converter = spec["converter"]
    staging_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in files:
        if source.name in published:
            continue
        lora_header(source)
        staging = staging_dir / source.name
        destination = output_dir / source.name
        if staging.exists():
            staging.unlink()
        subprocess.run([sys.executable, converter, str(source), str(staging)], check=True)
        tensor_count = lora_header(staging)
        if destination.exists():
            lora_header(destination)
            if tensor_fingerprint(staging) != tensor_fingerprint(destination):
                fail(f"published output conflicts with converted native weight: {destination}")
            staging.unlink()
            published.add(source.name)
            continue
        os.replace(staging, destination)
        published.add(source.name)
        print(json.dumps({"path": str(destination), "kind": "lora", "tensor_count": tensor_count}, ensure_ascii=False), flush=True)


def train_and_publish(spec):
    native_dir = pathlib.Path(spec["native_dir"])
    recovery_dir = pathlib.Path(spec["recovery_dir"])
    output_name = spec["output_name"]
    child = subprocess.Popen(spec["train_argv"])
    forwarded_signal = None

    def forward(signum, _frame):
        nonlocal forwarded_signal
        forwarded_signal = signum
        if child.poll() is None:
            child.send_signal(signum)

    previous = {signum: signal.signal(signum, forward) for signum in (signal.SIGINT, signal.SIGTERM)}
    published = set()
    try:
        while child.poll() is None:
            try:
                candidates = native_files(native_dir, output_name, require_final=False)
            except OSError:
                # Network-backed RunPod storage may transiently fail a
                # directory listing. Retry without terminating healthy
                # training; the post-exit listing remains strict.
                time.sleep(1)
                continue
            ready = []
            for path in candidates:
                try:
                    lora_header(path)
                except (OSError, RuntimeError):
                    # sd-scripts may still be writing the newly visible file.
                    # Retry it while training is alive; after the child exits,
                    # validation is strict and any corrupt retained file fails.
                    continue
                ready.append(path)
            try:
                publish(ready, spec, published)
            except (OSError, RuntimeError, subprocess.CalledProcessError):
                # A file may change between the readiness check and converter
                # open, especially on network-backed RunPod storage. Do not
                # kill healthy training for a transient publication race.
                # The post-exit pass below is strict.
                pass
            time.sleep(1)
        returncode = child.wait()
        files = native_files(native_dir, output_name, require_final=returncode == 0)
        publish(files, spec, published)
        if returncode != 0:
            recover(files, recovery_dir)
        if forwarded_signal is not None:
            return 128 + forwarded_signal
        return returncode
    except BaseException:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        recover(native_files(native_dir, output_name, require_final=False), recovery_dir)
        raise
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main():
    spec = json.loads(sys.argv[1])
    if spec.get("train_argv"):
        raise SystemExit(train_and_publish(spec))
    native_dir = pathlib.Path(spec["native_dir"])
    files = native_files(native_dir, spec["output_name"], require_final=True)
    try:
        publish(files, spec, set())
    except BaseException:
        recover(files, pathlib.Path(spec["recovery_dir"]))
        raise


if __name__ == "__main__":
    main()

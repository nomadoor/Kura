# This script runs inside training containers with stdlib only.
# Do not import kura here; it is delivered as `python -c` source text.

import hashlib
import json
import os
import pathlib
import sys


def fail(message):
    raise SystemExit("[kura] training state verification failed: " + message)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(value):
    path = pathlib.PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        fail(f"unsafe locked file path: {value!r}")
    return path


def reject_symlinks(root, path):
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            fail(f"symlink is not allowed in training state: {current}")


def main():
    if len(sys.argv) != 3:
        fail("usage: training_state_verify.py LOCK_PATH WORKSPACE")
    lock_path = pathlib.Path(sys.argv[1])
    workspace = pathlib.Path(sys.argv[2]).resolve()
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read source lock: {exc}")
    artifact_id = lock.get("artifact_id")
    native = lock.get("native_state_path")
    if not isinstance(artifact_id, str) or not artifact_id:
        fail("source lock has no artifact_id")
    expected_native = f"/workspace/artifacts/training-state/{artifact_id}/payload"
    if native != expected_native:
        fail(f"unexpected native_state_path: {native!r}")
    payload = workspace / "artifacts" / "training-state" / artifact_id / "payload"
    expected_root = (workspace / "artifacts" / "training-state").resolve()
    try:
        payload.resolve().relative_to(expected_root)
    except (OSError, ValueError):
        fail("training state payload escapes the artifact store")
    reject_symlinks(workspace, payload)
    if not payload.is_dir():
        fail(f"training state payload is missing: {payload}")
    files = lock.get("files")
    if not isinstance(files, list) or not files:
        fail("source lock has no file inventory")
    locked_paths = []
    for item in files:
        if not isinstance(item, dict):
            fail("source lock file inventory contains a non-object")
        relative = safe_relative(item.get("path")) if isinstance(item.get("path"), str) else fail("locked file has no path")
        path = payload.joinpath(*relative.parts)
        reject_symlinks(payload, path)
        if not path.is_file():
            fail(f"locked file is missing: {relative}")
        expected_size = item.get("size")
        if not isinstance(expected_size, int) or path.stat().st_size != expected_size:
            fail(f"size mismatch: {relative}")
        expected_digest = item.get("sha256")
        if not isinstance(expected_digest, str) or sha256_file(path) != expected_digest:
            fail(f"digest mismatch: {relative}")
        locked_paths.append(relative.as_posix())
    actual_paths = []
    for directory, directories, names in os.walk(payload, followlinks=False):
        base = pathlib.Path(directory)
        for name in directories:
            reject_symlinks(payload, base / name)
        for name in names:
            path = base / name
            reject_symlinks(payload, path)
            actual_paths.append(path.relative_to(payload).as_posix())
    if sorted(actual_paths) != sorted(locked_paths):
        fail("payload inventory differs from the compiled source lock")
    print(f"[kura] training state verified: {artifact_id}", flush=True)


if __name__ == "__main__":
    main()

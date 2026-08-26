# This script runs inside training containers with stdlib only.
# Do not import kura here; it is delivered as `python -c` source text.

import json
import hashlib
import os
import pathlib
import shutil
import sys


def fail(message):
    raise SystemExit("[kura] sd-scripts dataset staging failed: " + message)


def contained(root, path):
    try:
        pathlib.Path(path).resolve().relative_to(pathlib.Path(root).resolve())
        return True
    except ValueError:
        return False


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    lock_path = pathlib.Path(sys.argv[1])
    workspace = pathlib.Path(sys.argv[2])
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    stage_root = workspace / payload["stage_root"]
    run_cache_root = workspace / "runs"
    datasets_root = workspace / "datasets"
    if not contained(run_cache_root, stage_root) or tuple(stage_root.parts[-3:]) != ("cache", "sd-scripts", "datasets"):
        fail(f"stage_root must be a run-scoped sd-scripts dataset cache: {payload['stage_root']}")
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True)
    for item in payload.get("files", []):
        source = workspace / item["source"]
        destination = workspace / item["destination"]
        if not contained(datasets_root, source):
            fail(f"source escapes datasets/: {item['source']}")
        if not contained(stage_root, destination):
            fail(f"destination escapes run cache: {item['destination']}")
        if not source.is_file():
            fail(f"frozen source is missing: {item['source']}")
        identity = item.get("identity") or {}
        if source.stat().st_size != identity.get("size_bytes"):
            fail(f"frozen source size changed: {item['source']}")
        digest = sha256_file(source)
        if digest != identity.get("sha256"):
            fail(f"frozen source digest changed: {item['source']}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        relative = os.path.relpath(source, destination.parent)
        destination.symlink_to(relative)
    for controls in payload.get("effective_controls", []):
        dataset_index = controls.get("dataset_index")
        subset_index = controls.get("subset_index")
        for key in sorted(set(controls) - {"dataset_index", "subset_index"}):
            value = json.dumps(controls[key], ensure_ascii=False, separators=(",", ":"))
            print(
                f"[kura] sd-scripts dataset {dataset_index} subset {subset_index} {key}: {value}",
                flush=True,
            )
    print(f"[kura] sd-scripts dataset staged: {len(payload.get('files', []))} files", flush=True)


if __name__ == "__main__":
    main()

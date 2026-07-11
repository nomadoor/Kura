# This script runs inside training containers with stdlib plus huggingface_hub.
# Do not import kura here; it is delivered as `python -c` source text.

import json
import os
import shutil
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
DOWNLOAD_RESERVE_BYTES = 1024**3


CHILD = r"""
import json
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import hf_hub_download

item = json.loads(sys.argv[1])
cache_dir = os.environ.get("HF_HOME")
if not cache_dir:
    raise SystemExit("[kura] HF_HOME is required before downloading models")
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


def progress_bytes(total, baseline, expected_size):
    downloaded = max(0, total - baseline)
    if isinstance(expected_size, int):
        downloaded = min(downloaded, expected_size)
    return downloaded


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


def stable_link_target(path, link_path):
    target = workspace_mapped_path(path)
    link = os.path.abspath(link_path)
    try:
        target_workspace = os.path.commonpath([target, "/workspace"]) == "/workspace"
        link_workspace = os.path.commonpath([link, "/workspace"]) == "/workspace"
    except ValueError as exc:
        raise SystemExit(f"[kura] cannot map downloaded model path into workspace: {path}") from exc
    if target_workspace and link_workspace:
        return os.path.relpath(target, os.path.dirname(link))
    if os.path.isabs(path):
        raise SystemExit(f"[kura] cannot map downloaded model path into workspace: {path}")
    return path


def workspace_mapped_path(path):
    try:
        target = os.path.abspath(path)
        raw_maps = os.environ.get("KURA_WORKSPACE_PATH_MAPS") or "[]"
        try:
            mappings = json.loads(raw_maps)
        except json.JSONDecodeError:
            mappings = []
        for item in mappings:
            container = os.path.abspath(str(item.get("container", "")))
            workspace = os.path.abspath(str(item.get("workspace", "")))
            if not container or not workspace:
                continue
            if target == container or target.startswith(container.rstrip("/") + "/"):
                suffix = target[len(container):].lstrip("/")
                target = os.path.join(workspace, suffix)
                break
    except ValueError as exc:
        raise SystemExit(f"[kura] cannot map downloaded model path into workspace: {path}") from exc
    return target


def require_cache_mappable(cache_dir, link_path):
    link = os.path.abspath(link_path)
    try:
        link_workspace = os.path.commonpath([link, "/workspace"]) == "/workspace"
        cache_workspace = os.path.commonpath([workspace_mapped_path(cache_dir), "/workspace"]) == "/workspace"
    except ValueError:
        link_workspace = False
        cache_workspace = False
    if not cache_workspace:
        raise SystemExit(
            "[kura] HF_HOME must be inside /workspace or covered by KURA_WORKSPACE_PATH_MAPS before downloading models: "
            f"{cache_dir}"
        )


def metadata_failure_kind(exc):
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code in (401, 403):
        return "authentication"
    if status_code == 404:
        return "missing-artifact"
    name = type(exc).__name__.lower()
    if "timeout" in name or "connection" in name:
        return "connectivity"
    return "metadata"


def preflight_downloads(items):
    cache_dir = os.environ.get("HF_HOME")
    if not cache_dir:
        raise SystemExit("[kura] HF_HOME is required before downloading models")
    os.makedirs(cache_dir, exist_ok=True)

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import get_hf_file_metadata, hf_hub_url, try_to_load_from_cache

    required_bytes = 0
    for item in items:
        require_cache_mappable(cache_dir, item["link_path"])
        label = f"{item['key']}:{item['filename']}"
        cached = try_to_load_from_cache(
            item["repo_id"],
            item["filename"],
            cache_dir=cache_dir,
            revision=item.get("revision") or "main",
            repo_type=item.get("repo_type"),
        )
        if isinstance(cached, str) and os.path.isfile(cached):
            size = os.path.getsize(cached)
            item["_size_bytes"] = size
            print(f"[kura] hf preflight item {label} status=cached size_bytes={size}", flush=True)
            continue
        url = hf_hub_url(
            repo_id=item["repo_id"],
            filename=item["filename"],
            revision=item.get("revision") or "main",
            repo_type=item.get("repo_type"),
        )
        token = os.environ.get("HF_TOKEN")
        try:
            metadata = get_hf_file_metadata(url, token=token)
        except Exception as exc:
            kind = metadata_failure_kind(exc)
            raise SystemExit(f"[kura] hf preflight {kind} failure for {label}: {type(exc).__name__}: {exc}") from exc
        size = getattr(metadata, "size", None)
        if not isinstance(size, int) or size < 0:
            raise SystemExit(f"[kura] hf preflight metadata has no usable size for {label}")
        item["_size_bytes"] = size
        required_bytes += size
        print(f"[kura] hf preflight item {label} status=missing size_bytes={size}", flush=True)

    free_bytes = shutil.disk_usage(cache_dir).free
    print(
        f"[kura] hf preflight summary files={len(items)} download_bytes={required_bytes} "
        f"free_bytes={free_bytes} reserve_bytes={DOWNLOAD_RESERVE_BYTES}",
        flush=True,
    )
    if required_bytes + DOWNLOAD_RESERVE_BYTES > free_bytes:
        raise SystemExit(
            "[kura] hf preflight insufficient disk: "
            f"download_bytes={required_bytes} reserve_bytes={DOWNLOAD_RESERVE_BYTES} free_bytes={free_bytes}"
        )


def run_one(item):
    link_path = item["link_path"]
    cache_dir = os.environ.get("HF_HOME")
    if not cache_dir:
        raise SystemExit("[kura] HF_HOME is required before downloading models")
    require_cache_mappable(cache_dir, link_path)
    link_dir = os.path.dirname(link_path)
    os.makedirs(link_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    label = f"{item['key']}:{item['filename']}"
    progress_dirs = repo_cache_dirs(cache_dir, item)
    baseline_total = sum(tree_snapshot(directory)[0] for directory in progress_dirs)
    baseline_count = sum(tree_snapshot(directory)[2] for directory in progress_dirs)
    last_total = baseline_total
    last_mtime = max((tree_snapshot(directory)[1] for directory in progress_dirs), default=0.0)
    expected_size = item.get("_size_bytes")
    for attempt in range(1, ATTEMPTS + 1):
        print(f"[kura] hf download start {label} attempt {attempt}/{ATTEMPTS}", flush=True)
        process = subprocess.Popen([sys.executable, "-c", CHILD, json.dumps(item, ensure_ascii=False)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        last_progress = time.monotonic()
        while process.poll() is None:
            time.sleep(POLL_SEC)
            snapshots = [tree_snapshot(directory) for directory in progress_dirs]
            total = sum(snapshot[0] for snapshot in snapshots)
            newest = max((snapshot[1] for snapshot in snapshots), default=0.0)
            count = sum(snapshot[2] for snapshot in snapshots)
            downloaded = progress_bytes(total, baseline_total, expected_size)
            created_files = max(0, count - baseline_count)
            if total != last_total or newest != last_mtime:
                last_total, last_mtime = total, newest
                last_progress = time.monotonic()
                print(f"[kura] hf download progress {label} files={created_files} bytes={downloaded}", flush=True)
                continue
            idle = int(time.monotonic() - last_progress)
            print(f"[kura] hf download idle {label} idle={idle}s bytes={downloaded}", flush=True)
            if idle >= NO_PROGRESS_SEC:
                process.kill()
                process.wait(timeout=30)
                removed = remove_incomplete_files(repo_cache_dirs(cache_dir, item))
                print(f"[kura] hf download stalled {label}; removed {removed} incomplete file(s); retrying", flush=True)
                snapshots = [tree_snapshot(directory) for directory in progress_dirs]
                last_total = sum(snapshot[0] for snapshot in snapshots)
                last_mtime = max((snapshot[1] for snapshot in snapshots), default=0.0)
                baseline_total = last_total
                baseline_count = sum(snapshot[2] for snapshot in snapshots)
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
                os.symlink(stable_link_target(path, link_path), link_path)
            print(f"[kura] downloaded {item['key']} -> {path}", flush=True)
            print(f"[kura] linked {item['key']} -> {link_path}", flush=True)
            return
        if output.strip():
            print(output.strip(), flush=True)
        if attempt < ATTEMPTS:
            time.sleep(min(30, POLL_SEC))
    raise SystemExit(f"[kura] hf download failed after {ATTEMPTS} attempts: {label}")


def main():
    items = json.loads(sys.argv[1])
    preflight_downloads(items)
    for item in items:
        run_one(item)


if __name__ == "__main__":
    main()

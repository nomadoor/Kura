# This script runs inside training containers with stdlib plus huggingface_hub.
# Do not import kura here; it is delivered as `python -c` source text.

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


def stable_link_target(path, link_path):
    try:
        target = os.path.abspath(path)
        link = os.path.abspath(link_path)
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
        target_workspace = os.path.commonpath([target, "/workspace"]) == "/workspace"
        link_workspace = os.path.commonpath([link, "/workspace"]) == "/workspace"
    except ValueError:
        return path
    if target_workspace and link_workspace:
        return os.path.relpath(target, os.path.dirname(link))
    return path


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
    for item in json.loads(sys.argv[1]):
        run_one(item)


if __name__ == "__main__":
    main()

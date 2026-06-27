"""Run a Kura job on ephemeral RunPod storage with S3-compatible staging.

The container disk is disposable. This wrapper downloads the staged Kura
workspace prefix into /workspace, runs the backend command, and uploads the run
directory back to the same object store before exiting with the backend code.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import boto3
from botocore.config import Config


def env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=env("KURA_OBJECT_ENDPOINT_URL"),
        region_name=os.environ.get("KURA_OBJECT_REGION", "auto"),
        aws_access_key_id=env("KURA_OBJECT_ACCESS_KEY_ID"),
        aws_secret_access_key=env("KURA_OBJECT_SECRET_ACCESS_KEY"),
        config=Config(retries={"max_attempts": 10, "mode": "standard"}, read_timeout=7200),
    )


def download_prefix(client, bucket: str, prefix: str, workspace: Path) -> int:
    count = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for item in page.get("Contents", []):
            key = item["Key"]
            relative = key[len(prefix) + 1:]
            if not relative or relative.endswith("/"):
                continue
            target = workspace / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(target))
            count += 1
    return count


def upload_tree(client, bucket: str, prefix: str, workspace: Path, relative_root: str) -> int:
    root = workspace / relative_root
    if not root.exists():
        return 0
    count = 0
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            key = f"{prefix}/{path.relative_to(workspace).as_posix()}"
            client.upload_file(str(path), bucket, key)
            count += 1
    return count


def append_jsonl(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwd", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise RuntimeError("backend command is required")

    bucket = env("KURA_OBJECT_BUCKET")
    prefix = env("KURA_OBJECT_PREFIX").strip("/")
    run_id = env("KURA_RUN_ID")
    workspace = Path(os.environ.get("KURA_WORKSPACE", "/workspace"))
    log_path = Path(os.environ.get("KURA_LOG_PATH", str(workspace / "runs" / run_id / "logs" / "stdout.log")))
    events_path = workspace / "runs" / run_id / "logs" / "events.jsonl"
    client = s3_client()

    exit_code = 1
    started = datetime.now().astimezone().isoformat()
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        downloaded = download_prefix(client, bucket, prefix, workspace)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        append_jsonl(events_path, {"event": "remote_workspace_downloaded", "timestamp": started, "object_prefix": prefix, "files": downloaded})
        with log_path.open("ab") as log:
            result = subprocess.run(command, cwd=args.cwd, stdout=log, stderr=subprocess.STDOUT, check=False)
        exit_code = int(result.returncode)
    except Exception:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
        exit_code = 1
    finally:
        ended = datetime.now().astimezone().isoformat()
        exit_record = workspace / "runs" / run_id / "realizations" / f"remote-exit-{ended.replace(':', '').replace('.', '-')}.json"
        exit_record.parent.mkdir(parents=True, exist_ok=True)
        exit_record.write_text(json.dumps({"event": "remote_exit", "timestamp": ended, "exit_code": exit_code}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        append_jsonl(events_path, {"event": "remote_workspace_uploading", "timestamp": ended, "object_prefix": prefix, "exit_code": exit_code})
        try:
            upload_tree(client, bucket, prefix, workspace, f"runs/{run_id}")
        except Exception:
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exc(file=log)
            return 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

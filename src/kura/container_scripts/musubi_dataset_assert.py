# This script runs inside training containers with stdlib only.
# Do not import kura here; it is delivered as `python -c` source text.

from __future__ import annotations

import json
import sys
from pathlib import Path

import tomllib


IMAGE_SUFFIXES = {".avif", ".bmp", ".jpeg", ".jpg", ".png", ".webp"}
VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}


def die(message):
    raise SystemExit(f"[kura] {message}")


def media_count(directory, suffixes, label):
    try:
        return sum(1 for item in directory.iterdir() if item.is_file() and item.suffix.lower() in suffixes)
    except OSError as exc:
        die(f"cannot read Musubi {label} {directory}: {exc}")


def jsonl_count(path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError as exc:
        die(f"cannot read Musubi image_jsonl_file {path}: {exc}")


def main():
    if len(sys.argv) != 2:
        die("usage: musubi_dataset_assert.py DATASET_TOML")
    config_path = Path(sys.argv[1])
    try:
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        die(f"cannot parse Musubi dataset config {config_path}: {exc}")
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        die(f"Musubi dataset config has no [[datasets]] entries: {config_path}")
    summary = []
    for index, item in enumerate(datasets, start=1):
        if not isinstance(item, dict):
            die(f"Musubi dataset entry #{index} is not a table")
        image_directory = item.get("image_directory")
        video_directory = item.get("video_directory")
        image_jsonl_file = item.get("image_jsonl_file")
        video_jsonl_file = item.get("video_jsonl_file")
        if isinstance(image_directory, str) and image_directory:
            count = media_count(Path(image_directory), IMAGE_SUFFIXES, "image_directory")
            if count <= 0:
                die(f"Musubi dataset entry #{index} has no images in image_directory: {image_directory}")
            summary.append({"index": index, "image_directory": image_directory, "images": count})
            continue
        if isinstance(video_directory, str) and video_directory:
            count = media_count(Path(video_directory), VIDEO_SUFFIXES, "video_directory")
            if count <= 0:
                die(f"Musubi dataset entry #{index} has no videos in video_directory: {video_directory}")
            summary.append({"index": index, "video_directory": video_directory, "videos": count})
            continue
        if isinstance(image_jsonl_file, str) and image_jsonl_file:
            count = jsonl_count(Path(image_jsonl_file))
            if count <= 0:
                die(f"Musubi dataset entry #{index} has no rows in image_jsonl_file: {image_jsonl_file}")
            summary.append({"index": index, "image_jsonl_file": image_jsonl_file, "rows": count})
            continue
        if isinstance(video_jsonl_file, str) and video_jsonl_file:
            count = jsonl_count(Path(video_jsonl_file))
            if count <= 0:
                die(f"Musubi dataset entry #{index} has no rows in video_jsonl_file: {video_jsonl_file}")
            summary.append({"index": index, "video_jsonl_file": video_jsonl_file, "rows": count})
            continue
        summary.append({"index": index, "source": "unknown; deferred to Musubi"})
    print(f"[kura] musubi dataset ok {json.dumps(summary, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()

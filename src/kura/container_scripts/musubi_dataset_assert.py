# This script is delivered as `python -c` source text inside the pinned Musubi
# container. Do not import Kura here. Video checks deliberately use Musubi's
# own pinned loader so their frame-count semantics cannot drift from training.

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
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


def video_jsonl_paths(path):
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]
    except (OSError, json.JSONDecodeError) as exc:
        die(f"cannot read Musubi video_jsonl_file {path}: {exc}")
    videos = []
    for index, row in enumerate(rows, start=1):
        value = row.get("video_path") if isinstance(row, dict) else None
        if not isinstance(value, str) or not value:
            die(f"Musubi video_jsonl_file {path} row {index} has no video_path")
        videos.append(Path(value))
    return videos


def realization_record_path():
    workspace = os.environ.get("KURA_WORKSPACE")
    run_id = os.environ.get("KURA_RUN_ID")
    realization_id = os.environ.get("KURA_REALIZATION_ID")
    if not all(isinstance(value, str) and value for value in (workspace, run_id, realization_id)):
        die("Musubi video frame preflight requires KURA_WORKSPACE, KURA_RUN_ID, and KURA_REALIZATION_ID")
    return Path(workspace) / "runs" / run_id / "realizations" / f"{realization_id}.musubi-video-preflight.json"


def write_record(payload):
    path = realization_record_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sample_id_for_input(input_id, semantic):
    if not isinstance(input_id, str):
        return None
    parts = input_id.split(":")
    if len(parts) < 2 or not parts[0].startswith("d") or not parts[1].startswith("s"):
        return None
    try:
        dataset_index = int(parts[0][1:])
        sample_index = int(parts[1][1:])
        sample = semantic["datasets"][dataset_index]["samples"][sample_index]
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    sample_id = sample.get("id") if isinstance(sample, dict) else None
    return sample_id if isinstance(sample_id, str) else None


def input_context_by_view_path():
    workspace = os.environ.get("KURA_WORKSPACE")
    run_id = os.environ.get("KURA_RUN_ID")
    if not isinstance(workspace, str) or not isinstance(run_id, str):
        return {}
    lock_path = Path(workspace) / "runs" / run_id / "resolved" / "dataset-input.lock.json"
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    semantic = lock.get("semantic") if isinstance(lock, dict) else None
    contexts = {}
    for view in lock.get("views", []) if isinstance(lock, dict) else []:
        for link in view.get("links", []) if isinstance(view, dict) else []:
            path = link.get("path") if isinstance(link, dict) else None
            target = link.get("target") if isinstance(link, dict) else None
            if not isinstance(path, str) or not isinstance(target, str):
                continue
            contexts[str(Path(workspace) / path)] = {
                "source": target,
                "sample_id": sample_id_for_input(link.get("input_id"), semantic),
            }
    return contexts


def video_frame_preflight(entries, config_path):
    try:
        from musubi_tuner.dataset.media_utils import load_video
    except (ImportError, ModuleNotFoundError) as exc:
        die(f"pinned Musubi video loader is unavailable: {exc}")
    architecture = os.environ.get("KURA_MUSUBI_ARCHITECTURE")
    try:
        architecture_target_fps = float(os.environ["KURA_MUSUBI_TARGET_FPS"])
    except (KeyError, ValueError) as exc:
        die(f"Musubi video frame preflight has no target fps for architecture {architecture!r}: {exc}")
    contexts = input_context_by_view_path()
    measured = []
    errors = []
    for entry in entries:
        required = max(entry["target_frames"])
        source_fps = entry.get("source_fps")
        target_fps = architecture_target_fps if source_fps is not None else None
        for video in entry["videos"]:
            context = contexts.get(str(video), {})
            try:
                frames = load_video(
                    str(video),
                    0,
                    required,
                    source_fps=source_fps,
                    target_fps=target_fps,
                    bucket_reso=(64, 64),
                )
                effective_frames = len(frames)
                item = {
                    "dataset_index": entry["index"],
                    "video": str(video),
                    "source": context.get("source") or (os.readlink(video) if video.is_symlink() else str(video.resolve())),
                    "sample_id": context.get("sample_id"),
                    "effective_frames": effective_frames,
                    "required_frames": required,
                    "source_fps": source_fps,
                    "target_fps": target_fps,
                    "passed": effective_frames >= required,
                }
                measured.append(item)
            except Exception as exc:
                errors.append({
                    "dataset_index": entry["index"],
                    "video": str(video),
                    "source": context.get("source"),
                    "sample_id": context.get("sample_id"),
                    "error": f"{type(exc).__name__}: {exc}",
                })
    failures = [item for item in measured if not item["passed"]]
    status = "passed" if not failures and not errors else "failed"
    record = {
        "schema_version": 1,
        "event": "musubi_video_frame_preflight",
        "status": status,
        "observed_at": datetime.now().astimezone().isoformat(),
        "dataset_config": str(config_path),
        "architecture": architecture,
        "target_fps": architecture_target_fps,
        "videos": measured,
        "errors": errors,
    }
    write_record(record)
    if errors:
        details = "\n".join(
            f"- {item['video']} (source {item.get('source') or 'unknown'}, "
            f"sample {item.get('sample_id') or 'unknown'}): {item['error']}"
            for item in errors
        )
        die("Musubi video frame preflight could not measure every selected video:\n" + details)
    if failures:
        details = "\n".join(
            f"- {item['video']}: {item['effective_frames']} converted frames "
            f"(requires {item['required_frames']})"
            f"; source {item['source']}; sample {item.get('sample_id') or 'unknown'}"
            for item in failures
        )
        die("Musubi video frame preflight found videos shorter than max(target_frames):\n" + details)
    return record


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
    video_entries = []
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
            target_frames = item.get("target_frames")
            if (
                not isinstance(target_frames, list)
                or not target_frames
                or not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in target_frames)
            ):
                die(f"Musubi dataset entry #{index} video_directory requires positive target_frames")
            video_entries.append({
                "index": index,
                "video_directory": video_directory,
                "videos": sorted(
                    video for video in Path(video_directory).iterdir()
                    if video.is_file() and video.suffix.lower() in VIDEO_SUFFIXES
                ),
                "target_frames": target_frames,
                "source_fps": item.get("source_fps"),
            })
            summary.append({"index": index, "video_directory": video_directory, "videos": count})
            continue
        if isinstance(image_jsonl_file, str) and image_jsonl_file:
            count = jsonl_count(Path(image_jsonl_file))
            if count <= 0:
                die(f"Musubi dataset entry #{index} has no rows in image_jsonl_file: {image_jsonl_file}")
            summary.append({"index": index, "image_jsonl_file": image_jsonl_file, "rows": count})
            continue
        if isinstance(video_jsonl_file, str) and video_jsonl_file:
            videos = video_jsonl_paths(Path(video_jsonl_file))
            count = len(videos)
            if count <= 0:
                die(f"Musubi dataset entry #{index} has no rows in video_jsonl_file: {video_jsonl_file}")
            target_frames = item.get("target_frames")
            if (
                not isinstance(target_frames, list)
                or not target_frames
                or not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in target_frames)
            ):
                die(f"Musubi dataset entry #{index} video_jsonl_file requires positive target_frames")
            video_entries.append({
                "index": index,
                "video_jsonl_file": video_jsonl_file,
                "videos": videos,
                "target_frames": target_frames,
                "source_fps": item.get("source_fps"),
            })
            summary.append({"index": index, "video_jsonl_file": video_jsonl_file, "rows": count})
            continue
        summary.append({"index": index, "source": "unknown; deferred to Musubi"})
    if video_entries:
        record = video_frame_preflight(video_entries, config_path)
        print(
            f"[kura] musubi video frame preflight passed "
            f"{json.dumps({'videos': len(record['videos']), 'record': str(realization_record_path())}, ensure_ascii=False)}",
            flush=True,
        )
    print(f"[kura] musubi dataset ok {json.dumps(summary, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()

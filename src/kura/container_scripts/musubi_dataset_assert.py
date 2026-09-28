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


def die(message):
    raise SystemExit(f"[kura] {message}")


def frozen_suffixes(name):
    try:
        values = json.loads(os.environ[name])
    except (KeyError, json.JSONDecodeError) as exc:
        die(f"Musubi dataset preflight requires valid {name}: {exc}")
    if (
        not isinstance(values, list)
        or not values
        or not all(
            isinstance(value, str)
            and value.startswith(".")
            and value == value.lower()
            for value in values
        )
    ):
        die(f"Musubi dataset preflight requires {name} to be a non-empty suffix list")
    return frozenset(values)


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


def video_jsonl_inputs(path):
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]
    except (OSError, json.JSONDecodeError) as exc:
        die(f"cannot read Musubi video_jsonl_file {path}: {exc}")
    inputs = []
    for index, row in enumerate(rows, start=1):
        value = row.get("video_path") if isinstance(row, dict) else None
        if not isinstance(value, str) or not value:
            die(f"Musubi video_jsonl_file {path} row {index} has no video_path")
        inputs.append({
            "video": Path(value),
            "control": (
                Path(row["control_path"])
                if isinstance(row.get("control_path"), str) and row["control_path"]
                else None
            ),
            "explicit_audio": isinstance(row.get("audio_path"), str) and bool(row["audio_path"]),
        })
    return inputs


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
    contexts = {}
    for view in lock.get("views", []) if isinstance(lock, dict) else []:
        for link in view.get("links", []) if isinstance(view, dict) else []:
            path = link.get("path") if isinstance(link, dict) else None
            target = link.get("target") if isinstance(link, dict) else None
            if not isinstance(path, str) or not isinstance(target, str):
                continue
            contexts[str(Path(workspace) / path)] = {
                "source": target,
                "dataset_id": link.get("dataset"),
                "sample_id": link.get("sample"),
            }
    return contexts


def video_frame_preflight(entries, config_path, audio_suffixes):
    try:
        from musubi_tuner.dataset.media_utils import load_video
    except (ImportError, ModuleNotFoundError) as exc:
        die(f"pinned Musubi video loader is unavailable: {exc}")
    architecture = os.environ.get("KURA_MUSUBI_ARCHITECTURE")
    native_dataset_architecture = os.environ.get("KURA_MUSUBI_NATIVE_DATASET_ARCHITECTURE")
    profiles = os.environ.get("KURA_MUSUBI_PROFILES")
    fps_resample_mode = os.environ.get("KURA_MUSUBI_FPS_RESAMPLE_MODE")
    strict_timestamp_fps = fps_resample_mode == "timestamps"
    if fps_resample_mode not in {"timestamps", "source-fps-when-declared"}:
        die(
            "Musubi video frame preflight has no verified profile resampling mode "
            f"for profile(s) {profiles!r}: {fps_resample_mode!r}"
        )
    try:
        architecture_target_fps = float(os.environ["KURA_MUSUBI_TARGET_FPS"])
    except (KeyError, ValueError) as exc:
        die(f"Musubi video frame preflight has no target fps for architecture {architecture!r}: {exc}")
    contexts = input_context_by_view_path()
    measured = []
    errors = []
    for entry in entries:
        frame_extraction = entry.get("frame_extraction") or "head"
        latent_window_size = entry.get("fp_latent_window_size")
        full_framepack = (
            architecture == "framepack"
            and frame_extraction == "full"
            and isinstance(latent_window_size, int)
        )
        required = (
            int(entry.get("max_frames") or 129)
            if full_framepack
            else max(entry["target_frames"])
        )
        source_fps = entry.get("source_fps")
        target_fps = architecture_target_fps if strict_timestamp_fps or source_fps is not None else None
        video_inputs = entry.get("video_inputs")
        if not isinstance(video_inputs, list):
            video_inputs = [{"video": video, "explicit_audio": False} for video in entry["videos"]]
        for video_input in video_inputs:
            video = video_input["video"]
            context = contexts.get(str(video), {})
            try:
                if strict_timestamp_fps and not video_input.get("explicit_audio"):
                    resolved_video = video.resolve()
                    sidecars = sorted(
                        candidate for candidate in resolved_video.parent.iterdir()
                        if candidate.is_file()
                        and candidate.stem == resolved_video.stem
                        and candidate.suffix.lower() in audio_suffixes
                    )
                    if sidecars:
                        raise ValueError(
                            "implicit same-stem audio sidecar beside the resolved JSONL video path: "
                            + ", ".join(str(path) for path in sidecars)
                            + "; declare the selected audio as the sample's audio role"
                        )
                kwargs = {"target_fps": target_fps, "bucket_reso": (64, 64)}
                if strict_timestamp_fps:
                    kwargs["fps_resample_mode"] = "timestamps"
                else:
                    kwargs["source_fps"] = source_fps
                frames = load_video(str(video), 0, required, **kwargs)
                loaded_frames = len(frames)
                control_item = None
                control = video_input.get("control")
                if isinstance(control, Path):
                    control_context = contexts.get(str(control), {})
                    try:
                        control_frames = load_video(str(control), 0, required, **kwargs)
                        control_loaded_frames = len(control_frames)
                        if control_loaded_frames <= 0:
                            raise ValueError("control input has no readable frames")
                        control_item = {
                            "video": str(control),
                            "source": control_context.get("source") or (
                                os.readlink(control) if control.is_symlink() else str(control.resolve())
                            ),
                            "sample_id": control_context.get("sample_id"),
                            "loaded_frames": control_loaded_frames,
                            "target_frames": loaded_frames,
                            "length_policy": "trim-or-repeat-last-to-target",
                            "passed": True,
                        }
                    except Exception as exc:
                        errors.append({
                            "dataset_index": entry["index"],
                            "video": str(control),
                            "source": control_context.get("source"),
                            "sample_id": control_context.get("sample_id"),
                            "input_kind": "control",
                            "error": f"{type(exc).__name__}: {exc}",
                        })
                        continue
                if full_framepack:
                    from musubi_tuner.dataset.architectures import round_down_frame_count

                    if not native_dataset_architecture:
                        raise ValueError("Musubi native dataset architecture is unavailable")
                    effective_frames = round_down_frame_count(loaded_frames, native_dataset_architecture, 4)
                else:
                    effective_frames = loaded_frames
                minimum_frames = latent_window_size * 4 + 1 if full_framepack else required
                item = {
                    "dataset_index": entry["index"],
                    "video": str(video),
                    "source": context.get("source") or (os.readlink(video) if video.is_symlink() else str(video.resolve())),
                    "sample_id": context.get("sample_id"),
                    "effective_frames": effective_frames,
                    "loaded_frames": loaded_frames,
                    "required_frames": minimum_frames,
                    "frame_extraction": frame_extraction,
                    "max_frames": entry.get("max_frames"),
                    "fp_latent_window_size": latent_window_size,
                    "source_fps": source_fps,
                    "target_fps": target_fps,
                    "fps_resample_mode": "timestamps" if strict_timestamp_fps else None,
                    "audio_selection": (
                        "explicit-jsonl-path"
                        if video_input.get("explicit_audio")
                        else "verified-no-sidecar; embedded-or-silence"
                    ),
                    "control": control_item,
                    "passed": effective_frames >= minimum_frames,
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
        "profiles": profiles.split(",") if profiles else [],
        "target_fps": architecture_target_fps,
        "fps_resample_mode": fps_resample_mode,
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
        if any(item.get("frame_extraction") == "full" for item in failures):
            die("Musubi video frame preflight found videos shorter than FramePack's full-window minimum:\n" + details)
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
    image_suffixes = frozen_suffixes("KURA_MUSUBI_IMAGE_SUFFIXES")
    video_suffixes = frozen_suffixes("KURA_MUSUBI_VIDEO_SUFFIXES")
    audio_suffixes = frozen_suffixes("KURA_MUSUBI_AUDIO_SUFFIXES")
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
            count = media_count(Path(image_directory), image_suffixes, "image_directory")
            if count <= 0:
                die(f"Musubi dataset entry #{index} has no images in image_directory: {image_directory}")
            summary.append({"index": index, "image_directory": image_directory, "images": count})
            continue
        if isinstance(video_directory, str) and video_directory:
            count = media_count(Path(video_directory), video_suffixes, "video_directory")
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
                    if video.is_file() and video.suffix.lower() in video_suffixes
                ),
                "target_frames": target_frames,
                "frame_extraction": item.get("frame_extraction"),
                "max_frames": item.get("max_frames"),
                "fp_latent_window_size": item.get("fp_latent_window_size"),
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
            video_inputs = video_jsonl_inputs(Path(video_jsonl_file))
            videos = [item["video"] for item in video_inputs]
            count = len(video_inputs)
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
                "video_inputs": video_inputs,
                "target_frames": target_frames,
                "frame_extraction": item.get("frame_extraction"),
                "max_frames": item.get("max_frames"),
                "fp_latent_window_size": item.get("fp_latent_window_size"),
                "source_fps": item.get("source_fps"),
            })
            summary.append({"index": index, "video_jsonl_file": video_jsonl_file, "rows": count})
            continue
        summary.append({"index": index, "source": "unknown; deferred to Musubi"})
    if video_entries:
        record = video_frame_preflight(video_entries, config_path, audio_suffixes)
        print(
            f"[kura] musubi video frame preflight passed "
            f"{json.dumps({'videos': len(record['videos']), 'record': str(realization_record_path())}, ensure_ascii=False)}",
            flush=True,
        )
    print(f"[kura] musubi dataset ok {json.dumps(summary, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()

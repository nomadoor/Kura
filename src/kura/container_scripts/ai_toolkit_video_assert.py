"""Fail before model acquisition when selected AI-Toolkit audio is unusable."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".webm", ".mkv", ".wmv", ".m4v", ".flv"}


def _record_path() -> Path:
    workspace = os.environ.get("KURA_WORKSPACE")
    run_id = os.environ.get("KURA_RUN_ID")
    realization_id = os.environ.get("KURA_REALIZATION_ID")
    if not workspace or not run_id or not realization_id:
        raise SystemExit(
            "AI-Toolkit embedded-audio preflight requires KURA_WORKSPACE, "
            "KURA_RUN_ID, and KURA_REALIZATION_ID"
        )
    return (
        Path(workspace) / "runs" / run_id / "realizations"
        / f"{realization_id}.ai-toolkit-video-preflight.json"
    )


def _input_contexts() -> dict[str, dict]:
    workspace = Path(os.environ["KURA_WORKSPACE"])
    lock = workspace / "runs" / os.environ["KURA_RUN_ID"] / "resolved" / "dataset-input.lock.json"
    try:
        payload = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    contexts = {}
    for view in payload.get("views", []) if isinstance(payload, dict) else []:
        for link in view.get("links", []) if isinstance(view, dict) else []:
            path = link.get("path")
            if isinstance(path, str):
                contexts[str(workspace / path)] = {
                    "input_id": link.get("input_id"),
                    "dataset_id": link.get("dataset"),
                    "sample_id": link.get("sample"),
                    "source": link.get("target"),
                }
    return contexts


def _video_paths(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def _probe_with_pinned_loader(path: Path, native: dict) -> None:
    # This intentionally invokes the pinned trainer's own video/audio loader.
    # The small spatial crop limits preflight memory without changing frame or
    # audio selection.
    from toolkit.config_modules import DatasetConfig
    from toolkit.data_transfer_object.data_loader import FileItemDTO

    config = DatasetConfig(**native)
    item = FileItemDTO(
        path=str(path), dataset_config=config, dataset_root=str(path.parent),
        scale_to_width=64, scale_to_height=64, crop_width=64, crop_height=64,
    )
    item.load_and_process_video(None)
    audio = getattr(item, "audio_tensor", None)
    if audio is None or not callable(getattr(audio, "numel", None)) or audio.numel() <= 0:
        raise ValueError("the pinned AI-Toolkit loader produced no usable audio tensor")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: ai_toolkit_video_assert.py AI_TOOLKIT_YAML")
    import yaml

    config_path = Path(sys.argv[1])
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    processes = payload.get("config", {}).get("process", []) if isinstance(payload, dict) else []
    process = processes[0] if isinstance(processes, list) and processes else {}
    datasets = process.get("datasets", []) if isinstance(process, dict) else []
    contexts = _input_contexts()
    results = []
    failures = []
    for dataset_index, native in enumerate(datasets):
        if not isinstance(native, dict) or native.get("do_audio") is not True:
            continue
        folder = native.get("folder_path")
        if not isinstance(folder, str):
            failures.append({"dataset_index": dataset_index, "error": "folder_path is missing"})
            continue
        videos = _video_paths(Path(folder))
        if not videos:
            failures.append({"dataset_index": dataset_index, "error": "no selected video files"})
            continue
        for video in videos:
            context = contexts.get(str(video), {})
            result = {
                "dataset_index": dataset_index,
                "view_path": str(video),
                **context,
            }
            try:
                _probe_with_pinned_loader(video, native)
                result["status"] = "usable"
            except Exception as error:
                result.update({"status": "unusable", "error": f"{type(error).__name__}: {error}"})
                failures.append(result)
            results.append(result)
    record = {
        "schema_version": 1,
        "event": "ai_toolkit_embedded_audio_preflight",
        "config": str(config_path),
        "videos": results,
        "status": "failed" if failures else "passed",
    }
    destination = _record_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    if failures:
        details = "\n".join(
            f"- sample={item.get('sample_id')!r} source={item.get('source')!r} "
            f"view={item.get('view_path')!r}: {item.get('error')}"
            for item in failures
        )
        raise SystemExit("AI-Toolkit embedded-audio preflight failed:\n" + details)
    print(
        "[kura] AI-Toolkit embedded-audio preflight passed "
        + json.dumps({"videos": len(results), "record": str(destination)}),
        flush=True,
    )


if __name__ == "__main__":
    main()

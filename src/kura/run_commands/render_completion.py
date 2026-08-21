"""Concise provenance-rich completion output for render runs."""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any

import yaml

from kura.run_commands.experiment import _format_duration


_STEP_RE = re.compile(r"(?:^|[-_])step0*([0-9]+)(?:[-_.]|$)", re.IGNORECASE)


def _mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return {}
    return value if isinstance(value, dict) else {}


def _json_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _image_records(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return []
    records: list[dict[str, Any]] = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _checkpoint_label(workspace: Path, frozen: dict[str, Any], records: list[dict[str, Any]]) -> str:
    inputs = frozen.get("inputs") if isinstance(frozen.get("inputs"), dict) else {}
    train_run = inputs.get("train_run")
    record_paths = list(dict.fromkeys(
        str(record["checkpoint_path"])
        for record in records
        if isinstance(record.get("checkpoint_path"), str) and record["checkpoint_path"]
    ))
    if len(record_paths) > 1:
        pieces = [str(train_run)] if isinstance(train_run, str) and train_run else []
        pieces.append(f"{len(record_paths)} checkpoints vary by case")
        return " · ".join(pieces)
    checkpoint = inputs.get("checkpoint") if isinstance(inputs.get("checkpoint"), dict) else {}
    path_value = record_paths[0] if record_paths else checkpoint.get("path")
    pieces = [str(train_run)] if isinstance(train_run, str) and train_run else []
    if isinstance(path_value, str) and path_value:
        name = Path(path_value).name
        pieces.append(name)
        match = _STEP_RE.search(name)
        if match:
            pieces.append(f"step {int(match.group(1))}")
        elif isinstance(train_run, str) and train_run:
            train_status = _json_mapping(workspace / "runs" / train_run / "status.json")
            outputs = train_status.get("outputs") if isinstance(train_status.get("outputs"), list) else []
            output_names = {Path(item).name for item in outputs if isinstance(item, str)}
            last_step = train_status.get("last_step")
            if name in output_names and isinstance(last_step, int):
                pieces.append(f"final step {last_step}")
    return " · ".join(pieces) or "(not recorded)"


def _application_label(application: dict[str, Any]) -> str:
    kind = application.get("kind")
    class_type = application.get("class_type")
    if kind in {"lora_insert", "lora_binding"}:
        mode = "LoRA model+CLIP" if class_type == "LoraLoader" else "LoRA model only"
        strengths = []
        if isinstance(application.get("strength_model"), (int, float)):
            strengths.append(f"model {application['strength_model']:g}")
        if isinstance(application.get("strength_clip"), (int, float)):
            strengths.append(f"clip {application['strength_clip']:g}")
        return mode + (" · strength " + " / ".join(strengths) if strengths else " · strength workflow-defined")
    if kind == "model_patch_binding":
        return "model patch binding"
    if kind == "checkpoint_binding":
        return "checkpoint binding"
    if kind == "checkpoint_reference":
        return "checkpoint reference"
    return "none recorded"


def _quoted(value: Any, *, width: int = 120) -> str:
    if value is None:
        return "(workflow-owned)"
    text = " ".join(str(value).split())
    shortened = textwrap.shorten(text, width=width, placeholder="…") if text else ""
    return json.dumps(shortened, ensure_ascii=False)


def format_render_completion(workspace: Path, run_dir: Path, *, exit_code: int | None = None) -> str:
    """Format terminal render facts without defaulting to hashes or digests."""

    status = _json_mapping(run_dir / "status.json")
    frozen = _mapping(run_dir / "resolved" / "manifest.lock.yaml")
    records = _image_records(run_dir / "samples" / "images.jsonl")
    state = str(status.get("state") or "unknown")
    if exit_code is not None and state not in {"completed", "failed", "interrupted"}:
        state = "completed" if exit_code == 0 else "interrupted" if exit_code == 130 else "failed"
    code = status.get("exit_code") if status.get("exit_code") is not None else exit_code
    duration = _format_duration(status.get("started"), status.get("ended"))
    headline = f"{state}  exit {code if code is not None else '-'}"
    if duration:
        headline += f"  {duration}"

    inputs = frozen.get("inputs") if isinstance(frozen.get("inputs"), dict) else {}
    workflow = inputs.get("workflow") if isinstance(inputs.get("workflow"), dict) else {}
    workflow_name = Path(str(workflow.get("path") or "")).name or "(not recorded)"
    output_dir = frozen.get("render", {}).get("output_dir", "samples/images") if isinstance(frozen.get("render"), dict) else "samples/images"
    image_word = "image" if len(records) == 1 else "images"
    lines = [
        headline,
        f"rendered   {len(records)} {image_word}  {output_dir}",
        f"artifact   {_checkpoint_label(workspace, frozen, records)}",
        f"workflow   {workflow_name}",
    ]
    applications = [
        record["checkpoint_application"]
        for record in records
        if isinstance(record.get("checkpoint_application"), dict)
    ]
    if not applications:
        fallback = frozen.get("lora_insert") if isinstance(frozen.get("lora_insert"), dict) else {}
        applications = [{"kind": "lora_insert", **fallback}] if fallback else [{}]
    application_labels = list(dict.fromkeys(_application_label(item) for item in applications))
    if len(application_labels) == 1:
        lines.append(f"applied    {application_labels[0]}")
    else:
        lines.append(f"applied    {len(application_labels)} settings vary by case · samples/images.jsonl")

    cases: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any, Any]] = set()
    for record in records:
        identity = (record.get("case_id") or record.get("prompt_id"), record.get("prompt"), record.get("negative_prompt"), record.get("seed"))
        if identity not in seen:
            seen.add(identity)
            cases.append(record)
    lines.append(f"inputs     {len(cases)} case{'s' if len(cases) != 1 else ''}")
    if len(cases) == 1:
        record = cases[0]
        case_id = record.get("prompt_id") or "case"
        lines.append(f"  {case_id}  seed {record.get('seed') if record.get('seed') is not None else '(workflow-owned)'}")
        lines.append(f"    prompt    {_quoted(record.get('prompt'))}")
        lines.append(f"    negative  {_quoted(record.get('negative_prompt'))}")
    elif cases:
        source = inputs.get("cases") if isinstance(inputs.get("cases"), dict) else inputs.get("promptset")
        source = source if isinstance(source, dict) else {}
        source_name = Path(str(source.get("path") or "")).name or "resolved cases"
        for label, key in (("prompts", "prompt"), ("negative", "negative_prompt")):
            values = {json.dumps(record.get(key), ensure_ascii=False, sort_keys=True) for record in cases}
            if len(values) == 1:
                lines.append(f"  {label:<10}{_quoted(cases[0].get(key))}")
            else:
                lines.append(f"  {label:<10}vary by case · {source_name}")
        seeds = {record.get("seed") for record in cases}
        if len(seeds) == 1:
            seed = cases[0].get("seed")
            lines.append(f"  {'seed':<10}{seed if seed is not None else '(workflow-owned)'}")
        else:
            lines.append(f"  {'seeds':<10}vary by case")
        lines.append("  provenance samples/images.jsonl")
    return "\n".join(lines)

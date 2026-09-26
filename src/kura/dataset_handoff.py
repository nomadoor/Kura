"""Freeze complete backend projections and materialize disposable run views."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Any, Callable

from kura.dataset_manifest import measure_manifest
from kura.fsio import atomic_write_json


Project = Callable[[dict[str, Any]], dict[str, Any]]
_PORTABLE_STAT = ("size", "mtime_ns", "ctime_ns")
_TRAINER_MEDIA_SUFFIXES = frozenset({
    ".avif", ".avi", ".bmp", ".flac", ".gif", ".jpeg", ".jpg", ".m4v", ".mkv",
    ".mov", ".mp3", ".mp4", ".ogg", ".png", ".wav", ".webm", ".webp",
})
_PATH_LIKE_SUFFIXES = _TRAINER_MEDIA_SUFFIXES | frozenset({
    ".csv", ".json", ".jsonl", ".toml", ".txt", ".yaml", ".yml",
})


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _safe_workspace_relative(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"{context} must be a nonempty workspace-relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"{context} must stay inside the workspace")
    return path.as_posix()


def _manifest_selection(run: dict[str, Any], workspace: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    authored = run.get("datasets")
    if not isinstance(authored, list) or not authored:
        raise ValueError("training run requires datasets[]")
    datasets: list[dict[str, Any]] = []
    input_index: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    for dataset_index, selected in enumerate(authored):
        dataset_id = selected.get("id") if isinstance(selected, dict) else None
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError(f"datasets[{dataset_index}].id must be nonempty")
        if dataset_id in seen_ids:
            raise ValueError(f"dataset {dataset_id!r} is selected more than once")
        seen_ids.add(dataset_id)
        logical_root = workspace / "datasets" / dataset_id
        physical_root = logical_root.resolve(strict=True)
        measured = measure_manifest(logical_root)
        samples: list[dict[str, Any]] = []
        for sample_index, sample in enumerate(measured["identity"]["samples"]):
            files: list[dict[str, Any]] = []
            for file_index, reference in enumerate(sample["files"]):
                input_id = f"d{dataset_index}:s{sample_index}:f{file_index}"
                item = {**reference, "input_id": input_id}
                files.append(item)
                input_index[input_id] = {
                    "dataset": dataset_id,
                    "sample": sample["id"],
                    "kind": f"file[{file_index}] role={reference['role']}",
                    "binding_kind": f"file-role:{reference['role']}",
                    "container_source": f"/workspace/datasets/{dataset_id}/{reference['path']}",
                }
            caption_text = sample["caption"]
            caption = None
            if caption_text is not None:
                caption_id = f"d{dataset_index}:s{sample_index}:caption"
                caption = {"input_id": caption_id, "text": caption_text}
                input_index[caption_id] = {
                    "dataset": dataset_id,
                    "sample": sample["id"],
                    "kind": "caption",
                    "binding_kind": "caption",
                    "text": caption_text,
                }
            samples.append({
                "id": sample["id"], "group": sample.get("group"),
                "files": files, "caption": caption,
            })
        files = []
        for item in measured["files"]:
            files.append({
                "source": f"datasets/{dataset_id}/{item['path']}",
                "container_source": f"/workspace/datasets/{dataset_id}/{item['path']}",
                "sha256": item["sha256"],
                "stat": {key: item["stat"][key] for key in _PORTABLE_STAT},
            })
        datasets.append({
            "id": dataset_id,
            "logical_root": f"datasets/{dataset_id}",
            "physical_root": str(physical_root),
            "identity": measured["identity"],
            "identity_sha256": measured["identity_sha256"],
            "samples": samples,
            "files": files,
            "authoring_files": [
                {
                    "source": f"datasets/{dataset_id}/{item['path']}",
                    "stat": {key: item["stat"][key] for key in _PORTABLE_STAT},
                }
                for item in measured.get("authoring_files", [])
            ],
            "excluded_files": measured["excluded_files"],
            "excluded_directories": measured["excluded_directories"],
        })
    return {"schema_version": 1, "run_id": run.get("id"), "datasets": datasets}, input_index


def _projection_error(input_index: dict[str, dict[str, Any]], input_id: Any, reason: str) -> ValueError:
    identity = input_index.get(input_id) if isinstance(input_id, str) else None
    if identity is None:
        return ValueError(f"projection reports unknown input {input_id!r}: {reason}")
    return ValueError(
        f"dataset {identity['dataset']!r} sample {identity['sample']!r} "
        f"{identity['kind']} is unrepresentable: {reason}"
    )


def _validate_bindings(
    dataset_id: str,
    view_root: str,
    bindings: Any,
    placements: dict[str, str],
    input_index: dict[str, dict[str, Any]],
) -> None:
    if not isinstance(bindings, list):
        raise ValueError(f"backend projection for dataset {dataset_id!r} has no input bindings")
    bound: list[str] = []
    seen_keys: set[str] = set()
    bound_samples: set[str] = set()
    roots_by_kind: dict[str, str] = {}
    for index, binding in enumerate(bindings):
        if (
            not isinstance(binding, dict)
            or set(binding) != {"rule", "key", "members"}
            or binding.get("rule") != "same-relative-stem"
        ):
            raise ValueError(f"projection binding {index} has an unsupported rule or shape")
        key = binding.get("key")
        members = binding.get("members")
        if not isinstance(key, str) or not key:
            raise ValueError(f"projection binding {index} has no association key")
        key = _safe_workspace_relative(key, context=f"projection binding {index} association key")
        if not isinstance(members, list) or not members:
            raise ValueError(f"projection binding {index} must name one or more members")
        input_ids: list[str] = []
        member_roots: list[str] = []
        relative_stems: list[str] = []
        for member_index, member in enumerate(members):
            if not isinstance(member, dict) or set(member) != {"input_id", "root"}:
                raise ValueError(f"projection binding {index} member {member_index} has an invalid shape")
            input_id = member.get("input_id")
            member_root = _safe_workspace_relative(
                member.get("root"), context=f"projection binding {index} member {member_index} root",
            )
            if not isinstance(input_id, str):
                raise ValueError(f"projection binding {index} member {member_index} has an invalid input")
            if member_root != view_root and not member_root.startswith(view_root + "/"):
                raise ValueError(f"projection binding {index} member {member_index} root is outside its view")
            try:
                placement = placements[input_id]
            except KeyError as exc:
                raise ValueError(f"projection binding {index} names an input absent from the view") from exc
            if not placement.startswith(member_root + "/"):
                raise ValueError(f"projection binding {index} member {member_index} is outside its role root")
            relative = PurePosixPath(placement).relative_to(PurePosixPath(member_root))
            relative_stems.append(relative.with_suffix("").as_posix())
            input_ids.append(input_id)
            member_roots.append(member_root)
        identities = [input_index.get(item) for item in input_ids]
        if any(not isinstance(item, dict) for item in identities):
            raise ValueError(f"projection binding {index} names an unknown input")
        if any(item.get("dataset") != dataset_id for item in identities if isinstance(item, dict)):
            raise ValueError(f"projection binding {index} crosses dataset boundaries")
        if len({item.get("sample") for item in identities if isinstance(item, dict)}) != 1:
            raise ValueError(f"projection binding {index} crosses sample boundaries")
        if any(stem != key for stem in relative_stems):
            raise ValueError(f"projection binding {index} violates its association key")
        if key in seen_keys:
            raise ValueError(f"projection binding {index} has duplicate association key {key!r}")
        seen_keys.add(key)
        sample = identities[0].get("sample") if isinstance(identities[0], dict) else None
        if not isinstance(sample, str):
            raise ValueError(f"projection binding {index} has no sample identity")
        if sample in bound_samples:
            raise ValueError(f"projection sample {sample!r} has more than one binding")
        bound_samples.add(sample)
        for identity, member_root in zip(identities, member_roots, strict=True):
            binding_kind = identity.get("binding_kind") if isinstance(identity, dict) else None
            if not isinstance(binding_kind, str):
                raise ValueError(f"projection binding {index} input has no binding kind")
            previous_root = roots_by_kind.setdefault(binding_kind, member_root)
            if previous_root != member_root:
                raise ValueError(
                    f"projection input kind {binding_kind!r} uses multiple role roots: "
                    f"{previous_root!r} and {member_root!r}"
                )
        bound.extend(input_ids)
    if Counter(bound) != Counter(placements.keys()):
        raise ValueError(f"projection bindings do not cover the exact dataset {dataset_id!r} view inputs")


def _pointer_escape(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _pointer_tokens(pointer: Any, *, context: str) -> list[str]:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError(f"{context} must be a non-root JSON Pointer")
    tokens: list[str] = []
    for raw in pointer[1:].split("/"):
        index = 0
        while index < len(raw):
            if raw[index] == "~" and (index + 1 >= len(raw) or raw[index + 1] not in {"0", "1"}):
                raise ValueError(f"{context} has an invalid JSON Pointer escape")
            index += 2 if raw[index] == "~" else 1
        tokens.append(raw.replace("~1", "/").replace("~0", "~"))
    return tokens


def _pointer_value(value: Any, pointer: Any, *, context: str) -> Any:
    current = value
    for token in _pointer_tokens(pointer, context=context):
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdecimal() and int(token) < len(current):
            current = current[int(token)]
        else:
            raise ValueError(f"{context} does not exist in the generated row")
    return current


def _pointer_replace(value: Any, pointer: Any, replacement: Any, *, context: str) -> None:
    tokens = _pointer_tokens(pointer, context=context)
    current = value
    for token in tokens[:-1]:
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdecimal() and int(token) < len(current):
            current = current[int(token)]
        else:
            raise ValueError(f"{context} does not exist in the generated row")
    leaf = tokens[-1]
    if isinstance(current, dict) and leaf in current:
        current[leaf] = replacement
    elif isinstance(current, list) and leaf.isdecimal() and int(leaf) < len(current):
        current[int(leaf)] = replacement
    else:
        raise ValueError(f"{context} does not exist in the generated row")


def _string_leaf_pointers(value: Any, pointer: str = "") -> set[str]:
    if isinstance(value, str):
        return {pointer}
    if isinstance(value, dict):
        found: set[str] = set()
        for key, child in value.items():
            found.update(_string_leaf_pointers(child, pointer + "/" + _pointer_escape(str(key))))
        return found
    if isinstance(value, list):
        found = set()
        for index, child in enumerate(value):
            found.update(_string_leaf_pointers(child, pointer + f"/{index}"))
        return found
    return set()


def _looks_like_path(value: str) -> bool:
    return (
        "/" in value
        or "\\" in value
        or value.startswith((".", "~"))
        or PurePosixPath(value).suffix.lower() in _PATH_LIKE_SUFFIXES
    )


def _jsonl_rows(text: str, *, context: str) -> list[str]:
    """Split JSONL only at LF row separators, never at Unicode line characters."""
    if "\r" in text:
        raise ValueError(f"{context} must use LF row separators")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if any(not line.strip() for line in lines):
        raise ValueError(f"{context} contains a blank row")
    return lines


def _validate_native_jsonl_files(
    dataset: dict[str, Any],
    root: str,
    native_files: Any,
    placements: dict[str, str],
    input_index: dict[str, dict[str, Any]],
    seen_paths: set[str],
) -> tuple[list[dict[str, Any]], set[str]]:
    if native_files is None:
        return [], set()
    if not isinstance(native_files, list):
        raise ValueError(f"backend projection for dataset {dataset['id']!r} native files must be a list")
    sample_ids = {item["id"] for item in dataset["samples"]}
    inputs_by_sample = {
        sample["id"]: [
            *[reference["input_id"] for reference in sample["files"]],
            *([sample["caption"]["input_id"]] if sample["caption"] is not None else []),
        ]
        for sample in dataset["samples"]
    }
    row_ids: set[str] = set()
    rows_by_sample: dict[str, list[Any]] = {sample_id: [] for sample_id in sample_ids}
    all_referenced: Counter[str] = Counter()
    path_referenced: Counter[str] = Counter()
    verified: list[dict[str, Any]] = []
    for file_index, native in enumerate(native_files):
        if (
            not isinstance(native, dict)
            or set(native) != {"path", "text", "format", "literal_string_fields", "rows"}
        ):
            raise ValueError(f"projection native file {file_index} has an invalid shape")
        path = _safe_workspace_relative(native.get("path"), context=f"projection native file {file_index}")
        text = native.get("text")
        rows = native.get("rows")
        literal_fields = native.get("literal_string_fields")
        if not path.startswith(root + "/") or path in seen_paths or not isinstance(text, str):
            raise ValueError(f"projection native file {path!r} is outside its view, duplicated, or non-text")
        if (
            native.get("format") != "jsonl"
            or not isinstance(rows, list)
            or not isinstance(literal_fields, list)
            or not all(isinstance(pointer, str) for pointer in literal_fields)
            or len(set(literal_fields)) != len(literal_fields)
        ):
            raise ValueError(f"projection native file {path!r} must declare JSONL rows")
        for pointer in literal_fields:
            _pointer_tokens(pointer, context=f"projection native JSONL {path!r} literal field")
        lines = _jsonl_rows(text, context=f"projection native JSONL {path!r}")
        try:
            parsed_rows = [json.loads(line) for line in lines]
        except json.JSONDecodeError as exc:
            raise ValueError(f"projection native JSONL {path!r} is invalid on line {exc.lineno}") from exc
        if len(parsed_rows) != len(rows):
            raise ValueError(f"projection native JSONL {path!r} row count differs from its report")
        for row_index, (parsed, reported) in enumerate(zip(parsed_rows, rows, strict=True)):
            context = f"projection native JSONL {path!r} row {row_index}"
            if not isinstance(parsed, dict):
                raise ValueError(f"{context} must be a JSON object")
            if (
                not isinstance(reported, dict)
                or set(reported) != {"row_id", "sample_id", "repeat", "references", "literal_strings"}
            ):
                raise ValueError(f"{context} report has an invalid shape")
            row_id, sample_id = reported.get("row_id"), reported.get("sample_id")
            if not isinstance(row_id, str) or not row_id or row_id in row_ids:
                raise ValueError(f"{context} has a missing or duplicate row identity")
            row_ids.add(row_id)
            if not isinstance(sample_id, str) or sample_id not in sample_ids:
                raise ValueError(f"{context} names an unknown sample {sample_id!r}")
            repeat = reported.get("repeat")
            if repeat is not None and (
                not isinstance(repeat, dict)
                or set(repeat) != {"index", "count"}
                or not isinstance(repeat.get("index"), int)
                or isinstance(repeat.get("index"), bool)
                or not isinstance(repeat.get("count"), int)
                or isinstance(repeat.get("count"), bool)
                or repeat["index"] < 0
                or repeat["count"] < 1
            ):
                raise ValueError(f"{context} has an invalid repeat declaration")
            rows_by_sample[sample_id].append(repeat)
            references, literals = reported.get("references"), reported.get("literal_strings")
            if not isinstance(references, list) or not references or not isinstance(literals, list):
                raise ValueError(f"{context} must report references and literal strings")
            classified: set[str] = set()
            row_inputs: list[str] = []
            for reference_index, reference in enumerate(references):
                reference_context = f"{context} reference {reference_index}"
                if not isinstance(reference, dict) or reference.get("kind") not in {"path", "caption-text"}:
                    raise ValueError(f"{reference_context} has an invalid shape")
                kind = reference["kind"]
                expected_shape = {"kind", "pointer", "input_id", "path"} if kind == "path" else {
                    "kind", "pointer", "input_id",
                }
                if set(reference) != expected_shape:
                    raise ValueError(f"{reference_context} has an invalid shape")
                pointer, input_id = reference.get("pointer"), reference.get("input_id")
                if not isinstance(pointer, str) or pointer in classified:
                    raise ValueError(f"{reference_context} has a missing or duplicate pointer")
                identity = input_index.get(input_id) if isinstance(input_id, str) else None
                if (
                    not isinstance(identity, dict)
                    or identity.get("dataset") != dataset["id"]
                    or identity.get("sample") != sample_id
                ):
                    raise ValueError(f"{reference_context} does not match its manifest sample input")
                actual = _pointer_value(parsed, pointer, context=reference_context + " pointer")
                if kind == "path":
                    target = _safe_workspace_relative(reference.get("path"), context=reference_context + " path")
                    if placements.get(input_id) != target:
                        raise ValueError(f"{reference_context} does not match its projected view entry")
                    if actual != "/workspace/" + target:
                        raise ValueError(f"{reference_context} value does not resolve to its projected view entry")
                    path_referenced[input_id] += 1
                elif identity.get("kind") != "caption" or identity.get("text") != actual:
                    raise ValueError(f"{reference_context} does not preserve its caption input")
                row_inputs.append(input_id)
                all_referenced[input_id] += 1
                classified.add(pointer)
            if Counter(row_inputs) != Counter(inputs_by_sample[sample_id]):
                raise ValueError(f"{context} does not include every sample input exactly once")
            for literal_index, literal in enumerate(literals):
                literal_context = f"{context} literal string {literal_index}"
                if not isinstance(literal, dict) or set(literal) != {"pointer", "value"}:
                    raise ValueError(f"{literal_context} has an invalid shape")
                pointer, expected = literal.get("pointer"), literal.get("value")
                if not isinstance(pointer, str) or pointer in classified or not isinstance(expected, str):
                    raise ValueError(f"{literal_context} has an invalid or duplicate pointer")
                actual = _pointer_value(parsed, pointer, context=literal_context + " pointer")
                if actual != expected:
                    raise ValueError(f"{literal_context} differs from the generated row")
                if pointer not in literal_fields:
                    raise ValueError(f"{literal_context} literal field is not declared by the adapter")
                if _looks_like_path(actual):
                    raise ValueError(f"{literal_context} classifies a path-like value as a literal")
                classified.add(pointer)
            missing_literals = sorted(set(literal_fields) - {item.get("pointer") for item in literals})
            if missing_literals:
                raise ValueError(f"{context} omits declared literal field {missing_literals[0]!r}")
            unclassified = sorted(_string_leaf_pointers(parsed) - classified)
            if unclassified:
                raise ValueError(f"{context} has an unclassified string field {unclassified[0]!r}")
            extra = sorted(classified - _string_leaf_pointers(parsed))
            if extra:
                raise ValueError(f"{context} reports a non-string field {extra[0]!r}")
        seen_paths.add(path)
        verified.append(deepcopy(native))
    missing_samples = sorted(sample_id for sample_id, repeats in rows_by_sample.items() if not repeats)
    if missing_samples:
        raise ValueError(f"projection native JSONL has no row for sample {missing_samples[0]!r}")
    for sample_id, repeats in rows_by_sample.items():
        if len(repeats) == 1:
            if repeats[0] not in (None, {"index": 0, "count": 1}):
                raise ValueError(f"sample {sample_id!r} repeat declaration is incomplete")
            continue
        if any(repeat is None for repeat in repeats):
            raise ValueError(f"sample {sample_id!r} repeat must be declared for every generated row")
        counts = {repeat["count"] for repeat in repeats}
        indices = {repeat["index"] for repeat in repeats}
        if counts != {len(repeats)} or indices != set(range(len(repeats))):
            raise ValueError(f"sample {sample_id!r} repeat declaration is incomplete")
    expected_path_counts = Counter({
        input_id: len(rows_by_sample[identity["sample"]])
        for input_id, identity in input_index.items()
        if input_id in placements and identity.get("dataset") == dataset["id"]
    })
    if path_referenced != expected_path_counts:
        raise ValueError(f"projection native JSONL does not reference every view input exactly once per row")
    return verified, set(all_referenced)


def _native_jsonl_semantic(view: dict[str, Any]) -> list[dict[str, Any]]:
    root = PurePosixPath(view["root"])
    semantic: list[dict[str, Any]] = []
    for native in view.get("native_files", []):
        lines = _jsonl_rows(native["text"], context=f"projection native JSONL {native['path']!r}")
        parsed_rows = [json.loads(line) for line in lines]
        stable_rows: list[dict[str, Any]] = []
        for parsed, reported in zip(parsed_rows, native["rows"], strict=True):
            canonical = deepcopy(parsed)
            for reference in reported["references"]:
                replacement = {"$kura_input_id": reference["input_id"], "$kura_kind": reference["kind"]}
                if reference["kind"] == "path":
                    replacement["$kura_view_path"] = (
                        PurePosixPath(reference["path"]).relative_to(root).as_posix()
                    )
                _pointer_replace(
                    canonical,
                    reference["pointer"],
                    replacement,
                    context=f"native JSONL row {reported['row_id']!r} reference",
                )
            stable_rows.append({
                "row_id": reported["row_id"],
                "sample_id": reported["sample_id"],
                "repeat": reported["repeat"],
                "value": canonical,
            })
        semantic.append({
            "path": PurePosixPath(native["path"]).relative_to(root).as_posix(),
            "format": native["format"],
            "rows": stable_rows,
        })
    return semantic


def _validate_view(run_id: str, dataset: dict[str, Any], projected: dict[str, Any],
                   input_index: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, str], set[str]]:
    view = projected.get("view")
    if not isinstance(view, dict):
        raise ValueError(f"backend projection for dataset {dataset['id']!r} has no run view")
    root = _safe_workspace_relative(view.get("root"), context="projection view root")
    expected_prefix = f"runs/{run_id}/cache/dataset-view/"
    if not root.startswith(expected_prefix):
        raise ValueError("projection view root must be under the run-owned cache/dataset-view directory")
    links: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    placements: dict[str, str] = {}
    for index, link in enumerate(view.get("links", [])):
        if not isinstance(link, dict):
            raise ValueError(f"projection view link {index} must be a mapping")
        path = _safe_workspace_relative(link.get("path"), context=f"projection view link {index}")
        if not path.startswith(root + "/") or path in seen_paths:
            raise ValueError(f"projection view link {path!r} is outside its root or duplicated")
        input_id, target = link.get("input_id"), link.get("target")
        identity = input_index.get(input_id)
        if (
            not isinstance(identity, dict)
            or not isinstance(target, str)
            or identity.get("container_source") != target
        ):
            raise ValueError(f"projection view link {path!r} has an invalid input or target")
        if input_id in placements:
            raise ValueError(f"projection places input {input_id!r} more than once")
        seen_paths.add(path)
        placements[input_id] = path
        links.append({"path": path, "target": target, "input_id": input_id})
    files: list[dict[str, str]] = []
    for index, generated in enumerate(view.get("files", [])):
        if not isinstance(generated, dict):
            raise ValueError(f"projection generated file {index} must be a mapping")
        path = _safe_workspace_relative(generated.get("path"), context=f"projection generated file {index}")
        text, input_id = generated.get("text"), generated.get("input_id")
        if not path.startswith(root + "/") or path in seen_paths or not isinstance(text, str):
            raise ValueError(f"projection generated file {path!r} is outside its root, duplicated, or non-text")
        identity = input_index.get(input_id)
        if not isinstance(identity, dict):
            raise ValueError(f"projection generated file {path!r} names an unknown input")
        if identity.get("kind") != "caption" or identity.get("text") != text:
            raise ValueError(f"projection generated file {path!r} does not preserve its caption input")
        if input_id in placements:
            raise ValueError(f"projection places input {input_id!r} more than once")
        seen_paths.add(path)
        placements[input_id] = path
        files.append({"path": path, "text": text, "input_id": input_id})
    _validate_bindings(dataset["id"], root, projected.get("bindings"), placements, input_index)
    native_files, native_inputs = _validate_native_jsonl_files(
        dataset, root, view.get("native_files"), placements, input_index, seen_paths,
    )
    return {
        "dataset": dataset["id"], "root": root, "links": links, "files": files,
        "native_files": native_files,
    }, placements, native_inputs or set(placements)


def freeze_dataset_handoff(
    run: dict[str, Any], workspace: Path, resolved: Path, *, backend: str, project: Project,
) -> dict[str, Any]:
    """Measure v2 selection once, require a total projection, and freeze both locks."""
    selection, input_index = _manifest_selection(run, workspace)
    report = project(deepcopy(selection))
    if not isinstance(report, dict) or report.get("schema_version") != 1 or report.get("backend") != backend:
        raise ValueError(f"{backend} returned an invalid dataset projection report")
    projected_datasets = report.get("datasets")
    if not isinstance(projected_datasets, list):
        raise ValueError(f"{backend} projection report must contain datasets")
    by_id: dict[str, dict[str, Any]] = {}
    for projected in projected_datasets:
        dataset_id = projected.get("id") if isinstance(projected, dict) else None
        if not isinstance(dataset_id, str) or dataset_id in by_id:
            raise ValueError(f"{backend} projection has a missing or duplicate dataset id")
        by_id[dataset_id] = projected
    selected_ids = [item["id"] for item in selection["datasets"]]
    if set(by_id) != set(selected_ids):
        raise ValueError(f"{backend} projection dataset selection differs from run datasets")
    consumed: list[str] = []
    views: list[dict[str, Any]] = []
    views_by_dataset: dict[str, dict[str, Any]] = {}
    for dataset in selection["datasets"]:
        projected = by_id[dataset["id"]]
        failures = projected.get("unrepresentable", [])
        if not isinstance(failures, list):
            raise ValueError(f"{backend} projection unrepresentable inputs must be a list")
        for failure in failures:
            if not isinstance(failure, dict) or not isinstance(failure.get("reason"), str):
                raise ValueError(f"{backend} projection has an invalid unrepresentable input")
            raise _projection_error(input_index, failure.get("input_id"), failure["reason"])
        dataset_consumed = projected.get("consumed")
        if not isinstance(dataset_consumed, list) or not all(isinstance(item, str) for item in dataset_consumed):
            raise ValueError(f"{backend} projection consumed inputs must be a list of IDs")
        consumed.extend(dataset_consumed)
        view, placements, represented = _validate_view(str(run.get("id")), dataset, projected, input_index)
        if Counter(dataset_consumed) != Counter(represented):
            raise ValueError(
                f"{backend} projection consumed inputs differ from verified native inputs for dataset {dataset['id']!r}"
            )
        views.append(view)
        views_by_dataset[dataset["id"]] = view
    duplicates = sorted(item for item, count in Counter(consumed).items() if count > 1)
    unknown = sorted(set(consumed) - set(input_index))
    missing = sorted(set(input_index) - set(consumed))
    if duplicates:
        raise ValueError(f"{backend} projection consumes input more than once: {duplicates[0]}")
    if unknown:
        raise ValueError(f"{backend} projection consumes unknown input: {unknown[0]}")
    if missing:
        identity = input_index[missing[0]]
        raise ValueError(
            f"{backend} projection has unconsumed dataset {identity['dataset']!r} "
            f"sample {identity['sample']!r} {identity['kind']}"
        )
    all_files = [item for dataset in selection["datasets"] for item in dataset["files"]]
    authoring_files = [item for dataset in selection["datasets"] for item in dataset["authoring_files"]]
    dataset_roots = [
        {"dataset": dataset["id"], "logical": dataset["logical_root"], "physical": dataset["physical_root"]}
        for dataset in selection["datasets"]
    ]
    stable_projection = []
    for dataset_id in selected_ids:
        projected = by_id[dataset_id]
        projection_semantic = projected.get("semantic")
        if not isinstance(projection_semantic, dict):
            raise ValueError(f"{backend} projection for dataset {dataset_id!r} has no stable semantic identity")
        native_runtime = projected.get("native_runtime")
        native = projected.get("native")
        if not isinstance(native_runtime, dict) or not isinstance(native, dict):
            raise ValueError(f"{backend} projection for dataset {dataset_id!r} has an invalid native handoff")
        overlap = set(projection_semantic) & set(native_runtime)
        if overlap or native != {**projection_semantic, **native_runtime}:
            raise ValueError(
                f"{backend} projection for dataset {dataset_id!r} native handoff must be derived "
                "exactly from semantic and runtime-only fields"
            )
        stable_item: dict[str, Any] = {"id": dataset_id, "semantic": projection_semantic}
        view = views_by_dataset[dataset_id]
        native_jsonl = _native_jsonl_semantic(view)
        if native_jsonl:
            stable_item["native_jsonl"] = native_jsonl
        stable_projection.append(stable_item)
    semantic = {
        "schema_version": 1,
        "datasets": [dataset["identity"] for dataset in selection["datasets"]],
        "projection": stable_projection,
    }
    lock = {
        "schema_version": 2,
        "backend": backend,
        "verification": "content-hash-at-compile",
        "files": all_files,
        "authoring_files": authoring_files,
        "dataset_roots": dataset_roots,
        "views": views,
        "input_sha256": _digest(semantic),
        "semantic": semantic,
    }
    resolved.mkdir(parents=True, exist_ok=True)
    atomic_write_json(resolved / "dataset-projection.lock.json", report)
    atomic_write_json(resolved / "dataset-input.lock.json", lock)
    return lock


def _current_stat(path: Path) -> dict[str, int]:
    observed = path.stat()
    return {"size": observed.st_size, "mtime_ns": observed.st_mtime_ns, "ctime_ns": observed.st_ctime_ns}


def _source_changes(workspace: Path, lock: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    roots: dict[str, Path] = {}
    for item in lock.get("dataset_roots", []):
        if not isinstance(item, dict):
            changes.append("invalid dataset root lock")
            continue
        dataset, logical, physical = item.get("dataset"), item.get("logical"), item.get("physical")
        if not all(isinstance(value, str) and value for value in (dataset, logical, physical)):
            changes.append("invalid dataset root lock")
            continue
        try:
            current = (workspace / logical).resolve(strict=True)
        except (OSError, RuntimeError):
            changes.append(f"dataset root missing: {dataset}")
            continue
        expected = Path(physical)
        if current != expected:
            changes.append(f"dataset root changed: {dataset}")
            continue
        roots[dataset] = expected
    for item in [*lock.get("files", []), *lock.get("authoring_files", [])]:
        source, expected = item.get("source"), item.get("stat")
        if not isinstance(source, str) or not isinstance(expected, dict):
            changes.append("invalid input lock file")
            continue
        path = workspace / source
        try:
            physical = path.resolve(strict=True)
            parts = PurePosixPath(source).parts
            if len(parts) < 3 or parts[0] != "datasets" or parts[1] not in roots:
                raise ValueError("source has no frozen dataset root")
            physical.relative_to(roots[parts[1]])
            actual = _current_stat(physical)
        except (OSError, RuntimeError, ValueError):
            changes.append(f"missing or unsafe source: {source}")
            continue
        if actual != {key: expected.get(key) for key in _PORTABLE_STAT}:
            changes.append(f"changed since compile: {source}")
    return changes


def inspect_dataset_sources(workspace: Path, lock: dict[str, Any]) -> list[str]:
    """Compare compile-time source roots and portable stat without touching the run view."""
    if lock.get("schema_version") != 2:
        raise ValueError("dataset handoff inspection requires input lock schema_version 2")
    return _source_changes(workspace, lock)


def _view_path(workspace: Path, relative: str, root: str) -> Path:
    logical = _safe_workspace_relative(relative, context="view path")
    if logical != root and not logical.startswith(root + "/"):
        raise ValueError(f"view path escapes its root: {relative}")
    return workspace / logical


def _generated_view_files(view: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item for key in ("files", "native_files") for item in view.get(key, [])
        if isinstance(item, dict)
    ]


def _view_requires_rebuild(workspace: Path, view: dict[str, Any]) -> bool:
    root_relative = view["root"]
    root = workspace / root_relative
    if not root.exists() and not root.is_symlink():
        return False
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"dataset view root is not a real directory: {root_relative}")
    expected_links = {
        item["path"]: item["target"] for item in view.get("links", [])
        if isinstance(item, dict) and isinstance(item.get("path"), str) and isinstance(item.get("target"), str)
    }
    actual_links = {
        path.relative_to(workspace).as_posix(): os.readlink(path)
        for path in root.rglob("*") if path.is_symlink()
    }
    if actual_links != expected_links:
        return True
    expected_generated = {
        item["path"]: str(item.get("text")).encode("utf-8")
        for item in _generated_view_files(view) if isinstance(item.get("path"), str)
    }
    for relative, content in expected_generated.items():
        path = workspace / relative
        if not path.is_file() or path.is_symlink() or path.read_bytes() != content:
            return True
    for path in root.rglob("*"):
        relative = path.relative_to(workspace).as_posix()
        if (
            path.is_file()
            and not path.is_symlink()
            and path.suffix.lower() in _TRAINER_MEDIA_SUFFIXES
            and relative not in expected_generated
        ):
            return True
    return False


def inspect_dataset_view(workspace: Path, lock: dict[str, Any]) -> list[str]:
    """Compare the materialized view with the exact compiled native handoff."""
    changes: list[str] = []
    for view in lock.get("views", []):
        if not isinstance(view, dict) or not isinstance(view.get("root"), str):
            changes.append("invalid view lock")
            continue
        root_relative = view["root"]
        root = workspace / root_relative
        expected_links = {
            item["path"]: item["target"] for item in view.get("links", [])
            if isinstance(item, dict) and isinstance(item.get("path"), str) and isinstance(item.get("target"), str)
        }
        actual_links: dict[str, str] = {}
        if root.is_dir():
            for path in root.rglob("*"):
                if path.is_symlink():
                    actual_links[path.relative_to(workspace).as_posix()] = os.readlink(path)
        for path, target in expected_links.items():
            if path not in actual_links:
                changes.append(f"missing view link: {path}")
            elif actual_links[path] != target:
                changes.append(f"retargeted view link: {path}")
        for path in sorted(set(actual_links) - set(expected_links)):
            changes.append(f"unexpected view link: {path}")
        expected_generated = {
            item["path"] for item in _generated_view_files(view)
            if isinstance(item, dict) and isinstance(item.get("path"), str)
        }
        if root.is_dir():
            for path in root.rglob("*"):
                relative = path.relative_to(workspace).as_posix()
                if (
                    path.is_file()
                    and not path.is_symlink()
                    and path.suffix.lower() in _TRAINER_MEDIA_SUFFIXES
                    and relative not in expected_generated
                ):
                    changes.append(f"unexpected regular media in view: {relative}")
        for generated in _generated_view_files(view):
            if not isinstance(generated, dict) or not isinstance(generated.get("path"), str):
                changes.append("invalid generated view file")
                continue
            path = _view_path(workspace, generated["path"], root_relative)
            if not path.is_file() or path.is_symlink():
                changes.append(f"missing generated view file: {generated['path']}")
            elif path.read_bytes() != str(generated.get("text")).encode("utf-8"):
                changes.append(f"changed generated view file: {generated['path']}")
    return changes


def inspect_dataset_handoff(workspace: Path, lock: dict[str, Any]) -> list[str]:
    """Compare portable source stat and the exact compiled native handoff."""
    return [*_source_changes(workspace, lock), *inspect_dataset_view(workspace, lock)]


def remove_dataset_views(workspace: Path, run_dir: Path, lock: dict[str, Any]) -> dict[str, Any]:
    """Remove only the disposable view root owned by this terminal run."""
    relative = f"runs/{run_dir.name}/cache/dataset-view"
    roots = [item.get("root") for item in lock.get("views", []) if isinstance(item, dict)]
    if not roots:
        return {"status": "not-required", "path": relative}
    if any(not isinstance(root, str) or not root.startswith(relative + "/") for root in roots):
        raise ValueError("dataset view cleanup is outside the run-owned dataset-view root")
    target = workspace / relative
    current = workspace
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"dataset view cleanup path contains a symlink: {current}")
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise ValueError("dataset view cleanup target is not a real directory")
    if not target.exists():
        return {"status": "already-absent", "path": relative}
    shutil.rmtree(target)
    return {"status": "removed", "path": relative}


def materialize_dataset_view(workspace: Path, lock: dict[str, Any]) -> Path:
    """Create the frozen links/generated files after source stat passes."""
    source_changes = _source_changes(workspace, lock)
    if source_changes:
        raise ValueError("compiled dataset input changed: " + "; ".join(source_changes))
    roots: list[Path] = []
    for view in lock.get("views", []):
        if not isinstance(view, dict) or not isinstance(view.get("root"), str):
            raise ValueError("invalid dataset view lock")
        root_relative = _safe_workspace_relative(view["root"], context="view root")
        root = workspace / root_relative
        if _view_requires_rebuild(workspace, view):
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        roots.append(root)
        for link in view.get("links", []):
            path = _view_path(workspace, link["path"], root_relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_symlink():
                if os.readlink(path) != link["target"]:
                    raise ValueError(f"retargeted view link: {link['path']}")
            elif path.exists():
                raise ValueError(f"view link destination already exists: {link['path']}")
            else:
                path.symlink_to(link["target"])
        for generated in _generated_view_files(view):
            path = _view_path(workspace, generated["path"], root_relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if path.is_symlink() or path.read_bytes() != generated["text"].encode("utf-8"):
                    raise ValueError(f"generated view file differs: {generated['path']}")
            else:
                path.write_bytes(generated["text"].encode("utf-8"))
    changes = inspect_dataset_handoff(workspace, lock)
    if changes:
        raise ValueError("dataset view verification failed: " + "; ".join(changes))
    if len(roots) != 1:
        raise ValueError("materialized dataset handoff must currently contain exactly one view")
    return roots[0]


def _container_path(value: Any, *, context: str) -> PurePosixPath:
    if not isinstance(value, str) or not value.startswith("/") or "\\" in value or "\x00" in value:
        raise ValueError(f"{context} must be an absolute POSIX path")
    normalized = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in value.split("/")[1:]):
        raise ValueError(f"{context} must be normalized")
    return normalized


def _overlaps(left: PurePosixPath, right: PurePosixPath) -> bool:
    return left == right or left in right.parents or right in left.parents


def local_training_mounts(
    workspace: Path,
    run_dir: Path,
    lock: dict[str, Any],
    *,
    configured: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Build the closed local-Docker mount table for a manifest-v2 handoff."""
    workspace = workspace.resolve()
    run_dir = run_dir.resolve()
    resolved = (run_dir / "resolved").resolve()
    cache = (workspace / "cache").resolve()
    mounts: list[dict[str, str]] = []
    protected: list[PurePosixPath] = [PurePosixPath("/workspace/datasets"), PurePosixPath(f"/workspace/runs/{run_dir.name}/resolved")]
    protected_sources: list[Path] = []
    for item in lock.get("dataset_roots", []):
        if not isinstance(item, dict):
            raise ValueError("invalid dataset root lock")
        dataset, physical = item.get("dataset"), item.get("physical")
        if not isinstance(dataset, str) or not dataset or not isinstance(physical, str) or not physical:
            raise ValueError("invalid dataset root lock")
        target = _container_path(f"/workspace/datasets/{dataset}", context="dataset mount target")
        protected.append(target)
        physical_root = Path(physical).resolve(strict=True)
        protected_sources.append(physical_root)
        mounts.append({"source": str(physical_root), "target": target.as_posix(), "mode": "ro"})

    for source, label in ((run_dir, "run directory"), (cache, "workspace cache")):
        if any(source == root or source in root.parents or root in source.parents for root in protected_sources):
            raise ValueError(f"managed writable {label} overlaps a protected dataset root: {source}")

    mounts.extend([
        {"source": str(run_dir), "target": f"/workspace/runs/{run_dir.name}", "mode": "rw"},
        {"source": str(resolved), "target": f"/workspace/runs/{run_dir.name}/resolved", "mode": "ro"},
        {"source": str(cache), "target": "/workspace/cache", "mode": "rw"},
    ])
    seen_targets = {item["target"] for item in mounts}
    for index, item in enumerate(configured):
        if not isinstance(item, dict):
            raise ValueError(f"configured mount {index} must be a mapping")
        source, target_value, mode = item.get("source"), item.get("target"), item.get("mode", "rw")
        if not isinstance(source, str) or not source or mode not in {"ro", "rw"}:
            raise ValueError(f"configured mount {index} has an invalid source or mode")
        if target_value == "/root/.cache/huggingface":
            target_value = "/workspace/cache/huggingface"
        target = _container_path(target_value, context=f"configured mount {index} target")
        if target == PurePosixPath("/workspace"):
            raise ValueError("configured mount must not reintroduce the broad /workspace mount")
        if any(_overlaps(target, path) for path in protected):
            raise ValueError(f"configured mount overlaps a protected dataset mount: {target}")
        source_path = Path(source).expanduser()
        if not source_path.is_absolute():
            source_path = workspace / source_path
        source_path = source_path.resolve(strict=False)
        if mode == "rw" and any(
            source_path == root or source_path in root.parents or root in source_path.parents
            for root in protected_sources
        ):
            raise ValueError(f"configured writable mount source overlaps a protected dataset root: {source}")
        if target.as_posix() in seen_targets:
            raise ValueError(f"configured mount duplicates managed target: {target}")
        seen_targets.add(target.as_posix())
        mounts.append({"source": source, "target": target.as_posix(), "mode": mode})
    return mounts

"""Freeze complete backend projections and materialize disposable run views."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Any, Callable

import yaml

from kura.dataset_manifest import measure_manifest
from kura.fsio import atomic_write_json
from kura.media_types import KNOWN_MEDIA_SUFFIXES
from kura.paths import to_workspace_relative


Project = Callable[[dict[str, Any]], dict[str, Any]]
_PORTABLE_STAT = ("size", "mtime_ns", "ctime_ns")
_EXTENSION_LITERAL = re.compile(r"^\.[A-Za-z0-9]+$")
RUNPOD_MANIFEST_V2_UNSUPPORTED = (
    "RunPod selected-file transfer for manifest-v2 datasets is not implemented; "
    "use executor.name=docker until selected-file transfer and Pod-side hash "
    "verification are available"
)


def require_dataset_transfer_supported(*, executor: str, input_schema_version: Any) -> None:
    """Reject unsupported transfers before legacy whole-dataset staging can run."""

    if executor == "runpod" and input_schema_version == 2:
        raise ValueError(RUNPOD_MANIFEST_V2_UNSUPPORTED)


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
                    "manifest_position": file_index,
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


def project_caption_text(text: str, transform: str) -> str:
    """Apply a small, backend-declared caption text projection."""
    if transform == "identity":
        return text
    if transform == "strip":
        return text.strip()
    # Match Python text-file universal-newline handling without treating Unicode
    # line separators as physical caption-file lines.
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if transform == "first-line-strip":
        return lines[0].strip()
    if transform == "nonempty-lines-strip":
        return "\n".join(line.strip() for line in lines if line.strip())
    raise ValueError(f"projection declares unsupported caption transform {transform!r}")


def _validate_bindings(
    dataset_id: str,
    view_root: str,
    bindings: Any,
    placements: dict[str, str],
    input_index: dict[str, dict[str, Any]],
) -> set[str]:
    if not isinstance(bindings, list):
        raise ValueError(f"backend projection for dataset {dataset_id!r} has no input bindings")
    bound: list[str] = []
    seen_keys: set[str] = set()
    bound_samples: set[str] = set()
    roots_by_kind: dict[tuple[str, int | None], str] = {}
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
        binding_slots: dict[str, list[int | None]] = {}
        binding_positions: dict[str, list[tuple[int | None, int | None]]] = {}
        member_slots: list[int | None] = []
        for member_index, member in enumerate(members):
            if (
                not isinstance(member, dict)
                or set(member) not in ({"input_id", "root"}, {"input_id", "root", "slot"})
            ):
                raise ValueError(f"projection binding {index} member {member_index} has an invalid shape")
            input_id = member.get("input_id")
            member_root = _safe_workspace_relative(
                member.get("root"), context=f"projection binding {index} member {member_index} root",
            )
            if not isinstance(input_id, str):
                raise ValueError(f"projection binding {index} member {member_index} has an invalid input")
            slot = member.get("slot")
            if slot is not None and (
                isinstance(slot, bool) or not isinstance(slot, int) or slot < 0
            ):
                raise ValueError(
                    f"projection binding {index} member {member_index} slot must be a non-negative integer"
                )
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
            member_slots.append(slot)
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
        for identity, slot in zip(identities, member_slots, strict=True):
            binding_kind = identity.get("binding_kind") if isinstance(identity, dict) else None
            if not isinstance(binding_kind, str):
                raise ValueError(f"projection binding {index} input has no binding kind")
            binding_slots.setdefault(binding_kind, []).append(slot)
            manifest_position = identity.get("manifest_position") if isinstance(identity, dict) else None
            binding_positions.setdefault(binding_kind, []).append((slot, manifest_position))
        for binding_kind, slots in binding_slots.items():
            if all(slot is None for slot in slots):
                if len(slots) > 1:
                    raise ValueError(
                        f"projection binding {index} input kind {binding_kind!r} has multiple "
                        "members but does not declare an ordered slot for every member"
                    )
                continue
            if any(slot is None for slot in slots):
                raise ValueError(
                    f"projection binding {index} input kind {binding_kind!r} must either omit "
                    "slots or declare an ordered slot for every member"
                )
            concrete = [slot for slot in slots if slot is not None]
            if sorted(concrete) != list(range(len(concrete))):
                raise ValueError(
                    f"projection binding {index} input kind {binding_kind!r} slots must be "
                    "unique and contiguous from zero"
                )
            positions = binding_positions[binding_kind]
            manifest_positions = [position for _slot, position in positions]
            if all(isinstance(position, int) for position in manifest_positions):
                by_slot = [
                    position
                    for _slot, position in sorted(
                        positions,
                        key=lambda item: item[0] if item[0] is not None else -1,
                    )
                ]
                if by_slot != sorted(manifest_positions):
                    raise ValueError(
                        f"projection binding {index} input kind {binding_kind!r} slot order "
                        "does not match manifest file order"
                    )
        for identity, member_root, slot in zip(
            identities, member_roots, member_slots, strict=True,
        ):
            binding_kind = identity.get("binding_kind") if isinstance(identity, dict) else None
            if not isinstance(binding_kind, str):
                raise ValueError(f"projection binding {index} input has no binding kind")
            root_key = (binding_kind, slot)
            previous_root = roots_by_kind.setdefault(root_key, member_root)
            if previous_root != member_root:
                rendered_kind = (
                    binding_kind if slot is None else f"{binding_kind} slot {slot}"
                )
                raise ValueError(
                    f"projection input kind {rendered_kind!r} uses multiple role roots: "
                    f"{previous_root!r} and {member_root!r}"
                )
        bound.extend(input_ids)
    if Counter(bound) != Counter(placements.keys()):
        raise ValueError(f"projection bindings do not cover the exact dataset {dataset_id!r} view inputs")
    return set(roots_by_kind.values())


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
    if "/" in value or "\\" in value or value.startswith("~"):
        return True
    if value.startswith(".") and _EXTENSION_LITERAL.fullmatch(value) is None:
        return True
    return PurePosixPath(value).suffix.lower() in KNOWN_MEDIA_SUFFIXES


def _validate_native_string_classification(
    dataset_id: str,
    projected: dict[str, Any],
    native: dict[str, Any],
    views: list[dict[str, Any]],
) -> list[str]:
    declared = projected.get("native_string_fields")
    if not isinstance(declared, list) or not all(isinstance(item, str) for item in declared):
        raise ValueError(
            f"backend projection for dataset {dataset_id!r} must declare native string fields"
        )
    if len(set(declared)) != len(declared):
        raise ValueError(f"backend projection for dataset {dataset_id!r} has duplicate native string fields")

    structural: set[str] = set()
    for view in views:
        structural.update(consumer["native_pointer"] for consumer in view["consumers"])
        structural.update(write_root["native_pointer"] for write_root in view["write_roots"])
        if "repeat_pointer" in view:
            structural.add(view["repeat_pointer"])

    overlap = structural & set(declared)
    if overlap:
        raise ValueError(
            f"backend projection for dataset {dataset_id!r} classifies native pointer "
            f"{sorted(overlap)[0]!r} as both structural and other string"
        )
    for pointer in declared:
        value = _pointer_value(
            native, pointer, context=f"projection dataset {dataset_id!r} declared native string",
        )
        if not isinstance(value, str):
            raise ValueError(
                f"backend projection for dataset {dataset_id!r} native string field {pointer!r} is not a string"
            )
        if _looks_like_path(value):
            raise ValueError(
                f"backend projection for dataset {dataset_id!r} native string field {pointer!r} "
                "classifies a path-like value as another string"
            )

    string_pointers = _string_leaf_pointers(native)
    classified_strings = (structural & string_pointers) | set(declared)
    unclassified = sorted(string_pointers - classified_strings)
    if unclassified:
        pointer = unclassified[0]
        value = _pointer_value(native, pointer, context="unclassified native string")
        kind = "path-like string" if _looks_like_path(value) else "string"
        raise ValueError(
            f"backend projection for dataset {dataset_id!r} native {kind} {pointer!r} is not classified"
        )
    return list(declared)


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
    if native_files is None or native_files == []:
        return [], set()
    if not isinstance(native_files, list):
        raise ValueError(f"backend projection for dataset {dataset['id']!r} native files must be a list")
    sample_ids = {item["id"] for item in dataset["samples"]}
    view_sample_ids = {
        input_index[input_id]["sample"]
        for input_id in placements
        if input_id in input_index and input_index[input_id].get("dataset") == dataset["id"]
    }
    inputs_by_sample = {
        sample["id"]: [
            *[reference["input_id"] for reference in sample["files"]],
            *([sample["caption"]["input_id"]] if sample["caption"] is not None else []),
        ]
        for sample in dataset["samples"]
    }
    row_ids: set[str] = set()
    rows_by_sample: dict[str, list[Any]] = {sample_id: [] for sample_id in view_sample_ids}
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
        used_literal_fields: set[str] = set()
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
            if sample_id not in view_sample_ids:
                raise ValueError(f"{context} names a sample outside its native view")
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
                if not isinstance(reference, dict) or reference.get("kind") not in {
                    "path", "caption-text", "caption-text-strip",
                }:
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
                else:
                    expected_caption = identity.get("text") if identity.get("kind") == "caption" else None
                    if kind == "caption-text-strip" and isinstance(expected_caption, str):
                        expected_caption = expected_caption.strip()
                    if expected_caption != actual:
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
                used_literal_fields.add(pointer)
            unclassified = sorted(_string_leaf_pointers(parsed) - classified)
            if unclassified:
                raise ValueError(f"{context} has an unclassified string field {unclassified[0]!r}")
            extra = sorted(classified - _string_leaf_pointers(parsed))
            if extra:
                raise ValueError(f"{context} reports a non-string field {extra[0]!r}")
        unused_literal_fields = sorted(set(literal_fields) - used_literal_fields)
        if unused_literal_fields:
            raise ValueError(
                f"projection native JSONL {path!r} declares unused literal field "
                f"{unused_literal_fields[0]!r}"
            )
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


def _validate_view(
    run_id: str,
    dataset: dict[str, Any],
    view: Any,
    native: dict[str, Any],
    input_index: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, str], set[str]]:
    if not isinstance(view, dict):
        raise ValueError(f"backend projection for dataset {dataset['id']!r} has an invalid run view")
    allowed_view_keys = {
        "id", "root", "links", "files", "native_files", "write_roots", "consumers", "repeat",
        "repeat_pointer", "bindings",
    }
    required_view_keys = allowed_view_keys - {"bindings", "repeat_pointer"}
    if set(view) - allowed_view_keys or not required_view_keys <= set(view):
        raise ValueError(f"backend projection for dataset {dataset['id']!r} view has an invalid shape")
    view_id = view.get("id")
    if (
        not isinstance(view_id, str)
        or not view_id
        or _safe_workspace_relative(view_id, context="projection view id") != view_id
        or "/" in view_id
    ):
        raise ValueError(f"backend projection for dataset {dataset['id']!r} has an invalid view id")
    root = _safe_workspace_relative(view.get("root"), context="projection view root")
    expected_prefix = f"runs/{run_id}/cache/dataset-view/"
    if not root.startswith(expected_prefix):
        raise ValueError("projection view root must be under the run-owned cache/dataset-view directory")
    repeat = view.get("repeat")
    if not isinstance(repeat, int) or isinstance(repeat, bool) or repeat < 1:
        raise ValueError(f"projection view {view_id!r} repeat must be a positive integer")
    repeat_pointer = view.get("repeat_pointer")
    if repeat > 1 and not isinstance(repeat_pointer, str):
        raise ValueError(f"projection view {view_id!r} repeat has no native pointer")
    if repeat_pointer is not None:
        if not isinstance(repeat_pointer, str):
            raise ValueError(f"projection view {view_id!r} repeat pointer is invalid")
        actual_repeat = _pointer_value(
            native, repeat_pointer, context=f"projection view {view_id!r} native repeat",
        )
        if actual_repeat != repeat:
            raise ValueError(f"projection view {view_id!r} native repeat differs from its declared repeat")
    write_roots = view.get("write_roots")
    if not isinstance(write_roots, list) or not write_roots:
        raise ValueError(f"projection view {view_id!r} must declare write roots")
    verified_write_roots: list[dict[str, str]] = []
    seen_write_root_paths: set[str] = set()
    for index, value in enumerate(write_roots):
        if not isinstance(value, dict) or set(value) != {"path", "native_pointer"}:
            raise ValueError(f"projection view {view_id!r} write root {index} has an invalid shape")
        write_root = _safe_workspace_relative(
            value.get("path"), context=f"projection view {view_id!r} write root {index}",
        )
        pointer = value.get("native_pointer")
        if not isinstance(pointer, str):
            raise ValueError(f"projection view {view_id!r} write root {index} has no native pointer")
        if write_root != root and not write_root.startswith(root + "/"):
            raise ValueError(f"projection view {view_id!r} write root is outside its view")
        actual = _pointer_value(
            native, pointer, context=f"projection view {view_id!r} native write root {index}",
        )
        expected = "/workspace/" + write_root
        if not isinstance(actual, str) or (actual != expected and not actual.startswith(expected + "/")):
            raise ValueError(f"projection view {view_id!r} native write root does not match its declared path")
        verified = {"path": write_root, "native_pointer": pointer}
        if write_root in seen_write_root_paths:
            raise ValueError(f"projection view {view_id!r} has a duplicate write root")
        seen_write_root_paths.add(write_root)
        verified_write_roots.append(verified)
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
        links.append({
            "path": path,
            "target": target,
            "input_id": input_id,
            "dataset": identity["dataset"],
            "sample": identity["sample"],
        })
    files: list[dict[str, str]] = []
    for index, generated in enumerate(view.get("files", [])):
        if not isinstance(generated, dict):
            raise ValueError(f"projection generated file {index} must be a mapping")
        path = _safe_workspace_relative(generated.get("path"), context=f"projection generated file {index}")
        text, input_id = generated.get("text"), generated.get("input_id")
        caption_transform = generated.get("caption_transform", "identity")
        if not path.startswith(root + "/") or path in seen_paths or not isinstance(text, str):
            raise ValueError(f"projection generated file {path!r} is outside its root, duplicated, or non-text")
        identity = input_index.get(input_id)
        if not isinstance(identity, dict):
            raise ValueError(f"projection generated file {path!r} names an unknown input")
        source_text = identity.get("text") if identity.get("kind") == "caption" else None
        if (
            not isinstance(caption_transform, str)
            or not isinstance(source_text, str)
            or project_caption_text(source_text, caption_transform) != text
        ):
            raise ValueError(f"projection generated file {path!r} does not preserve its caption input")
        if input_id in placements:
            raise ValueError(f"projection places input {input_id!r} more than once")
        seen_paths.add(path)
        placements[input_id] = path
        files.append({
            "path": path,
            "text": text,
            "input_id": input_id,
            **({"caption_transform": caption_transform} if caption_transform != "identity" else {}),
        })
    native_files, native_inputs = _validate_native_jsonl_files(
        dataset, root, view.get("native_files"), placements, input_index, seen_paths,
    )
    consumers = view.get("consumers")
    if not isinstance(consumers, list) or not consumers:
        raise ValueError(f"projection view {view_id!r} has no native consumers")
    kinds = {item.get("kind") for item in consumers if isinstance(item, dict)}
    if len(kinds) != 1:
        raise ValueError(f"projection view {view_id!r} mixes native consumer kinds")
    kind = next(iter(kinds)) if kinds else None
    verified_bindings: list[dict[str, Any]] | None = None
    if kind == "recursive-directory":
        if native_files:
            raise ValueError(f"projection view {view_id!r} recursive consumer cannot use native JSONL rows")
        bindings = view.get("bindings")
        role_roots = _validate_bindings(dataset["id"], root, bindings, placements, input_index)
        ordered_roots = sorted(PurePosixPath(item) for item in role_roots)
        for index, left in enumerate(ordered_roots):
            for right in ordered_roots[index + 1:]:
                if left in right.parents or right in left.parents:
                    raise ValueError(
                        f"projection view {view_id!r} has nested role roots for a recursive consumer"
                    )
        consumed_by_directories: list[str] = []
        consumer_paths: list[PurePosixPath] = []
        seen_consumer_ids: set[str] = set()
        for index, consumer in enumerate(consumers):
            if not isinstance(consumer, dict) or set(consumer) != {
                "id", "kind", "native_pointer", "path", "input_ids",
            }:
                raise ValueError(f"projection view {view_id!r} recursive consumer {index} has an invalid shape")
            consumer_id = consumer.get("id")
            if not isinstance(consumer_id, str) or not consumer_id or consumer_id in seen_consumer_ids:
                raise ValueError(f"projection view {view_id!r} has a missing or duplicate consumer id")
            seen_consumer_ids.add(consumer_id)
            pointer = consumer.get("native_pointer")
            consumer_path = _safe_workspace_relative(
                consumer.get("path"), context=f"projection view {view_id!r} consumer {consumer_id!r} path",
            )
            input_ids = consumer.get("input_ids")
            if not isinstance(pointer, str) or not isinstance(input_ids, list) or not input_ids:
                raise ValueError(f"projection view {view_id!r} consumer {consumer_id!r} is incomplete")
            if consumer_path != root and not consumer_path.startswith(root + "/"):
                raise ValueError(f"projection view {view_id!r} consumer {consumer_id!r} is outside its view")
            actual = _pointer_value(
                native, pointer, context=f"projection view {view_id!r} consumer {consumer_id!r}",
            )
            if actual != "/workspace/" + consumer_path:
                raise ValueError(f"projection view {view_id!r} native consumer does not point at its view path")
            path_object = PurePosixPath(consumer_path)
            if any(path_object == other or path_object in other.parents or other in path_object.parents
                   for other in consumer_paths):
                raise ValueError(f"projection view {view_id!r} has overlapping recursive consumers")
            consumer_paths.append(path_object)
            for input_id in input_ids:
                placement = placements.get(input_id) if isinstance(input_id, str) else None
                if placement is None or not placement.startswith(consumer_path + "/"):
                    raise ValueError(
                        f"projection view {view_id!r} consumer {consumer_id!r} names an input outside its path"
                    )
                consumed_by_directories.append(input_id)
        if Counter(consumed_by_directories) != Counter(placements.keys()):
            raise ValueError(f"projection view {view_id!r} consumers do not cover every view input exactly once")
        verified_bindings = deepcopy(bindings)
        represented = set(placements)
    elif kind == "jsonl":
        if "bindings" in view:
            raise ValueError(f"projection view {view_id!r} JSONL consumer must not declare directory bindings")
        consumed_native_files: list[str] = []
        seen_consumer_ids: set[str] = set()
        for index, consumer in enumerate(consumers):
            if not isinstance(consumer, dict) or set(consumer) != {
                "id", "kind", "native_pointer", "native_file",
            }:
                raise ValueError(f"projection view {view_id!r} JSONL consumer {index} has an invalid shape")
            consumer_id = consumer.get("id")
            pointer = consumer.get("native_pointer")
            if (
                not isinstance(consumer_id, str)
                or not consumer_id
                or consumer_id in seen_consumer_ids
                or not isinstance(pointer, str)
            ):
                raise ValueError(f"projection view {view_id!r} has an invalid JSONL consumer identity")
            seen_consumer_ids.add(consumer_id)
            native_file = _safe_workspace_relative(
                consumer.get("native_file"), context=f"projection view {view_id!r} native JSONL file",
            )
            if native_file not in {item["path"] for item in native_files}:
                raise ValueError(f"projection view {view_id!r} JSONL consumer names an unverified native file")
            actual = _pointer_value(
                native, pointer, context=f"projection view {view_id!r} consumer {consumer_id!r}",
            )
            if actual != "/workspace/" + native_file:
                raise ValueError(f"projection view {view_id!r} native consumer does not point at its JSONL file")
            consumed_native_files.append(native_file)
        if Counter(consumed_native_files) != Counter(item["path"] for item in native_files):
            raise ValueError(f"projection view {view_id!r} consumers do not cover every native JSONL file")
        represented = native_inputs
    else:
        raise ValueError(f"projection view {view_id!r} has an unsupported native consumer")
    verified_view: dict[str, Any] = {
        "id": view_id,
        "dataset": dataset["id"],
        "root": root,
        "links": links,
        "files": files,
        "native_files": native_files,
        "write_roots": verified_write_roots,
        "consumers": deepcopy(consumers),
        "repeat": repeat,
    }
    if repeat_pointer is not None:
        verified_view["repeat_pointer"] = repeat_pointer
    if verified_bindings is not None:
        verified_view["bindings"] = verified_bindings
    return {
        **verified_view,
    }, placements, represented


def _view_semantic(view: dict[str, Any]) -> dict[str, Any]:
    root = PurePosixPath(view["root"])
    consumers = deepcopy(view["consumers"])
    for consumer in consumers:
        if consumer.get("kind") == "jsonl":
            consumer["native_file"] = PurePosixPath(consumer["native_file"]).relative_to(root).as_posix()
        elif consumer.get("kind") == "recursive-directory":
            consumer["path"] = PurePosixPath(consumer["path"]).relative_to(root).as_posix()
    stable: dict[str, Any] = {
        "id": view["id"],
        "repeat": view["repeat"],
        "consumers": consumers,
        "write_roots": [
            {
                "path": PurePosixPath(item["path"]).relative_to(root).as_posix(),
                "native_pointer": item["native_pointer"],
            }
            for item in view["write_roots"]
        ],
        "entries": [
            {
                "input_id": item["input_id"],
                "path": PurePosixPath(item["path"]).relative_to(root).as_posix(),
                "kind": "link" if "target" in item else "generated",
                **(
                    {"caption_transform": item["caption_transform"]}
                    if "caption_transform" in item else {}
                ),
            }
            for item in [*view.get("links", []), *view.get("files", [])]
        ],
    }
    if "repeat_pointer" in view:
        stable["repeat_pointer"] = view["repeat_pointer"]
    if "bindings" in view:
        stable["bindings"] = [
            {
                "rule": binding["rule"],
                "key": binding["key"],
                "members": [
                    {
                        "input_id": member["input_id"],
                        "root": PurePosixPath(member["root"]).relative_to(root).as_posix(),
                    }
                    for member in binding["members"]
                ],
            }
            for binding in view["bindings"]
        ]
    native_jsonl = _native_jsonl_semantic(view)
    if native_jsonl:
        stable["native_jsonl"] = native_jsonl
    return stable


def _merge_projection_native(semantic: Any, runtime: Any) -> Any:
    """Merge disjoint semantic/runtime leaves through matching containers."""
    if isinstance(semantic, dict) and isinstance(runtime, dict):
        merged: dict[str, Any] = {}
        for key in semantic.keys() | runtime.keys():
            if key in semantic and key in runtime:
                merged[key] = _merge_projection_native(semantic[key], runtime[key])
            elif key in semantic:
                merged[key] = deepcopy(semantic[key])
            else:
                merged[key] = deepcopy(runtime[key])
        return merged
    if isinstance(semantic, list) and isinstance(runtime, list):
        if len(semantic) != len(runtime):
            raise ValueError("semantic and runtime native lists differ in length")
        return [
            _merge_projection_native(semantic_item, runtime_item)
            for semantic_item, runtime_item in zip(semantic, runtime)
        ]
    raise ValueError("semantic and runtime native leaves overlap")


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
    views_by_dataset: dict[str, list[dict[str, Any]]] = {}
    seen_view_ids: set[tuple[str, str]] = set()
    seen_view_roots: set[str] = set()
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
        projection_semantic = projected.get("semantic")
        if not isinstance(projection_semantic, dict):
            raise ValueError(f"{backend} projection for dataset {dataset['id']!r} has no stable semantic identity")
        projection_policy = projected.get("policy")
        if projection_policy is not None and not isinstance(projection_policy, dict):
            raise ValueError(f"{backend} projection for dataset {dataset['id']!r} has an invalid projection policy")
        native_runtime = projected.get("native_runtime")
        native = projected.get("native")
        if not isinstance(native_runtime, dict) or not isinstance(native, dict):
            raise ValueError(f"{backend} projection for dataset {dataset['id']!r} has an invalid native handoff")
        try:
            derived_native = _merge_projection_native(
                projection_semantic, native_runtime,
            )
        except ValueError as error:
            raise ValueError(
                f"{backend} projection for dataset {dataset['id']!r} native handoff must be derived "
                "exactly from semantic and runtime-only fields"
            ) from error
        if native != derived_native:
            raise ValueError(
                f"{backend} projection for dataset {dataset['id']!r} native handoff must be derived "
                "exactly from semantic and runtime-only fields"
            )
        projected_views = projected.get("views")
        if not isinstance(projected_views, list) or not projected_views:
            raise ValueError(f"backend projection for dataset {dataset['id']!r} has no run views")
        consumed.extend(dataset_consumed)
        dataset_views: list[dict[str, Any]] = []
        dataset_represented: list[str] = []
        for projected_view in projected_views:
            view, _placements, represented = _validate_view(
                str(run.get("id")), dataset, projected_view, native, input_index,
            )
            view_identity = (dataset["id"], view["id"])
            if view_identity in seen_view_ids:
                raise ValueError(f"{backend} projection has a duplicate native view id {view['id']!r}")
            if view["root"] in seen_view_roots:
                raise ValueError(f"{backend} projection has a duplicate native view root {view['root']!r}")
            view_root_path = PurePosixPath(view["root"])
            if any(
                view_root_path in PurePosixPath(existing).parents
                or PurePosixPath(existing) in view_root_path.parents
                for existing in seen_view_roots
            ):
                raise ValueError(f"{backend} projection has overlapping native view roots")
            seen_view_ids.add(view_identity)
            seen_view_roots.add(view["root"])
            dataset_represented.extend(represented)
            dataset_views.append(view)
            views.append(view)
        if Counter(dataset_consumed) != Counter(dataset_represented):
            raise ValueError(
                f"{backend} projection consumed inputs differ from verified native inputs for dataset {dataset['id']!r}"
            )
        native_string_fields = _validate_native_string_classification(
            dataset["id"], projected, native, dataset_views,
        )
        projected["native_string_fields"] = native_string_fields
        views_by_dataset[dataset["id"]] = dataset_views
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
        stable_item: dict[str, Any] = {
            "id": dataset_id,
            "semantic": projection_semantic,
            "native_string_fields": projected["native_string_fields"],
            "views": [_view_semantic(view) for view in views_by_dataset[dataset_id]],
        }
        if projected.get("policy") is not None:
            stable_item["policy"] = deepcopy(projected["policy"])
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
            and path.suffix.lower() in KNOWN_MEDIA_SUFFIXES
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
                    and path.suffix.lower() in KNOWN_MEDIA_SUFFIXES
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
    """Create every frozen view after source stat passes; return the first root for compatibility."""
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
    if not roots:
        raise ValueError("materialized dataset handoff has no views")
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


def _resume_state_mount(workspace: Path, run_dir: Path) -> dict[str, str] | None:
    lock_path = run_dir / "resolved" / "training-state-source.lock.json"
    if not lock_path.is_file():
        return None
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read the frozen training-state source lock") from exc
    artifact_id = lock.get("artifact_id") if isinstance(lock, dict) else None
    expected = (
        f"/workspace/artifacts/training-state/{artifact_id}/payload"
        if isinstance(artifact_id, str) and artifact_id
        else None
    )
    if expected is None or lock.get("native_state_path") != expected:
        raise ValueError("training-state source lock has an invalid native state path")
    source = workspace / "artifacts" / "training-state" / artifact_id
    payload = source / "payload"
    if not payload.is_dir():
        raise ValueError(f"required training-state payload does not exist: {payload}")
    return {
        "source": str(source.resolve(strict=True)),
        "target": f"/workspace/artifacts/training-state/{artifact_id}",
        "mode": "ro",
    }


def _local_model_requirements(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "resolved" / "model-requirements.lock.yaml"
    if not path.is_file():
        return []
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError("cannot read the frozen model requirements") from exc
    requirements = document.get("requirements") if isinstance(document, dict) else None
    if not isinstance(requirements, list):
        raise ValueError("frozen model requirements must contain requirements[]")
    return [
        item for item in requirements
        if isinstance(item, dict) and item.get("acquisition") == "local-path"
    ]


def _configured_model_source(
    workspace: Path,
    runtime: PurePosixPath,
    configured: list[dict[str, str]],
) -> Path | None:
    matches: list[tuple[int, Path]] = []
    for index, item in enumerate(configured):
        if not isinstance(item, dict):
            continue
        target_value, source_value = item.get("target"), item.get("source")
        if not isinstance(target_value, str) or not isinstance(source_value, str):
            continue
        target = _container_path(target_value, context=f"configured mount {index} target")
        if runtime != target and target not in runtime.parents:
            continue
        source = Path(source_value).expanduser()
        if not source.is_absolute():
            source = workspace / source
        suffix = runtime.relative_to(target)
        matches.append((len(target.parts), source.joinpath(*suffix.parts)))
    if not matches:
        return None
    return max(matches, key=lambda item: item[0])[1]


def _local_model_mounts(
    workspace: Path,
    run_dir: Path,
    configured: list[dict[str, str]],
) -> list[dict[str, str]]:
    mounts: list[dict[str, str]] = []
    seen_targets: set[str] = set()
    workspace_root = PurePosixPath("/workspace")
    for requirement in _local_model_requirements(run_dir):
        role = requirement.get("role")
        reference = requirement.get("runtime_reference")
        if not isinstance(reference, str) or not reference:
            raise ValueError(f"local-path model {role!r} has no runtime reference")
        runtime = _container_path(reference, context=f"local-path model {role!r}")
        relative = to_workspace_relative(runtime.as_posix(), workspace=workspace)
        source = workspace / relative if relative is not None else _configured_model_source(
            workspace, runtime, configured,
        )
        if source is None:
            raise ValueError(
                f"local-path model {role!r} at {reference} is not covered by the local Docker mount table"
            )
        try:
            physical = source.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ValueError(f"local-path model {role!r} does not exist: {source}") from exc
        target = runtime.as_posix()
        if target in seen_targets:
            continue
        seen_targets.add(target)
        mounts.append({"source": str(physical), "target": target, "mode": "ro"})
    return mounts


def _reject_writable_model_aliases(
    workspace: Path,
    configured: list[dict[str, str]],
    model_mounts: list[dict[str, str]],
) -> None:
    for index, item in enumerate(configured):
        if not isinstance(item, dict) or item.get("mode", "rw") != "rw":
            continue
        source_value, target_value = item.get("source"), item.get("target")
        if not isinstance(source_value, str) or not isinstance(target_value, str):
            continue
        source = Path(source_value).expanduser()
        if not source.is_absolute():
            source = workspace / source
        source = source.resolve(strict=False)
        target = _container_path(target_value, context=f"configured mount {index} target")
        for model in model_mounts:
            model_source = Path(model["source"]).resolve(strict=False)
            model_target = PurePosixPath(model["target"])
            source_overlap = (
                source == model_source
                or source in model_source.parents
                or model_source in source.parents
            )
            protected_by_overlay = model_target == target or target in model_target.parents
            if source_overlap and not protected_by_overlay:
                raise ValueError(
                    "configured writable mount re-exposes a local-path model through "
                    f"another target: {target}"
                )


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
    resume_mount = _resume_state_mount(workspace, run_dir)
    if resume_mount is not None:
        if resume_mount["target"] in seen_targets:
            raise ValueError("configured mount duplicates managed training-state target")
        seen_targets.add(resume_mount["target"])
        mounts.append(resume_mount)
    model_mounts = _local_model_mounts(workspace, run_dir, configured)
    _reject_writable_model_aliases(workspace, configured, model_mounts)
    for mount in model_mounts:
        target = mount["target"]
        if target in seen_targets:
            existing = next(item for item in mounts if item["target"] == target)
            existing_source = Path(existing["source"]).expanduser()
            if not existing_source.is_absolute():
                existing_source = workspace / existing_source
            if (
                existing.get("mode") != "ro"
                or existing_source.resolve(strict=False) != Path(mount["source"])
            ):
                raise ValueError(f"local-path model target conflicts with another mount: {target}")
            continue
        seen_targets.add(target)
        mounts.append(mount)
    return mounts

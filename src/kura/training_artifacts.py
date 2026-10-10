"""Durable training-state artifacts and Resume lineage helpers."""

from __future__ import annotations

import hashlib
import json
import os
import pickletools
import re
import shutil
import tempfile
import zipfile
from functools import wraps
from pathlib import Path
from typing import Any, Iterable, NamedTuple

import yaml

from kura.fsio import atomic_write_json, file_lock
from kura.install_source import kura_continuity
from kura.run_envelope import final_step, resume_intent, training_state_policy, validated_recipe


ARTIFACT_SCHEMA_VERSION = 1


class _ReferenceInspectionError(ValueError):
    """Signal that retention must skip deletion without invalidating publication."""


def training_state_reference_lock(workspace: Path):
    """Serialize artifact selection/reference creation with publication retention."""

    return file_lock(_artifact_root(workspace) / ".store.lock")


def _locked_artifact_store(function):
    @wraps(function)
    def locked(workspace: Path, *args: Any, **kwargs: Any):
        with file_lock(_artifact_root(workspace) / ".store.lock"):
            return function(workspace, *args, **kwargs)

    return locked


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_DTYPE_BYTES = {
    "BOOL": 1, "I8": 1, "U8": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
    "F8_E4M3": 1, "F8_E5M2": 1,
}


def validate_safetensors_file(path: Path) -> None:
    """The one structural check for a weights file Kura keeps, wherever it came from.

    It reads the header only: unique keys, string metadata, and tensors that
    fill the data exactly once. A tensor's byte length is checked when Kura
    knows its dtype; a newer dtype is accepted on the other checks.
    """

    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"not a complete safetensors file: {path.name}")
        header_size = int.from_bytes(prefix, "little", signed=False)
        if header_size <= 0 or header_size > size - 8:
            raise ValueError(f"invalid safetensors header size: {path.name}")

        def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate safetensors header keys: {path.name}")
                result[key] = value
            return result

        try:
            header = json.loads(handle.read(header_size), object_pairs_hook=unique_keys)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid safetensors header: {path.name}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"non-object safetensors header: {path.name}")
    data_size = size - 8 - header_size
    intervals: list[tuple[int, int]] = []
    for key, value in header.items():
        if key == "__metadata__":
            if not isinstance(value, dict) or not all(isinstance(name, str) and isinstance(item, str) for name, item in value.items()):
                raise ValueError(f"invalid safetensors metadata: {path.name}")
            continue
        offsets = value.get("data_offsets") if isinstance(value, dict) else None
        shape = value.get("shape") if isinstance(value, dict) else None
        dtype = value.get("dtype") if isinstance(value, dict) else None
        if (
            not isinstance(dtype, str) or not dtype
            or not isinstance(shape, list)
            or not all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
        ):
            raise ValueError(f"invalid safetensors tensor entry: {path.name}")
        start, end = offsets
        if not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int) or isinstance(end, bool) or start < 0 or end < start or end > data_size:
            raise ValueError(f"safetensors tensor data is outside the file: {path.name}")
        if dtype in _DTYPE_BYTES:
            elements = 1
            for dimension in shape:
                elements *= dimension
            if end - start != elements * _DTYPE_BYTES[dtype]:
                raise ValueError(f"safetensors tensor byte size is invalid: {path.name}")
        intervals.append((start, end))
    if not intervals:
        raise ValueError(f"safetensors file contains no tensors: {path.name}")
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError(f"safetensors tensor data is not contiguous: {path.name}")
        cursor = end
    if cursor != data_size:
        raise ValueError(f"safetensors tensor data is not contiguous: {path.name}")


def _validate_torch_archive(path: Path) -> None:
    """Check modern torch.save ZIP structure and CRCs without loading pickle."""

    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if not names or archive.testzip() is not None:
                raise ValueError(f"training-state has a corrupt torch archive: {path.name}")
            data_pickle = [name for name in names if name.endswith("/data.pkl")]
            versions = [name for name in names if name.endswith("/version")]
            serialization_ids = [name for name in names if name.endswith("/.data/serialization_id")]
            if len(data_pickle) != 1 or len(versions) != 1 or len(serialization_ids) != 1:
                raise ValueError(f"training-state torch archive is missing canonical records: {path.name}")
            try:
                operations = list(pickletools.genops(archive.read(data_pickle[0])))
            except (ValueError, EOFError) as exc:
                raise ValueError(f"training-state torch archive has invalid pickle structure: {path.name}") from exc
            if not operations or operations[-1][0].name != "STOP" or not any(op.name in {"EMPTY_DICT", "DICT"} for op, _, _ in operations):
                raise ValueError(f"training-state torch archive does not contain a state mapping: {path.name}")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"training-state has an invalid torch archive: {path.name}") from exc


def _artifact_root(workspace: Path) -> Path:
    return workspace / "artifacts" / "training-state"


def _artifact_dir(workspace: Path, artifact_id: str) -> Path:
    if not artifact_id or Path(artifact_id).name != artifact_id:
        raise ValueError("training-state artifact ID must be a safe directory name")
    root = _artifact_root(workspace).resolve(strict=False)
    candidate = (root / artifact_id).resolve(strict=False)
    if candidate.parent != root:
        raise ValueError("training-state artifact must stay under artifacts/training-state")
    return candidate


def _load_manifest_path(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        manifest = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid training-state manifest: {path}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"unsupported training-state manifest: {path}")
    manifest["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    return manifest


def load_training_state(workspace: Path, artifact_id: str) -> dict[str, Any]:
    return _load_manifest_path(_artifact_dir(workspace, artifact_id) / "manifest.json")


def verify_training_state(workspace: Path, manifest: dict[str, Any]) -> Path:
    artifact_id = manifest.get("id")
    if not isinstance(artifact_id, str):
        raise ValueError("training-state manifest has no artifact ID")
    stored = load_training_state(workspace, artifact_id)
    expected_manifest = manifest.get("manifest_sha256")
    if isinstance(expected_manifest, str) and stored["manifest_sha256"] != expected_manifest:
        raise ValueError(f"training-state manifest digest mismatch: {artifact_id}")
    payload_value = stored.get("payload")
    expected_payload = training_state_payload(artifact_id, root=None)
    if payload_value != expected_payload:
        raise ValueError(f"training-state payload path is invalid: {artifact_id}")
    payload = workspace / expected_payload
    if not payload.is_dir():
        raise ValueError(f"training-state payload is missing: {artifact_id}")
    files = stored.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"training-state manifest has no files: {artifact_id}")
    expected_paths: set[str] = set()
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError(f"training-state manifest has an invalid file entry: {artifact_id}")
        relative = Path(item["path"])
        if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError(f"training-state manifest file escapes its payload: {item['path']}")
        path = payload / relative
        if any(parent.is_symlink() for parent in [path, *path.parents[:len(relative.parts)]]) or not path.is_file():
            raise ValueError(f"training-state file is missing: {item['path']}")
        size = path.stat().st_size
        if size != item.get("size"):
            raise ValueError(f"training-state size mismatch: {item['path']}")
        if _sha256_file(path) != item.get("sha256"):
            raise ValueError(f"training-state digest mismatch: {item['path']}")
        expected_paths.add(relative.as_posix())
    actual_paths: set[str] = set()
    for path in payload.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"training-state payload contains a symlink: {path.relative_to(payload)}")
        if path.is_file():
            actual_paths.add(path.relative_to(payload).as_posix())
    if actual_paths != expected_paths:
        raise ValueError(f"training-state payload inventory mismatch: {artifact_id}")
    return payload


def _protected_artifact_ids(workspace: Path) -> set[str]:
    protected: set[str] = set()
    for path in (workspace / "runs").glob("*/run.yaml"):
        try:
            import yaml

            run = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise _ReferenceInspectionError(f"cannot safely inspect training-state reference: {path}") from exc
        continuation = run.get("continuation") if isinstance(run, dict) else None
        source = continuation.get("source") if isinstance(continuation, dict) else None
        artifact_id = source.get("artifact_id") if isinstance(source, dict) else None
        if isinstance(artifact_id, str):
            protected.add(artifact_id)
    for path in (workspace / "runs").glob("*/resolved/training-state-source.lock.json"):
        try:
            lock = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise _ReferenceInspectionError(f"cannot safely inspect training-state reference: {path}") from exc
        artifact_id = lock.get("artifact_id") if isinstance(lock, dict) else None
        if isinstance(artifact_id, str):
            protected.add(artifact_id)
    return protected


def _manifest_observed_step(manifest: dict[str, Any]) -> int:
    value = manifest.get("observed_step")
    return value if isinstance(value, int) and not isinstance(value, bool) else -1


def _retained_steps(candidates: list[dict[str, Any]], keep_generations: int) -> set[int]:
    steps = {
        step
        for step in dict.fromkeys(_manifest_observed_step(item) for item in candidates)
        if step >= 0
    }
    return set(sorted(steps, reverse=True)[:keep_generations])


def training_state_retention_floor(workspace: Path, source_run: str, keep_generations: int) -> int | None:
    """Return the oldest logical step in the ordinary retention window."""

    if isinstance(keep_generations, bool) or not isinstance(keep_generations, int) or keep_generations <= 0:
        raise ValueError("training-state keep_generations must be a positive integer")
    candidates: list[dict[str, Any]] = []
    for path in _artifact_root(workspace).glob("*/manifest.json"):
        try:
            manifest = _load_manifest_path(path)
        except ValueError:
            continue
        if manifest.get("source_run") == source_run:
            candidates.append(manifest)
    retained = _retained_steps(candidates, keep_generations)
    return min(retained) if retained else None


def _apply_retention(workspace: Path, source_run: str, keep_generations: int) -> None:
    if isinstance(keep_generations, bool) or not isinstance(keep_generations, int) or keep_generations <= 0:
        raise ValueError("training-state keep_generations must be a positive integer")
    candidates: list[dict[str, Any]] = []
    for path in _artifact_root(workspace).glob("*/manifest.json"):
        try:
            manifest = _load_manifest_path(path)
        except ValueError:
            continue
        if manifest.get("source_run") == source_run:
            candidates.append(manifest)
    candidates.sort(key=lambda item: (_manifest_observed_step(item), str(item.get("id"))), reverse=True)
    retained_steps = _retained_steps(candidates, keep_generations)
    protected = _protected_artifact_ids(workspace)
    for manifest in candidates:
        if _manifest_observed_step(manifest) in retained_steps:
            continue
        artifact_id = manifest.get("id")
        if not isinstance(artifact_id, str) or artifact_id in protected:
            continue
        target = _artifact_dir(workspace, artifact_id)
        if target.is_dir():
            shutil.rmtree(target)


@_locked_artifact_store
def publish_training_state(
    workspace: Path,
    *,
    source_run: str,
    source_realization: str | None,
    backend: str,
    observed_step: int,
    candidate: Path,
    native_format: str,
    restoration_contract: dict[str, Any],
    runtime_identity: dict[str, Any] | None = None,
    compatibility: dict[str, Any] | None = None,
    save_event_id: str | None = None,
    keep_generations: int = 2,
) -> dict[str, Any]:
    """Copy one complete candidate into the protected store and publish last."""

    if not source_run or Path(source_run).name != source_run:
        raise ValueError("source_run must be a safe run ID")
    if isinstance(observed_step, bool) or not isinstance(observed_step, int) or observed_step < 0:
        raise ValueError("training-state observed_step must be a non-negative integer")
    if not candidate.is_dir():
        raise ValueError("training-state candidate must be a directory")
    model_file = candidate / "model.safetensors"
    if model_file.is_file():
        try:
            validate_safetensors_file(model_file)
        except ValueError as exc:
            raise ValueError(f"training-state has {exc}") from exc
    for path in candidate.rglob("*"):
        if path.is_file() and (path.name in {"optimizer.bin", "scheduler.bin", "optimizer.pt", "rng.pt"} or re.fullmatch(r"random_states_\d+\.pkl", path.name)):
            _validate_torch_archive(path)
    inventory: list[dict[str, Any]] = []
    for path in sorted(candidate.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"training-state candidate must not contain symlinks: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(candidate).as_posix()
        inventory.append({"path": relative, "size": path.stat().st_size, "sha256": _sha256_file(path)})
    if not inventory:
        raise ValueError("training-state candidate contains no files")
    identity = {
        "backend": backend,
        "source_run": source_run,
        "observed_step": observed_step,
        "files": inventory,
        "runtime_identity": runtime_identity or {},
        "compatibility": compatibility or {},
    }
    content_digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    artifact_id = f"state-step-{observed_step:08d}-{content_digest[:12]}"
    destination = _artifact_dir(workspace, artifact_id)
    if destination.exists():
        manifest = load_training_state(workspace, artifact_id)
        verify_training_state(workspace, manifest)
        return manifest
    root = _artifact_root(workspace)
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{artifact_id}.", dir=root))
    try:
        payload = staging / "payload"
        shutil.copytree(candidate, payload)
        copied_paths: set[str] = set()
        for item in inventory:
            copied = payload / item["path"]
            if copied.stat().st_size != item["size"] or _sha256_file(copied) != item["sha256"]:
                raise ValueError(f"training-state candidate changed during publication: {item['path']}")
            copied_paths.add(item["path"])
        actual_paths = {
            path.relative_to(payload).as_posix()
            for path in payload.rglob("*")
            if path.is_file()
        }
        if actual_paths != copied_paths:
            raise ValueError("training-state candidate inventory changed during publication")
        manifest = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "id": artifact_id,
            "kind": "training-state",
            "backend": backend,
            "format": native_format,
            "source_run": source_run,
            "source_realization": source_realization,
            "save_event_id": save_event_id or f"{source_run}:step:{observed_step}",
            "observed_step": observed_step,
            "payload": training_state_payload(artifact_id, root=None),
            "files": inventory,
            "runtime_identity": runtime_identity or {},
            "compatibility": compatibility or {},
            "restoration_contract": restoration_contract,
        }
        atomic_write_json(staging / "manifest.json", manifest)
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    published = load_training_state(workspace, artifact_id)
    verify_training_state(workspace, published)
    try:
        _apply_retention(workspace, source_run, keep_generations)
    except _ReferenceInspectionError:
        # Reference discovery failed closed: keep every generation, but do not
        # relabel an already-published and verified artifact as a sync failure.
        pass
    return published


def select_training_state(workspace: Path, source_run: str, artifact_id: str | None = None) -> dict[str, Any]:
    if artifact_id is not None:
        candidate = load_training_state(workspace, artifact_id)
        if candidate.get("source_run") != source_run:
            raise ValueError(f"training-state artifact {artifact_id} does not belong to source run {source_run}")
        verify_training_state(workspace, candidate)
        return candidate
    candidates: list[dict[str, Any]] = []
    for path in _artifact_root(workspace).glob("*/manifest.json"):
        try:
            candidate = _load_manifest_path(path)
            if candidate.get("source_run") != source_run:
                continue
            verify_training_state(workspace, candidate)
        except ValueError:
            continue
        candidates.append(candidate)
    if not candidates:
        raise ValueError(f"source run {source_run} has no recoverable training-state artifact")
    return max(candidates, key=lambda item: (_manifest_observed_step(item), str(item.get("id"))))


def training_state_at_step(
    workspace: Path,
    source_run: str,
    observed_step: int,
    *,
    verify_payload: bool = True,
) -> dict[str, Any] | None:
    for path in _artifact_root(workspace).glob("*/manifest.json"):
        try:
            candidate = _load_manifest_path(path)
            if candidate.get("source_run") != source_run or candidate.get("observed_step") != observed_step:
                continue
            if verify_payload:
                verify_training_state(workspace, candidate)
            return candidate
        except ValueError:
            continue
    return None


def recipe_fingerprint(run: dict[str, Any]) -> str:
    payload = {key: run.get(key) for key in ("backend", "model", "datasets", "recipe")}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def training_state_location(artifact_id: str, root: str | None = "/workspace") -> str:
    """Where a training-state artifact lives below a workspace root; `root=None` gives the relative path."""
    relative = f"artifacts/training-state/{artifact_id}"
    return relative if root is None else f"{root.rstrip('/')}/{relative}"


def training_state_payload(artifact_id: str, root: str | None = "/workspace") -> str:
    """The artifact's native state directory, which a trainer resumes from."""
    return f"{training_state_location(artifact_id, root)}/payload"


def verified_resume_source(workspace: Path, run: dict[str, Any], *, verify: bool = True) -> dict[str, Any] | None:
    """The training-state manifest a Resume run names, checked against its frozen digest.

    The one place every reader (compile, plan, staging, transfer) loads the source; a
    malformed continuation fails with `resume_intent`'s message. `verify=False` skips
    re-hashing the payload, for display-only sizing.
    """
    continuation = resume_intent(run)
    if continuation is None:
        return None
    source = continuation["source"]
    manifest = load_training_state(workspace, source["artifact_id"])
    if manifest["manifest_sha256"] != source["manifest_sha256"]:
        raise ValueError(f"training-state manifest digest mismatch: {source['artifact_id']}")
    if verify:
        verify_training_state(workspace, manifest)
    return manifest


def resume_steps(
    run: dict[str, Any],
    *,
    lock: dict[str, Any] | None = None,
    contract: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """The one Resume step arithmetic: logical source and target, and the trainer's native range.

    `native_progress` says whether the trainer counts its steps from zero in this process
    (`process_local`) or continues the logical count; `native_target` says the same for the
    target the trainer is given, which runs from `native_start` to `native_end`. Read from a
    frozen source lock when one is given, else from the run's continuation and its backend's
    training-state contract (a backend building its own command passes its contract).
    """
    if lock is not None:
        source_step, target_step = lock.get("source_step"), lock.get("target_step")
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in (source_step, target_step)):
            raise ValueError("the frozen training-state source lock has no integer source_step and target_step; recompile the run")
        progress = lock.get("native_progress", "logical")
        space = lock.get("native_target", "logical")
    else:
        continuation = resume_intent(run)
        if continuation is None:
            return None
        contract = training_state_contract(run) if contract is None else contract
        source_step, target_step = continuation["source"]["observed_step"], continuation["target_step"]
        progress = contract.get("native_progress", "logical")
        space = contract.get("native_target", "logical")
    process_local = space == "process_local"
    return {
        "source_step": source_step,
        "target_step": target_step,
        "additional_steps": target_step - source_step,
        "native_progress": progress,
        "native_target": space,
        "native_start": 0 if process_local else source_step,
        "native_end": target_step - source_step if process_local else target_step,
    }


def trained_steps(run: dict[str, Any]) -> int | None:
    """The optimizer steps a training run trains: the steps a Resume adds, else the recipe's
    steps. Every checkpoint and disk estimate counts from it. An invalid continuation raises;
    a run without a positive step count gives None."""
    steps = resume_steps(run, contract={})
    return steps["additional_steps"] if steps is not None else final_step(run)


def logical_step(native_step: int, steps: dict[str, Any] | None) -> int:
    """A trainer-reported step as a logical step: process-local progress starts at the source step.

    `steps` is a frozen source lock or `resume_steps`; None (not a Resume) leaves the step as is.
    """
    if steps is not None and steps.get("native_progress") == "process_local":
        return steps["source_step"] + native_step
    return native_step


def frozen_resume_steps(run_dir: Path, run: dict[str, Any]) -> dict[str, Any] | None:
    """A run's Resume steps, from its frozen source lock when compile wrote a readable one."""
    lock = read_resume_lock(run_dir)
    return resume_steps(run, lock=lock) if lock is not None else resume_steps(run)


def read_resume_lock(run_dir: Path) -> dict[str, Any] | None:
    """The run's frozen training-state source lock, or None when it has none.

    A lock that exists but cannot be read raises: reading it as no lock would treat a
    Resume as a fresh run and record its native steps as logical ones.
    """
    path = run_dir / "resolved" / "training-state-source.lock.json"
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read the frozen training-state source lock; recompile the run") from exc
    if not isinstance(loaded, dict):
        raise ValueError("the frozen training-state source lock is not a mapping; recompile the run")
    return loaded


def resume_artifact_directory(workspace: Path, run: dict[str, Any]) -> Path | None:
    if verified_resume_source(workspace, run) is None:
        return None
    return _artifact_dir(workspace, run["continuation"]["source"]["artifact_id"])


def compile_resume_lock(
    workspace: Path,
    run: dict[str, Any],
    resolved: Path,
    *,
    target_runtime_identity: dict[str, Any] | None = None,
    target_input_lock: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    continuation = resume_intent(run)
    if continuation is None:
        return None
    source = continuation["source"]
    current_fingerprint = recipe_fingerprint(run)
    if current_fingerprint != source["recipe_sha256"]:
        raise ValueError("Resume training recipe changed after the derived run was created; create a Fork from Weight instead")
    manifest = verified_resume_source(workspace, run)
    if manifest.get("source_run") != run.get("parent_run"):
        raise ValueError("Resume artifact source run does not match parent_run")
    backend = run.get("backend") if isinstance(run.get("backend"), dict) else {}
    if manifest.get("backend") != backend.get("name"):
        raise ValueError("Resume artifact backend does not match the derived run")
    if manifest.get("observed_step") != source["observed_step"]:
        raise ValueError("Resume artifact observed step does not match authored intent")
    compatibility = manifest.get("compatibility") if isinstance(manifest.get("compatibility"), dict) else {}
    expected_recipe = compatibility.get("recipe_sha256")
    if isinstance(expected_recipe, str) and expected_recipe != current_fingerprint:
        raise ValueError("Resume artifact recipe changed from the published compatibility fingerprint")
    dataset_input: dict[str, Any] | None = None
    if target_input_lock is not None:
        source_path = workspace / "runs" / str(run.get("parent_run")) / "resolved" / "dataset-input.lock.json"
        if source_path.is_file():
            source_input_lock = json.loads(source_path.read_text(encoding="utf-8"))
            source_identity = source_input_lock.get("input_sha256")
            target_identity = target_input_lock.get("input_sha256")
            if isinstance(source_identity, str) and isinstance(target_identity, str):
                if source_identity != target_identity:
                    raise ValueError("Resume dataset input changed from the source run")
                dataset_input = {"status": "content-matched", "input_sha256": target_identity}
            elif source_input_lock.get("verification") == "unverified-native-source" and source_identity is None:
                # The old recipe fingerprint still guards the declared dataset
                # digest; do not misrepresent it as media-content verification.
                dataset_input = {"status": "source-unverified", "detail": "media identity is unverified in the source input lock"}
            else:
                raise ValueError("Resume dataset input identity is unverified in a lock-bearing run")
        else:
            dataset_input = {"status": "legacy-unverified", "detail": "media identity is unverified (old dataset digest only)"}
    if manifest.get("restoration_contract") != continuation.get("restoration_contract"):
        raise ValueError("Resume restoration contract does not match the selected artifact")
    source_runtime = manifest.get("runtime_identity") if isinstance(manifest.get("runtime_identity"), dict) else {}
    if target_runtime_identity is not None and source_runtime:
        source_adapter = source_runtime.get("adapter_source")
        target_adapter = target_runtime_identity.get("adapter_source")
        if source_adapter is not None and source_adapter != target_adapter:
            raise ValueError("Resume backend adapter identity differs from the source training state")
        source_executor = source_runtime.get("actual_executor")
        target_executor = target_runtime_identity.get("declared_executor")
        source_image = source_runtime.get("actual_image_identity")
        target_image = target_runtime_identity.get("selected_image_identity")
        source_pinning = source_image.get("pinning") if isinstance(source_image, dict) else None
        target_pinning = target_image.get("pinning") if isinstance(target_image, dict) else None
        if source_executor == target_executor:
            if source_image is not None and source_image != target_image:
                raise ValueError("Resume runtime image identity differs from the source training state")
        elif source_executor is not None:
            if not isinstance(target_pinning, dict) or target_pinning.get("strength") != "content-hash":
                raise ValueError("cross-executor Resume target runtime image must have an observed or declared content hash")
            if not isinstance(source_pinning, dict) or source_pinning.get("strength") != "content-hash":
                raise ValueError("cross-executor Resume source runtime image has no observed or declared content hash")
            source_contract = source_runtime.get("runtime_contract_sha256")
            target_contract = target_runtime_identity.get("runtime_contract_sha256")
            if not isinstance(source_contract, str) or source_contract != target_contract:
                raise ValueError("Resume cross-executor runtime pair is not verified compatible")
    artifact_id = manifest["id"]
    steps = resume_steps(run)
    lock = {
        "schema_version": 1,
        "mode": "resume",
        "source_run": run["parent_run"],
        "artifact_id": artifact_id,
        "manifest_sha256": manifest["manifest_sha256"],
        "source_realization": manifest.get("source_realization"),
        "source_step": steps["source_step"],
        "target_step": steps["target_step"],
        "additional_steps": steps["additional_steps"],
        "native_progress": steps["native_progress"],
        "native_target": steps["native_target"],
        "native_state_path": training_state_payload(artifact_id),
        "restoration_contract": manifest["restoration_contract"],
        "runtime_identity": manifest.get("runtime_identity") or {},
        "compatibility": compatibility,
        "dataset_input": dataset_input,
        "kura": _kura_continuity(workspace / "runs" / str(run["parent_run"]) / "resolved" / "env.lock"),
        "files": manifest["files"],
    }
    resolved.mkdir(parents=True, exist_ok=True)
    atomic_write_json(resolved / "training-state-source.lock.json", lock)
    return lock


def _kura_continuity(source_env_lock: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(source_env_lock.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        loaded = None
    return kura_continuity(loaded if isinstance(loaded, dict) else {})


def training_state_contract(run: dict[str, Any]) -> dict[str, Any]:
    """Return the active adapter's recovery contract without owning backend policy here."""

    from kura.backends import get_backend

    backend = run.get("backend") if isinstance(run.get("backend"), dict) else {}
    adapter = get_backend(backend.get("name"))
    if adapter.training_state is None:
        raise ValueError(f"backend {adapter.name!r} has no training-state contract")
    return adapter.training_state(run)


# The step a backend writes into a checkpoint name: `-step00000100` (sd-scripts, Musubi
# Tuner) or `_` plus exactly nine digits (AI-Toolkit). Final weights carry none; a shorter
# number after `_` is part of the name, such as a run ID's four hex digits.
_CHECKPOINT_STEP = re.compile(r"(?:(?:^|[-_])step0*(\d+)|_(\d{9}))(?=\.safetensors$|[-_.]|$)", re.IGNORECASE)


def checkpoint_step(name: str) -> int | None:
    """The training step in a checkpoint or state directory name, for every backend; None for final weights."""
    matches = _CHECKPOINT_STEP.findall(name)
    if not matches:
        return None
    step_form, underscore_form = matches[-1]
    return int(step_form or underscore_form)


class CheckpointFiles(NamedTuple):
    """The weight files a run saved: step-named checkpoints and the unstepped final weights."""

    stepped: list[Path]
    final: list[Path]

    @property
    def saved(self) -> int:
        """How many checkpoints the run saved: its step-named ones, or its final weights when none has a step."""
        return len(self.stepped) or len(self.final)


def checkpoint_files(paths: Iterable[Path], run_id: str) -> CheckpointFiles:
    """Sort a run's outputs, relative to its outputs directory, into checkpoints, for every reader that counts them.

    A checkpoint is a .safetensors file outside a native training-state
    directory; its name says whether it is a step's checkpoint or the final weights.
    An older AI-Toolkit layout wrote below `<run id>/`; when weights also sit at the
    top, those are copies and are not counted again. The state check reads a path
    below `<run id>/` from there, so a run ID ending in `-state` is not a state directory.
    """
    legacy_root = Path(run_id)
    weights = [path for path in paths if path.suffix.lower() == ".safetensors"]
    if any(len(path.parts) == 1 for path in weights):
        weights = [path for path in weights if not path.is_relative_to(legacy_root)]
    weights = [
        path
        for path in weights
        if not is_training_state_output(path.relative_to(legacy_root) if path.is_relative_to(legacy_root) else path)
    ]
    return CheckpointFiles(
        stepped=[path for path in weights if checkpoint_step(path.name) is not None],
        final=[path for path in weights if checkpoint_step(path.name) is None],
    )


# What every executor records when a finished run that must leave training state left none.
MISSING_STATE_PUBLICATION_ERROR = "required training-state artifact is not published"
MISSING_STATE_SYNC_ERROR = (
    "the run's terminal snapshot has no valid training-state artifact; "
    "inspect the backend state output before relying on Resume"
)


def missing_training_state_error(capture_required: bool, *, trainer_completed: bool) -> str | None:
    """What a run that left no valid training state records, on every executor.

    Only a trainer that completed must have saved one; a run that failed or was
    stopped may have ended before its first save, and has nothing to resume from.
    """
    return MISSING_STATE_SYNC_ERROR if capture_required and trainer_completed else None


# Native save flags that count epochs; managed state is retained by steps, so they are refused.
EPOCH_SAVE_FLAGS = frozenset({"--save_every_n_epochs", "--save_last_n_epochs", "--save_last_n_epochs_state", "--save_n_epoch_ratio"})


def managed_state_cadence(run: dict[str, Any], configured_cadence: int | None, *, contract: dict[str, Any] | None = None) -> int | None:
    """The state save cadence Kura sets: the configured one (None when unset, so each backend
    keeps its own default) on a fresh run; on a process-local Resume, the configured cadence or
    the recipe's steps capped at the steps the run adds, so it saves at least once."""
    steps = resume_steps(run, contract=contract)
    if steps is None or steps["native_progress"] != "process_local":
        return configured_cadence
    cadence = configured_cadence if configured_cadence is not None else validated_recipe(run, required=True)["steps"]
    return min(cadence, steps["additional_steps"])


def managed_state_save_args(
    run: dict[str, Any],
    configured_cadence: int | None,
    extra_args: list[str],
    *,
    contract: dict[str, Any] | None = None,
) -> list[str]:
    """The save flags of an accelerate trainer (Musubi Tuner, sd-scripts) whose state Kura manages.

    The trainer saves state at `managed_state_cadence` (named only when Kura sets one) and at
    the end, and keeps the states of the last cadence, or of the recipe's steps when none is
    set (two generations), or only the newest (one generation). `configured_cadence` is the
    backend's validated `save_every_n_steps`; a backend building its own command passes its
    training-state contract.
    """
    epoch_flags = sorted({arg.split("=", 1)[0] for arg in extra_args} & EPOCH_SAVE_FLAGS)
    if epoch_flags:
        backend = run.get("backend") if isinstance(run.get("backend"), dict) else {}
        raise ValueError(
            f"{backend.get('name')} epoch save flags are incompatible with managed training-state retention; "
            "use backend.config.save_every_n_steps instead: " + ", ".join(epoch_flags)
        )
    cadence = managed_state_cadence(run, configured_cadence, contract=contract)
    if training_state_policy(run)["keep_generations"] == 2:
        window = cadence if cadence is not None else validated_recipe(run, required=True)["steps"]
    else:
        window = 1
    every = [] if cadence is None else ["--save_every_n_steps", str(cadence)]
    return [*every, "--save_state", "--save_state_on_train_end", "--save_last_n_steps_state", str(window)]


def run_output_name(run: dict[str, Any]) -> str:
    """The name a run's trainer gives its outputs: the run ID on a Resume, which owns a new
    output namespace, else the configured `output_name`, else the run ID."""
    backend = run.get("backend") if isinstance(run.get("backend"), dict) else {}
    config = backend.get("config") if isinstance(backend.get("config"), dict) else {}
    if resume_intent(run) is not None:
        return str(run["id"])
    return str(config.get("output_name") or run["id"])


# What `state_directory_step` gives the final state directory, whose name carries no step:
# only a verified step marker inside it can place it (`publish_training_state_candidate`).
FINAL_STATE_STEP = -1


def state_directory_step(name: str, output_name: str, *, allow_final: bool) -> int | None:
    """The native step a run's state directory name gives, for every executor.

    `{output_name}-stepNNNN-state` (four or more digits) gives NNNN; `{output_name}-state`
    gives FINAL_STATE_STEP when the final state may be published; anything else, including
    another output name's directories, gives None.
    """
    prefix = f"{output_name}-"
    if not name.startswith(prefix) or not name.endswith("-state"):
        return None
    middle = name[len(prefix):-len("-state")]
    if middle.startswith("step") and len(middle) >= 8 and middle[4:].isdigit() and middle[4:].isascii():
        return int(middle[4:])
    if allow_final and name == f"{output_name}-state":
        return FINAL_STATE_STEP
    return None


def training_state_managed(run: dict[str, Any], contract: dict[str, Any] | None = None, *, frozen: bool = False) -> bool:
    """The one rule for whether Kura manages a run's training state.

    The run asks for it (`recovery.training_state.enabled`), Kura builds its
    command (a custom `backend.config.command` has no state contract), and the
    backend can restore state for this architecture and mode. Backends save
    state only then, and every executor requires a completed run to leave state
    only then. A backend building its own command passes its contract; others
    look it up. `frozen` marks a compiled manifest: compile always freezes
    `recovery`, so one without it was compiled before Kura managed state.
    """
    if frozen and not isinstance(run.get("recovery"), dict):
        return False
    backend = run.get("backend") if isinstance(run.get("backend"), dict) else {}
    config = backend.get("config") if isinstance(backend.get("config"), dict) else {}
    if config.get("command") is not None:
        return False
    if not training_state_policy(run)["enabled"]:
        return False
    contract = training_state_contract(run) if contract is None else contract
    return contract.get("capability") != "unsupported"


def training_state_capture_required(run_dir: Path) -> bool:
    """Return whether a completed run is expected to publish recoverable state."""

    run, _, _ = _published_run_context(run_dir)
    return training_state_managed(run, frozen=True)


def _published_run_context(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest_path = run_dir / "resolved" / "manifest.lock.yaml"
    try:
        run = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot inspect training-state candidates for {run_dir.name}") from exc
    if not isinstance(run, dict):
        raise ValueError(f"cannot inspect training-state candidates for {run_dir.name}")
    try:
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        status = {}
    runtime_identity: dict[str, Any] = {}
    env_lock = run_dir / "resolved" / "env.lock"
    if env_lock.is_file():
        try:
            loaded = yaml.safe_load(env_lock.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            loaded = None
        if isinstance(loaded, dict):
            runtime_identity = loaded
    realization_ref = status.get("last_realization")
    if isinstance(realization_ref, str):
        realization_path = run_dir / realization_ref
        try:
            realization = json.loads(realization_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            realization = None
        if isinstance(realization, dict):
            runtime_identity["actual_executor"] = realization.get("executor")
            if isinstance(realization.get("image_identity"), dict):
                runtime_identity["actual_image_identity"] = realization["image_identity"]
            if isinstance(realization.get("adapter_source"), dict):
                runtime_identity["adapter_source"] = realization["adapter_source"]
    return run, status, runtime_identity


def state_logical_step(
    run_dir: Path, run: dict[str, Any], contract: dict[str, Any], native_step: int | None, marked_step: int | None,
) -> int | None:
    """The logical step of one state directory, for every executor that places one.

    A backend whose step marker counts logical steps is placed by the marker alone: whether
    the native name counts process-local or logical steps depends on the dataset's shape. Any
    other backend's native step is mapped through the run's Resume steps. None means the
    directory cannot be placed yet.
    """
    marker = contract.get("state_step") if isinstance(contract.get("state_step"), dict) else None
    if marker is not None and marker.get("space") == "logical":
        return marked_step
    if native_step is None:
        return None
    return logical_step(native_step, frozen_resume_steps(run_dir, run))


def publish_training_state_candidate(workspace: Path, run_dir: Path, candidate: Path, observed_step: int) -> dict[str, Any] | None:
    """Publish one complete state directory at its logical step, or return None.

    `observed_step` is the native step its name gives (`state_directory_step`). A backend's
    verified step marker decides the step when it has one, and is the only way to place the
    final state directory (FINAL_STATE_STEP), whose name carries no step.
    """
    run, status, runtime_identity = _published_run_context(run_dir)
    policy = training_state_policy(run)
    contract = training_state_contract(run)
    if not training_state_managed(run, contract):
        return None
    native_format = contract["native_format"]
    required = contract["required_files"]
    restoration = contract["restoration_contract"]
    if any(not (candidate / name).is_file() for name in required):
        return None
    marker = contract.get("state_step") if isinstance(contract.get("state_step"), dict) else None
    marked_step = _read_candidate_step(candidate, marker) if marker is not None else None
    if marker is not None and marked_step is None:
        return None
    native_step = marked_step if observed_step == FINAL_STATE_STEP else observed_step
    step = state_logical_step(run_dir, run, contract, native_step, marked_step)
    if step is None:
        return None
    backend = run.get("backend") if isinstance(run.get("backend"), dict) else {}
    backend_name = str(backend.get("name"))
    if backend_name == "sd-scripts" and (candidate / "train_state.json").is_file():
        try:
            train_state = json.loads((candidate / "train_state.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(train_state, dict) or train_state.get("current_step") != step:
            return None
    return publish_training_state(
        workspace,
        source_run=run_dir.name,
        source_realization=status.get("last_realization") if isinstance(status.get("last_realization"), str) else None,
        backend=backend_name,
        observed_step=step,
        candidate=candidate,
        native_format=native_format,
        restoration_contract=restoration,
        runtime_identity=runtime_identity,
        compatibility={"recipe_sha256": recipe_fingerprint(run)},
        keep_generations=policy["keep_generations"],
    )


def _read_candidate_step(candidate: Path, marker: dict[str, Any] | None) -> int | None:
    if marker is None or not isinstance(marker.get("path"), str) or not isinstance(marker.get("field"), str):
        return None
    try:
        document = json.loads((candidate / marker["path"]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(document, dict):
        return None
    expected_schema = marker.get("schema_version")
    if expected_schema is not None and document.get("schema_version") != expected_schema:
        return None
    expected_backend = marker.get("backend")
    if expected_backend is not None and document.get("backend") != expected_backend:
        return None
    digests = marker.get("digests")
    if digests is not None:
        if not isinstance(digests, dict):
            return None
        for digest_field, relative_name in digests.items():
            if not isinstance(digest_field, str) or not isinstance(relative_name, str):
                return None
            expected_digest = document.get(digest_field)
            payload = candidate / relative_name
            if not isinstance(expected_digest, str) or not payload.is_file() or _sha256_file(payload) != expected_digest:
                return None
    value = document.get(marker["field"])
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def publish_completed_training_states(
    workspace: Path,
    run_dir: Path,
    *,
    allow_final_state: bool = False,
) -> list[dict[str, Any]]:
    """Publish structurally complete step-state directories already on local disk."""

    run, _, _ = _published_run_context(run_dir)
    contract = training_state_contract(run)
    if not training_state_managed(run, contract):
        return []
    output_name = run_output_name(run)
    published: list[dict[str, Any]] = []
    outputs = run_dir / "outputs"
    for candidate in sorted(outputs.glob("*-state")) if outputs.is_dir() else []:
        observed_step = state_directory_step(candidate.name, output_name, allow_final=allow_final_state)
        if observed_step is None or not candidate.is_dir():
            continue
        manifest = publish_training_state_candidate(workspace, run_dir, candidate, observed_step)
        if manifest is not None:
            published.append(manifest)
    unique: dict[str, dict[str, Any]] = {}
    for item in published:
        if _artifact_dir(workspace, item["id"]).is_dir():
            unique[item["id"]] = item
    return list(unique.values())


def is_training_state_output(path: Path) -> bool:
    """Return whether an output file is nested below a native state directory."""

    return any(part.endswith("-state") for part in path.parts[:-1])

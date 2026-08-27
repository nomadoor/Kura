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
from typing import Any

import yaml

from kura.fsio import atomic_write_json, file_lock
from kura.run_envelope import common_recipe, resume_intent, training_state_policy


ARTIFACT_SCHEMA_VERSION = 1


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


def _validate_safetensors_file(path: Path) -> None:
    """Validate the safe container structure without deserializing tensor data."""

    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"training-state has an incomplete safetensors file: {path.name}")
        header_size = int.from_bytes(prefix, "little", signed=False)
        if header_size <= 0 or header_size > size - 8:
            raise ValueError(f"training-state has an invalid safetensors header size: {path.name}")
        try:
            header = json.loads(handle.read(header_size))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"training-state has an invalid safetensors header: {path.name}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"training-state has a non-object safetensors header: {path.name}")
    data_size = size - 8 - header_size
    dtype_bytes = {
        "BOOL": 1, "I8": 1, "U8": 1,
        "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
        "I32": 4, "U32": 4, "F32": 4,
        "I64": 8, "U64": 8, "F64": 8,
        "F8_E4M3": 1, "F8_E5M2": 1,
    }
    intervals: list[tuple[int, int]] = []
    for key, value in header.items():
        if key == "__metadata__":
            continue
        offsets = value.get("data_offsets") if isinstance(value, dict) else None
        shape = value.get("shape") if isinstance(value, dict) else None
        dtype = value.get("dtype") if isinstance(value, dict) else None
        if (
            dtype not in dtype_bytes
            or not isinstance(shape, list)
            or not all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
        ):
            raise ValueError(f"training-state has an invalid safetensors tensor entry: {path.name}")
        start, end = offsets
        if not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int) or isinstance(end, bool) or start < 0 or end < start or end > data_size:
            raise ValueError(f"training-state safetensors data is out of bounds: {path.name}")
        elements = 1
        for dimension in shape:
            elements *= dimension
        if end - start != elements * dtype_bytes[dtype]:
            raise ValueError(f"training-state safetensors tensor byte size is invalid: {path.name}")
        intervals.append((start, end))
    if not intervals or sorted(intervals)[0][0] != 0 or any(left[1] != right[0] for left, right in zip(sorted(intervals), sorted(intervals)[1:])) or sorted(intervals)[-1][1] != data_size:
        raise ValueError(f"training-state safetensors data is not complete: {path.name}")


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


def _manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


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
    expected_payload = f"artifacts/training-state/{artifact_id}/payload"
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
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"cannot safely inspect training-state reference: {path}") from exc
        continuation = run.get("continuation") if isinstance(run, dict) else None
        source = continuation.get("source") if isinstance(continuation, dict) else None
        artifact_id = source.get("artifact_id") if isinstance(source, dict) else None
        if isinstance(artifact_id, str):
            protected.add(artifact_id)
    for path in (workspace / "runs").glob("*/resolved/training-state-source.lock.json"):
        try:
            lock = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot safely inspect training-state reference: {path}") from exc
        artifact_id = lock.get("artifact_id") if isinstance(lock, dict) else None
        if isinstance(artifact_id, str):
            protected.add(artifact_id)
    return protected


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
    candidates.sort(key=lambda item: (int(item.get("observed_step") or -1), str(item.get("id"))), reverse=True)
    retained_steps = {
        step
        for step in dict.fromkeys(int(item.get("observed_step") or -1) for item in candidates)
        if step >= 0
    }
    retained_steps = set(sorted(retained_steps, reverse=True)[:keep_generations])
    protected = _protected_artifact_ids(workspace)
    for manifest in candidates:
        if int(manifest.get("observed_step") or -1) in retained_steps:
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
        _validate_safetensors_file(model_file)
    for path in candidate.rglob("*"):
        if path.is_file() and (path.name in {"optimizer.bin", "scheduler.bin", "optimizer.pt"} or re.fullmatch(r"random_states_\d+\.pkl", path.name)):
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
            "payload": f"artifacts/training-state/{artifact_id}/payload",
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
    _apply_retention(workspace, source_run, keep_generations)
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
    return max(candidates, key=lambda item: (int(item.get("observed_step") or -1), str(item.get("id"))))


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


def resume_artifact_directory(workspace: Path, run: dict[str, Any]) -> Path | None:
    continuation = run.get("continuation")
    if not isinstance(continuation, dict) or continuation.get("mode") != "resume":
        return None
    source = continuation.get("source")
    artifact_id = source.get("artifact_id") if isinstance(source, dict) else None
    expected = source.get("manifest_sha256") if isinstance(source, dict) else None
    if not isinstance(artifact_id, str) or not isinstance(expected, str):
        raise ValueError("Resume continuation requires source artifact_id and manifest_sha256")
    manifest = load_training_state(workspace, artifact_id)
    if manifest["manifest_sha256"] != expected:
        raise ValueError(f"training-state manifest digest mismatch: {artifact_id}")
    verify_training_state(workspace, manifest)
    return _artifact_dir(workspace, artifact_id)


def compile_resume_lock(
    workspace: Path,
    run: dict[str, Any],
    resolved: Path,
    *,
    target_runtime_identity: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    continuation = resume_intent(run)
    if continuation is None:
        return None
    source = continuation["source"]
    current_fingerprint = recipe_fingerprint(run)
    if current_fingerprint != source["recipe_sha256"]:
        raise ValueError("Resume training recipe changed after the derived run was created; create a Fork from Weight instead")
    manifest = load_training_state(workspace, source["artifact_id"])
    if manifest["manifest_sha256"] != source["manifest_sha256"]:
        raise ValueError(f"training-state manifest digest mismatch: {source['artifact_id']}")
    verify_training_state(workspace, manifest)
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
    contract = training_state_contract(run)
    lock = {
        "schema_version": 1,
        "mode": "resume",
        "source_run": run["parent_run"],
        "artifact_id": artifact_id,
        "manifest_sha256": manifest["manifest_sha256"],
        "source_realization": manifest.get("source_realization"),
        "source_step": manifest["observed_step"],
        "target_step": continuation["target_step"],
        "additional_steps": continuation["target_step"] - manifest["observed_step"],
        "native_progress": contract.get("native_progress", "logical"),
        "native_target": contract.get("native_target", "logical"),
        "native_state_path": f"/workspace/artifacts/training-state/{artifact_id}/payload",
        "restoration_contract": manifest["restoration_contract"],
        "runtime_identity": manifest.get("runtime_identity") or {},
        "compatibility": compatibility,
        "files": manifest["files"],
    }
    resolved.mkdir(parents=True, exist_ok=True)
    atomic_write_json(resolved / "training-state-source.lock.json", lock)
    return lock


def training_state_contract(run: dict[str, Any]) -> dict[str, Any]:
    """Return the active adapter's recovery contract without owning backend policy here."""

    from kura.backends import get_backend

    backend = run.get("backend") if isinstance(run.get("backend"), dict) else {}
    adapter = get_backend(backend.get("name"))
    if adapter.training_state is None:
        raise ValueError(f"backend {adapter.name!r} has no training-state contract")
    return adapter.training_state(run)


def training_state_capture_required(run_dir: Path) -> bool:
    """Return whether a completed run is expected to publish recoverable state."""

    run, _, _ = _published_run_context(run_dir)
    if not training_state_policy(run)["enabled"]:
        return False
    return training_state_contract(run).get("capability") != "unsupported"


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


def publish_training_state_candidate(workspace: Path, run_dir: Path, candidate: Path, observed_step: int) -> dict[str, Any] | None:
    run, status, runtime_identity = _published_run_context(run_dir)
    policy = training_state_policy(run)
    if not policy["enabled"]:
        return None
    contract = training_state_contract(run)
    native_format = contract["native_format"]
    required = contract["required_files"]
    restoration = contract["restoration_contract"]
    continuation = resume_intent(run)
    if any(not (candidate / name).is_file() for name in required):
        return None
    marker = contract.get("state_step") if isinstance(contract.get("state_step"), dict) else None
    marked_step = _read_candidate_step(candidate, marker) if marker is not None else None
    if marker is not None and marked_step is None:
        return None
    if marker is not None and marker.get("space") == "logical":
        logical_step = marked_step
    else:
        logical_step = observed_step
        if continuation is not None and contract.get("native_progress") == "process_local":
            logical_step = continuation["source"]["observed_step"] + observed_step
    if not isinstance(logical_step, int):
        return None
    backend = run.get("backend") if isinstance(run.get("backend"), dict) else {}
    backend_name = str(backend.get("name"))
    if backend_name == "sd-scripts" and (candidate / "train_state.json").is_file():
        try:
            train_state = json.loads((candidate / "train_state.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(train_state, dict) or train_state.get("current_step") != logical_step:
            return None
    return publish_training_state(
        workspace,
        source_run=run_dir.name,
        source_realization=status.get("last_realization") if isinstance(status.get("last_realization"), str) else None,
        backend=backend_name,
        observed_step=logical_step,
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
    policy = training_state_policy(run)
    if not policy["enabled"]:
        return []
    backend = run.get("backend") if isinstance(run.get("backend"), dict) else {}
    contract = training_state_contract(run)
    config = backend.get("config") if isinstance(backend.get("config"), dict) else {}
    continuation = resume_intent(run)
    output_name = str((run.get("id") if continuation is not None else config.get("output_name")) or run.get("id") or run_dir.name)
    pattern = re.compile(rf"^{re.escape(output_name)}-step(?P<step>\d{{4,}})-state$")
    final_step = continuation["target_step"] if continuation is not None else common_recipe(run).get("steps")
    published: list[dict[str, Any]] = []
    outputs = run_dir / "outputs"
    for candidate in sorted(outputs.glob("*-state")) if outputs.is_dir() else []:
        match = pattern.fullmatch(candidate.name)
        if not candidate.is_dir():
            continue
        if match is not None:
            observed_step = int(match.group("step"))
        elif allow_final_state and candidate.name == f"{output_name}-state" and isinstance(final_step, int):
            marker = contract.get("state_step") if isinstance(contract.get("state_step"), dict) else None
            observed_step = _read_candidate_step(candidate, marker)
            if observed_step is None:
                continue
        else:
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

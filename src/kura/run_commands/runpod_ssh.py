"""RunPod SSH, SCP, upload, pull, and download helpers."""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import stat
import tarfile
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from kura.container_scripts import script_source
from kura.executors.runpod import project_runpod_dataset_handoff
from kura.dataset_transfer import StagedTransferChanged, TransferRefused, verify_pinned_transfer
from kura.media_types import KNOWN_MEDIA_SUFFIXES, frozen_suffixes

from kura.artifact_publication import output_contract, publish_outputs, record_publication_failure
from kura.executors import _materialize_stdout_progress, _redact_secret_text, _redact_secrets
from kura.executors.runpod import POD_SELF_DELETE_FUNCTION
from kura.fsio import atomic_write_json
from kura.records import record
from kura.workspace import load_yaml as _load_yaml
from kura.workspace import run_path as _run_path
from kura.workspace import workspace_config as _workspace_config
from kura.run_envelope import common_recipe, resume_intent, training_state_policy
from kura.executors.common import _OperationBusy, _mutate_run_status, _record_progress, check_stop, sleep_checking_stop, _run_operation_lock, append_run_event, record_launch_phase, run_events
from kura.run_commands.common import _load_frozen_command, _safe_error
from kura.run_commands.plan import _configured_download_min_free_bytes, _ensure_free_bytes
from kura.training_artifacts import is_training_state_output, load_training_state, publish_completed_training_states, publish_training_state_candidate, select_training_state, training_state_at_step, training_state_contract, training_state_retention_floor, verify_training_state
from kura.runtime_io import validated_write_roots


RUNPOD_TRANSFER_TIMEOUT_SEC = 600


def _run_bounded(command: list[str], *, context: str, timeout: int = RUNPOD_TRANSFER_TIMEOUT_SEC, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
    try:
        return subprocess.run(command, check=False, timeout=timeout, **kwargs)
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"{context} timed out after {exc.timeout} seconds") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_or_copy_snapshot_file(
    source: Path,
    target: Path,
    *,
    free_space_root: Path,
    required_free_bytes: int,
) -> None:
    """Reuse immutable bytes, accounting for disk only when linking is unavailable."""

    try:
        # Both the publication source and terminal snapshot are immutable Kura
        # artifacts; linking avoids duplicating large protected state payloads.
        os.link(source, target)
    except OSError:
        _ensure_free_bytes(
            free_space_root,
            required_free_bytes + source.stat().st_size,
            context="RunPod delta download reusable copy fallback",
        )
        shutil.copy2(source, target)


def _validated_snapshot_manifest(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        raise ValueError("remote snapshot manifest did not return a file list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("remote snapshot manifest contains a non-object entry")
        path_value = item.get("path")
        size = item.get("size")
        mtime_ns = item.get("mtime_ns")
        digest = item.get("sha256")
        if not isinstance(path_value, str):
            raise ValueError("remote snapshot manifest entry has no path")
        relative = PurePosixPath(path_value)
        if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError(f"remote snapshot manifest has an unsafe path: {path_value}")
        canonical = relative.as_posix()
        if canonical in seen:
            raise ValueError(f"remote snapshot manifest contains a duplicate path: {canonical}")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"remote snapshot manifest has an invalid size: {canonical}")
        if not isinstance(mtime_ns, int) or isinstance(mtime_ns, bool) or mtime_ns < 0:
            raise ValueError(f"remote snapshot manifest has an invalid mtime: {canonical}")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"remote snapshot manifest has an invalid SHA-256: {canonical}")
        seen.add(canonical)
        normalized.append({"path": canonical, "size": size, "mtime_ns": mtime_ns, "sha256": digest})
    return sorted(normalized, key=lambda item: item["path"])


def _runpod_remote_snapshot_manifest(
    details: dict[str, Any],
    *,
    workspace: str,
    run_id: str,
    timeout_sec: int = RUNPOD_TRANSFER_TIMEOUT_SEC,
) -> list[dict[str, Any]]:
    remote_root = f"{workspace.rstrip('/')}/runs/{run_id}"
    script = f"""
export PATH="/opt/conda/bin:/usr/local/bin:$PATH"
python - <<'PY'
import hashlib
import json
import os

root = {remote_root!r}
items = []
errors = []
for current, dirs, files in os.walk(root, topdown=True, followlinks=False):
    relative_root = os.path.relpath(current, root)
    if relative_root == ".":
        dirs[:] = [name for name in dirs if name not in {{"cache", "transfer"}}]
    kept_dirs = []
    for name in sorted(dirs):
        path = os.path.join(current, name)
        if os.path.islink(path):
            errors.append("symlink directory: " + os.path.relpath(path, root))
        else:
            kept_dirs.append(name)
    dirs[:] = kept_dirs
    for name in sorted(files):
        path = os.path.join(current, name)
        relative = os.path.relpath(path, root).replace(os.sep, "/")
        if os.path.islink(path) or not os.path.isfile(path):
            errors.append("non-regular file: " + relative)
            continue
        stat = os.stat(path)
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        items.append({{"path": relative, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}})
print(json.dumps({{"files": items, "errors": errors}}))
PY
""".strip()
    result = _run_bounded(
        [*_ssh_base(details), script],
        context="remote snapshot inventory",
        timeout=timeout_sec,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise ValueError(_redact_secret_text(result.stderr.strip() or result.stdout.strip() or "remote snapshot inventory failed"))
    payload = json.loads(result.stdout or "{}")
    if not isinstance(payload, dict):
        raise ValueError("remote snapshot inventory did not return an object")
    errors = payload.get("errors")
    if isinstance(errors, list) and errors:
        raise ValueError(f"remote snapshot contains unsupported entries: {errors[0]}")
    return _validated_snapshot_manifest(payload.get("files"))


def _transfer_runpod_snapshot_delta(
    details: dict[str, Any],
    *,
    remote_root: str,
    run_id: str,
    files: list[dict[str, Any]],
    destination: Path,
) -> None:
    (destination / run_id).mkdir(parents=True, exist_ok=True)
    if not files:
        return
    paths = [item["path"] for item in files]
    archive_nonce = secrets.token_hex(6)
    remote_archive = f"/tmp/kura-download-{run_id}-{archive_nonce}.tar.gz"
    local_archive = destination / f".kura-download-{run_id}-{archive_nonce}.tar.gz"
    script = f"""
export PATH="/opt/conda/bin:/usr/local/bin:$PATH"
python - <<'PY'
import json
import os
import pathlib
import tarfile

root = pathlib.Path({remote_root!r})
run_id = {run_id!r}
paths = json.loads({json.dumps(json.dumps(paths))})
archive = {remote_archive!r}
with tarfile.open(archive, "w:gz") as handle:
    for value in paths:
        relative = pathlib.PurePosixPath(value)
        if relative.is_absolute() or not relative.parts or any(part in {{"", ".", ".."}} for part in relative.parts):
            raise SystemExit("unsafe snapshot path: " + value)
        source = root
        for part in relative.parts:
            source = source / part
            if source.is_symlink():
                raise SystemExit("snapshot path traverses a symlink: " + value)
        if not source.is_file():
            raise SystemExit("snapshot file is missing: " + value)
        handle.add(source, arcname=str(pathlib.PurePosixPath(run_id) / relative), recursive=False)
PY
""".strip()
    try:
        packed = _run_bounded([*_ssh_base(details), script], context="remote delta archive packing")
        if packed.returncode:
            raise ValueError(f"remote delta archive packing failed with exit code {packed.returncode}")
        copied = _run_bounded(
            [
                "scp",
                *_ssh_transport_options(details),
                "-P", str(details["port"]),
                "-i", str(details["key"]),
                f"root@{details['ip']}:{remote_archive}",
                str(local_archive),
            ],
            context="scp delta snapshot",
        )
        if copied.returncode:
            raise ValueError(f"scp delta snapshot failed with exit code {copied.returncode}")
        _extract_snapshot_delta_archive(
            local_archive,
            destination,
            run_id=run_id,
            expected_paths={f"{run_id}/{item['path']}" for item in files},
        )
    finally:
        local_archive.unlink(missing_ok=True)
        try:
            _run_bounded(
                [*_ssh_base(details), f"rm -f {shlex.quote(remote_archive)}"],
                context="remote delta archive cleanup",
            )
        except (OSError, ValueError, subprocess.TimeoutExpired):
            # The disposable Pod cleanup path removes /tmp. Do not mask the
            # actual packing, transfer, or verification result with a failed
            # best-effort temporary-file cleanup.
            pass


def _extract_snapshot_delta_archive(
    archive_path: Path,
    destination: Path,
    *,
    run_id: str,
    expected_paths: set[str],
) -> None:
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            file_members: dict[str, tarfile.TarInfo] = {}
            for member in archive.getmembers():
                relative = PurePosixPath(member.name)
                if relative.is_absolute() or not relative.parts or relative.parts[0] != run_id or any(part in {"", ".", ".."} for part in relative.parts):
                    raise ValueError(f"download archive contains an unsafe path: {member.name}")
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError(f"download archive contains a non-regular entry: {member.name}")
                canonical = relative.as_posix()
                if canonical in file_members:
                    raise ValueError(f"download archive contains a duplicate path: {canonical}")
                file_members[canonical] = member
            if set(file_members) != expected_paths:
                raise ValueError("download archive inventory does not match the requested delta")
            for relative, member in file_members.items():
                target = destination.joinpath(*PurePosixPath(relative).parts)
                if target.exists() or target.is_symlink():
                    raise ValueError(f"download archive collides with a reusable file: {relative}")
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"download archive file cannot be read: {relative}")
                temporary = target.with_name(f".{target.name}.partial-{secrets.token_hex(4)}")
                try:
                    with source, temporary.open("xb") as output:
                        shutil.copyfileobj(source, output)
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
    except tarfile.TarError as exc:
        raise ValueError("invalid delta snapshot archive") from exc


def _local_reusable_snapshot_source(
    run_dir: Path,
    item: dict[str, Any],
    *,
    remote_root: str,
) -> Path | None:
    relative = PurePosixPath(item["path"])
    if len(relative.parts) != 2 or relative.parts[0] != "outputs" or not relative.name.endswith(".safetensors"):
        return None
    try:
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    mirrored = status.get("mirrored_outputs") if isinstance(status.get("mirrored_outputs"), list) else []
    publication = next(
        (
            candidate
            for candidate in mirrored
            if isinstance(candidate, dict)
            and candidate.get("name") == relative.name
            and candidate.get("path") == relative.as_posix()
        ),
        None,
    )
    if not isinstance(publication, dict):
        return None
    if (
        publication.get("size") != item["size"]
        or publication.get("remote_mtime_ns") != item["mtime_ns"]
        or publication.get("remote_path") != f"{remote_root.rstrip('/')}/{relative.as_posix()}"
    ):
        return None
    candidate = run_dir.joinpath(*relative.parts)
    if candidate.is_symlink() or not candidate.is_file():
        return None
    try:
        if candidate.stat().st_size != item["size"] or _sha256_file(candidate) != item["sha256"]:
            return None
        if candidate.suffix == ".safetensors":
            _validate_safetensors_file(candidate)
    except (OSError, ValueError):
        return None
    return candidate


def _local_reusable_training_state_sources(
    run_dir: Path,
    snapshot_manifest: list[dict[str, Any]],
) -> dict[str, Path]:
    """Map complete remote state directories to identical protected payloads."""

    groups: dict[str, list[dict[str, Any]]] = {}
    for item in snapshot_manifest:
        parts = PurePosixPath(item["path"]).parts
        if len(parts) < 3 or parts[0] != "outputs" or not parts[1].endswith("-state"):
            continue
        groups.setdefault("/".join(parts[:2]), []).append(item)
    if not groups:
        return {}
    workspace = run_dir.parent.parent
    artifacts: list[tuple[dict[str, Any], Path]] = []
    for manifest_path in sorted((workspace / "artifacts" / "training-state").glob("*/manifest.json")):
        try:
            manifest = load_training_state(workspace, manifest_path.parent.name)
            if manifest.get("source_run") != run_dir.name:
                continue
            payload = verify_training_state(workspace, manifest)
        except (OSError, ValueError):
            continue
        artifacts.append((manifest, payload))
    reusable: dict[str, Path] = {}
    for state_root, remote_files in groups.items():
        remote_inventory = {
            "/".join(PurePosixPath(item["path"]).parts[2:]): (item["size"], item["sha256"])
            for item in remote_files
        }
        for artifact, payload in artifacts:
            local_inventory = {
                item["path"]: (item.get("size"), item.get("sha256"))
                for item in artifact.get("files", [])
                if isinstance(item, dict) and isinstance(item.get("path"), str)
            }
            if local_inventory != remote_inventory:
                continue
            for relative in remote_inventory:
                reusable[f"{state_root}/{relative}"] = payload.joinpath(*PurePosixPath(relative).parts)
            break
    return reusable


def _verify_snapshot_tree(snapshot: Path, manifest: list[dict[str, Any]]) -> None:
    expected = {item["path"]: item for item in manifest}
    actual: set[str] = set()
    for path in snapshot.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"downloaded snapshot contains a symlink: {path.relative_to(snapshot)}")
        if not path.is_file():
            continue
        relative = path.relative_to(snapshot).as_posix()
        item = expected.get(relative)
        if item is None:
            raise ValueError(f"downloaded snapshot contains an unexpected file: {relative}")
        if path.stat().st_size != item["size"] or _sha256_file(path) != item["sha256"]:
            raise ValueError(f"downloaded snapshot does not match the remote manifest: {relative}")
        actual.add(relative)
    missing = sorted(set(expected) - actual)
    if missing:
        raise ValueError(f"downloaded snapshot is missing a remote file: {missing[0]}")


def _latest_runpod_transfer(run_dir: Path) -> dict[str, Any]:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    realization_ref = status.get("last_realization")
    if not isinstance(realization_ref, str):
        raise ValueError("run has no RunPod realization")
    realization = json.loads((run_dir / realization_ref).read_text(encoding="utf-8"))
    transfer = realization.get("transfer")
    if realization.get("executor") != "runpod" or not isinstance(transfer, dict):
        raise ValueError("latest realization has no RunPod upload transfer")
    return transfer


def _latest_runpod_stage(run_dir: Path) -> dict[str, Any]:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    stage_ref = status.get("last_stage")
    if not isinstance(stage_ref, str):
        raise ValueError("run has no RunPod upload stage")
    stage = json.loads((run_dir / stage_ref).read_text(encoding="utf-8"))
    if stage.get("storage_mode") != "upload":
        raise ValueError("latest RunPod stage is not an upload bundle")
    return stage


def cmd_run_upload(args: argparse.Namespace) -> int:
    try:
        run_dir = _run_path(args.run_id)
        transfer = _latest_runpod_transfer(run_dir)
        archive = transfer.get("archive")
        upload_code = transfer.get("upload_code")
        if not isinstance(archive, str) or not isinstance(upload_code, str):
            raise ValueError("latest realization has no upload archive/code")
        archive_path = run_dir / archive
        if not archive_path.is_file():
            raise ValueError(f"upload archive is missing: {archive_path}")
        if not shutil.which("runpodctl"):
            raise ValueError("runpodctl is not installed locally; install it before uploading")
        return subprocess.run(["runpodctl", "send", str(archive_path), "--code", upload_code], check=False).returncode
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"cannot upload run bundle: {_safe_error(exc)}", file=sys.stderr)
        return 1


def download_run(run_id: str, *, force: bool = False) -> int:
    try:
        run_dir = _run_path(run_id)
        with _run_operation_lock(run_dir, "download", blocking=False):
            return _download_run_unlocked(run_id, force=force)
    except (OSError, ValueError) as exc:
        print(f"cannot download run outputs: {_safe_error(exc)}", file=sys.stderr)
        return 1


def _download_run_unlocked(run_id: str, *, force: bool = False) -> int:
    collecting_on: dict[str, Any] | None = None
    try:
        run_dir = _run_path(run_id)
        destination = run_dir / "downloads"
        downloaded_run = destination / run_id
        try:
            manifest = _load_yaml(run_dir / "resolved" / "manifest.lock.yaml")
        except (OSError, ValueError, yaml.YAMLError):
            manifest = {}
        if not isinstance(manifest, dict):
            manifest = {}
        backend = manifest.get("backend") if isinstance(manifest.get("backend"), dict) else {}
        backend_name = backend.get("name") if isinstance(backend.get("name"), str) else None

        def materialize_primary_outputs(output_dir: Path) -> list[str]:
            primary = run_dir / "outputs"
            outputs: list[str] = []
            if not output_dir.exists():
                return outputs
            primary.mkdir(parents=True, exist_ok=True)
            publications: dict[Path, Path] = {}
            legacy_publications: dict[Path, Path] = {}
            for source in sorted(path for path in output_dir.rglob("*") if path.is_file()):
                relative = source.relative_to(output_dir)
                if is_training_state_output(relative):
                    continue
                if backend_name == "ai-toolkit" and len(relative.parts) > 1 and relative.parts[0] == run_id:
                    relative = Path(*relative.parts[1:])
                    legacy_publications[relative] = source
                if relative in publications:
                    raise ValueError(f"download outputs collide at canonical path {relative}")
                publications[relative] = source
            for relative, source in publications.items():
                target = primary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(f".{target.name}.partial-{secrets.token_hex(4)}")
                try:
                    try:
                        os.link(source, temporary)
                    except OSError:
                        shutil.copy2(source, temporary)
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
                outputs.append(str((primary / relative).relative_to(run_dir)))
            legacy_root = primary / run_id
            if legacy_publications and legacy_root.is_dir():
                legacy_files = {
                    path.relative_to(legacy_root): path
                    for path in legacy_root.rglob("*")
                    if path.is_file()
                }
                if legacy_files.keys() == legacy_publications.keys() and all(
                    filecmp.cmp(legacy_files[relative], source, shallow=False)
                    for relative, source in legacy_publications.items()
                ):
                    shutil.rmtree(legacy_root)
            return outputs

        def record_recovery_download(recovery_artifacts: list[str]) -> None:
            if not recovery_artifacts:
                return
            if any(
                prior.get("event") == "run_recovery_artifacts_downloaded" and prior.get("artifacts") == recovery_artifacts
                for prior in run_events(run_dir)
            ):
                return
            append_run_event(run_dir, {"event": "run_recovery_artifacts_downloaded", "timestamp": datetime.now().astimezone().isoformat(), "kind": "non-final-intermediate", "artifacts": recovery_artifacts})

        def materialize_downloaded_status() -> tuple[bool, list[str]]:
            exits = sorted((downloaded_run / "realizations").glob("remote-exit-*.json"))
            if not exits:
                return False, []
            remote_exit = json.loads(exits[-1].read_text(encoding="utf-8"))
            exit_code = remote_exit.get("exit_code")
            if not isinstance(exit_code, int):
                return False, []
            output_dir = downloaded_run / "outputs"
            state_capture_required = _directory_training_state_sync_enabled(run_dir)
            published_states = (
                publish_completed_training_states(
                    run_dir.parent.parent,
                    downloaded_run,
                    allow_final_state=exit_code == 0,
                )
                if backend_name in {"ai-toolkit", "musubi-tuner", "sd-scripts"}
                and output_dir.is_dir()
                and any(output_dir.glob("*-state"))
                else []
            )
            if state_capture_required and not published_states:
                try:
                    published_states = [select_training_state(run_dir.parent.parent, run_id)]
                except ValueError:
                    published_states = []
            state_sync_error: str | None = None
            if state_capture_required and not published_states:
                if exit_code == 0:
                    # A completed trainer must leave a durable state; keep the Pod.
                    raise ValueError(
                        "downloaded run snapshot has no valid training-state artifact; "
                        "keep the Pod until recovery files are inspected or downloaded"
                    )
                # A failed trainer may never have written state. The snapshot
                # already holds everything the Pod had, so holding the Pod
                # would only bill; record the gap as Docker does.
                state_sync_error = (
                    "remote run failed and its downloaded snapshot has no valid training-state artifact; "
                    "inspect the backend state output before relying on Resume"
                )
            outputs = materialize_primary_outputs(output_dir)
            publication_manifest: str | None = None
            contract: dict[str, Any] | None = None
            if exit_code == 0:
                current_status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
                realization_ref = current_status.get("last_realization")
                realization_id = Path(realization_ref).stem if isinstance(realization_ref, str) else None
                try:
                    contract = output_contract(run_dir)
                    if contract is not None:
                        if realization_id is None:
                            raise ValueError("cannot publish downloaded outputs without a realization")
                        publication_manifest, outputs = publish_outputs(
                            run_dir, realization_id, contract, candidate_paths=outputs
                        )
                except (OSError, ValueError) as exc:
                    error = _safe_error(exc)
                    attempt = record_publication_failure(run_dir, realization_id, error) if realization_id else None

                    def record_blocked(current: dict[str, Any]) -> None:
                        current.update({
                            "state": "recovery_required", "execution_state": "completed",
                            "exit_code": exit_code, "ended": remote_exit.get("timestamp"),
                            "publication_state": "blocked", "publication_error": error,
                            "recovery_required": True,
                            "remote_state": "completed", "remote_exit_code": exit_code,
                            "remote_exit": str(exits[-1].relative_to(run_dir)),
                            "remote_ended": remote_exit.get("timestamp"),
                        })
                        if attempt:
                            current["last_publication_attempt"] = attempt

                    _mutate_run_status(run_dir, record_blocked)
                    raise
            recovery_root = downloaded_run / "recovery"
            recovery_artifacts = [
                str(path.relative_to(run_dir))
                for path in sorted(recovery_root.rglob("*"))
                if path.is_file()
            ] if recovery_root.is_dir() else []
            steps: int | None = None
            if exit_code == 0:
                continuation = manifest.get("continuation") if isinstance(manifest.get("continuation"), dict) else {}
                configured_steps = continuation.get("target_step") if continuation.get("mode") == "resume" else None
                if not isinstance(configured_steps, int):
                    try:
                        configured_steps = common_recipe(manifest).get("steps")
                    except ValueError:
                        configured_steps = None
                if isinstance(configured_steps, int) and configured_steps > 0:
                    steps = configured_steps

            realization_ref = json.loads((run_dir / "status.json").read_text(encoding="utf-8")).get("last_realization")
            input_postflight = (
                project_runpod_dataset_handoff(run_dir, downloaded_run, Path(realization_ref).stem)
                if isinstance(realization_ref, str) else None
            )

            def mutate(status: dict[str, Any]) -> None:
                if input_postflight is not None:
                    status["dataset_input_postflight"] = input_postflight
                status.update({"state": "completed" if exit_code == 0 else "failed", "exit_code": exit_code, "ended": remote_exit.get("timestamp"), "outputs": outputs, "recovery_artifacts": recovery_artifacts, "downloaded_run": str(downloaded_run.relative_to(run_dir)), "remote_exit": str(exits[-1].relative_to(run_dir)), "remote_state": "completed" if exit_code == 0 else "failed", "remote_exit_code": exit_code, "remote_ended": remote_exit.get("timestamp"), "recovery_required": False})
                status["execution_state"] = "completed" if exit_code == 0 else "failed"
                if state_sync_error is not None:
                    status["training_state_sync_error"] = state_sync_error
                else:
                    # A final download that obtained (or never needed) state
                    # supersedes any error recorded by an earlier mid-run sync.
                    status.pop("training_state_sync_error", None)
                status["publication_state"] = "completed" if contract else "legacy-unverified" if exit_code == 0 else "not-required"
                status.pop("publication_error", None)
                if publication_manifest:
                    status["publication_manifest"] = publication_manifest
                if published_states:
                    status["recoverable_training_states"] = [
                        {
                            "artifact_id": item["id"],
                            "observed_step": item["observed_step"],
                            "manifest_sha256": item["manifest_sha256"],
                            "restoration_level": item["restoration_contract"]["level"],
                        }
                        for item in published_states
                    ]
                if steps is not None:
                    status["last_step"] = steps
                    status["total_steps"] = steps
                    if continuation.get("mode") == "resume" and isinstance(continuation.get("source"), dict):
                        source_step = continuation["source"].get("observed_step")
                        if isinstance(source_step, int):
                            status["current_run_step"] = steps - source_step
                            status["current_run_total_steps"] = steps - source_step
                    _record_progress(run_dir, status)

            _mutate_run_status(run_dir, mutate)
            return True, recovery_artifacts

        if downloaded_run.exists() and not force:
            materialized, recovery_artifacts = materialize_downloaded_status()
            if materialized:
                record_recovery_download(recovery_artifacts)
                print(json.dumps(json.loads((run_dir / "status.json").read_text(encoding="utf-8")), indent=2))
                return 0
            raise ValueError("downloaded run snapshot is missing remote-exit; use --force to retry or inspect the Pod before stopping it")
        if not shutil.which("runpodctl"):
            raise ValueError("runpodctl is not installed locally; install it before downloading")
        config = _workspace_config()
        min_download_free = _configured_download_min_free_bytes(config)
        details = _runpod_ssh_details(run_dir, timeout_sec=60, interval_sec=2)
        _start_ssh_master(details)
        _mark_runpod_outputs_collecting(details, run_id)
        collecting_on = details
        destination.mkdir(exist_ok=True)
        workspace = _runpod_workspace_for_run(run_dir)
        remote_run_dir = f"{workspace.rstrip('/')}/runs/{run_id}"
        snapshot_manifest = _runpod_remote_snapshot_manifest(details, workspace=workspace, run_id=run_id)
        reusable = _local_reusable_training_state_sources(run_dir, snapshot_manifest)
        pending: list[dict[str, Any]] = []
        for item in snapshot_manifest:
            if item["path"] in reusable:
                continue
            source = _local_reusable_snapshot_source(run_dir, item, remote_root=remote_run_dir)
            if source is None:
                pending.append(item)
            else:
                reusable[item["path"]] = source
        pending_size = sum(item["size"] for item in pending)
        terminal_download = {
            "files": len(snapshot_manifest),
            "bytes": sum(item["size"] for item in snapshot_manifest),
            "reused_files": len(reusable),
            "reused_bytes": sum(item["size"] for item in snapshot_manifest if item["path"] in reusable),
            "transferred_files": len(pending),
            "transferred_bytes": pending_size,
        }
        required_free_bytes = max(min_download_free, pending_size * 2 + 5 * 1024**3)
        _ensure_free_bytes(
            destination,
            required_free_bytes,
            context="RunPod delta download",
        )
        staging = destination / f".{run_id}.partial-{secrets.token_hex(6)}"
        backup: Path | None = None
        recovery_artifacts: list[str] = []
        try:
            snapshot = staging / run_id
            snapshot.mkdir(parents=True)
            for relative, source in reusable.items():
                target = snapshot.joinpath(*PurePosixPath(relative).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                _link_or_copy_snapshot_file(
                    source,
                    target,
                    free_space_root=destination,
                    required_free_bytes=required_free_bytes,
                )
            _transfer_runpod_snapshot_delta(
                details,
                remote_root=remote_run_dir,
                run_id=run_id,
                files=pending,
                destination=staging,
            )
            refreshed = _runpod_remote_snapshot_manifest(details, workspace=workspace, run_id=run_id)
            if refreshed != snapshot_manifest:
                raise ValueError("remote run changed while its terminal snapshot was being downloaded")
            _verify_snapshot_tree(snapshot, snapshot_manifest)
            if downloaded_run.exists():
                backup = destination / f".{run_id}.previous-{secrets.token_hex(6)}"
                os.replace(downloaded_run, backup)
            try:
                os.replace(snapshot, downloaded_run)
            except BaseException:
                if backup is not None and backup.exists() and not downloaded_run.exists():
                    os.replace(backup, downloaded_run)
                raise
            try:
                materialized, recovery_artifacts = materialize_downloaded_status()
                if not materialized:
                    raise ValueError("downloaded run snapshot is missing remote-exit; remote completion is not confirmed")
            except BaseException:
                if backup is not None and backup.exists():
                    rejected = destination / f".{run_id}.rejected-{secrets.token_hex(6)}"
                    os.replace(downloaded_run, rejected)
                    os.replace(backup, downloaded_run)
                    backup = None
                    # The rejected tree is not a valid reusable snapshot; keeping
                    # it would bypass the next remote-manifest verification pass.
                    shutil.rmtree(rejected)
                raise
            if backup is not None:
                shutil.rmtree(backup)
                backup = None
        finally:
            if staging.exists():
                shutil.rmtree(staging)
            if backup is not None and backup.exists() and not downloaded_run.exists():
                os.replace(backup, downloaded_run)
        _mutate_run_status(run_dir, lambda status: status.__setitem__("terminal_download", terminal_download))
        append_run_event(
            run_dir,
            {
                "event": "run_terminal_snapshot_downloaded",
                "timestamp": datetime.now().astimezone().isoformat(),
                **terminal_download,
            },
        )
        record_recovery_download(recovery_artifacts)
        _mark_runpod_outputs_collected(details, run_id)
        return 0
    except (OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        print(f"cannot download run outputs: {_safe_error(exc)}", file=sys.stderr)
        if collecting_on is not None:
            # A failed download is not in progress; keep the unattended timer armed.
            _clear_runpod_mark(collecting_on, _runpod_collecting_mark(run_id))
        return 1


def cmd_run_download(args: argparse.Namespace) -> int:
    return download_run(args.run_id, force=args.force)


def _runpod_workspace_for_run(run_dir: Path) -> str:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    realization_ref = status.get("last_realization")
    if isinstance(realization_ref, str):
        try:
            realization = json.loads((run_dir / realization_ref).read_text(encoding="utf-8"))
            workspace = realization.get("request", {}).get("env", {}).get("KURA_WORKSPACE")
            if isinstance(workspace, str) and workspace:
                return workspace
        except (OSError, json.JSONDecodeError):
            pass
    return "/workspace"


def _checkpoint_step(name: str) -> int | None:
    matches = re.findall(r"(?:step|_)(\d{4,})(?=\.safetensors$|[-_.])", name)
    if not matches:
        return None
    return int(matches[-1])


def _select_remote_outputs(items: list[dict[str, Any]], *, step: int | None = None, since_step: int | None = None, all_outputs: bool = False) -> list[dict[str, Any]]:
    candidates = [item for item in items if isinstance(item.get("name"), str)]
    for item in candidates:
        if not isinstance(item.get("step"), int):
            item["step"] = _checkpoint_step(str(item["name"]))
    if step is not None:
        return [item for item in candidates if item.get("step") == step]
    if since_step is not None:
        return [item for item in candidates if isinstance(item.get("step"), int) and item["step"] >= since_step]
    if all_outputs:
        return candidates
    stepped = [item for item in candidates if isinstance(item.get("step"), int)]
    if stepped:
        latest = max(int(item["step"]) for item in stepped)
        return [item for item in stepped if item.get("step") == latest]
    return candidates[-1:] if candidates else []


def _same_remote_output_version(before: dict[str, Any], after: dict[str, Any] | None) -> bool:
    """Return true only when a remote file did not change across a transfer."""

    if after is None:
        return False
    return all(before.get(key) == after.get(key) for key in ("path", "size", "mtime_ns"))


def _same_remote_training_state_version(before: dict[str, Any], after: dict[str, Any] | None) -> bool:
    """Require an identical recursive inventory across a directory transfer."""

    if after is None or before.get("path") != after.get("path"):
        return False
    before_files = before.get("files")
    after_files = after.get("files")
    if not isinstance(before_files, list) or not isinstance(after_files, list):
        return False
    def normalize(items: list[Any]) -> list[tuple[Any, Any, Any]]:
        return sorted(
            (item.get("path"), item.get("size"), item.get("mtime_ns"))
            for item in items
            if isinstance(item, dict)
        )
    return (
        len(before_files) == len(after_files)
        and normalize(before_files) == normalize(after_files)
        and before.get("logical_step") == after.get("logical_step")
    )


def _validate_safetensors_file(path: Path) -> None:
    """Reject truncated or structurally invalid safetensors before publication."""

    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"checkpoint is not a complete safetensors file: {path.name}")
        header_size = int.from_bytes(prefix, "little", signed=False)
        if header_size <= 0 or header_size > size - 8:
            raise ValueError(f"checkpoint has an invalid safetensors header size: {path.name}")
        try:
            def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
                result: dict[str, Any] = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError(f"checkpoint has duplicate safetensors header keys: {path.name}")
                    result[key] = value
                return result

            header = json.loads(handle.read(header_size), object_pairs_hook=reject_duplicate_keys)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"checkpoint has an invalid safetensors header: {path.name}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"checkpoint has a non-object safetensors header: {path.name}")
    data_size = size - 8 - header_size
    intervals: list[tuple[int, int]] = []
    for key, value in header.items():
        if key == "__metadata__":
            if not isinstance(value, dict) or not all(isinstance(name, str) and isinstance(item, str) for name, item in value.items()):
                raise ValueError(f"checkpoint has invalid safetensors metadata: {path.name}")
            continue
        if not isinstance(value, dict) or not isinstance(value.get("dtype"), str) or not value["dtype"]:
            raise ValueError(f"checkpoint has an invalid tensor entry: {path.name}")
        shape = value.get("shape")
        if not isinstance(shape, list) or not all(isinstance(dimension, int) and not isinstance(dimension, bool) and dimension >= 0 for dimension in shape):
            raise ValueError(f"checkpoint has an invalid tensor entry: {path.name}")
        if not isinstance(value.get("data_offsets"), list) or len(value["data_offsets"]) != 2:
            raise ValueError(f"checkpoint has an invalid tensor entry: {path.name}")
        start, end = value["data_offsets"]
        if not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int) or isinstance(end, bool) or start < 0 or end < start or end > data_size:
            raise ValueError(f"checkpoint has tensor data outside the file: {path.name}")
        intervals.append((start, end))
    if not intervals:
        raise ValueError(f"checkpoint contains no tensors: {path.name}")
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError(f"checkpoint tensor data is not contiguous: {path.name}")
        cursor = end
    if cursor != data_size:
        raise ValueError(f"checkpoint tensor data is not contiguous: {path.name}")


def _runpod_remote_outputs(details: dict[str, Any], *, workspace: str, run_id: str, timeout_sec: int = 30) -> list[dict[str, Any]]:
    remote_outputs = f"{workspace.rstrip('/')}/runs/{run_id}/outputs"
    script = f"""
export PATH="/opt/conda/bin:/usr/local/bin:$PATH"
python - <<'PY'
import glob
import json
import os
import re

directory = {remote_outputs!r}
items = []
for path in sorted(glob.glob(os.path.join(directory, "*.safetensors"))):
    name = os.path.basename(path)
    stat = os.stat(path)
    matches = re.findall(r"(?:step|_)(\\d{{4,}})(?=\\.safetensors$|[-_.])", name)
    step = int(matches[-1]) if matches else None
    items.append({{"path": path, "name": name, "step": step, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}})
print(json.dumps(items))
PY
""".strip()
    result = subprocess.run([*_ssh_base(details), script], text=True, capture_output=True, check=False, timeout=timeout_sec)
    if result.returncode:
        raise ValueError(_redact_secret_text(result.stderr.strip() or result.stdout.strip() or "remote output listing failed"))
    data = json.loads(result.stdout or "[]")
    if not isinstance(data, list):
        raise ValueError("remote output listing did not return a list")
    return [item for item in data if isinstance(item, dict)]


def _runpod_remote_training_states(details: dict[str, Any], *, workspace: str, run_id: str, timeout_sec: int = 30) -> list[dict[str, Any]]:
    remote_outputs = f"{workspace.rstrip('/')}/runs/{run_id}/outputs"
    script = f"""
export PATH="/opt/conda/bin:/usr/local/bin:$PATH"
python - <<'PY'
import glob
import json
import os
import re

directory = {remote_outputs!r}
items = []
for state_dir in sorted(glob.glob(os.path.join(directory, "*-step*-state"))):
    if not os.path.isdir(state_dir) or os.path.islink(state_dir):
        continue
    name = os.path.basename(state_dir)
    match = re.search(r"-step(\\d{{4,}})-state$", name)
    if not match:
        continue
    files = []
    logical_step = None
    for root, dirs, names in os.walk(state_dir):
        dirs[:] = sorted(item for item in dirs if not os.path.islink(os.path.join(root, item)))
        for filename in sorted(names):
            path = os.path.join(root, filename)
            if os.path.islink(path) or not os.path.isfile(path):
                continue
            stat = os.stat(path)
            files.append({{"path": os.path.relpath(path, state_dir), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}})
    for marker_name in ("kura-state-info.json", "state-info.json"):
        marker_path = os.path.join(state_dir, marker_name)
        try:
            with open(marker_path, encoding="utf-8") as handle:
                marked = json.load(handle).get("logical_step")
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(marked, int) and not isinstance(marked, bool) and marked >= 0:
            logical_step = marked
            break
    items.append({{"path": state_dir, "name": name, "step": int(match.group(1)), "logical_step": logical_step, "files": files}})
print(json.dumps(items))
PY
""".strip()
    result = subprocess.run([*_ssh_base(details), script], text=True, capture_output=True, check=False, timeout=timeout_sec)
    if result.returncode:
        raise ValueError(_redact_secret_text(result.stderr.strip() or result.stdout.strip() or "remote training-state listing failed"))
    data = json.loads(result.stdout or "[]")
    if not isinstance(data, list):
        raise ValueError("remote training-state listing did not return a list")
    return [item for item in data if isinstance(item, dict)]


def _pull_remote_training_state_items(
    run_dir: Path,
    details: dict[str, Any],
    *,
    workspace: str,
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    host_workspace = run_dir.parent.parent
    published: list[dict[str, Any]] = []
    pending_root = run_dir / "recovery" / "training-state-pull"
    pending_root.mkdir(parents=True, exist_ok=True)
    try:
        run = _load_yaml(run_dir / "resolved" / "manifest.lock.yaml")
        continuation = resume_intent(run)
        contract = training_state_contract(run)
        retention_floor = training_state_retention_floor(
            host_workspace,
            run_dir.name,
            training_state_policy(run)["keep_generations"],
        )
    except (OSError, ValueError, yaml.YAMLError):
        continuation = None
        contract = {}
        retention_floor = None
    for item in items:
        name, remote_path, step, files = item.get("name"), item.get("path"), item.get("step"), item.get("files")
        if not isinstance(name, str) or Path(name).name != name or not isinstance(remote_path, str):
            continue
        if isinstance(step, bool) or not isinstance(step, int) or not isinstance(files, list) or not files:
            continue
        logical_step = item.get("logical_step")
        if isinstance(logical_step, bool) or not isinstance(logical_step, int):
            logical_step = step
            if continuation is not None and contract.get("native_progress") == "process_local":
                logical_step = continuation["source"]["observed_step"] + step
        if retention_floor is not None and logical_step < retention_floor:
            continue
        existing = training_state_at_step(host_workspace, run_dir.name, logical_step, verify_payload=False)
        if existing is not None:
            published.append(existing)
            continue
        total_size = sum(entry.get("size") for entry in files if isinstance(entry, dict) and isinstance(entry.get("size"), int))
        _ensure_free_bytes(pending_root, total_size + 1024**3, context="RunPod training-state pull")
        partial = pending_root / f".{name}.partial"
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir()
        try:
            result = _run_bounded(
                [
                    "scp", "-r",
                    *_ssh_transport_options(details),
                    "-P", str(details["port"]),
                    "-i", str(details["key"]),
                    f"root@{details['ip']}:{remote_path}/.",
                    str(partial),
                ],
                context="scp training-state pull",
                text=True,
                capture_output=True,
            )
            if result.returncode:
                raise ValueError(f"scp training-state pull failed with exit code {result.returncode}: {name}")
            refreshed = _runpod_remote_training_states(details, workspace=workspace, run_id=run_dir.name)
            after = next((candidate for candidate in refreshed if candidate.get("path") == remote_path), None)
            if not _same_remote_training_state_version(item, after):
                raise ValueError(f"remote training state changed while it was being copied: {name}")
            expected = {
                entry["path"]: entry["size"]
                for entry in files
                if isinstance(entry, dict) and isinstance(entry.get("path"), str) and isinstance(entry.get("size"), int)
            }
            actual = {
                path.relative_to(partial).as_posix(): path.stat().st_size
                for path in partial.rglob("*")
                if path.is_file() and not path.is_symlink()
            }
            if actual != expected:
                raise ValueError(f"local training-state inventory does not match the remote source: {name}")
            manifest = publish_training_state_candidate(host_workspace, run_dir, partial, step)
            if manifest is None:
                raise ValueError(f"remote training state is missing required backend files: {name}")
            published.append(manifest)
        finally:
            if partial.exists():
                shutil.rmtree(partial)
    return published


def _pull_remote_output_items(
    run_dir: Path,
    details: dict[str, Any],
    *,
    workspace: str,
    items: list[dict[str, Any]],
    force: bool = False,
) -> list[dict[str, Any]]:
    """Copy stable remote checkpoints without exposing partial local files."""

    if not items:
        return []
    config = _workspace_config()
    min_download_free = _configured_download_min_free_bytes(config)
    destination = run_dir / "outputs"
    destination.mkdir(parents=True, exist_ok=True)
    try:
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        status = {}
    previous_outputs = status.get("mirrored_outputs") if isinstance(status.get("mirrored_outputs"), list) else []
    previous_by_name = {item.get("name"): item for item in previous_outputs if isinstance(item, dict) and isinstance(item.get("name"), str)}
    pulled: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for item in items:
        name = item.get("name")
        size = item.get("size")
        if isinstance(name, str):
            local_path = destination / name
            previous = previous_by_name.get(name)
            metadata_matches = isinstance(previous, dict) and previous.get("remote_path") == item.get("path") and previous.get("remote_mtime_ns") == item.get("mtime_ns")
            if local_path.exists() and isinstance(size, int) and local_path.stat().st_size == size and metadata_matches and not force:
                try:
                    _validate_safetensors_file(local_path)
                except (OSError, ValueError):
                    pass
                else:
                    pulled.append({"name": name, "path": str(local_path.relative_to(run_dir)), "step": item.get("step"), "size": size, "remote_path": item.get("path"), "remote_mtime_ns": item.get("mtime_ns"), "skipped": True})
                    continue
        pending.append(item)
    pending_size = sum(item.get("size") for item in pending if isinstance(item.get("size"), int))
    if pending:
        _ensure_free_bytes(destination, max(min(10 * 1024**3, min_download_free), pending_size + 5 * 1024**3), context="RunPod output pull")
    for item in pending:
        name = item.get("name")
        remote_path = item.get("path")
        size = item.get("size")
        if not isinstance(name, str) or not isinstance(remote_path, str):
            continue
        local_path = destination / name
        partial_path = local_path.with_name(f".{local_path.name}.partial")
        partial_path.unlink(missing_ok=True)
        try:
            result = _run_bounded([
                "scp",
                *_ssh_transport_options(details),
                "-P", str(details["port"]),
                "-i", str(details["key"]),
                f"root@{details['ip']}:{remote_path}",
                str(partial_path),
            ], context="scp output pull")
            if result.returncode:
                raise ValueError(f"scp output pull failed with exit code {result.returncode}: {name}")
            refreshed = _runpod_remote_outputs(details, workspace=workspace, run_id=run_dir.name)
            after = next((candidate for candidate in refreshed if candidate.get("path") == remote_path), None)
            if not _same_remote_output_version(item, after) or not isinstance(size, int) or partial_path.stat().st_size != size:
                raise ValueError(f"remote checkpoint changed while it was being copied: {name}; wait for the save to finish and retry")
            _validate_safetensors_file(partial_path)
            os.replace(partial_path, local_path)
        finally:
            partial_path.unlink(missing_ok=True)
        published = {"name": name, "path": str(local_path.relative_to(run_dir)), "step": item.get("step"), "size": local_path.stat().st_size, "remote_path": remote_path, "remote_mtime_ns": item.get("mtime_ns"), "skipped": False}
        pulled.append(published)
        # Persist each verified publication immediately. A later item in the
        # same batch may fail, but that must not orphan an already-local file
        # from status.json and force a multi-GB transfer again next cycle.
        _record_pulled_outputs(run_dir, [published])
    return pulled


def _record_pulled_outputs(run_dir: Path, pulled: list[dict[str, Any]], *, emit_event: bool = True) -> None:
    def mutate(status: dict[str, Any]) -> None:
        status.pop("checkpoint_sync_error", None)
        if not pulled:
            return
        previous = status.get("mirrored_outputs") if isinstance(status.get("mirrored_outputs"), list) else []
        merged = {item.get("name"): item for item in previous if isinstance(item, dict) and isinstance(item.get("name"), str)}
        for item in pulled:
            if isinstance(item.get("name"), str):
                merged[item["name"]] = item
        status["mirrored_outputs"] = list(merged.values())
        status["mirrored_outputs_synced_at"] = datetime.now().astimezone().isoformat()

    _mutate_run_status(run_dir, mutate)
    copied = [item for item in pulled if not item.get("skipped")]
    if copied and emit_event:
        append_run_event(run_dir, {"event": "run_outputs_pulled", "timestamp": datetime.now().astimezone().isoformat(), "count": len(copied), "outputs": copied})


def _record_pulled_training_states(run_dir: Path, manifests: list[dict[str, Any]]) -> None:
    changed = False

    def mutate(status: dict[str, Any]) -> None:
        nonlocal changed
        status.pop("training_state_sync_error", None)
        if not manifests:
            return
        recoverable = [
            {
                "artifact_id": item["id"],
                "manifest_sha256": item["manifest_sha256"],
                "observed_step": item["observed_step"],
                "restoration_level": item["restoration_contract"]["level"],
            }
            for item in sorted(manifests, key=lambda value: int(value["observed_step"]))
        ]
        # The sync loop re-lists published states every poll; only a new or
        # retired state is a fact worth a status write and an event.
        if status.get("recoverable_training_states") == recoverable:
            return
        changed = True
        status["recoverable_training_states"] = recoverable
        status["training_states_synced_at"] = datetime.now().astimezone().isoformat()

    _mutate_run_status(run_dir, mutate)
    if changed:
        append_run_event(
            run_dir,
            {
                "event": "run_training_states_pulled",
                "timestamp": datetime.now().astimezone().isoformat(),
                "artifacts": [item["id"] for item in manifests],
            },
        )


def _directory_training_state_sync_enabled(run_dir: Path) -> bool:
    manifest_path = run_dir / "resolved" / "manifest.lock.yaml"
    try:
        run = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return False
    if not isinstance(run, dict):
        return False
    backend = run.get("backend") if isinstance(run.get("backend"), dict) else {}
    recovery = run.get("recovery")
    return (
        isinstance(recovery, dict)
        and "training_state" in recovery
        and backend.get("name") in {"ai-toolkit", "musubi-tuner", "sd-scripts"}
        and training_state_policy(run)["enabled"]
    )


def _try_sync_runpod_checkpoints(run_dir: Path, details: dict[str, Any], *, workspace: str, run_id: str) -> bool:
    """Best-effort checkpoint mirror used by the normal RunPod lifecycle."""

    try:
        with _run_operation_lock(run_dir, "checkpoint-pull", blocking=False):
            items = _runpod_remote_outputs(details, workspace=workspace, run_id=run_id)
            pulled = _pull_remote_output_items(run_dir, details, workspace=workspace, items=items)
            # Newly published files were recorded immediately so partial
            # success survives a later transfer failure. Merge skipped items
            # and clear stale errors without emitting the same event twice.
            _record_pulled_outputs(run_dir, pulled, emit_event=False)
            if _directory_training_state_sync_enabled(run_dir):
                try:
                    state_items = _runpod_remote_training_states(details, workspace=workspace, run_id=run_id)
                    training_states = _pull_remote_training_state_items(run_dir, details, workspace=workspace, items=state_items)
                    _record_pulled_training_states(run_dir, training_states)
                except (OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
                    state_error = _safe_error(exc)
                    try:
                        _mutate_run_status(
                            run_dir,
                            lambda status: status.__setitem__("training_state_sync_error", state_error),
                        )
                    except (OSError, json.JSONDecodeError):
                        pass
                    return False
        return True
    except _OperationBusy:
        return True
    except (OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        error_message = _safe_error(exc)
        try:
            _mutate_run_status(
                run_dir,
                lambda status: status.__setitem__("checkpoint_sync_error", error_message),
            )
        except (OSError, json.JSONDecodeError):
            pass
        return False


def cmd_run_pull(args: argparse.Namespace) -> int:
    try:
        run_dir = _run_path(args.run_id)
        if not shutil.which("runpodctl"):
            raise ValueError("runpodctl is not installed locally; install it before pulling outputs")
        workspace = _runpod_workspace_for_run(run_dir)
        details = _runpod_ssh_details(run_dir, timeout_sec=args.ssh_timeout, interval_sec=2)
        items = _runpod_remote_outputs(details, workspace=workspace, run_id=args.run_id)
        selected = _select_remote_outputs(items, step=args.step, since_step=args.since_step, all_outputs=args.all)
        if not selected:
            raise ValueError("no matching remote .safetensors outputs found")
        with _run_operation_lock(run_dir, "checkpoint-pull"):
            pulled = _pull_remote_output_items(run_dir, details, workspace=workspace, items=selected, force=args.force)
            _record_pulled_outputs(run_dir, pulled, emit_event=False)
        print(json.dumps({"run_id": args.run_id, "destination": str(run_dir / "outputs"), "pulled": pulled}, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        print(f"cannot pull run outputs: {_safe_error(exc)}", file=sys.stderr)
        return 1


def _runpod_upload_process(run_dir: Path, code_seed: str, timeout_sec: int) -> tuple[subprocess.Popen[str], str]:
    stage = _latest_runpod_stage(run_dir)
    archive = stage.get("archive")
    if not isinstance(archive, str):
        raise ValueError("latest RunPod stage has no upload archive")
    archive_path = run_dir / archive
    if not archive_path.is_file():
        raise ValueError(f"upload archive is missing: {archive_path}")
    if not shutil.which("runpodctl"):
        raise ValueError("runpodctl is not installed locally")
    process = subprocess.Popen(["runpodctl", "send", str(archive_path), "--code", code_seed], text=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    time.sleep(min(max(timeout_sec, 0), 1))
    if process.poll() is not None:
        raise ValueError(f"runpodctl send exited early with exit code {process.returncode}")
    return process, code_seed


def _wait_process(process: subprocess.Popen[str], timeout_sec: int) -> None:
    try:
        process.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        process.terminate()
        raise ValueError("runpodctl send did not complete before timeout") from exc
    if process.returncode:
        raise ValueError(f"runpodctl send failed with exit code {process.returncode}")


def remote_job_pid(run_dir: Path, pid_path: str, *, timeout_sec: int = 120) -> str | None:
    """The pid a started remote job left on its Pod, or None when the Pod has no such file.

    Raises ValueError when the Pod cannot be asked, so an unanswered question is
    never taken for "the job did not start".
    """
    details = _runpod_ssh_details(run_dir, timeout_sec=timeout_sec, interval_sec=5)
    script = f"if [ -f {shlex.quote(pid_path)} ]; then cat {shlex.quote(pid_path)}; else echo __KURA_NO_PID__; fi"
    try:
        result = subprocess.run([*_ssh_base(details), script], text=True, capture_output=True, check=False, timeout=60)
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"checking the remote job timed out after {exc.timeout} seconds") from exc
    if result.returncode:
        raise ValueError(_redact_secret_text(result.stderr.strip() or "checking the remote job failed"))
    value = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    return None if value in ("", "__KURA_NO_PID__") else value


def _runpod_ssh_details(run_dir: Path, *, timeout_sec: int, interval_sec: int = 10) -> dict[str, Any]:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    pod_id = status.get("pod_id")
    if not isinstance(pod_id, str):
        raise ValueError("run has no RunPod pod ID")
    deadline = time.monotonic() + timeout_sec
    last_error = ""
    while time.monotonic() < deadline:
        try:
            result = subprocess.run(["runpodctl", "pod", "get", pod_id], text=True, capture_output=True, check=False, timeout=max(interval_sec * 3, 1))
        except subprocess.TimeoutExpired:
            last_error = "runpodctl pod get timed out"
            time.sleep(interval_sec)
            continue
        if result.returncode:
            last_error = _redact_secret_text(result.stderr.strip() or result.stdout.strip())
        else:
            try:
                pod = json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                last_error = _safe_error(exc)
            else:
                ssh = pod.get("ssh") if isinstance(pod.get("ssh"), dict) else {}
                ip, port = ssh.get("ip"), ssh.get("port")
                key = ssh.get("ssh_key", {}).get("path") if isinstance(ssh.get("ssh_key"), dict) else None
                if isinstance(ip, str) and isinstance(port, int) and isinstance(key, str):
                    # When RunPod reports it, the container start time splits the
                    # Pod startup wait into allocation plus image pull, and boot to SSH.
                    started = pod.get("lastStartedAt")
                    return {"pod_id": pod_id, "ip": ip, "port": port, "key": key, "container_started_at": started if isinstance(started, str) and started else None}
                last_error = str(ssh.get("error") or "pod SSH is not ready")
        sleep_checking_stop(interval_sec)
    raise ValueError(f"pod SSH did not become ready before timeout: {last_error}")


# One persistent ssh connection per Pod lets every later ssh/scp skip its own
# handshake. Reuse is only an optimization: whenever the private socket
# directory is unusable, commands connect directly exactly as before.
_SSH_SOCKET_PATH_LIMIT = 100  # sun_path is 104-108 bytes; ssh adds a suffix while binding


def _ssh_control_dir(*, create: bool = False) -> Path | None:
    if os.name != "posix":
        return None
    directory = Path.home() / ".ssh" / "kura-mux"
    try:
        if create:
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = directory.lstat()
    except OSError:
        return None
    # The socket carries secrets on stdin, so only a private, real directory
    # owned by this user is acceptable.
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        return None
    return directory


def _ssh_control_path(details: dict[str, Any], *, create: bool = False) -> str | None:
    directory = _ssh_control_dir(create=create)
    if directory is None:
        return None
    path = str(directory / f"{details['ip']}_{details['port']}")
    return path if len(path) + 8 <= _SSH_SOCKET_PATH_LIMIT else None


def _ssh_transport_options(details: dict[str, Any]) -> list[str]:
    """Options every ssh/scp to a Pod shares; rides a live master when one exists."""
    options = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]
    control_path = _ssh_control_path(details)
    if control_path is not None:
        options += ["-o", f"ControlPath={control_path}", "-o", "ControlMaster=no"]
    return options


def _start_ssh_master(details: dict[str, Any]) -> None:
    """Open one persistent connection for later ssh/scp to reuse; best effort."""
    control_path = _ssh_control_path(details, create=True)
    if control_path is None:
        return
    target = f"root@{details['ip']}"
    port = str(details["port"])
    try:
        check = subprocess.run(
            ["ssh", "-o", f"ControlPath={control_path}", "-p", port, "-O", "check", target],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=10,
        )
        if check.returncode == 0:
            return
        # No live master answered, so a leftover socket (a crash, or an earlier
        # Pod on the same address) is stale; remove it or the new master
        # cannot bind and every client keeps hitting the dead socket.
        Path(control_path).unlink(missing_ok=True)
        subprocess.run(
            [
                "ssh", "-N", "-f",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=20",
                "-o", f"ControlPath={control_path}",
                "-o", "ControlMaster=yes",
                "-o", "ControlPersist=600",
                "-o", "ServerAliveInterval=30",
                "-o", "ServerAliveCountMax=2",
                "-i", str(details["key"]),
                "-p", port,
                target,
            ],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return


def _ssh_base(details: dict[str, Any]) -> list[str]:
    return [
        "ssh",
        *_ssh_transport_options(details),
        "-o", "ConnectTimeout=20",
        "-i", str(details["key"]),
        "-p", str(details["port"]),
        f"root@{details['ip']}",
    ]


def _scp_to_runpod(details: dict[str, Any], source: Path, target: str) -> None:
    command = [
        "scp",
        "-o", "BatchMode=yes",
        *_ssh_transport_options(details),
        "-o", "ConnectTimeout=20",
        "-P", str(details["port"]),
        "-i", str(details["key"]),
        str(source),
        f"root@{details['ip']}:{target}",
    ]
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=600)
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"scp upload timed out after {exc.timeout} seconds") from exc
    if result.returncode:
        detail = _redact_secret_text(result.stderr.strip() or result.stdout.strip())
        suffix = f": {detail}" if detail else ""
        raise ValueError(f"scp upload failed with exit code {result.returncode}{suffix}")


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_http_ready(endpoint: str, *, timeout_sec: int = 180) -> None:
    deadline = time.monotonic() + timeout_sec
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(endpoint.rstrip("/") + "/system_stats", timeout=5) as response:
                response.read()
            return
        except (OSError, urllib.error.URLError) as exc:
            last_error = _safe_error(exc)
        time.sleep(2)
    raise ValueError(f"ComfyUI endpoint did not become ready before timeout: {last_error}")


def _start_runpod_session_lease_guard(details: dict[str, Any], *, workspace: str, run_id: str, max_lease_sec: int = 12 * 3600) -> None:
    """Start the Pod-side lease fuse before any render setup or uploads."""

    if max_lease_sec <= 0:
        return
    pod_id = details.get("pod_id")
    pod_id_value = pod_id if isinstance(pod_id, str) else ""
    log_path = f"{workspace.rstrip('/')}/runs/{run_id}/logs/stdout.log"
    script = "\n".join([
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(str(PurePosixPath(log_path).parent))}",
        f"touch {shlex.quote(log_path)}",
        _runpod_lease_guard_shell(max_lease_sec=max_lease_sec, pod_id=pod_id_value, log_path=log_path),
    ])
    try:
        result = subprocess.run([*_ssh_base(details), script], text=True, capture_output=True, check=False, timeout=60)
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"remote lease guard setup timed out after {exc.timeout} seconds") from exc
    if result.returncode:
        detail = _redact_secret_text(result.stderr.strip() or result.stdout.strip() or "lease guard setup failed")
        raise ValueError(f"remote lease guard setup failed with exit code {result.returncode}: {detail}")


def _sync_runpod_remote_stdout(run_dir: Path, details: dict[str, Any], *, workspace: str, run_id: str, timeout_sec: int = 30) -> bool:
    """Mirror remote stdout progress into local run artifacts."""

    try:
        with _run_operation_lock(run_dir, "remote-log"):
            return _sync_runpod_remote_stdout_unlocked(run_dir, details, workspace=workspace, run_id=run_id, timeout_sec=timeout_sec)
    except (OSError, ValueError):
        return False


def _sync_runpod_remote_stdout_unlocked(run_dir: Path, details: dict[str, Any], *, workspace: str, run_id: str, timeout_sec: int = 30) -> bool:
    """Perform one remote-log sync while the per-run log lock is held."""

    status_path = run_dir / "status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    offset = status.get("remote_log_bytes")
    if not isinstance(offset, int) or offset < 0:
        offset = 0
    remote_log = f"{workspace.rstrip('/')}/runs/{run_id}/logs/stdout.log"
    marker = "__KURA_LOG_SIZE__:"
    script = f"""
set -u
log={shlex.quote(remote_log)}
offset={offset}
if [ -f "$log" ]; then
  size=$(wc -c < "$log" | tr -d ' ')
  if [ "$size" -lt "$offset" ]; then
    offset=0
  fi
  if [ "$size" -gt "$offset" ]; then
    tail -c +$((offset + 1)) "$log"
  fi
  printf '\\n{marker}%s\\n' "$size"
else
  printf '\\n{marker}0\\n'
fi
""".strip()
    try:
        result = subprocess.run([*_ssh_base(details), script], capture_output=True, check=False, timeout=timeout_sec)
    except (OSError, subprocess.TimeoutExpired):
        return False
    if result.returncode:
        return False
    sentinel = ("\n" + marker).encode("utf-8")
    if sentinel not in result.stdout:
        return False
    payload, suffix = result.stdout.rsplit(sentinel, 1)
    first = suffix.splitlines()[0] if suffix.splitlines() else b""
    try:
        remote_size = int(first.decode("ascii", errors="replace").strip())
    except ValueError:
        return False
    if remote_size < offset:
        offset = 0
    if payload.startswith(b"\n") and offset == remote_size:
        payload = payload[1:]
    if payload:
        log_path = run_dir / "logs" / "stdout.log"
        log_path.parent.mkdir(exist_ok=True)
        with log_path.open("ab") as handle:
            handle.write(payload)
    def mutate(current: dict[str, Any]) -> None:
        current["remote_log_bytes"] = remote_size
        current["remote_log_synced_at"] = datetime.now().astimezone().isoformat()
        _materialize_stdout_progress(run_dir, current, state=str(current.get("state") or "running"))

    try:
        _mutate_run_status(run_dir, mutate)
    except (OSError, json.JSONDecodeError):
        return False
    return True


def _try_sync_runpod_remote_stdout(run_dir: Path, *, ssh_timeout_sec: int = 10) -> bool:
    try:
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        realization_ref = status.get("last_realization")
        if not isinstance(realization_ref, str):
            return False
        realization = json.loads((run_dir / realization_ref).read_text(encoding="utf-8"))
        workspace = realization.get("request", {}).get("env", {}).get("KURA_WORKSPACE", "/workspace")
        if not isinstance(workspace, str):
            workspace = "/workspace"
        details = _runpod_ssh_details(run_dir, timeout_sec=ssh_timeout_sec, interval_sec=2)
        return _sync_runpod_remote_stdout(run_dir, details, workspace=workspace, run_id=run_dir.name)
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _read_runpod_remote_exit(details: dict[str, Any], *, workspace: str, run_id: str, timeout_sec: int = 30) -> dict[str, Any] | None:
    remote_dir = f"{workspace.rstrip('/')}/runs/{run_id}/realizations"
    script = f"""
set -u
dir={shlex.quote(remote_dir)}
latest=$(ls -1 "$dir"/remote-exit-*.json 2>/dev/null | sort | tail -n 1 || true)
if [ -n "$latest" ]; then
  cat "$latest"
fi
""".strip()
    try:
        result = subprocess.run([*_ssh_base(details), script], text=True, capture_output=True, check=False, timeout=timeout_sec)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode or not result.stdout.strip():
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _record_remote_exit_observation(run_dir: Path, exit_record: dict[str, Any]) -> None:
    """Append a remote-completion fact while local output recovery is pending."""

    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    exit_code = exit_record.get("exit_code")
    if not isinstance(exit_code, int):
        return
    if status.get("remote_exit_code") == exit_code and status.get("last_remote_exit_observation"):
        return
    realization_ref = status.get("last_realization")
    realization_id = Path(realization_ref).stem if isinstance(realization_ref, str) else "runpod"
    observed_at = datetime.now().astimezone().isoformat()
    compact = re.sub(r"[^0-9]", "", observed_at)[:20]
    observation_path = run_dir / "realizations" / f"{realization_id}.remote-exit-observed-{compact}.json"
    observation = {
        "event": "remote_exit_observed",
        "realization_id": realization_id,
        "observed_at": observed_at,
        "remote_state": "completed" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "remote_timestamp": exit_record.get("timestamp"),
        "recovery_required": True,
    }
    atomic_write_json(observation_path, _redact_secrets(record("remote_exit_observation", observation)))
    def mutate(current: dict[str, Any]) -> None:
        current.update({
            "remote_state": observation["remote_state"],
            "remote_exit_code": exit_code,
            "remote_ended": exit_record.get("timestamp"),
            "recovery_required": True,
            "last_remote_exit_observation": str(observation_path.relative_to(run_dir)),
        })

    _mutate_run_status(run_dir, mutate)
    append_run_event(run_dir, observation, best_effort=True)


def _try_observe_runpod_remote_exit(run_dir: Path, *, ssh_timeout_sec: int = 10) -> bool:
    try:
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        realization_ref = status.get("last_realization")
        if not isinstance(realization_ref, str):
            return False
        realization = json.loads((run_dir / realization_ref).read_text(encoding="utf-8"))
        workspace = realization.get("request", {}).get("env", {}).get("KURA_WORKSPACE", "/workspace")
        if not isinstance(workspace, str):
            workspace = "/workspace"
        details = _runpod_ssh_details(run_dir, timeout_sec=ssh_timeout_sec, interval_sec=2)
        exit_record = _read_runpod_remote_exit(details, workspace=workspace, run_id=run_dir.name, timeout_sec=30)
        if exit_record is None:
            return False
        _record_remote_exit_observation(run_dir, exit_record)
        return True
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _runpod_secret_env_payload(*, remote_notify: bool = False) -> str | None:
    lines: list[str] = []
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        quoted = shlex.quote(hf_token)
        lines.extend([
            f"export HF_TOKEN={quoted}",
            f"export HUGGINGFACE_HUB_TOKEN={quoted}",
        ])
    if remote_notify and os.environ.get("KURA_NTFY_TOPIC"):
        lines.append("export KURA_REMOTE_NOTIFY_NTFY=1")
        for key in ("KURA_NTFY_TOPIC", "KURA_NTFY_SERVER", "KURA_NTFY_TOKEN", "KURA_NTFY_PRIORITY"):
            value = os.environ.get(key)
            if value:
                lines.append(f"export {key}={shlex.quote(value)}")
    if not lines:
        return None
    return "\n".join([*lines, ""])


# Without its controller, a finished Pod waits this long at least for the
# outputs to be collected (docs/adr/runpod-unattended-completion.md).
UNATTENDED_MIN_WAIT_SEC = 2 * 3600


def _runpod_collected_mark(run_id: str) -> str:
    return f"/tmp/kura-jobs/{run_id}.collected"


def _runpod_collecting_mark(run_id: str) -> str:
    return f"/tmp/kura-jobs/{run_id}.collecting"


def _unattended_completion_shell(*, wait_sec: int | None, collected_mark: str, collecting_mark: str | None = None) -> str:
    """Shell that starts the post-training timer; ``wait_sec=None`` is the automatic wait."""
    if wait_sec == 0:
        return 'echo "[kura] unattended completion is off; only the maximum lease bounds this Pod" >> "$KURA_LOG_PATH" 2>&1 || true'
    fixed = "" if wait_sec is None else str(int(wait_sec))
    return f"""
kura_elapsed=$(( $(date +%s) - KURA_JOB_STARTED_EPOCH ))
kura_wait={fixed}
if [ -z "$kura_wait" ]; then
  if [ "$kura_elapsed" -gt {UNATTENDED_MIN_WAIT_SEC} ]; then kura_wait=$kura_elapsed; else kura_wait={UNATTENDED_MIN_WAIT_SEC}; fi
fi
echo "[kura] unattended completion: the Pod deletes itself in ${{kura_wait}}s unless Kura collects the outputs first" >> "$KURA_LOG_PATH" 2>&1 || true
(
  sleep "$kura_wait"
  # A controller that started collecting keeps the Pod; if it dies mid-way,
  # the maximum lease still deletes the Pod.
  while [ -e {shlex.quote(collecting_mark or collected_mark + ".collecting")} ] && [ ! -e {shlex.quote(collected_mark)} ]; do sleep 60; done
  if [ -e {shlex.quote(collected_mark)} ]; then exit 0; fi
  echo "[kura] unattended wait of ${{kura_wait}}s expired before the outputs were collected; deleting the Pod" >> "$KURA_LOG_PATH" 2>&1 || true
  kura_pod_self_delete "$KURA_LOG_PATH" || true
) </dev/null >/dev/null 2>&1 &
""".strip()


# Where the Pod keeps its lease deadline, in seconds since the epoch.
LEASE_DEADLINE_PATH = "/tmp/kura-lease-deadline"


def record_lease_deadline(run_dir: Path, deadline_epoch: int, *, reason: str, previous_epoch: int | None = None) -> Path:
    """Record the Pod's lease deadline in the run, so a follower can compare it with the training left."""
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    reference = status.get("last_realization")
    realization_id = Path(reference).stem if isinstance(reference, str) else "unrecorded"
    stamp = datetime.now().astimezone()
    path = run_dir / "realizations" / f"{realization_id}.lease-{stamp.strftime('%Y%m%d-%H%M%S-%f')}.json"
    atomic_write_json(path, record("lease", {
        "realization_id": realization_id, "at": stamp.isoformat(), "reason": reason, "deadline_epoch": deadline_epoch,
        "deadline": datetime.fromtimestamp(deadline_epoch).astimezone().isoformat(),
        **({"previous_deadline_epoch": previous_epoch} if previous_epoch is not None else {}),
    }))
    return path


def latest_lease_deadline(run_dir: Path, realization_id: str) -> int | None:
    """The lease deadline last recorded for this realization, in seconds since the epoch."""
    found = sorted((run_dir / "realizations").glob(f"{realization_id}.lease-*.json"))
    for path in reversed(found):
        try:
            recorded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        value = recorded.get("deadline_epoch") if recorded.get("kind") == "lease" else None
        if isinstance(value, int):
            return value
    return None


def _runpod_lease_guard_shell(*, max_lease_sec: int, pod_id: str, log_path: str) -> str:
    """The maximum lease: delete the Pod after ``max_lease_sec`` whatever the controller does."""
    if max_lease_sec <= 0:
        return ""
    pod_export = f"RUNPOD_POD_ID={shlex.quote(pod_id)}; export RUNPOD_POD_ID" if pod_id else ":"
    deadline_file = shlex.quote(LEASE_DEADLINE_PATH)
    # The deadline lives in a file, so `kura run lease` can move it; an unreadable
    # file falls back to the deadline set here.
    return f"""
{POD_SELF_DELETE_FUNCTION}
kura_lease_initial=$(( $(date +%s) + {int(max_lease_sec)} ))
# A guard started again on the same Pod never moves a deadline already set.
[ -s {deadline_file} ] || {{ echo "$kura_lease_initial" > {deadline_file}.tmp && mv {deadline_file}.tmp {deadline_file}; }}
(
  set +e
  {pod_export}
  while :; do
    kura_lease_deadline=$(cat {deadline_file} 2>/dev/null)
    # Anything but a plausible epoch (digits, at most 11 of them) falls back to the armed deadline.
    case "$kura_lease_deadline" in ''|*[!0-9]*|????????????*) kura_lease_deadline=$kura_lease_initial ;; esac
    [ "$(date +%s)" -ge "$kura_lease_deadline" ] && break
    sleep 30
  done
  mkdir -p "$(dirname {shlex.quote(log_path)})" || true
  echo "[kura] the maximum lease ended; deleting the Pod" >> {shlex.quote(log_path)} 2>&1 || true
  kura_pod_self_delete {shlex.quote(log_path)} || true
) </dev/null >/dev/null 2>&1 &
""".strip()


def _mark_runpod_outputs_collecting(details: dict[str, Any], run_id: str) -> None:
    """Tell the Pod-side timer that a download is underway.

    Without the mark the timer could delete the Pod mid-download, so a
    download that cannot place it does not start; the caller retries.
    """
    if not _touch_runpod_mark(details, _runpod_collecting_mark(run_id)):
        raise ValueError("cannot mark the RunPod outputs as being collected")


def _mark_runpod_outputs_collected(details: dict[str, Any], run_id: str) -> None:
    """Tell the Pod-side timer that the controller has the outputs; best effort."""
    _touch_runpod_mark(details, _runpod_collected_mark(run_id))


def _touch_runpod_mark(details: dict[str, Any], mark: str) -> bool:
    return _run_runpod_mark_command(details, f"mkdir -p /tmp/kura-jobs && touch {shlex.quote(mark)}")


def _clear_runpod_mark(details: dict[str, Any], mark: str) -> bool:
    return _run_runpod_mark_command(details, f"rm -f {shlex.quote(mark)}")


def _run_runpod_mark_command(details: dict[str, Any], command: str) -> bool:
    try:
        result = subprocess.run([*_ssh_base(details), command], text=True, capture_output=True, check=False, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _runpod_remote_job_script(
    *,
    workspace: str,
    run_id: str,
    realization_id: str,
    remote_secret_path: str,
    archive_name: str,
    remote_archive: str,
    cwd: str,
    command: str,
    write_roots: list[dict[str, str]] | None = None,
    transfer_manifest: str | None = None,
    transfer_manifest_sha256: str | None = None,
    command_env: dict[str, str] | None = None,
    unattended_wait_sec: int | None = None,
) -> str:
    declared_roots = write_roots or []
    # An SSH session does not inherit the Pod's create-time environment, so
    # the frozen command's env is exported here; Kura's own values below win.
    command_exports = []
    for key, value in sorted((command_env or {}).items()):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or not isinstance(value, str):
            raise ValueError(f"frozen command env has an invalid entry: {key!r}")
        command_exports.append(f"export {key}={shlex.quote(value)}")
    command_env_block = "\n".join(command_exports)
    if transfer_manifest is None:
        receive_inputs = (
            f'tar -xzf {shlex.quote(remote_archive)} -C "$KURA_WORKSPACE" >> "$KURA_LOG_PATH" 2>&1 || exit_code=$?'
        )
        inputs_postflight = ""
    else:
        # Selected-file transfers are verified before anything reaches the
        # workspace; a failure leaves exit_code non-zero so the trainer (and
        # its model acquisition) never starts.
        verifier = script_source("runpod_input_verify.py")
        media = shlex.quote(frozen_suffixes(KNOWN_MEDIA_SUFFIXES))
        receive_inputs = (
            f"export KURA_KNOWN_MEDIA_SUFFIXES={media}\n"
            f"python - {shlex.quote(remote_archive)} {shlex.quote(transfer_manifest)} "
            f"{shlex.quote(str(transfer_manifest_sha256))} "
            f">> \"$KURA_LOG_PATH\" 2>&1 <<'KURA_RUNPOD_INPUT_VERIFY' || exit_code=$?\n"
            f"{verifier}\n"
            "KURA_RUNPOD_INPUT_VERIFY\n"
            "inputs_verified=$exit_code"
        )
        # Runs after the trainer whatever its exit code; it only records drift.
        inputs_postflight = (
            'if [ "$inputs_verified" -eq 0 ]; then\n'
            "python - --postflight >> \"$KURA_LOG_PATH\" 2>&1 <<'KURA_RUNPOD_INPUT_POSTFLIGHT' || true\n"
            f"{verifier}\n"
            "KURA_RUNPOD_INPUT_POSTFLIGHT\n"
            "fi"
        )
    paths = validated_write_roots(
        {"env": {item["env"]: item["path"] for item in declared_roots}, "write_roots": declared_roots},
        workspace_path=workspace,
    )
    prepare_write_roots = "\n".join(
        f'mkdir -p "{path}" && test -w "{path}" || exit 1' for path in paths
    )
    return f"""
set -u
secret_file={shlex.quote(remote_secret_path)}
cleanup() {{
  rm -f "$secret_file"
}}
trap cleanup EXIT
if [ -f "$secret_file" ]; then
  . "$secret_file"
fi
{command_env_block}
export PATH="/opt/conda/bin:/usr/local/bin:$PATH"
export KURA_WORKSPACE={shlex.quote(workspace)}
export KURA_RUN_ID={shlex.quote(run_id)}
export KURA_REALIZATION_ID={shlex.quote(realization_id)}
export KURA_LOG_PATH={shlex.quote(workspace + '/runs/' + run_id + '/logs/stdout.log')}
export KURA_JOB_STARTED_EPOCH=$(date +%s)
{POD_SELF_DELETE_FUNCTION}
export HF_HOME="$KURA_WORKSPACE/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$KURA_WORKSPACE/runs/$KURA_RUN_ID/logs"
mkdir -p "$KURA_WORKSPACE/runs/$KURA_RUN_ID/outputs" "$KURA_WORKSPACE/runs/$KURA_RUN_ID/checkpoints" "$KURA_WORKSPACE/runs/$KURA_RUN_ID/samples" "$KURA_WORKSPACE/runs/$KURA_RUN_ID/metrics"
mkdir -p "$HF_HUB_CACHE" "$KURA_WORKSPACE/cache/models"
{prepare_write_roots}
case "$HF_HOME" in "$KURA_WORKSPACE"/*) ;; *) echo "[kura] HF_HOME must be under KURA_WORKSPACE before remote job start: $HF_HOME" >&2; exit 1 ;; esac
case "$HF_HUB_CACHE" in "$HF_HOME"/*) ;; *) echo "[kura] HF_HUB_CACHE must be under HF_HOME before remote job start: $HF_HUB_CACHE" >&2; exit 1 ;; esac
touch "$KURA_LOG_PATH"
echo "Kura controller uploaded {shlex.quote(archive_name)}" >> "$KURA_LOG_PATH"
read_cgroup_value() {{
  file="$1"
  key="${{2:-}}"
  if [ ! -r "$file" ]; then
    printf '%s' "unknown"
    return
  fi
  if [ -n "$key" ]; then
    value=$(awk -v key="$key" '$1 == key {{ print $2; exit }}' "$file" 2>/dev/null || true)
  else
    value=$(cat "$file" 2>/dev/null || true)
  fi
  printf '%s' "${{value:-unknown}}"
}}
read_memory_current() {{
  value=$(read_cgroup_value /sys/fs/cgroup/memory.current)
  if [ "$value" = unknown ]; then value=$(read_cgroup_value /sys/fs/cgroup/memory/memory.usage_in_bytes); fi
  printf '%s' "$value"
}}
read_memory_peak() {{
  value=$(read_cgroup_value /sys/fs/cgroup/memory.peak)
  if [ "$value" = unknown ]; then value=$(read_cgroup_value /sys/fs/cgroup/memory/memory.max_usage_in_bytes); fi
  printf '%s' "$value"
}}
read_memory_max() {{
  value=$(read_cgroup_value /sys/fs/cgroup/memory.max)
  if [ "$value" = unknown ]; then value=$(read_cgroup_value /sys/fs/cgroup/memory/memory.limit_in_bytes); fi
  printf '%s' "$value"
}}
read_oom_kill() {{
  value=$(read_cgroup_value /sys/fs/cgroup/memory.events oom_kill)
  if [ "$value" = unknown ]; then value=$(read_cgroup_value /sys/fs/cgroup/memory/memory.oom_control oom_kill); fi
  printf '%s' "$value"
}}
collect_runtime_diagnostics() {{
  phase="$1"
  {{
    echo "[kura] runtime diagnostics $phase"
    echo "[kura] cgroup memory.current=$(read_memory_current)"
    echo "[kura] cgroup memory.peak=$(read_memory_peak)"
    echo "[kura] cgroup memory.max=$(read_memory_max)"
    echo "[kura] cgroup memory.events"
    if [ -r /sys/fs/cgroup/memory.events ]; then
      sed 's/^/[kura]   /' /sys/fs/cgroup/memory.events
    elif [ -r /sys/fs/cgroup/memory/memory.oom_control ]; then
      sed 's/^/[kura]   /' /sys/fs/cgroup/memory/memory.oom_control
      echo "[kura]   failcnt $(read_cgroup_value /sys/fs/cgroup/memory/memory.failcnt)"
    else
      echo "[kura]   unavailable"
    fi
    echo "[kura] proc meminfo"
    if [ -r /proc/meminfo ]; then grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree):' /proc/meminfo | sed 's/^/[kura]   /'; else echo "[kura]   unavailable"; fi
    echo "[kura] cpu_count=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo unknown)"
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader,nounits 2>&1 | sed 's/^/[kura] gpu /'
    else
      echo "[kura] gpu nvidia-smi unavailable"
    fi
    df -Pk "$KURA_WORKSPACE" 2>&1 | sed 's/^/[kura] disk /'
  }} >> "$KURA_LOG_PATH" 2>&1
}}
export KURA_CGROUP_OOM_KILL_BEFORE=$(read_oom_kill)
collect_runtime_diagnostics before_backend
exit_code=0
{receive_inputs}
if [ "$exit_code" -eq 0 ]; then
  cd {shlex.quote(cwd)} || exit_code=$?
fi
if [ "$exit_code" -eq 0 ]; then
  {command} >> "$KURA_LOG_PATH" 2>&1
  exit_code=$?
fi
collect_runtime_diagnostics after_backend
{inputs_postflight}
export KURA_CGROUP_OOM_KILL_AFTER=$(read_oom_kill)
export KURA_CGROUP_MEMORY_PEAK=$(read_memory_peak)
export KURA_EXIT_CODE="$exit_code"
mkdir -p "$KURA_WORKSPACE/runs/$KURA_RUN_ID/realizations"
python - <<'PY'
import json, os, urllib.request
from datetime import datetime
run_id = os.environ["KURA_RUN_ID"]
workspace = os.environ.get("KURA_WORKSPACE", "/workspace")
now = datetime.now().astimezone().isoformat()
exit_code = int(os.environ.get("KURA_EXIT_CODE", "0"))
def optional_int(name):
    try:
        return int(os.environ.get(name, ""))
    except ValueError:
        return None
oom_before = optional_int("KURA_CGROUP_OOM_KILL_BEFORE")
oom_after = optional_int("KURA_CGROUP_OOM_KILL_AFTER")
memory_peak = optional_int("KURA_CGROUP_MEMORY_PEAK")
path = f"{{workspace}}/runs/{{run_id}}/realizations/remote-exit-{{now.replace(':', '').replace('.', '-')}}.json"
with open(path, "w", encoding="utf-8") as handle:
    json.dump({{
        "kind": "remote_exit",
        "schema_version": 1,
        "event": "remote_exit",
        "timestamp": now,
        "exit_code": exit_code,
        "diagnostics": {{
            "cgroup_oom_kill_before": oom_before,
            "cgroup_oom_kill_after": oom_after,
            "cgroup_oom_kill_delta": oom_after - oom_before if oom_before is not None and oom_after is not None else None,
            "cgroup_memory_peak_bytes": memory_peak,
        }},
    }}, handle, ensure_ascii=False, indent=2)
    handle.write("\\n")
if os.environ.get("KURA_REMOTE_NOTIFY_NTFY") == "1" and os.environ.get("KURA_NTFY_TOPIC"):
    try:
        server = os.environ.get("KURA_NTFY_SERVER", "https://ntfy.sh").rstrip("/")
        topic = os.environ["KURA_NTFY_TOPIC"].lstrip("/")
        title = f"Kura remote finished: {{run_id}}"
        body = f"Remote training finished with exit code {{exit_code}}. Pod may still be billing until the controller downloads outputs and stops it."
        headers = {{"Title": title, "Tags": "warning" if exit_code else "white_check_mark", "Priority": os.environ.get("KURA_NTFY_PRIORITY", "4")}}
        token = os.environ.get("KURA_NTFY_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {{token}}"
        request = urllib.request.Request(f"{{server}}/{{topic}}", data=body.encode("utf-8"), method="POST", headers=headers)
        with urllib.request.urlopen(request, timeout=20) as response:
            response.read()
    except Exception:
        pass
PY
{_unattended_completion_shell(wait_sec=unattended_wait_sec, collected_mark=_runpod_collected_mark(run_id), collecting_mark=_runpod_collecting_mark(run_id))}
exit "$exit_code"
""".strip()


def _prepare_remote_upload(run_dir: Path) -> dict[str, Any]:
    """Decide everything the upload needs before the first SSH action.

    Nothing has run on the Pod yet, so any failure here is a refusal: the
    caller stops the unused Pod instead of leaving it billing.
    """
    try:
        return _prepare_remote_upload_unchecked(run_dir)
    except TransferRefused:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError) as error:
        raise TransferRefused(f"remote job preparation failed before upload: {_safe_error(error)}") from error


def _prepare_remote_upload_unchecked(run_dir: Path) -> dict[str, Any]:
    stage = _latest_runpod_stage(run_dir)
    archive = stage.get("archive")
    archive_name = stage.get("archive_name")
    if not isinstance(archive, str) or not isinstance(archive_name, str):
        raise ValueError("latest RunPod stage has no upload archive")
    archive_path = run_dir / archive
    if not archive_path.is_file():
        raise TransferRefused(f"upload archive is missing: {archive_path}")
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    realization_ref = status.get("last_realization")
    if not isinstance(realization_ref, str):
        raise ValueError("run has no RunPod realization")
    realization = json.loads((run_dir / realization_ref).read_text(encoding="utf-8"))
    workspace = realization.get("request", {}).get("env", {}).get("KURA_WORKSPACE", "/workspace")
    run_id = run_dir.name
    cwd = realization.get("container_cwd")
    argv = realization.get("backend_command")
    if not isinstance(workspace, str) or not isinstance(cwd, str) or not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
        raise ValueError("latest realization has no runnable RunPod command")
    selected_files = stage.get("transfer") == "selected-files"
    remote_manifest_sha256: str | None = None
    pinned_manifest: Path | None = None
    if selected_files:
        # Refusals here happen before any upload, so the Pod is safe to stop.
        if workspace != "/workspace":
            # Frozen view links target /workspace/datasets/...
            raise TransferRefused("selected-file RunPod transfer requires KURA_WORKSPACE=/workspace")
        # Send only what launch pinned before the Pod existed, after proving the
        # stage still equals it; the Pod trusts nothing but the pinned digest.
        pin = realization.get("transfer") if isinstance(realization.get("transfer"), dict) else {}
        pinned, remote_manifest_sha256 = pin.get("pinned_manifest"), pin.get("manifest_sha256")
        if not isinstance(pinned, str) or not isinstance(remote_manifest_sha256, str) or pin.get("stage") != status.get("last_stage"):
            raise StagedTransferChanged("the realization has no pin for the current stage")
        locked_run = _load_yaml(run_dir / "resolved" / "manifest.lock.yaml")
        pinned_manifest = run_dir / pinned
        verify_pinned_transfer(
            run_dir.parent.parent, run_dir, locked_run, stage, pinned_manifest, remote_manifest_sha256,
        )
    frozen = _load_frozen_command(run_dir, _load_yaml(run_dir / "resolved" / "manifest.lock.yaml"))
    command_env = frozen.get("env") if isinstance(frozen.get("env"), dict) else {}
    return {
        "command_env": command_env,
        "stage": stage, "archive_path": archive_path, "archive_name": archive_name, "status": status,
        "realization": realization, "realization_ref": realization_ref, "workspace": workspace,
        "run_id": run_id, "cwd": cwd, "argv": argv, "selected_files": selected_files,
        "pinned_manifest": pinned_manifest, "remote_manifest_sha256": remote_manifest_sha256,
    }


def _runpod_run_over_ssh(run_dir: Path, *, ssh_timeout_sec: int, job_timeout_sec: int | None, remote_notify: bool = False, max_lease_sec: int = 12 * 3600, unattended_wait_sec: int | None = None, notify_channels: Any = None) -> int:
    prepared_upload = _prepare_remote_upload(run_dir)
    status = prepared_upload["status"]
    realization = prepared_upload["realization"]
    realization_ref = prepared_upload["realization_ref"]
    workspace = prepared_upload["workspace"]
    run_id = prepared_upload["run_id"]
    cwd = prepared_upload["cwd"]
    argv = prepared_upload["argv"]
    archive_path = prepared_upload["archive_path"]
    archive_name = prepared_upload["archive_name"]
    selected_files = prepared_upload["selected_files"]
    pinned_manifest = prepared_upload["pinned_manifest"]
    remote_manifest_sha256 = prepared_upload["remote_manifest_sha256"]
    command_env = prepared_upload["command_env"]
    realization_id = str(realization.get("id") or Path(realization_ref).stem)
    details = _runpod_ssh_details(run_dir, timeout_sec=ssh_timeout_sec, interval_sec=3)
    # The lease starts at first contact, so a Pod that fails before its job starts is still bounded.
    lease_pod_id = status.get("pod_id")
    _start_runpod_session_lease_guard({**details, "pod_id": lease_pod_id if isinstance(lease_pod_id, str) else ""},
                                      workspace=workspace, run_id=run_id, max_lease_sec=max_lease_sec)
    if max_lease_sec > 0:
        record_lease_deadline(run_dir, int(time.time()) + max_lease_sec, reason="armed")
    record_launch_phase(run_dir, realization_id, "ssh_ready", container_started_at=details.get("container_started_at"))
    _start_ssh_master(details)
    remote_dir = f"{workspace}/.kura-transfer/{run_id}" if selected_files else workspace
    remote_archive = f"{remote_dir}/{archive_name}"
    remote_manifest: str | None = None
    uploads = [(archive_path, remote_archive)]
    if pinned_manifest is not None:
        remote_manifest = f"{remote_dir}/transfer-manifest.json"
        uploads.append((pinned_manifest, remote_manifest))
    try:
        upload_bytes: int | None = sum(local.stat().st_size for local, _ in uploads)
    except OSError:
        upload_bytes = None
    record_launch_phase(run_dir, realization_id, "upload_started", bytes=upload_bytes)
    prepared = _run_bounded([*_ssh_base(details), f"mkdir -p {shlex.quote(remote_dir)}"], context="ssh workspace preparation")
    if prepared.returncode:
        raise ValueError(f"ssh workspace preparation failed with exit code {prepared.returncode}")
    for local, remote in uploads:
        scp = [
            "scp",
            *_ssh_transport_options(details),
            "-P", str(details["port"]),
            "-i", str(details["key"]),
            str(local),
            f"root@{details['ip']}:{remote}",
        ]
        uploaded = _run_bounded(scp, context="scp upload")
        if uploaded.returncode:
            raise ValueError(f"scp upload failed with exit code {uploaded.returncode}")
    record_launch_phase(run_dir, realization_id, "upload_finished")
    command = " ".join(shlex.quote(arg) for arg in argv)
    remote_secret_path = f"/tmp/kura-secrets/{run_id}.env"
    secret_payload = _runpod_secret_env_payload(remote_notify=remote_notify)
    if secret_payload is not None:
        install_secret_script = f"""
set -euo pipefail
umask 077
mkdir -p /tmp/kura-secrets
cat > {shlex.quote(remote_secret_path)}
chmod 600 {shlex.quote(remote_secret_path)}
""".strip()
        installed = _run_bounded([*_ssh_base(details), install_secret_script], input=secret_payload, text=True, context="ssh secret preparation")
        if installed.returncode:
            raise ValueError(f"ssh secret preparation failed with exit code {installed.returncode}")
    remote_job_script = _runpod_remote_job_script(
        workspace=workspace,
        run_id=run_id,
        realization_id=realization_id,
        remote_secret_path=remote_secret_path,
        archive_name=archive_name,
        remote_archive=remote_archive,
        cwd=cwd,
        command=command,
        write_roots=realization.get("write_roots"),
        transfer_manifest=remote_manifest,
        transfer_manifest_sha256=remote_manifest_sha256,
        command_env=command_env,
        unattended_wait_sec=unattended_wait_sec,
    )
    remote_job_path = f"/tmp/kura-jobs/{run_id}.sh"
    remote_controller_log = f"/tmp/kura-jobs/{run_id}.controller.log"
    remote_pid_path = f"/tmp/kura-jobs/{run_id}.{realization_id}.pid"
    start_script = f"""
set -euo pipefail
mkdir -p /tmp/kura-jobs
cat > {shlex.quote(remote_job_path)}
chmod 700 {shlex.quote(remote_job_path)}
nohup sh {shlex.quote(remote_job_path)} </dev/null >{shlex.quote(remote_controller_log)} 2>&1 &
echo $! > {shlex.quote(remote_pid_path)} || true
echo $!
""".strip()
    # Intent before effect: a crash after this line leaves the pid file on the
    # Pod to show whether the job started; it is never started a second time.
    intent_path = run_dir / "realizations" / f"{realization_id}.remote-job-intent.json"
    atomic_write_json(intent_path, record("remote_job_intent", {
        "realization_id": realization_id, "requested_at": datetime.now().astimezone().isoformat(),
        "job_path": remote_job_path, "pid_path": remote_pid_path,
    }))
    started = _run_bounded([*_ssh_base(details), start_script], input=remote_job_script, text=True, capture_output=True, context="remote job start")
    if started.returncode:
        detail = _redact_secret_text(started.stderr.strip() or started.stdout.strip() or "remote job start failed")
        raise ValueError(f"remote job start failed with exit code {started.returncode}: {detail}")
    remote_pid = started.stdout.strip().splitlines()[-1] if started.stdout.strip() else None
    remote_job_started_at = datetime.now().astimezone().isoformat()
    try:
        atomic_write_json(run_dir / "realizations" / f"{realization_id}.remote-job.json", record("remote_job", {
            "realization_id": realization_id, "started_at": remote_job_started_at, "pid": remote_pid,
            "job_path": remote_job_path, "pid_path": remote_pid_path, "intent": intent_path.name,
        }))
    except OSError as exc:
        # The job is running; following it matters more than this record, and
        # the launch phase below still marks it started for a later reattach.
        print(f"warning: could not record the started remote job: {exc}", file=sys.stderr)
    record_launch_phase(
        run_dir, realization_id, "remote_job_started", at=remote_job_started_at,
        max_lease_sec=max_lease_sec,
        unattended_wait="auto" if unattended_wait_sec is None else unattended_wait_sec,
    )
    try:
        def mutate(status: dict[str, Any]) -> None:
            status["remote_pid"] = remote_pid
            status["remote_job_started_at"] = remote_job_started_at

        _mutate_run_status(run_dir, mutate)
    except (OSError, json.JSONDecodeError):
        pass
    return _follow_runpod_job(run_dir, details, workspace=workspace, run_id=run_id, realization_id=realization_id, job_timeout_sec=job_timeout_sec, notify_channels=notify_channels)


# Time collection needs after training: the download and the Pod's deletion.
LEASE_MARGIN_SEC = 15 * 60
# The longest lease one change may set; longer ones are extended again later.
MAX_LEASE_CHANGE_SEC = 7 * 24 * 3600


def _training_left_sec(run_dir: Path) -> float | None:
    try:
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    step, total, per_step = status.get("last_step"), status.get("total_steps"), status.get("seconds_per_iter")
    if isinstance(step, int) and isinstance(total, int) and isinstance(per_step, (int, float)) and total > step:
        return (total - step) * float(per_step)
    return None


def lease_shortfall(run_dir: Path, realization_id: str, *, now: float | None = None) -> dict[str, Any] | None:
    """The training time left and the lease left, when training plus collection would outlast the lease."""
    training_left = _training_left_sec(run_dir)
    deadline = latest_lease_deadline(run_dir, realization_id)
    if training_left is None or not deadline:
        return None
    now = time.time() if now is None else now
    if now + training_left + LEASE_MARGIN_SEC <= deadline:
        return None
    return {"training_left_sec": training_left, "lease_left_sec": deadline - now, "deadline_epoch": deadline}


def _warn_if_lease_short(run_dir: Path, realization_id: str, run_id: str, notify_channels: Any) -> None:
    """Warn once per deadline when the lease looks too short; Kura never extends it itself."""
    shortfall = lease_shortfall(run_dir, realization_id)
    if shortfall is None:
        return
    marker = run_dir / "realizations" / f"{realization_id}.leasewarning-{shortfall['deadline_epoch']}.json"
    if marker.exists():
        return
    suggested_hours = int((shortfall["training_left_sec"] + LEASE_MARGIN_SEC) // 3600) + 2
    message = (
        f"training needs about {_format_remaining(shortfall['training_left_sec'])} more, but the Pod's lease ends in "
        f"{_format_remaining(shortfall['lease_left_sec'])}; extend it with `kura run lease {run_id} {suggested_hours}h`, "
        "or the Pod is deleted before training finishes"
    )
    atomic_write_json(marker, record("lease_warning", {"realization_id": realization_id, "at": datetime.now().astimezone().isoformat(), "message": message, **shortfall}))
    print(f"[kura] warning: {message}", file=sys.stderr, flush=True)
    from kura.notifications import notify

    try:
        notify(notify_channels, subject=f"Kura run lease too short: {run_id}", body=message, priority="4")
    except Exception as exc:  # a notification never fails the run
        print(f"[kura] notification failed: {exc}", file=sys.stderr)


def _follow_runpod_job(run_dir: Path, details: dict[str, Any], *, workspace: str, run_id: str, realization_id: str, job_timeout_sec: int | None, notify_channels: Any = None) -> int:
    """Follow a started remote job until its exit record appears; return its exit code."""
    deadline = time.monotonic() + job_timeout_sec if job_timeout_sec and job_timeout_sec > 0 else None
    # The Pod bills until the controller notices the job ended, so the cheap
    # remote-exit check runs every few seconds; the heavier log and
    # checkpoint sync keeps its own slower cadence.
    next_sync = 0.0
    sync_interval_sec = 20.0
    exit_check_interval_sec = 4.0
    next_exit_check = 0.0
    while True:
        check_stop()
        now = time.monotonic()
        if deadline is not None and now >= deadline:
            raise subprocess.TimeoutExpired(["runpod-remote-job", run_id], job_timeout_sec)
        if now >= next_sync:
            _sync_runpod_remote_stdout(run_dir, details, workspace=workspace, run_id=run_id, timeout_sec=30)
            _try_sync_runpod_checkpoints(run_dir, details, workspace=workspace, run_id=run_id)
            _warn_if_lease_short(run_dir, realization_id, run_id, notify_channels)
            next_sync = now + sync_interval_sec
        if now >= next_exit_check:
            exit_record = _read_runpod_remote_exit(details, workspace=workspace, run_id=run_id, timeout_sec=30)
            if exit_record is not None:
                record_launch_phase(run_dir, realization_id, "remote_exit_observed", remote_timestamp=exit_record.get("timestamp"))
                _sync_runpod_remote_stdout(run_dir, details, workspace=workspace, run_id=run_id, timeout_sec=30)
                _try_sync_runpod_checkpoints(run_dir, details, workspace=workspace, run_id=run_id)
                _record_remote_exit_observation(run_dir, exit_record)
                exit_code = exit_record.get("exit_code")
                return int(exit_code) if isinstance(exit_code, int) else 1
            next_exit_check = now + exit_check_interval_sec
        sleep_checking_stop(2)


def follow_running_runpod_job(run_dir: Path, *, ssh_timeout_sec: int, job_timeout_sec: int | None, notify_channels: Any = None) -> int:
    """Pick up a remote job an earlier controller started, and follow it to its exit.

    The job keeps running on the Pod when the controller that started it
    dies; following it again needs only the Pod and the job's exit record.
    """
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    realization_ref = status.get("last_realization")
    realization_id = Path(realization_ref).stem if isinstance(realization_ref, str) else ""
    workspace = _runpod_workspace_for_run(run_dir)
    details = _runpod_ssh_details(run_dir, timeout_sec=ssh_timeout_sec, interval_sec=3)
    _start_ssh_master(details)
    if realization_id:
        record_launch_phase(run_dir, realization_id, "controller_reattached")
    return _follow_runpod_job(run_dir, details, workspace=workspace, run_id=run_dir.name, realization_id=realization_id, job_timeout_sec=job_timeout_sec, notify_channels=notify_channels)


def download_with_retries(run_id: str, attempts: int, interval_sec: int) -> int:
    for _ in range(attempts):
        code = download_run(run_id, force=True)
        if code == 0:
            return 0
        sleep_checking_stop(interval_sec)
    return 1


def _download_with_retries(run_id: str, attempts: int, interval_sec: int) -> int:
    return download_with_retries(run_id, attempts, interval_sec)


def _remote_path_size(details: dict[str, Any], path: str, *, timeout_sec: int = 60) -> int | None:
    script = f"du -sb {shlex.quote(path)} 2>/dev/null | awk '{{print $1}}'"
    result = subprocess.run([*_ssh_base(details), script], text=True, capture_output=True, check=False, timeout=timeout_sec)
    if result.returncode:
        return None
    try:
        return int(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def _format_remaining(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours, rest = divmod(seconds, 3600)
    return f"{hours}h {rest // 60:02d}m"


def change_runpod_lease(run_dir: Path, duration_sec: int, *, yes: bool, input_stream: Any = None) -> int:
    """Move a running Pod's lease deadline to `duration_sec` from now, after the user confirms."""
    if duration_sec <= 0:
        raise ValueError("the new lease must be longer than zero; use `kura run stop` to end the Pod now")
    if duration_sec > MAX_LEASE_CHANGE_SEC:
        raise ValueError(f"a lease longer than {MAX_LEASE_CHANGE_SEC // 3600}h is refused; set a shorter one and extend it again later if needed")
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    reference = status.get("last_realization")
    realization = json.loads((run_dir / reference).read_text(encoding="utf-8")) if isinstance(reference, str) else {}
    pod = realization.get("pod") if isinstance(realization.get("pod"), dict) else None
    if realization.get("executor") != "runpod" or not isinstance(status.get("pod_id"), str) or pod is None:
        raise ValueError("this run has no RunPod Pod to change the lease of")
    if status.get("pod_stopped_at") or status.get("pod_missing_at"):
        raise ValueError("this run's Pod is already stopped")
    if realization.get("purpose") == "comfyui-render":
        # A render session Pod also carries a fixed timer set when it was created, which this cannot move.
        raise ValueError("a render Pod's lease cannot be changed yet; its creation-time timer still ends it")
    details = _runpod_ssh_details(run_dir, timeout_sec=120, interval_sec=5)
    path = shlex.quote(LEASE_DEADLINE_PATH)
    # The Pod's clock decides when the guard fires, so deadlines are computed there.
    read = subprocess.run([*_ssh_base(details), f"date +%s; cat {path} 2>/dev/null || true"], text=True, capture_output=True, check=False, timeout=60)
    if read.returncode:
        raise ValueError(_redact_secret_text(read.stderr.strip() or f"SSH to the Pod failed with exit code {read.returncode}"))
    lines = read.stdout.split()
    now = int(lines[0]) if lines and lines[0].isdigit() else int(time.time())
    current = int(lines[1]) if len(lines) > 1 and lines[1].isdigit() else None
    new = now + duration_sec
    price = pod.get("cost_per_h")
    print(f"Pod {status['pod_id']}: the lease now ends "
          + (f"in {_format_remaining(current - now)} ({datetime.fromtimestamp(current).astimezone():%Y-%m-%d %H:%M})" if current else "at an unknown time (the Pod predates changeable leases)")
          + f"; it would end in {_format_remaining(duration_sec)} ({datetime.fromtimestamp(new).astimezone():%Y-%m-%d %H:%M})."
          + (f" Hourly price: ${price:.3f}." if isinstance(price, (int, float)) else ""), file=sys.stderr)
    if current is None:
        raise ValueError("this Pod's lease cannot be changed; it was created before Kura kept the deadline in a file")
    training = _training_left_sec(run_dir)
    if training is not None and new < now + training + LEASE_MARGIN_SEC:
        print(f"note: training needs about {_format_remaining(training)} more, so this lease ends before it finishes", file=sys.stderr)
    if not yes:
        stream = input_stream or sys.stdin
        if not stream.isatty():
            raise ValueError("changing the lease changes what the Pod may bill; confirm in a terminal, or pass --yes only on the user's instruction")
        print("Change the lease? [y/N] ", end="", file=sys.stderr, flush=True)
        if stream.readline().strip().lower() != "y":
            print("the lease is unchanged", file=sys.stderr)
            return 1
    write = subprocess.run(
        [*_ssh_base(details), f"new=$(( $(date +%s) + {int(duration_sec)} )); echo $new > {path}.tmp && mv {path}.tmp {path} && cat {path}"],
        text=True, capture_output=True, check=False, timeout=60,
    )
    kept = write.stdout.strip()
    if write.returncode or not kept.isdigit():
        raise ValueError(_redact_secret_text(write.stderr.strip() or "the Pod did not keep the new lease"))
    new = int(kept)
    record_lease_deadline(run_dir, new, reason="changed by kura run lease", previous_epoch=current)
    print(f"the Pod now deletes itself at {datetime.fromtimestamp(new).astimezone():%Y-%m-%d %H:%M} at the latest", file=sys.stderr)
    return 0


def cmd_run_lease(args: argparse.Namespace) -> int:
    from kura.run_commands.plan import _parse_duration_seconds

    try:
        return change_runpod_lease(_run_path(args.run_id), _parse_duration_seconds(args.duration), yes=bool(args.yes))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"cannot change the lease: {_safe_error(exc)}", file=sys.stderr)
        return 1

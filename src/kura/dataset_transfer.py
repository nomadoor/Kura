"""Selected-file transfer inventory and archive for manifest-v2 remote runs.

The inventory is derived only from the bound input lock and projection that
freeze wrote (through ``load_frozen_dataset_handoff``), the run envelope, and
an approved Resume artifact. Every entry names one explicit namespace and one
workspace-relative destination; the remote side verifies this same inventory
and never infers a destination of its own.

Two properties are kept apart on purpose. The archive is reproducible (sorted
entries, fixed time/owner/mode) so equal inputs give equal bytes. Integrity is
proven separately by per-file SHA-256 against the compile-time lock and by the
archive's own SHA-256; reproducibility is never treated as that proof.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from kura.dataset_handoff import inspect_dataset_sources, load_frozen_dataset_handoff
from kura.fsio import atomic_write_json
from kura.training_artifacts import load_training_state, verify_training_state

TRANSFER_SCHEMA_VERSION = 1
_CHUNK = 1024 * 1024


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or "\x00" in value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or path.as_posix() != value
    ):
        raise ValueError(f"transfer path is not a safe relative path: {value!r}")
    return value


def _safe_destination(value: str, *, prefix: str) -> str:
    if not (value.startswith(prefix + "/")):
        raise ValueError(f"transfer destination is not a safe path under {prefix}: {value!r}")
    try:
        return _safe_relative(value)
    except ValueError as error:
        raise ValueError(f"transfer destination is not a safe path under {prefix}: {value!r}") from error


def _entry(
    namespace: str, destination: str, root: Path, relative: str, *, size: int | None, sha256: str | None,
) -> dict[str, Any]:
    """One archive member: read from ``root/relative`` without following links below root."""
    _safe_relative(relative)
    return {
        "namespace": namespace,
        "archive_name": f"{namespace}/{destination}",
        "destination": destination,
        "root": str(root),
        "relative": relative,
        "size": size,
        "sha256": sha256,
    }


def _open_beneath(root: str, relative: str) -> int:
    """Open ``root/relative`` read-only, refusing a symlink at every component below root."""
    if not hasattr(os, "O_NOFOLLOW") or os.open not in os.supports_dir_fd:
        raise ValueError("selected-file transfer requires no-follow directory-relative opens")
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        *parents, name = relative.split("/")
        for part in parents:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        return os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
    finally:
        os.close(directory)


def _envelope_entries(workspace: Path, run_dir: Path) -> list[dict[str, Any]]:
    root = run_dir.resolve(strict=True)
    prefix = f"runs/{run_dir.name}"
    relatives = ["run.yaml"]
    for path in sorted((root / "resolved").rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"run envelope contains a non-regular entry: {path.relative_to(root)}")
        if path.is_file():
            relatives.append(path.relative_to(root).as_posix())
    return [
        _entry(
            "envelope", _safe_destination(f"{prefix}/{relative}", prefix=prefix), root, relative,
            size=None, sha256=None,
        )
        for relative in relatives
    ]


def _source_entries(lock: dict[str, Any]) -> list[dict[str, Any]]:
    roots = {item["logical"]: Path(item["physical"]).resolve(strict=True) for item in lock["dataset_roots"]}
    entries = []
    for item in lock["files"]:
        destination = _safe_destination(item["source"], prefix="datasets")
        _, dataset_id, relative = destination.split("/", 2) if destination.count("/") >= 2 else ("", "", "")
        root = roots.get(f"datasets/{dataset_id}")
        if root is None or not relative:
            raise ValueError(f"transfer source is outside every frozen dataset root: {destination}")
        # A manifest may name an in-root symlink; transfer the bytes of the file
        # it resolves to, under the manifest's logical destination.
        resolved = (root / relative).resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError(f"transfer source escapes its dataset root: {destination}")
        entries.append(_entry(
            "source", destination, root, resolved.relative_to(root).as_posix(),
            size=item["stat"]["size"], sha256=item["sha256"],
        ))
    return entries


def _resume_entries(
    workspace: Path, run: dict[str, Any], *, verify: bool,
) -> tuple[list[dict[str, Any]], dict[str, str] | None]:
    continuation = run.get("continuation")
    if not isinstance(continuation, dict) or continuation.get("mode") != "resume":
        return [], None
    source = continuation.get("source") if isinstance(continuation.get("source"), dict) else {}
    artifact_id = source.get("artifact_id")
    manifest = load_training_state(workspace, str(artifact_id))
    if manifest["manifest_sha256"] != source.get("manifest_sha256"):
        raise ValueError(f"training-state manifest digest mismatch: {artifact_id}")
    if verify:
        verify_training_state(workspace, manifest)
    prefix = f"artifacts/training-state/{artifact_id}"
    root = (workspace / prefix).resolve(strict=True)
    entries = [_entry(
        "resume", f"{prefix}/manifest.json", root, "manifest.json",
        size=None, sha256=manifest["manifest_sha256"],
    )]
    for item in manifest["files"]:
        relative = f"payload/{item['path']}"
        destination = _safe_destination(f"{prefix}/{relative}", prefix=f"{prefix}/payload")
        entries.append(_entry(
            "resume", destination, root, relative, size=item["size"], sha256=item["sha256"],
        ))
    return entries, {"artifact_id": str(artifact_id), "manifest_sha256": manifest["manifest_sha256"]}


def build_transfer_inventory(
    workspace: Path, run_dir: Path, run: dict[str, Any], *, verify_resume: bool = True,
) -> dict[str, Any]:
    """Return the exact file inventory a remote run receives.

    ``verify_resume=False`` is for display-only sizing; staging and launch
    always verify the Resume artifact, and archiving re-hashes every file.
    """
    lock, _ = load_frozen_dataset_handoff(run_dir / "resolved")
    resume_entries, resume = _resume_entries(workspace, run, verify=verify_resume)
    entries = [
        *_envelope_entries(workspace, run_dir),
        *_source_entries(lock),
        *resume_entries,
    ]
    seen: dict[str, str] = {}
    for item in entries:
        folded = item["destination"].casefold()
        if folded in seen:
            raise ValueError(
                f"transfer destinations collide: {seen[folded]!r} and {item['destination']!r}"
            )
        seen[folded] = item["destination"]
    entries.sort(key=lambda item: item["archive_name"])
    return {
        "schema_version": TRANSFER_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "input_sha256": lock["input_sha256"],
        "projection_sha256": lock["projection_sha256"],
        "resume": resume,
        "entries": entries,
    }


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.type = tarfile.REGTYPE
    return info


def _padded(size: int) -> int:
    return -(-size // tarfile.BLOCKSIZE) * tarfile.BLOCKSIZE


def _tar_bytes(members: list[tuple[str, int]]) -> int:
    """The exact size of the reproducible PAX tar holding these (name, size) members."""
    total = sum(
        len(_tar_info(name, size).tobuf(tarfile.PAX_FORMAT, tarfile.ENCODING, "surrogateescape")) + _padded(size)
        for name, size in members
    )
    total += 2 * tarfile.BLOCKSIZE
    return -(-total // tarfile.RECORDSIZE) * tarfile.RECORDSIZE


def estimate_transfer(inventory: dict[str, Any]) -> dict[str, int]:
    """Return payload bytes and the exact uncompressed tar size of an inventory."""
    members = [
        (item["archive_name"], item["size"] if isinstance(item["size"], int) else _entry_stat(item).st_size)
        for item in inventory["entries"]
    ]
    payload = sum(size for _, size in members)
    archive = _tar_bytes(members)
    return {
        "payload_bytes": payload,
        "tar_bytes": archive,
        "local_stage_free_bytes": archive,
        # The uploaded tar and the verified extracted tree coexist until the
        # tar is removed after publication.
        "remote_peak_bytes": archive + payload,
    }


class _HashingReader:
    def __init__(self, handle: BinaryIO) -> None:
        self._handle = handle
        self.digest = hashlib.sha256()
        self.read_bytes = 0

    def read(self, size: int = -1) -> bytes:
        data = self._handle.read(size)
        self.digest.update(data)
        self.read_bytes += len(data)
        return data


class _HashingWriter:
    def __init__(self, handle: BinaryIO) -> None:
        self._handle = handle
        self.digest = hashlib.sha256()
        self.written = 0

    def write(self, data: bytes) -> int:
        self.digest.update(data)
        self.written += len(data)
        return self._handle.write(data)

    def tell(self) -> int:
        return self.written


def _stat_key(observed: os.stat_result) -> tuple[int, int, int]:
    return observed.st_size, observed.st_mtime_ns, observed.st_ctime_ns


def _add_entry(archive: tarfile.TarFile, item: dict[str, Any]) -> dict[str, Any]:
    try:
        descriptor = _open_beneath(item["root"], item["relative"])
    except OSError as error:
        raise ValueError(f"cannot read transfer source {item['destination']}: {error}") from error
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"transfer source is not a regular file: {item['destination']}")
        if isinstance(item["size"], int) and before.st_size != item["size"]:
            raise ValueError(f"transfer source size differs from the lock: {item['destination']}")
        reader = _HashingReader(handle)
        archive.addfile(_tar_info(item["archive_name"], before.st_size), reader)
        after = os.fstat(handle.fileno())
    digest = reader.digest.hexdigest()
    if _stat_key(before) != _stat_key(after) or reader.read_bytes != before.st_size:
        raise ValueError(f"transfer source changed while it was archived: {item['destination']}")
    if item["sha256"] is not None and digest != item["sha256"]:
        raise ValueError(f"transfer source content differs from the lock: {item['destination']}")
    return {
        "namespace": item["namespace"],
        "archive_name": item["archive_name"],
        "destination": item["destination"],
        "size": before.st_size,
        "sha256": digest,
    }


def write_transfer_archive(
    workspace: Path, run_dir: Path, inventory: dict[str, Any], archive_path: Path,
) -> dict[str, Any]:
    """Write the reproducible tar and return the proven transfer record.

    Source stat is checked immediately before writing; while writing, every
    file's SHA-256 must equal the compile-time lock and its stat must not
    change between the start and end of its read.
    """
    lock, _ = load_frozen_dataset_handoff(run_dir / "resolved")
    if lock["input_sha256"] != inventory["input_sha256"]:
        raise ValueError("transfer inventory belongs to a different compile; rebuild it")
    changes = inspect_dataset_sources(workspace, lock)
    if changes:
        raise ValueError("compiled dataset input changed before transfer; recompile the run: " + "; ".join(changes[:5]))
    temporary = archive_path.with_name(f".{archive_path.name}.partial")
    entries: list[dict[str, Any]] = []
    try:
        with temporary.open("wb") as raw:
            writer = _HashingWriter(raw)
            with tarfile.open(fileobj=writer, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for item in inventory["entries"]:
                    entries.append(_add_entry(archive, item))
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, archive_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "schema_version": TRANSFER_SCHEMA_VERSION,
        "run_id": inventory["run_id"],
        "input_sha256": inventory["input_sha256"],
        "projection_sha256": inventory["projection_sha256"],
        "resume": inventory["resume"],
        "archive_sha256": writer.digest.hexdigest(),
        "tar_bytes": writer.written,
        "payload_bytes": sum(item["size"] for item in entries),
        "local_source_stat_verification": "matched",
        "entries": entries,
    }


_ENTRY_KEYS = ("namespace", "archive_name", "destination", "size", "sha256")


def _proven_entry(item: dict[str, Any]) -> dict[str, Any]:
    """The entry a correct stage records for this inventory item."""
    return {
        "namespace": item["namespace"],
        "archive_name": item["archive_name"],
        "destination": item["destination"],
        "size": item["size"] if isinstance(item["size"], int) else _entry_stat(item).st_size,
        "sha256": item["sha256"] if item["sha256"] is not None else _sha256_entry(item),
    }


def verify_stage_matches_compile(workspace: Path, run_dir: Path, run: dict[str, Any], record: dict[str, Any]) -> None:
    """Allow a remote launch only when the staged transfer is exactly this compile.

    Every recorded fact is recomputed from the current compile and compared
    exactly, and the tar on disk is read once to prove each member's name,
    type, size, and content and the archive digest, all before a Pod exists.
    """
    current = build_transfer_inventory(workspace, run_dir, run)
    entries = [_proven_entry(item) for item in current["entries"]]
    payload = sum(item["size"] for item in entries)
    tar_bytes = _tar_bytes([(item["archive_name"], item["size"]) for item in entries])
    expected = {
        "executor": "runpod",
        "storage_mode": "upload",
        "transfer": "selected-files",
        "schema_version": TRANSFER_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "input_sha256": current["input_sha256"],
        "projection_sha256": current["projection_sha256"],
        "resume": current["resume"],
        "local_source_stat_verification": "matched",
        "entries": entries,
        "payload_bytes": payload,
        "total_bytes": payload,
        "tar_bytes": tar_bytes,
        "remote_peak_bytes": tar_bytes + payload,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"staged transfer {key} differs from the compiled run; stage it again")
    _verify_staged_files(run_dir, record)


_MANIFEST_KEYS = (
    "schema_version", "run_id", "input_sha256", "projection_sha256",
    "resume", "archive_sha256", "tar_bytes", "payload_bytes", "entries",
)


def _staged_file(run_dir: Path, value: Any, expected_name: str) -> Path:
    if value != f"transfer/{expected_name}" or "/" in expected_name:
        raise ValueError("staged transfer names an unexpected file; stage it again")
    path = run_dir / "transfer" / expected_name
    try:
        observed = path.lstat()
    except OSError as error:
        raise ValueError(f"staged transfer file is missing: {value}; stage it again") from error
    if not stat.S_ISREG(observed.st_mode):
        raise ValueError(f"staged transfer file is not a regular file: {value}; stage it again")
    return path


def _verify_staged_files(run_dir: Path, record: dict[str, Any]) -> None:
    name = record.get("archive_name")
    if not isinstance(name, str) or name != f"kura-upload-{run_dir.name}.tar":
        raise ValueError("staged transfer archive name is invalid; stage it again")
    archive = _staged_file(run_dir, record.get("archive"), name)
    if archive.stat().st_size != record.get("tar_bytes"):
        raise ValueError("staged transfer archive size differs from its record; stage it again")
    entries = record["entries"]
    with os.fdopen(_open_beneath(str(archive.parent), archive.name), "rb") as handle:
        reader = _HashingReader(handle)
        try:
            with tarfile.open(fileobj=reader, mode="r|") as stream:
                members = iter(stream)
                for item in entries:
                    member = next(members, None)
                    if (
                        member is None
                        or member.name != item["archive_name"]
                        or not member.isreg()
                        or member.size != item["size"]
                        or (member.mtime, member.uid, member.gid, member.mode) != (0, 0, 0, 0o644)
                    ):
                        raise ValueError(f"staged transfer archive member differs: {item['archive_name']}; stage it again")
                    extracted = stream.extractfile(member)
                    digest = hashlib.sha256()
                    for chunk in iter(lambda: extracted.read(_CHUNK), b""):
                        digest.update(chunk)
                    if digest.hexdigest() != item["sha256"]:
                        raise ValueError(f"staged transfer archive member content differs: {item['archive_name']}; stage it again")
                if next(members, None) is not None:
                    raise ValueError("staged transfer archive has extra members; stage it again")
        except tarfile.TarError as error:
            raise ValueError("staged transfer archive is not a readable tar; stage it again") from error
        while reader.read(_CHUNK):
            pass
    if reader.digest.hexdigest() != record.get("archive_sha256"):
        raise ValueError("staged transfer archive content differs from its record; stage it again")
    manifest = _staged_file(run_dir, record.get("manifest"), f"kura-upload-{run_dir.name}.manifest.json")
    try:
        written = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("staged transfer manifest is unreadable; stage it again") from error
    if written != {key: record.get(key) for key in _MANIFEST_KEYS}:
        raise ValueError("staged transfer manifest differs from its record; stage it again")


def _sha256_entry(item: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    with os.fdopen(_open_beneath(item["root"], item["relative"]), "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entry_stat(item: dict[str, Any]) -> os.stat_result:
    descriptor = _open_beneath(item["root"], item["relative"])
    try:
        return os.fstat(descriptor)
    finally:
        os.close(descriptor)


def write_transfer_manifest(path: Path, record: dict[str, Any]) -> None:
    """Write the inventory the Pod verifies, bound to the archive by its digest."""
    atomic_write_json(path, {key: record[key] for key in _MANIFEST_KEYS})

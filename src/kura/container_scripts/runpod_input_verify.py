"""Verify a selected-file transfer on the Pod before any model acquisition.

Usage: runpod_input_verify.py <archive> <manifest> <manifest-sha256>

The controller passes the manifest's SHA-256 inside the job script it sends,
so the manifest itself is bound before anything else is trusted.

The archive is streamed once into a fresh staging directory. Each member must
be exactly the manifest entry at the same position (name, regular type, size,
fixed metadata, SHA-256), and the whole archive must match the manifest's size
and digest. The staged run envelope's input lock must match the manifest's
input and projection digests. Only then are the verified trees renamed into
the workspace, the frozen dataset views are created from that lock, and their
exact link and generated-file inventory is checked. A realization record is
written on success and on failure; failure exits non-zero so the job never
reaches the trainer.
"""

import hashlib
import json
import os
import shutil
import stat
import sys
import tarfile
from datetime import datetime
from pathlib import Path, PurePosixPath

CHUNK = 1024 * 1024
CONTAINER_WORKSPACE = "/workspace/"


class TransferError(Exception):
    pass


def safe_relative(value):
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\x00" in value
        or value.startswith("/")
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise TransferError(f"unsafe transfer path: {value!r}")
    return value


def digest_json(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


class HashingReader:
    def __init__(self, handle):
        self.handle = handle
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size=-1):
        data = self.handle.read(size)
        self.digest.update(data)
        self.size += len(data)
        return data


def write_new_file(path, reader, expected_size):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    digest = hashlib.sha256()
    written = 0
    with os.fdopen(descriptor, "wb") as handle:
        for chunk in iter(lambda: reader.read(CHUNK), b""):
            digest.update(chunk)
            written += len(chunk)
            handle.write(chunk)
    if written != expected_size:
        raise TransferError(f"member size differs while extracting: {path}")
    return digest.hexdigest()


def extract_verified(archive_path, manifest, staging):
    entries = manifest["entries"]
    with open(archive_path, "rb") as raw:
        reader = HashingReader(raw)
        try:
            with tarfile.open(fileobj=reader, mode="r|") as stream:
                members = iter(stream)
                for entry in entries:
                    member = next(members, None)
                    if (
                        member is None
                        or member.name != entry["archive_name"]
                        or member.name != f"{entry['namespace']}/{entry['destination']}"
                        or not member.isreg()
                        or member.size != entry["size"]
                        or (member.mtime, member.uid, member.gid, member.mode) != (0, 0, 0, 0o644)
                    ):
                        raise TransferError(f"archive member differs from the manifest: {entry['archive_name']}")
                    destination = safe_relative(entry["destination"])
                    digest = write_new_file(staging / destination, stream.extractfile(member), entry["size"])
                    if digest != entry["sha256"]:
                        raise TransferError(f"archive member content differs from the manifest: {destination}")
                if next(members, None) is not None:
                    raise TransferError("archive has members the manifest does not list")
        except tarfile.TarError as error:
            raise TransferError(f"archive is not a readable tar: {error}") from error
        while reader.read(CHUNK):
            pass
    if reader.size != manifest["tar_bytes"] or reader.digest.hexdigest() != manifest["archive_sha256"]:
        raise TransferError("archive size or digest differs from the manifest")


def check_envelope(staging, manifest, run_id):
    resolved = staging / "runs" / run_id / "resolved"
    lock = json.loads((resolved / "dataset-input.lock.json").read_text(encoding="utf-8"))
    report = json.loads((resolved / "dataset-projection.lock.json").read_text(encoding="utf-8"))
    if (
        lock.get("input_sha256") != manifest["input_sha256"]
        or lock.get("projection_sha256") != manifest["projection_sha256"]
        or digest_json(report) != manifest["projection_sha256"]
        or lock.get("run_id") != run_id
    ):
        raise TransferError("staged input lock differs from the transfer manifest")
    return lock


def publish(staging, workspace, run_id):
    """Rename the verified trees into place as one unit; return an undo function.

    Every target is checked before the first rename, and a failure midway
    moves already-published trees back, so the workspace either has all
    verified inputs or none of them and the same Pod can retry.
    """
    moves = [
        (staging / "runs" / run_id / "run.yaml", workspace / "runs" / run_id / "run.yaml"),
        (staging / "runs" / run_id / "resolved", workspace / "runs" / run_id / "resolved"),
        (staging / "datasets", workspace / "datasets"),
        (staging / "artifacts", workspace / "artifacts"),
    ]
    staged_runs = {path.name for path in (staging / "runs").iterdir()} if (staging / "runs").is_dir() else set()
    if staged_runs - {run_id} or {path.name for path in staging.iterdir()} - {"runs", "datasets", "artifacts"}:
        raise TransferError("staging contains a tree outside the declared namespaces")
    moves = [(source, target) for source, target in moves if source.exists()]
    for _, target in moves:
        if target.exists() or target.is_symlink():
            raise TransferError(f"transfer target already exists: {target.relative_to(workspace)}")
    moved = []

    def undo():
        for source, target in reversed(moved):
            os.rename(target, source)
        moved.clear()

    try:
        for source, target in moves:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.rename(source, target)
            moved.append((source, target))
    except OSError:
        undo()
        raise
    return undo


def materialize_views(workspace, lock, run_id, created):
    """Create each frozen view; append the highest directory each view creates to ``created``."""
    datasets = workspace / "datasets"
    view_prefix = f"runs/{run_id}/cache/dataset-view/"
    links = 0
    generated = 0
    for view in lock["views"]:
        root_relative = safe_relative(view["root"])
        if not root_relative.startswith(view_prefix):
            raise TransferError(f"view root is outside the run-owned view area: {root_relative}")
        root = workspace / root_relative
        if root.exists() or root.is_symlink():
            raise TransferError(f"view root already exists: {root_relative}")
        # Record the highest directory this attempt creates, so rollback
        # removes the intermediate directories too.
        top = root
        while not top.parent.exists():
            top = top.parent
        root.mkdir(parents=True)
        created.append(top)
        expected_links = {}
        for link in view["links"]:
            path = safe_relative(link["path"])
            target = link["target"]
            if not path.startswith(root_relative + "/") or not target.startswith(CONTAINER_WORKSPACE + "datasets/"):
                raise TransferError(f"view link is outside its view or dataset area: {path}")
            local_target = workspace / safe_relative(target[len(CONTAINER_WORKSPACE):])
            resolved = Path(os.path.realpath(local_target))
            if not resolved.is_relative_to(Path(os.path.realpath(datasets))) or not resolved.is_file():
                raise TransferError(f"view link target is not a transferred dataset file: {path}")
            (workspace / path).parent.mkdir(parents=True, exist_ok=True)
            (workspace / path).symlink_to(target)
            expected_links[path] = target
        expected_files = {}
        for item in [*view.get("files", []), *view.get("native_files", [])]:
            path = safe_relative(item["path"])
            if not path.startswith(root_relative + "/"):
                raise TransferError(f"generated view file is outside its view: {path}")
            data = item["text"].encode("utf-8")
            (workspace / path).parent.mkdir(parents=True, exist_ok=True)
            with os.fdopen(os.open(workspace / path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644), "wb") as handle:
                handle.write(data)
            expected_files[path] = data
        links += len(expected_links)
        generated += len(expected_files)
    changes = view_changes(workspace, lock)
    if changes:
        raise TransferError("view inventory differs from the frozen lock: " + "; ".join(changes[:5]))
    return links, generated


def media_suffixes():
    """Kura's frozen media vocabulary, passed by the controller; required."""
    try:
        values = json.loads(os.environ["KURA_KNOWN_MEDIA_SUFFIXES"])
    except (KeyError, ValueError) as error:
        raise TransferError(f"KURA_KNOWN_MEDIA_SUFFIXES is missing or invalid: {error}") from error
    if not isinstance(values, list) or not values or not all(isinstance(item, str) for item in values):
        raise TransferError("KURA_KNOWN_MEDIA_SUFFIXES must be a non-empty suffix list")
    return frozenset(values)


def view_changes(workspace, lock):
    """Compare every view's exact link and generated-file inventory with the lock.

    Trainers may write caches beside view files, so only an unexpected media
    file counts as drift among regular files.
    """
    suffixes = media_suffixes()
    changes = []
    for view in lock["views"]:
        root_relative = safe_relative(view["root"])
        root = workspace / root_relative
        expected_links = {link["path"]: link["target"] for link in view["links"]}
        expected_files = {
            item["path"]: item["text"].encode("utf-8")
            for item in [*view.get("files", []), *view.get("native_files", [])]
        }
        actual_links = {}
        actual_files = set()
        if not root.is_dir() or root.is_symlink():
            changes.append(f"missing view root: {root_relative}")
            continue
        for path in root.rglob("*"):
            relative = path.relative_to(workspace).as_posix()
            if path.is_symlink():
                actual_links[relative] = os.readlink(path)
            elif path.is_file():
                actual_files.add(relative)
            elif not path.is_dir():
                changes.append(f"unexpected view entry: {relative}")
        for path in sorted(set(expected_links) | set(actual_links)):
            if expected_links.get(path) != actual_links.get(path):
                changes.append(f"view link differs: {path}")
        for path in sorted(set(expected_files) - actual_files):
            changes.append(f"missing generated view file: {path}")
        for path in sorted(actual_files - set(expected_files)):
            if Path(path).suffix.lower() in suffixes:
                changes.append(f"unexpected regular media in view: {path}")
        for path in sorted(set(expected_files) & actual_files):
            if (workspace / path).read_bytes() != expected_files[path]:
                changes.append(f"generated view file differs: {path}")
    return changes


def source_state(path):
    """A transferred source's identity for drift checks; a non-regular file never matches."""
    observed = path.lstat()
    if not stat.S_ISREG(observed.st_mode):
        return {"type": "not-regular"}
    return {"size": observed.st_size, "mtime_ns": observed.st_mtime_ns, "ctime_ns": observed.st_ctime_ns}


def source_baseline(workspace, manifest):
    return {
        entry["destination"]: source_state(workspace / entry["destination"])
        for entry in manifest["entries"]
        if entry["namespace"] == "source"
    }


def postflight():
    """After the trainer: re-check views and transferred sources; never fails the run."""
    workspace = Path(os.environ["KURA_WORKSPACE"])
    run_id = os.environ["KURA_RUN_ID"]
    realization_id = os.environ["KURA_REALIZATION_ID"]
    realizations = workspace / "runs" / run_id / "realizations"
    record = {"schema_version": 1, "realization_id": realization_id, "observed_at": datetime.now().astimezone().isoformat()}
    try:
        verified = json.loads((realizations / f"{realization_id}.runpod-input.json").read_text(encoding="utf-8"))
        if verified.get("status") != "verified":
            raise TransferError("inputs were never verified for this realization")
        lock = json.loads((workspace / "runs" / run_id / "resolved" / "dataset-input.lock.json").read_text(encoding="utf-8"))
        source_changes = []
        for destination, baseline in sorted(verified["source_baseline"].items()):
            try:
                observed = source_state(workspace / safe_relative(destination))
            except OSError:
                source_changes.append(f"missing transferred source: {destination}")
                continue
            if observed != baseline:
                source_changes.append(f"transferred source changed: {destination}")
        link_changes = view_changes(workspace, lock)
        record.update({
            "status": "changed" if source_changes or link_changes else "matched",
            "source_stat_verification": "changed" if source_changes else "matched",
            "view_link_verification": "changed" if link_changes else "matched",
            "source_changes": source_changes,
            "view_changes": link_changes,
        })
    except (TransferError, OSError, KeyError, TypeError, AttributeError, ValueError) as error:
        record.update({
            "status": "uncheckable",
            "source_stat_verification": "uncheckable",
            "view_link_verification": "uncheckable",
            "error": f"{type(error).__name__}: {error}",
        })
    realizations.mkdir(parents=True, exist_ok=True)
    (realizations / f"{realization_id}.runpod-input-postflight.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(f"[kura] selected-file transfer postflight: {record['status']}")


def main():
    if sys.argv[1:] == ["--postflight"]:
        postflight()
        return
    archive_path, manifest_path, manifest_sha256 = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    workspace = Path(os.environ["KURA_WORKSPACE"])
    run_id = os.environ["KURA_RUN_ID"]
    realization_id = os.environ["KURA_REALIZATION_ID"]
    record_path = workspace / "runs" / run_id / "realizations" / f"{realization_id}.runpod-input.json"
    staging = workspace / ".kura-transfer" / run_id / "staging"
    record = {
        "schema_version": 1,
        "realization_id": realization_id,
        "started_at": datetime.now().astimezone().isoformat(),
    }
    try:
        manifest_bytes = manifest_path.read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256:
            raise TransferError("transfer manifest differs from the one the controller verified")
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        if manifest.get("schema_version") != 1 or manifest.get("run_id") != run_id:
            raise TransferError("transfer manifest does not belong to this run")
        record.update({
            "archive_sha256": manifest["archive_sha256"],
            "input_sha256": manifest["input_sha256"],
            "projection_sha256": manifest["projection_sha256"],
        })
        if staging.is_symlink():
            raise TransferError("transfer staging path is a symlink")
        # A previous failed attempt on this Pod leaves only unpublished staging.
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        extract_verified(archive_path, manifest, staging)
        lock = check_envelope(staging, manifest, run_id)
        undo = publish(staging, workspace, run_id)
        created_views = []
        try:
            links, generated = materialize_views(workspace, lock, run_id, created_views)
            usage = shutil.disk_usage(workspace)
            record.update({
                "status": "verified",
                "entries": len(manifest["entries"]),
                "payload_bytes": manifest["payload_bytes"],
                "content_verification": "sha256-per-file-and-archive",
                "view_link_verification": "matched",
                "view_links": links,
                "generated_view_files": generated,
                "source_baseline": source_baseline(workspace, manifest),
                "disk": {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free},
                "finished_at": datetime.now().astimezone().isoformat(),
            })
            # Commit point: once this record exists the inputs are published.
            write_record(record_path, record)
        except (TransferError, OSError, KeyError, TypeError, AttributeError, ValueError) as error:
            # Remove only the view roots this attempt created, then unpublish,
            # so the workspace holds none of this attempt and it can retry.
            try:
                for root in reversed(created_views):
                    shutil.rmtree(root)
                undo()
            except OSError as rollback_error:
                raise TransferError(f"{error}; rollback also failed: {rollback_error}") from error
            raise
    except (TransferError, OSError, KeyError, TypeError, AttributeError, ValueError) as error:
        record.update({
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "finished_at": datetime.now().astimezone().isoformat(),
        })
        write_record(record_path, record)
        print(f"[kura] selected-file transfer verification failed: {record['error']}", file=sys.stderr)
        sys.exit(1)
    # After the commit: removing the uploaded tar and staging only frees disk,
    # so a failure here is logged and never turns a verified transfer into a failure.
    for label, cleanup in (("archive", archive_path.unlink), ("staging", lambda: shutil.rmtree(staging.parent))):
        try:
            cleanup()
        except OSError as cleanup_error:
            print(f"[kura] transfer {label} cleanup skipped: {cleanup_error}", file=sys.stderr)
    print(f"[kura] selected-file transfer verified: {record['entries']} files, {record['view_links']} view links")


def write_record(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


if __name__ == "__main__":
    main()

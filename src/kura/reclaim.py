"""What a user can free, in four classes by what deleting it costs (docs/adr/reclaimable-data.md).

`reclaim_inventory` is the one owner: `kura cleanup plan`, `kura cleanup`, and
`kura doctor disk` all read it. It reads existing records only (run status,
model-requirement locks, AI-Toolkit model records, the training-state store,
frozen image choices) and decides nothing by age: last use only sorts it.
"""

from __future__ import annotations

import json
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kura.container_scripts.ai_toolkit_models import RECORD_SUFFIX as AI_TOOLKIT_MODELS_RECORD_SUFFIX
from kura.container_scripts.ai_toolkit_models import list_model_files
from kura import doctor
from kura.backends import BACKENDS
from kura.executors.common import CLEANUP_ELIGIBLE_STATES, run_finished, unresolved_create_intents
from kura.fsio import atomic_write_json
from kura.hf_cache import hub_usage
from kura.images import PINNED_IMAGES, effective_image
from kura.paths import AI_TOOLKIT_MODELS_PATH, LEGACY_AI_TOOLKIT_CACHE, local_hf_cache
from kura.records import names_realization, realization_reference, record
from kura.training_artifacts import is_training_state_output, reclaimable_training_states
from kura.yamlio import YAMLError, load_yaml

CLASSES = ("free", "download", "resume", "outputs")
# The classes `kura cleanup plan` totals as reclaimable; outputs are listed only when asked for.
RECLAIMABLE_CLASSES = ("free", "download", "resume")
RECOVERY = {
    "free": "nothing: Kura or Docker recreates it when needed",
    "download": "a download the next time a run needs it",
    "resume": "resuming this run; every LoRA it saved stays",
    "outputs": "cannot be recreated: retraining does not reproduce it",
}
AI_TOOLKIT_MODELS_DIR = AI_TOOLKIT_MODELS_PATH.rsplit("/", 1)[-1]
_HUB_PREFIXES = {"models": "", "datasets": "datasets/", "spaces": "spaces/"}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _when(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.astimezone()


def _mtime(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")
    except OSError:
        return None


def read_run_status(run_dir: Path) -> dict[str, Any] | None:
    """A run's status.json, or None when it is missing or cannot be read."""
    try:
        value = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def run_done(run_dir: Path, status: dict[str, Any] | None) -> bool:
    """Whether Kura may offer or delete a run's data (the inventory and `kura run prune` ask here): its
    status reads, it ended in a cleanup state with publication no longer pending, and no Pod of it may
    still bill (`runner.pod_may_still_bill`), and no launch left a container or Pod unrecorded (an
    unresolved create intent of either executor). Unknown and recovery_required wait for a person; an
    unreadable status counts as running."""
    from kura.runner import pod_may_still_bill

    return (
        status is not None and str(status.get("state")) in CLEANUP_ELIGIBLE_STATES and run_finished(status)
        and pod_may_still_bill(run_dir, status) is None and not unresolved_create_intents(run_dir)
    )


def run_uses_inputs(status: dict[str, Any] | None) -> bool:
    """Whether a run uses its models and image now or at its next launch: an unreadable status, an
    unfinished run, or a compiled one that has not launched. A model or image it names is kept."""
    return status is None or not run_finished(status) or status.get("state") == "compiled"


def _render_checkpoint_paths(run_dir: Path) -> list[str] | None:
    """The checkpoint paths a compiled render froze (each case's `checkpoint.path` in `resolved/cases.jsonl`,
    and a legacy render's `inputs.checkpoint.path`); [] for a run that is not a render with frozen records, and None
    when its frozen records exist but cannot be read."""
    frozen = run_dir / "resolved" / "manifest.lock.yaml"
    if not frozen.is_file():
        return []
    try:
        run = load_yaml(frozen)
        cases_path = run_dir / "resolved" / "cases.jsonl"
        lines = cases_path.read_text(encoding="utf-8").splitlines() if cases_path.is_file() else []
        cases = [json.loads(line) for line in lines if line.strip()]
    except (OSError, ValueError, YAMLError):
        return None
    if not isinstance(run, dict):
        return None
    if run.get("type") != "render":
        return []
    inputs = run.get("inputs") if isinstance(run.get("inputs"), dict) else {}
    checkpoints = [case.get("checkpoint") for case in cases if isinstance(case, dict)] + [inputs.get("checkpoint")]
    return [item["path"] for item in checkpoints if isinstance(item, dict) and isinstance(item.get("path"), str)]


def render_users(workspace: Path, train_run: str) -> list[str]:
    """The render runs that read a training run's files now or at their next launch: a checkpoint path they
    froze is inside `runs/<train_run>/`, and they are unfinished, compiled, or unreadable (`run_uses_inputs`).
    A render whose frozen records exist but cannot be read counts too, as an unreadable record does for models."""
    root = (workspace / "runs" / train_run).resolve()
    users = []
    for run_dir in sorted((workspace / "runs").glob("*")):
        if not run_dir.is_dir() or run_dir.name == train_run or not run_uses_inputs(read_run_status(run_dir)):
            continue
        paths = _render_checkpoint_paths(run_dir)
        if paths is None or any(_workspace_file(workspace, path).is_relative_to(root) for path in paths):
            users.append(run_dir.name)
    return users


def _workspace_file(workspace: Path, value: str) -> Path:
    """A path a render names, relative to the workspace or absolute, resolved without requiring it to exist."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else workspace / path).resolve()


def _runs(workspace: Path) -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    for run_dir in sorted((workspace / "runs").glob("*")):
        if run_dir.is_dir():
            status = read_run_status(run_dir)
            runs[run_dir.name] = {
                "dir": run_dir, "status": status or {}, "state": str((status or {}).get("state") or "unreadable"),
                "done": run_done(run_dir, status), "uses_inputs": run_uses_inputs(status),
                "unfinished": status is None or not run_finished(status),
            }
    return runs


def became_in_use(workspace: Path, item: dict[str, Any]) -> list[str]:
    """The runs that use an item now, read again right before it is removed (a run may have been
    launched since the inventory was taken)."""
    if item["kind"] == "model-links":
        statuses = {path.name: read_run_status(path) for path in (workspace / "runs").glob("*") if path.is_dir()}
        return sorted(name for name, status in statuses.items() if status is None or not run_finished(status))
    if item["class"] == "download":
        # Every run, not only those the inventory saw: one compiled or launched since may name this item.
        runs = {path.name: path for path in (workspace / "runs").glob("*") if path.is_dir()}
        refs: set[str] = set()
        unreadable: list[str] = []
        for name, path in runs.items():
            if not run_uses_inputs(read_run_status(path)):
                continue
            try:
                repos, files, legacy = _model_references(path)
                image = _frozen_image(path)
            except UnreadableReference:
                unreadable.append(name)
                continue
            names = repos | files | ({LEGACY_AI_TOOLKIT_CACHE} if legacy else set())
            if item["id"] in names or (image is not None and image in item.get("docker_ids", [])) or name in item["referenced_by"]:
                refs.add(name)
        return sorted(refs | set(unreadable))
    if item["kind"] == "docker-containers":
        return sorted(run_id for run_id in item["referenced_by"]
                      if not run_done(workspace / "runs" / run_id, read_run_status(workspace / "runs" / run_id)))
    if item["class"] in ("resume", "outputs") or item["kind"] == "run-scratch":
        run_dir = workspace / "runs" / item["id"]
        if run_dir.is_dir() and not run_done(run_dir, read_run_status(run_dir)):
            return [item["id"]]
        # A render that will read this run's checkpoints keeps its outputs.
        return render_users(workspace, item["id"]) if item["class"] == "outputs" else []
    return []


def _last_used(refs: set[str], runs: dict[str, dict[str, Any]], fallback: Path | None) -> str | None:
    """The `ended` time of the newest referencing run, else the item's own modification time."""
    ended = [(when, value) for value in (runs[run]["status"].get("ended") for run in refs if run in runs) if (when := _when(value))]
    if ended:
        return max(ended)[1]
    return _mtime(fallback) if fallback is not None else None


def _in_use(refs: set[str], runs: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(run for run in refs if run in runs and runs[run]["uses_inputs"])


def _item(cls: str, kind: str, item_id: str, *, paths: list[Path] = (), size: int | None, refs: set[str] = frozenset(),
          last_used: str | None, runs: dict[str, dict[str, Any]], **extra: Any) -> dict[str, Any]:
    item = {
        "class": cls, "kind": kind, "id": item_id, "paths": [str(path) for path in paths], "size_bytes": size,
        "recovery": RECOVERY[cls], "referenced_by": sorted(refs), "last_used": last_used, **extra,
    }
    in_use = _in_use(set(refs), runs)
    if in_use:
        item["in_use_by"] = in_use
    return item


def last_observation_container_missing(run_dir: Path, status: dict[str, Any]) -> bool:
    """True only when the latest realization's own observation records a missing container."""
    reference = status.get("last_observation")
    realization = status.get("last_realization")
    if not isinstance(reference, str) or not isinstance(realization, str):
        return False
    observation = _read_json(run_dir / reference)
    return observation.get("container_missing") is True and names_realization(realization, str(observation.get("realization_id")))


def _sizes(paths: list[Path]) -> int | None:
    sizes = [doctor._path_size_bytes(path) for path in paths]
    return None if any(size is None for size in sizes) else sum(sizes)


def run_containers(run_dir: Path, containers: list[dict[str, Any]]) -> list[str]:
    """The IDs, among `docker ps` JSON entries, of the containers this run launched: their labels name the
    run and a realization it recorded. A copy of the workspace with the same run, or another workspace,
    never matches a container this run did not record."""
    found = []
    for container in containers:
        labels = dict(label.partition("=")[::2] for label in str(container.get("Labels") or "").split(","))
        realization = labels.get("io.kura.realization_id") or ""
        if container.get("ID") and labels.get("io.kura.run_id") == run_dir.name and realization \
                and Path(realization).name == realization and (run_dir / realization_reference(realization)).is_file():
            found.append(str(container["ID"]))
    return found


def _scratch_paths(run_dir: Path) -> list[Path]:
    """A finished run's `cache/`, `tmp/`, `.cache/`, and `transfer/` (its RunPod upload archive), less the
    directories where a trainer writes weights before Kura publishes them (each adapter's `cache_outputs`):
    those, and the directories above them, stay."""
    # Every adapter's directories, whichever backend the run used: the names are the trainers' own, so another
    # backend's run never has them, and reading the run's backend would add a way to fail for nothing.
    kept = {run_dir / relative for adapter in BACKENDS.values() for relative in adapter.cache_outputs
            if (run_dir / relative).exists() or (run_dir / relative).is_symlink()}

    def free(path: Path) -> list[Path]:
        if path in kept:
            return []
        if any(path in keep.parents for keep in kept) and path.is_dir() and not path.is_symlink():
            return [found for child in sorted(path.iterdir()) for found in free(child)]
        return [path]

    return [found for name in ("cache", "tmp", ".cache", "transfer") if (run_dir / name).exists() or (run_dir / name).is_symlink()
            for found in free(run_dir / name)]


def _free_items(workspace: Path, runs: dict[str, dict[str, Any]], docker: dict[str, Any] | None) -> list[dict[str, Any]]:
    items = []
    for run_id, run in runs.items():
        run_dir = run["dir"]
        if run["done"]:
            targets = _scratch_paths(run_dir)
            if targets:
                items.append(_item("free", "run-scratch", run_id, paths=targets, size=_sizes(targets), refs={run_id},
                                   last_used=_last_used({run_id}, runs, run_dir), runs=runs))
        elif (run_dir / "cache" / "dataset-view").is_dir() and last_observation_container_missing(run_dir, run["status"]):
            # Kept for recovery after Docker reported the container missing; launch rebuilds it from the frozen lock.
            view = run_dir / "cache" / "dataset-view"
            items.append(_item("free", "dataset-view", run_id, paths=[view], size=_sizes([view]),
                               last_used=_mtime(view), runs=runs))
    links = workspace / "cache" / "models"
    if links.is_dir() and any(links.iterdir()):
        # Links each launch writes again into the Hugging Face cache; a running run reads them.
        active = sorted(run_id for run_id, run in runs.items() if run["unfinished"])
        item = _item("free", "model-links", "cache/models", paths=[links], size=_sizes([links]), last_used=_mtime(links), runs=runs)
        if active:
            item["in_use_by"] = active
        items.append(item)
    if docker is not None:
        for usage in docker.get("usage") or []:
            if str(usage.get("Type", "")).lower() == "build cache" and (usage.get("reclaimable_bytes") or 0) > 0:
                items.append(_item("free", "docker-build-cache", "docker-build-cache", size=usage["reclaimable_bytes"], last_used=None, runs=runs,
                                   note="`docker builder prune` frees unused build cache of the whole Docker host, not only Kura's"))
        # Only a finished run's container of this workspace: an unfinished run still reads its container's exit
        # code, and a container of another workspace (labels name no run here) is that workspace's to remove.
        stopped = doctor.stopped_kura_containers()
        owned = {container: run_id for run_id, run in runs.items() if run["done"] for container in run_containers(run["dir"], stopped)}
        if owned:
            items.append(_item("free", "docker-containers", "docker-containers", size=None, refs=set(owned.values()),
                               last_used=None, runs=runs, docker_ids=list(owned)))
        volumes = [str(item["Name"]) for item in doctor.kura_docker_volumes() if item.get("Name")]
        if volumes:
            items.append(_item("free", "docker-volumes", "docker-volumes", size=None, last_used=None, runs=runs, docker_ids=volumes))
    return items


def _repo_of_selector(value: Any) -> str | None:
    """The repository an AI-Toolkit selector names (`org/name`, or a file below it), or None for a local path."""
    if not isinstance(value, str) or value.startswith(("/", ".", "~")):
        return None
    parts = value.split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 and all(parts[:2]) else None


class UnreadableReference(ValueError):
    """A record that may name a model or image cannot be read, so nothing it may name is offered."""


def _model_references(run_dir: Path) -> tuple[set[str], set[str], bool]:
    """The repositories a run's model-requirement lock names, the AI-Toolkit model files its records list,
    and whether those records list the legacy workspace directory. A record that exists but cannot be
    read raises UnreadableReference."""
    repos: set[str] = set()
    lock_path = run_dir / "resolved" / "model-requirements.lock.yaml"
    try:
        lock = load_yaml(lock_path) if lock_path.exists() else None
    except (OSError, YAMLError) as exc:
        raise UnreadableReference(f"{lock_path} cannot be read") from exc
    requirements = lock.get("requirements") if isinstance(lock, dict) else None
    for requirement in requirements if isinstance(requirements, list) else []:
        identity = requirement.get("identity") if isinstance(requirement, dict) else None
        if not isinstance(identity, dict):
            continue
        if isinstance(identity.get("repo_id"), str):
            repos.add(identity["repo_id"])
        elif identity.get("kind") == "backend-resolved-selector" and (repo := _repo_of_selector(identity.get("value"))):
            repos.add(repo)
    files: set[str] = set()
    legacy = False
    for path in (run_dir / "realizations").glob(f"*{AI_TOOLKIT_MODELS_RECORD_SUFFIX}"):
        try:
            models = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise UnreadableReference(f"{path} cannot be read") from exc
        models = models if isinstance(models, dict) else {}
        models_path = str(models.get("models_path") or "")
        listed = [entry["path"] for entry in models.get("files") or [] if isinstance(entry, dict) and isinstance(entry.get("path"), str)]
        if models_path.rstrip("/") == AI_TOOLKIT_MODELS_PATH:
            files.update(f"{AI_TOOLKIT_MODELS_DIR}/{name}" for name in listed)
        elif LEGACY_AI_TOOLKIT_CACHE in models_path and listed:
            legacy = True
    return repos, files, legacy


def _model_items(workspace: Path, hf_cache: Path, runs: dict[str, dict[str, Any]], unreadable: list[str], notes: list[str]) -> list[dict[str, Any]]:
    repo_refs: dict[str, set[str]] = {}
    file_refs: dict[str, set[str]] = {}
    legacy_refs: set[str] = set()
    for run_id, run in runs.items():
        try:
            repos, files, legacy = _model_references(run["dir"])
        except UnreadableReference as exc:
            unreadable.append(str(exc))
            continue
        for repo in repos:
            repo_refs.setdefault(repo, set()).add(run_id)
        for name in files:
            file_refs.setdefault(name, set()).add(run_id)
        if legacy:
            legacy_refs.add(run_id)
    items = []
    usage = hub_usage(hf_cache)

    def repo_id(name: str) -> str:
        """The id a listed repository goes by; another `hub/` entry that shares a payload is named by its path."""
        kind, _, rest = name.partition("--")
        return _HUB_PREFIXES[kind] + rest.replace("--", "/") if kind in _HUB_PREFIXES else f"hub/{name}"

    for name, repository in usage.repositories.items():
        repo_dir = repository.path
        kind = name.partition("--")[0]
        refs = repo_refs.get(repo_id(name), set()) if kind == "models" else set()
        extra: dict[str, Any] = {"link_to": str(repo_dir.resolve())} if repo_dir.is_symlink() else {}
        shared_bytes, shared_with = usage.shared(name)
        if shared_with:
            # Payloads in huggingface_hub's shared store that other repositories link to too: freed only with them.
            extra.update(shared_bytes=shared_bytes, shared_with=[repo_id(other) for other in shared_with])
        items.append(_item("download", "hf-repository", repo_id(name), paths=[repo_dir], size=usage.own_bytes(name), refs=refs,
                           last_used=_last_used(refs, runs, repo_dir), runs=runs, **extra))
    if not usage.complete:
        notes.append(f"part of the Hugging Face cache {hf_cache / 'hub'} cannot be read: a model's size may include files another "
                     "model shares, and no shared file is deleted until it can be read")
    if usage.unreferenced:
        # Shared-store payloads no repository links to any more (a removal interrupted, or one huggingface_hub kept).
        paths = sorted(usage.unreferenced)
        items.append(_item("download", "hf-unreferenced-blobs", "hub/blobs", paths=paths, size=sum(usage.unreferenced.values()),
                           last_used=max((stamp for path in paths if (stamp := _mtime(path))), default=None), runs=runs))
    models_root = hf_cache / AI_TOOLKIT_MODELS_DIR
    for entry in list_model_files(str(models_root)):
        name = f"{AI_TOOLKIT_MODELS_DIR}/{entry['path']}"
        refs = file_refs.get(name, set())
        path = models_root / entry["path"]
        items.append(_item("download", "ai-toolkit-model-file", name, paths=[path], size=entry["size_bytes"], refs=refs,
                           last_used=_last_used(refs, runs, path), runs=runs))
    legacy = workspace / LEGACY_AI_TOOLKIT_CACHE
    if legacy.is_dir():
        items.append(_item("download", "legacy-ai-toolkit-models", LEGACY_AI_TOOLKIT_CACHE, paths=[legacy], size=doctor._path_size_bytes(legacy),
                           refs=legacy_refs, last_used=_last_used(legacy_refs, runs, legacy), runs=runs))
    return items


def _frozen_image(run_dir: Path) -> str | None:
    path = run_dir / "resolved" / "env.lock"
    if not path.exists():
        return None
    try:
        lock = load_yaml(path)
    except (OSError, YAMLError) as exc:
        raise UnreadableReference(f"{path} cannot be read") from exc
    value = lock.get("selected_image") or lock.get("local_image") if isinstance(lock, dict) else None
    return value if isinstance(value, str) and value else None


def _image_items(config: dict[str, Any], runs: dict[str, dict[str, Any]], unreadable: list[str]) -> list[dict[str, Any]]:
    """Kura images on this Docker host that are old: not the current pin, not a workspace override, and
    not frozen in an unfinished run. One item per image ID, removed by every reference it has."""
    repositories = {reference.split("@", 1)[0] for reference in PINNED_IMAGES.values()}
    current = {effective_image(config, name)["reference"] for name in PINNED_IMAGES} | set(PINNED_IMAGES.values())
    frozen: dict[str, str] = {}
    for run_id, run in runs.items():
        try:
            image = _frozen_image(run["dir"])
        except UnreadableReference as exc:
            unreadable.append(str(exc))
            continue
        if image:
            frozen[run_id] = image
    unfinished = {frozen[run_id] for run_id, run in runs.items() if run_id in frozen and run["uses_inputs"]}
    by_id: dict[str, dict[str, Any]] = {}
    for line in doctor._docker_json_lines(["docker", "image", "ls", "--digests", "--format", "{{json .}}"]):
        repository = str(line.get("Repository") or "")
        if repository not in repositories:
            continue
        image = by_id.setdefault(str(line.get("ID") or repository), {"references": [], "size": line.get("Size"), "created": line.get("CreatedAt")})
        digest, tag = str(line.get("Digest") or ""), str(line.get("Tag") or "")
        if digest.startswith("sha256:"):
            image["references"].append(f"{repository}@{digest}")
        if tag and tag != "<none>":
            image["references"].append(f"{repository}:{tag}")
    items = []
    for image_id, image in sorted(by_id.items()):
        references = sorted(set(image["references"]))
        if not references or set(references) & (current | unfinished):
            continue
        refs = {run_id for run_id, frozen_image in frozen.items() if frozen_image in references}
        ended = _last_used(refs, runs, None)
        items.append(_item("download", "image", references[0], size=doctor._bytes_from_docker_size(image["size"]), refs=refs,
                           last_used=ended or (str(image["created"]) if image["created"] else None), runs=runs,
                           docker_ids=references, image_id=image_id))
    return items


def native_state_directories(run_dir: Path) -> list[Path]:
    """The trainer's own state folders in a run's outputs (every executor names them `*-state`)."""
    outputs = run_dir / "outputs"
    found: list[Path] = []
    for path in sorted(outputs.rglob("*-state")) if outputs.is_dir() else []:
        relative = path.relative_to(outputs)
        if path.is_dir() and not path.is_symlink() and is_training_state_output(relative / "x", run_dir.name) \
                and not any(relative.is_relative_to(parent.relative_to(outputs)) for parent in found):
            found.append(path)
    return found


def _resume_items(workspace: Path, runs: dict[str, dict[str, Any]], notes: list[str]) -> list[dict[str, Any]]:
    try:
        stored = reclaimable_training_states(workspace)
    except ValueError as exc:
        notes.append(f"stored training states are not listed: {exc}")
        stored = {}
    items = []
    for run_id in sorted({run_id for run_id, run in runs.items() if run["done"]} | (set(stored) - set(runs))):
        run = runs.get(run_id)
        native = native_state_directories(run["dir"]) if run else []
        artifacts = stored.get(run_id, [])
        paths = native + [workspace / "artifacts" / "training-state" / manifest["id"] for manifest in artifacts]
        if not paths:
            continue
        items.append(_item("resume", "training-state", run_id, paths=paths, size=_sizes(paths), refs={run_id} if run else set(),
                           last_used=_last_used({run_id}, runs, paths[0]), runs=runs,
                           artifact_ids=[manifest["id"] for manifest in artifacts], run_present=run is not None))
    return items


def _output_items(workspace: Path, runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for run_id, run in runs.items():
        run_dir = run["dir"]
        paths = [run_dir / name for name in ("outputs", "downloads") if (run_dir / name).exists()]
        if not run["done"] or not paths:
            continue
        native = native_state_directories(run_dir)
        size = _sizes(paths)
        native_size = _sizes(native)
        outputs = run_dir / "outputs"
        checkpoints = [
            {"path": path.relative_to(outputs).as_posix(), "size_bytes": path.stat().st_size}
            for path in sorted(outputs.rglob("*.safetensors")) if outputs.is_dir()
            and path.is_file() and not is_training_state_output(path.relative_to(outputs), run_id)
        ]
        item = _item("outputs", "run-outputs", run_id, paths=paths,
                     size=None if size is None or native_size is None else size - native_size, refs={run_id},
                     last_used=_last_used({run_id}, runs, run_dir), runs=runs, checkpoints=checkpoints)
        users = render_users(workspace, run_id)
        if users:
            item["in_use_by"] = users
        items.append(item)
    return items


def _totals(items: list[dict[str, Any]], classes: tuple[str, ...]) -> dict[str, dict[str, int]]:
    return {cls: {"count": sum(1 for item in items if item["class"] == cls),
                  "size_bytes": sum(item["size_bytes"] or 0 for item in items if item["class"] == cls)} for cls in classes}


def reclaim_inventory(workspace: Path, config: dict[str, Any], *, outputs: bool = False, docker: dict[str, Any] | None = None) -> dict[str, Any]:
    """Everything a user can free, by class, with each item's size, cost to recover, referencing runs,
    and last use. Outputs are listed only when `outputs` asks for them, and are never in the totals.
    `docker` is a `doctor._docker_storage_summary()` the caller already has."""
    runs = _runs(workspace)
    hf_cache = local_hf_cache(workspace, config)
    notes: list[str] = []
    unreadable: list[str] = []
    docker = docker if docker is not None else doctor._docker_storage_summary()
    if not shutil.which("docker"):
        # No Docker on this host (or not on PATH): nothing of Docker's to list, and nothing to run.
        docker = {"daemon_reachable": False, "diagnosis": "the docker command is not available on this host"}
    if not docker.get("daemon_reachable"):
        notes.append(f"Docker items are not listed: {docker.get('diagnosis') or 'the Docker daemon is not reachable'}")
        docker = None
    items = _free_items(workspace, runs, docker) + _model_items(workspace, hf_cache, runs, unreadable, notes)
    if docker is not None:
        items += _image_items(config, runs, unreadable)
    if unreadable:
        notes.append("models and images cannot be removed while a run record that may name them cannot be read: " + "; ".join(unreadable))
    items += _resume_items(workspace, runs, notes)
    if outputs:
        items += _output_items(workspace, runs)
    order = {cls: index for index, cls in enumerate(CLASSES)}
    items.sort(key=lambda item: (order[item["class"]], _when(item["last_used"]) or datetime.max.replace(tzinfo=timezone.utc), -(item["size_bytes"] or 0)))
    return {
        "items": items,
        "totals": _totals(items, RECLAIMABLE_CLASSES),
        "hf_cache": str(hf_cache),
        "hf_cache_inside_workspace": workspace.resolve() in hf_cache.parents,
        "notes": notes,
        "unreadable_references": unreadable,
        "unfinished_runs": sorted(run_id for run_id, run in runs.items() if run["unfinished"]),
    }


MODEL_KINDS = ("hf-repository", "hf-unreferenced-blobs", "ai-toolkit-model-file", "legacy-ai-toolkit-models")


def select_reclaim(
    inventory: dict[str, Any], *, models: list[str] = (), images: str | None = None, resume_data: list[str] = (),
    archive: str | None = None, archive_to: str | None = None, checkpoints: str | None = None, keep: list[str] = (),
) -> list[dict[str, Any]]:
    """What one `kura cleanup` invocation removes or moves: the free class when no option names anything
    else, otherwise only the items the options name. Raises naming every problem before anything is touched."""
    items = inventory["items"]
    problems: list[str] = []
    named = bool(models or images or resume_data or archive or checkpoints)
    actions: list[dict[str, Any]] = [] if named else [
        {"action": "free", "item": item} for item in items if item["class"] == "free" and "in_use_by" not in item
    ]
    if (models or images) and inventory.get("unreadable_references"):
        problems.append("models and images are not removed while a run record that may name them cannot be read: "
                        + "; ".join(inventory["unreadable_references"]) + "; repair or remove that run first")
    if "unreferenced" in models and inventory.get("unfinished_runs"):
        problems.append("--models unreferenced waits until no run is unfinished (a running trainer may be downloading models "
                        f"no record names yet): {', '.join(inventory['unfinished_runs'])}; run it after they finish, or name models by id")

    if "unreferenced" in models and not inventory["hf_cache_inside_workspace"]:
        # Only this workspace's records are read, so a model another workspace sharing the cache uses looks unreferenced.
        unnamed = [item["id"] for item in items if item["kind"] in MODEL_KINDS and not item["referenced_by"]]
        problems.append(f"--models unreferenced is refused: the Hugging Face cache {inventory['hf_cache']} is outside this workspace, "
                        "and a model another workspace using it needs is not named by any record here; name the models to remove "
                        f"by id, after checking no other workspace uses them: {', '.join(unnamed) or 'none is unreferenced here'}")

    def find(kinds: tuple[str, ...], item_id: str) -> dict[str, Any] | None:
        return next((item for item in items if item["kind"] in kinds and item["id"] == item_id), None)

    model_items = [item for item in items if item["kind"] in MODEL_KINDS]
    for token in models:
        chosen = [item for item in model_items if not item["referenced_by"] and "in_use_by" not in item] if token == "unreferenced" else [find(MODEL_KINDS, token)]
        if chosen == [None]:
            problems.append(f"model {token!r} is not in the inventory; `kura cleanup plan` lists model ids")
            continue
        for item in chosen:
            if "in_use_by" in item:
                problems.append(f"model {token!r} is in use by run(s) {', '.join(item['in_use_by'])} (unfinished, compiled, or unreadable); free it after they end")
            elif "link_to" in item:
                problems.append(f"model {token!r} is a link to {item['link_to']}; Kura removes only what is inside the Hugging Face cache")
            elif all(action["item"] is not item for action in actions):
                actions.append({"action": "models", "item": item})
    if images is not None:
        actions += [{"action": "images", "item": item} for item in items if item["kind"] == "image"]
    resume_items = [item for item in items if item["class"] == "resume"]
    for token in resume_data:
        chosen = resume_items if token == "finished" else [item for item in resume_items if item["id"] == token]
        if not chosen and token != "finished":
            problems.append(f"run {token!r} has no resume data to free (it is not finished, a continuation references its "
                            "training state, or it kept none); `kura cleanup plan` lists resume data")
        for item in chosen:
            if all(action["item"] is not item for action in actions):
                actions.append({"action": "resume-data", "item": item})
    outputs = {item["id"]: item for item in items if item["class"] == "outputs"}
    if keep and checkpoints is None:
        problems.append("--keep names the checkpoints to keep for --checkpoints <run>")
    if archive_to is not None and archive is None:
        problems.append("--to names where --archive-outputs <run> moves the run's outputs")
    for name in (checkpoints, archive):
        if name in outputs and "in_use_by" in outputs[name]:
            problems.append(f"run {name!r} has checkpoints that render run(s) {', '.join(outputs[name]['in_use_by'])} will read "
                            "(unfinished or compiled, or their frozen records cannot be read); change them after those renders end or are discarded")
    if checkpoints is not None:
        item = outputs.get(checkpoints)
        if item is None:
            problems.append(f"run {checkpoints!r} has no outputs Kura may change (it is not finished or has none); "
                            "`kura cleanup plan --outputs` lists runs and their checkpoints")
        elif not keep:
            problems.append("--checkpoints needs --keep <file>... naming the checkpoints to keep; "
                            "`kura run prune <run-id>` deletes a whole run")
        else:
            names = [entry["path"] for entry in item["checkpoints"]]
            unmatched = [name for name in keep if not any(name in (path, Path(path).name) for path in names)]
            if unmatched:
                problems.append(f"run {checkpoints!r} has no checkpoint named {', '.join(map(repr, unmatched))}; it has {', '.join(names) or 'none'}")
            else:
                removed = [path for path in names if path not in keep and Path(path).name not in keep]
                actions.append({"action": "remove-checkpoints", "item": item, "remove": removed,
                                "keep": [path for path in names if path not in removed],
                                "remove_bytes": sum(entry["size_bytes"] for entry in item["checkpoints"] if entry["path"] in removed)})
    if archive is not None:
        item = outputs.get(archive)
        if item is None:
            problems.append(f"run {archive!r} has no outputs Kura may move (it is not finished or has none); "
                            "`kura cleanup plan --outputs` lists runs")
        elif archive_to is None:
            problems.append("--archive-outputs needs --to <dir>, the directory that receives the run's outputs")
        else:
            destination = Path(archive_to).expanduser().resolve() / archive
            run_dir = Path(item["paths"][0]).parent.resolve()
            if destination.exists():
                problems.append(f"{destination} already exists; name another --to directory")
            elif destination.is_relative_to(run_dir):
                problems.append(f"{destination} is inside the run itself; name a directory outside {run_dir}")
            else:
                actions.append({"action": "archive-outputs", "item": item, "destination": str(destination)})
    if problems:
        raise ValueError("; ".join(problems))
    return actions


def record_reclaim(run_dir: Path, action: str, facts: dict[str, Any]) -> str:
    """Append what an archive or removal did to the run's records, beside its latest realization, so no
    record goes on naming a file as present without a later one saying it left. Status ignores it."""
    status = _read_json(run_dir / "status.json")
    realization = status.get("last_realization")
    prefix = Path(realization).stem if isinstance(realization, str) and realization else "run"
    timestamp = datetime.now().astimezone()
    path = run_dir / "realizations" / f"{prefix}.reclaim-{timestamp.strftime('%Y%m%d-%H%M%S-%f')}-{secrets.token_hex(3)}.json"
    atomic_write_json(path, record("reclaim", {"action": action, "at": timestamp.isoformat(), **facts}))
    return path.relative_to(run_dir).as_posix()

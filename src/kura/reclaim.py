"""What a user can free, in four classes by what deleting it costs (docs/adr/reclaimable-data.md).

`reclaim_inventory` is the one owner: `kura cleanup plan`, `kura cleanup`, and
`kura doctor disk` all read it. It reads existing records only (run status,
model-requirement locks, AI-Toolkit model records, the training-state store,
frozen image choices) and decides nothing by age: last use only sorts it.
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kura.container_scripts.ai_toolkit_models import RECORD_SUFFIX as AI_TOOLKIT_MODELS_RECORD_SUFFIX
from kura.container_scripts.ai_toolkit_models import list_model_files
from kura import doctor
from kura.executors.common import CLEANUP_ELIGIBLE_STATES, run_finished
from kura.fsio import atomic_write_json
from kura.images import PINNED_IMAGES, effective_image
from kura.paths import AI_TOOLKIT_MODELS_PATH, LEGACY_AI_TOOLKIT_CACHE, local_hf_cache
from kura.records import record
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


def _runs(workspace: Path) -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    for run_dir in sorted((workspace / "runs").glob("*")):
        if run_dir.is_dir():
            status = _read_json(run_dir / "status.json")
            state = str(status.get("state") or "unknown")
            # Finished, with publication no longer pending: what Kura may offer from a run (unknown and
            # recovery_required wait for a person).
            done = state in CLEANUP_ELIGIBLE_STATES and run_finished(status)
            runs[run_dir.name] = {"dir": run_dir, "status": status, "state": state, "done": done}
    return runs


def _last_used(refs: set[str], runs: dict[str, dict[str, Any]], fallback: Path | None) -> str | None:
    """The `ended` time of the newest referencing run, else the item's own modification time."""
    ended = [(when, value) for value in (runs[run]["status"].get("ended") for run in refs if run in runs) if (when := _when(value))]
    if ended:
        return max(ended)[1]
    return _mtime(fallback) if fallback is not None else None


def _in_use(refs: set[str], runs: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(run for run in refs if run in runs and not run_finished(runs[run]["status"]))


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
    return observation.get("container_missing") is True and f"realizations/{observation.get('realization_id')}.json" == realization


def _sizes(paths: list[Path]) -> int | None:
    sizes = [doctor._path_size_bytes(path) for path in paths]
    return None if any(size is None for size in sizes) else sum(sizes)


def _free_items(workspace: Path, runs: dict[str, dict[str, Any]], docker: dict[str, Any] | None) -> list[dict[str, Any]]:
    items = []
    for run_id, run in runs.items():
        run_dir = run["dir"]
        if run["done"]:
            targets = [run_dir / name for name in ("cache", "tmp", ".cache") if (run_dir / name).exists()]
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
        active = sorted(run_id for run_id, run in runs.items() if not run_finished(run["status"]))
        item = _item("free", "model-links", "cache/models", paths=[links], size=_sizes([links]), last_used=_mtime(links), runs=runs)
        if active:
            item["in_use_by"] = active
        items.append(item)
    if docker is not None:
        for usage in docker.get("usage") or []:
            if str(usage.get("Type", "")).lower() == "build cache" and (usage.get("reclaimable_bytes") or 0) > 0:
                items.append(_item("free", "docker-build-cache", "docker-build-cache", size=usage["reclaimable_bytes"], last_used=None, runs=runs))
        containers = doctor.stopped_kura_containers()
        if containers:
            ids = [str(item["ID"]) for item in containers if item.get("ID")]
            items.append(_item("free", "docker-containers", "docker-containers", size=None, last_used=None, runs=runs, docker_ids=ids))
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


def _model_references(run_dir: Path) -> tuple[set[str], set[str], bool]:
    """The repositories a run's model-requirement lock names, the AI-Toolkit model files its records list,
    and whether those records list the legacy workspace directory."""
    repos: set[str] = set()
    try:
        lock = load_yaml(run_dir / "resolved" / "model-requirements.lock.yaml")
    except (OSError, YAMLError):
        lock = None
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
        models = _read_json(path)
        models_path = str(models.get("models_path") or "")
        listed = [entry["path"] for entry in models.get("files") or [] if isinstance(entry, dict) and isinstance(entry.get("path"), str)]
        if models_path.rstrip("/") == AI_TOOLKIT_MODELS_PATH:
            files.update(f"{AI_TOOLKIT_MODELS_DIR}/{name}" for name in listed)
        elif LEGACY_AI_TOOLKIT_CACHE in models_path and listed:
            legacy = True
    return repos, files, legacy


def _model_items(workspace: Path, hf_cache: Path, runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    repo_refs: dict[str, set[str]] = {}
    file_refs: dict[str, set[str]] = {}
    legacy_refs: set[str] = set()
    for run_id, run in runs.items():
        repos, files, legacy = _model_references(run["dir"])
        for repo in repos:
            repo_refs.setdefault(repo, set()).add(run_id)
        for name in files:
            file_refs.setdefault(name, set()).add(run_id)
        if legacy:
            legacy_refs.add(run_id)
    items = []
    hub = hf_cache / "hub"
    for repo_dir in sorted(hub.glob("*--*")) if hub.is_dir() else []:
        kind, _, rest = repo_dir.name.partition("--")
        if kind not in _HUB_PREFIXES or not repo_dir.is_dir():
            continue
        repo_id = _HUB_PREFIXES[kind] + rest.replace("--", "/")
        refs = repo_refs.get(repo_id, set()) if kind == "models" else set()
        items.append(_item("download", "hf-repository", repo_id, paths=[repo_dir], size=doctor._path_size_bytes(repo_dir), refs=refs,
                           last_used=_last_used(refs, runs, repo_dir), runs=runs))
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
    try:
        lock = load_yaml(run_dir / "resolved" / "env.lock")
    except (OSError, YAMLError):
        return None
    value = lock.get("selected_image") or lock.get("local_image") if isinstance(lock, dict) else None
    return value if isinstance(value, str) and value else None


def _image_items(config: dict[str, Any], runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Kura images on this Docker host that are old: not the current pin, not a workspace override, and
    not frozen in an unfinished run. One item per image ID, removed by every reference it has."""
    repositories = {reference.split("@", 1)[0] for reference in PINNED_IMAGES.values()}
    current = {effective_image(config, name)["reference"] for name in PINNED_IMAGES} | set(PINNED_IMAGES.values())
    frozen = {run_id: image for run_id, run in runs.items() if (image := _frozen_image(run["dir"]))}
    unfinished = {frozen[run_id] for run_id, run in runs.items() if run_id in frozen and not run_finished(run["status"])}
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


def _output_items(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
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
        items.append(_item("outputs", "run-outputs", run_id, paths=paths,
                           size=None if size is None or native_size is None else size - native_size, refs={run_id},
                           last_used=_last_used({run_id}, runs, run_dir), runs=runs, checkpoints=checkpoints))
    return items


def _totals(items: list[dict[str, Any]], classes: tuple[str, ...]) -> dict[str, dict[str, int]]:
    return {cls: {"count": sum(1 for item in items if item["class"] == cls),
                  "size_bytes": sum(item["size_bytes"] or 0 for item in items if item["class"] == cls)} for cls in classes}


def reclaim_inventory(workspace: Path, config: dict[str, Any], *, outputs: bool = False) -> dict[str, Any]:
    """Everything a user can free, by class, with each item's size, cost to recover, referencing runs,
    and last use. Outputs are listed only when `outputs` asks for them, and are never in the totals."""
    runs = _runs(workspace)
    hf_cache = local_hf_cache(workspace, config)
    notes: list[str] = []
    docker = doctor._docker_storage_summary()
    if not docker.get("daemon_reachable"):
        notes.append(f"Docker items are not listed: {docker.get('diagnosis') or 'the Docker daemon is not reachable'}")
        docker = None
    items = _free_items(workspace, runs, docker) + _model_items(workspace, hf_cache, runs)
    if docker is not None:
        items += _image_items(config, runs)
    items += _resume_items(workspace, runs, notes)
    if outputs:
        items += _output_items(runs)
    order = {cls: index for index, cls in enumerate(CLASSES)}
    items.sort(key=lambda item: (order[item["class"]], _when(item["last_used"]) or datetime.max.replace(tzinfo=timezone.utc), -(item["size_bytes"] or 0)))
    return {
        "items": items,
        "totals": _totals(items, RECLAIMABLE_CLASSES),
        "hf_cache": str(hf_cache),
        "hf_cache_inside_workspace": workspace.resolve() in hf_cache.parents,
        "notes": notes,
    }


MODEL_KINDS = ("hf-repository", "ai-toolkit-model-file", "legacy-ai-toolkit-models")


def select_reclaim(
    inventory: dict[str, Any], *, models: list[str] = (), images: str | None = None, resume_data: list[str] = (),
    archive: str | None = None, archive_to: str | None = None, checkpoints: str | None = None, keep: list[str] = (),
) -> list[dict[str, Any]]:
    """What one `kura cleanup` invocation removes or moves: the free class always, and only the items the
    flags name from the other classes. Raises naming every problem before anything is touched."""
    items = inventory["items"]
    problems: list[str] = []
    actions: list[dict[str, Any]] = [
        {"action": "free", "item": item} for item in items if item["class"] == "free" and "in_use_by" not in item
    ]

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
                problems.append(f"model {token!r} is in use by unfinished run(s) {', '.join(item['in_use_by'])}; free it after they end")
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

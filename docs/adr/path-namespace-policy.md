# Path namespace policy

Status: accepted.

Date: 2026-07-03

## Context

Kura runs training inside Docker or RunPod while the host CLI, monitor, and
agents read workspace files directly. A single file can therefore have several
valid names: a host absolute path, a container path such as `/workspace/...`, and
a path through an extra Docker mount such as `/root/.cache/huggingface/...`.

The root problem is not Docker itself. The problem is persisting a path without
recording which namespace consumes it.

## Decision

Path namespace is defined by the artifact consumer:

| Artifact | Consumer | Persisted form |
| --- | --- | --- |
| `dataset.yaml`, `items.jsonl` | host CLI / agent / human | dataset-relative typed file references; `dataset.yaml` declares the items schema version |
| `resolved/dataset-input.lock.json` | host CLI / executor | workspace-relative source identities and explicitly namespaced native destinations |
| `resolved/backend-command.lock.json`, dataset TOML, training argv | container | container absolute paths, normally `/workspace/...` |
| run-scoped training views | trainer in container / executor | run-owned native files and source links resolving through the executor's source mapping |
| `status.json`, model lock files, indexes, workspace symlinks | host CLI / agent / human | workspace-relative paths, or host-resolvable symlinks |
| realization mounts | host CLI and executor | explicit source/target pairs |
| RunPod remote facts | host CLI, as remote facts | fields must make the remote namespace explicit |
| logs | humans | free text; Kura must not machine-interpret arbitrary log paths |

No Kura-owned workspace artifact may persist a container-private path such as
`/root/...`, `/opt/...`, `/tmp/...`, `/var/...`, or `/app/...` unless the field is
explicitly a container command/runtime fact. If Kura cannot map a path through
the workspace mount table, it must fail or treat the fact as unavailable. It
must not invent a mapping.

The authored dataset and its frozen run-selected projection have different
path consumers. A container-private native path in a command or dataset TOML
must not be mistaken for a host-resolvable source in the input lock. The
trainer receives a writable run-owned native view with links to selected
source files. Kura checks the link inventory and targets against the lock;
adjacent caches in the view are not authored sources. Local Docker maps the
authored dataset read-only so a write through a source link cannot alter it.
Host UID or file mode alone is not that protection. Hardlinks are not a
dataset-view fallback. A dataset-root symlink may resolve to an external host
source only when its target and selected files pass containment and source
mount-mapping checks; the trainer's view must resolve to the mounted source
path, not an unmounted host path.

`model-bundle.lock.yaml` is the source of truth for Musubi model provenance.
`cache/models/` is a convenience layer for container paths and may contain
symlinks. Host-side plan and monitor code must treat that symlink tree as
best-effort: a broken or un-mappable link means "not cached", never a crash.

Docker launch passes the resolved mount table to container helper scripts as
data. Container scripts do not import Kura.

Executor model-cache contract:

- Executors that run model download helpers must set `HF_HOME` and
  `HF_HUB_CACHE` explicitly. `HF_HOME` is the Hugging Face state root;
  repository snapshots and blobs use `HF_HUB_CACHE`.
- `HF_HOME` must be under the container workspace root, normally
  `/workspace/cache/huggingface`. `HF_HUB_CACHE` must be its `hub/` child or be
  covered by `KURA_WORKSPACE_PATH_MAPS`.
- Container helpers must treat missing or unmappable `HF_HUB_CACHE` as a contract
  error before downloading. They must not fall back to private locations such as
  `/root/.cache/huggingface` or `/tmp/...`.
- Local Docker may continue to expose a legacy Hugging Face cache mount through
  `KURA_WORKSPACE_PATH_MAPS`, but new executor paths should prefer a single
  workspace-visible cache location.

## Enforcement

- `src/kura/paths.py` owns namespace conversion helpers.
- Docker launch passes `KURA_WORKSPACE_PATH_MAPS` into the container.
- `hf_download.py` uses that map when creating stable workspace symlinks.
- `kura doctor disk` reports workspace symlinks with container-private or
  workspace-external absolute targets.
- `kura fix-links` is a dry-run-first repair command. It rewrites only links
  whose targets are covered by the workspace mount table; it reports unfixable
  links without deleting them.

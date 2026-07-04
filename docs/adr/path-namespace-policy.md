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
| `resolved/command.json`, dataset TOML, training argv | container | container absolute paths, normally `/workspace/...` |
| `status.json`, model lock files, indexes, workspace symlinks | host CLI / agent / human | workspace-relative paths, or host-resolvable symlinks |
| realization mounts | host CLI and executor | explicit source/target pairs |
| RunPod remote facts | host CLI, as remote facts | fields must make the remote namespace explicit |
| logs | humans | free text; Kura must not machine-interpret arbitrary log paths |

No Kura-owned workspace artifact may persist a container-private path such as
`/root/...`, `/opt/...`, `/tmp/...`, `/var/...`, or `/app/...` unless the field is
explicitly a container command/runtime fact. If Kura cannot map a path through
the workspace mount table, it must fail or treat the fact as unavailable. It
must not invent a mapping.

`model-bundle.lock.yaml` is the source of truth for Musubi model provenance.
`cache/models/` is a convenience layer for container paths and may contain
symlinks. Host-side plan and monitor code must treat that symlink tree as
best-effort: a broken or un-mappable link means "not cached", never a crash.

Docker launch passes the resolved mount table to container helper scripts as
data. Container scripts do not import Kura.

Executor model-cache contract:

- Executors that run model download helpers must set `HF_HOME` explicitly.
- `HF_HOME` must either be under the container workspace root, normally
  `/workspace/cache/huggingface`, or be covered by `KURA_WORKSPACE_PATH_MAPS`.
- Container helpers must treat missing or unmappable `HF_HOME` as a contract
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

# ADR: Local Docker disk safety

Kura local training can create large data in several places at once:

- workspace cache and run artifacts
- Hugging Face and backend model caches
- Docker images, build cache, containers, and volumes
- temporary files written through bind mounts

On WSL2 this is especially dangerous because the Linux distro and Docker
Desktop may live in VHDX files backed by Windows drives. Linux `df` reports the
logical ext4 filesystem, not necessarily the remaining free space on the
Windows drive that must grow the VHDX. VHDX files also do not automatically
shrink after files are deleted.

The accounting unit is therefore the physical backing store, not just a logical
path. Kura probes each relevant path and records both the Linux filesystem free
space and, when possible, the Windows backing drive free space. The effective
free space is the smaller of the two.

## Decision

Kura treats local training as a disk-sensitive operation, not just a process
launch.

- `kura doctor disk` reports workspace cache/runs, filesystem free space, Docker
  storage, cache-related environment variables, root-owned files, and storage
  backing confidence. It returns structured `issues`; blocking-risk findings are
  `warnings`, while cleanup hygiene such as a large but currently affordable
  cache is an `advisory`.
- `kura cleanup ...` defaults to dry-run. Destructive cleanup requires `--yes`.
  Whole-run deletion, including outputs/downloads, additionally requires
  `--delete-final-artifacts`.
- `kura fix-permissions` repairs root-owned Kura cache/run files, defaulting to
  dry-run and limiting its scope to Kura-managed workspace paths.
- Local Docker launch refuses to start when the workspace or writable mounts have
  less effective free space than the configured floor, currently 100GiB by
  default.
- For Musubi model downloads, local Docker launch estimates Hugging Face file
  sizes before starting when metadata is reachable. The configured free-space
  floor is treated as the post-write safety margin, so a run that may download
  33GiB requires roughly `floor + 33GiB` free on the cache backing store.
- Musubi model downloads above the safety threshold, 25GiB by default, require
  explicit run intent with `safety.allow_large_model_downloads: true`. A run can
  tune this threshold with `safety.large_model_download_gb`.
- When many checkpoints are explicitly allowed, local Docker launch adds a
  conservative checkpoint write budget to the workspace requirement.
- On WSL2, Kura auto-detects the distro backing drive from the WSL registry when
  Windows interop is available. If the backing drive cannot be resolved, local
  Docker launch fails safe unless the run explicitly sets
  `safety.allow_storage_risk: true`.
- `storage.host_drive` in `workspace.yaml` is an override for unusual WSL2
  setups, not a required normal setting.
- Large adapter smoke tests must run `kura doctor disk` before downloading
  multi-GB models.

## Safety map

| Operation | Check | Blocks on |
| --- | --- | --- |
| `kura doctor disk` | read-only inventory | exits non-zero on warning-severity issues |
| local Docker launch | `StorageStatus.effective_free_bytes` plus estimated writes for workspace, cache, and writable mounts | low effective free, unknown WSL2 backing, excessive Docker build cache |
| large Musubi model download | estimated new Hugging Face/model-cache writes after Kura cache hits | above `safety.large_model_download_gb` without `safety.allow_large_model_downloads: true` |
| checkpoint-heavy train run | run plan / launch preflight | many unpruned checkpoints unless explicitly allowed |
| RunPod download/pull | local destination free space | insufficient space for downloaded artifacts |
| cleanup | dry-run by default | destructive action requires `--yes`; final artifacts require extra flag |
| permission repair | scoped chown of Kura cache/runs | never deletes data |

## Follow-up

Future work may add cache indexing and better Docker Desktop backing-store
accounting. Those features should preserve the same principle: show where bytes
live before deleting or launching.

# ADR: Local Docker disk safety

Kura local training can create large data in several places at once:

- workspace cache and run artifacts
- Hugging Face and backend model caches
- Docker images, build cache, containers, and volumes
- temporary files written through bind mounts

On WSL2 this is especially dangerous because the Linux distro and Docker
Desktop may live in different VHDX files. A user can move the distro to a large
drive while Docker Desktop still consumes the system drive. VHDX files also do
not automatically shrink after files are deleted.

## Decision

Kura treats local training as a disk-sensitive operation, not just a process
launch.

- `kura doctor disk` reports workspace cache/runs, filesystem free space, Docker
  storage, cache-related environment variables, and root-owned files.
- `kura cleanup ...` starts as a dry-run inventory command. Destructive cleanup
  must stay guarded and explicit.
- Local Docker launch refuses to start when the workspace or writable mounts have
  less than 50GiB free.
- Large adapter smoke tests must run `kura doctor disk` before downloading
  multi-GB models.

## Follow-up

Future work may add guarded delete modes, permission repair, cache indexing, and
more precise per-run disk estimates. Those features should preserve the same
principle: show where bytes live before deleting or launching.

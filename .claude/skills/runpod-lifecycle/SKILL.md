---
name: runpod-lifecycle
description: RunPod remote training lifecycle and billing safety for Kura. Use when working on kura run remote, runpod staging/upload/download/pull/stop/reconcile, Pod cleanup, max lease, notifications, GPU selection, Network Volumes, or RunPod docs/README updates.
---

# RunPod Lifecycle

Use this skill for any RunPod remote execution or cleanup change.

## Standard remote flow

```text
compile
stage upload bundle
launch disposable Pod
upload over SSH
run backend command detached from SSH control
poll remote logs/exit record
download snapshot
hold for review
stop Pod
```

## Current defaults

- `kura run remote <run-id>`
- `--hold-for 30m`: normal post-download review window.
- `--max-lease 12h`: Pod-side best-effort billing fuse if the local controller dies.
- `--job-timeout 0`: wait until remote exit.
- `runpod.storage_mode: upload`: no Network Volume by default.

## Non-negotiables

- Do not stop a disposable Pod until remote exit and local download are confirmed.
- If download/completion is uncertain, leave the Pod running and print/notify recovery steps.
- Do not add unbounded keep-alive flags. Use bounded leases only.
- If review hold is interrupted, stop the Pod.
- `max-lease` is a billing safety fuse, not output preservation. Do not set it shorter than expected training plus review unless loss of container-disk outputs is acceptable.
- Do not put `HF_TOKEN`, RunPod keys, ntfy tokens, or object-store credentials in Pod create environment.

## Recovery commands

```sh
uv run kura doctor runpod
uv run kura run reconcile <run-id>
uv run kura run download <run-id> --force
uv run kura run pull <run-id> --since-step 1000
uv run kura run stop <run-id>
```

## Test expectations

Run lifecycle tests after changes:

```sh
uv run python -m unittest tests.test_cli
uv run kura run remote --help
```

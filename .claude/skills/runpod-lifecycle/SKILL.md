---
name: runpod-lifecycle
description: RunPod remote training lifecycle and billing safety for Kura. Use when working on kura run remote, runpod staging/upload/download/pull/stop/reconcile, Pod cleanup, max lease, notifications, GPU selection, Network Volumes, or RunPod docs/README updates.
---

# RunPod Lifecycle

Use this skill for any RunPod remote execution or cleanup change.

## Standard remote flow

```text
draft run plan: measure GPU stock/price
record the selected GPU and immediate/wait capacity policy
compile
final run plan and one approval
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

- `kura run execute <run-id>` is the normal entry point and honors the RunPod
  executor frozen in the compiled run.
- `kura run remote <run-id>` remains the low-level entry point for advanced
  lifecycle flags and recovery work.
- `--hold-for 30m`: normal post-download review window.
- `--max-lease 12h`: Pod-side best-effort billing fuse if the local controller dies.
- `--job-timeout 0`: wait until remote exit.
- `runpod.storage_mode: upload`: no Network Volume by default.
- GPU candidates default to `NVIDIA RTX A5000` then `NVIDIA A40` with custom priority.
- If a run explicitly sets `compute.gpu`, treat it as part of the user's run
  intent and use it before workspace-level candidates.
- Run `kura run plan` once while the RunPod run is still a draft so current
  stock and alternatives can inform `compute.gpu` and `compute.capacity`.
  Compile only after that choice, then show the final compiled plan for the
  single launch approval.
- `compute.capacity.mode=wait` is a bounded foreground policy. The default
  upload path cannot safely use RunPod's provider-side Deploy When Available
  subscription because the controller must still upload inputs, start training,
  and install the max-lease guard after Pod creation.
- Confirm a bounded capacity wait once before entering its wait loop so it can
  acquire unattended. The confirmation covers the configured creation-attempt
  sequence and must warn that the displayed hourly price may change while
  waiting; do not move the prompt to the eventual capacity-acquisition moment.

## Non-negotiables

- Never add `--yes` to a RunPod launch unless the user explicitly instructed
  that billed launch. In a non-interactive agent or script session, `--yes`
  records that explicit instruction; it is not a convenience flag for bypassing
  the launch gate. It skips only the question; Kura still prints the GPU, price,
  and maximum-lease summary.
- A local execution failure is not permission to switch providers. In
  particular, do not rewrite `run.yaml` from a local executor to `runpod`
  because Docker, ComfyUI, or another local service is unavailable. Switching
  local to RunPod creates a new cost decision: show the GPU, hourly price, and
  maximum lease, obtain user approval, then record and compile the approved
  executor change.
- Do not stop a disposable Pod until remote exit and local download are confirmed.
- If download/completion is uncertain, leave the Pod running and print/notify recovery steps.
- Do not add unbounded keep-alive flags. Use bounded leases only.
- If review hold is interrupted, stop the Pod.
- `max-lease` is a billing safety fuse, not output preservation. Do not set it shorter than expected training plus review unless loss of container-disk outputs is acceptable.
- Do not put `HF_TOKEN`, RunPod keys, ntfy tokens, or object-store credentials in Pod create environment.
- Every remote execution path must establish the executor contract before any
  work: `HF_HOME` set inside the workspace namespace (`$KURA_WORKSPACE/cache/huggingface`) and `HF_HUB_CACHE` set to its `hub/` child
  and `KURA_*` variables the scripts consume. This applies to any revived or
  new path (object staging included) — a path that forgets this repeats the
  2026-07-05 "download lands in container-private /root/.cache" incident.
- Treat compute choice as a constrained resource plan, not a convenience
  default. Start with the smallest candidate that should satisfy the declared
  training plan, then move up only when capacity, memory, or runtime evidence
  justifies it. The agent may tune execution accommodations within the same
  GPU class; a GPU-class/cost change or an expected elapsed-time increase beyond
  roughly 2x requires user approval and a new plan.

## Recovery commands

```sh
uv run kura doctor runpod
uv run kura run reconcile <run-id>
uv run kura run download <run-id> --force
uv run kura run pull <run-id> --since-step 1000
uv run kura run stop <run-id>
```

After a long unattended capacity wait, run `uv run kura doctor runpod` to
confirm that no unrecorded Pod remains before retrying or leaving RunPod.

If RunPod fails before receiving a request with an OS-level permission error,
the current agent process may lack network access. Use
`docs/external-access.md` for the agent-specific setup. Do not classify that as
a RunPod outage or add a Kura-side network bypass.

## Test expectations

Run lifecycle tests after changes:

```sh
uv run python -m unittest tests.test_cli
uv run kura run remote --help
```

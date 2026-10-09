---
name: runpod-lifecycle
description: RunPod remote training lifecycle and billing safety for Kura. Use when the user trains or renders on RunPod, or when a RunPod run needs staging, download, stop, reconcile, Pod cleanup, max lease, notifications, GPU selection, or Network Volumes.
---

# RunPod Lifecycle

Use this skill whenever a run executes on RunPod or a RunPod Pod needs recovery or cleanup.

## Standard remote flow

```text
draft run plan: measure GPU stock/price
record the selected GPU (the run waits for it by default; capacity mode: immediate fails instead)
compile
final run plan and one approval
stage upload bundle (manifest-v2: selected files only, hashed against the lock)
launch disposable Pod (manifest-v2: staged archive re-proven against the compile first)
upload over SSH
verify inputs on the Pod before model acquisition (manifest-v2)
run backend command detached from SSH control
poll remote logs/exit record
verify terminal manifest and download only the snapshot delta
stop Pod
```

## Current defaults

- `kura run execute <run-id>` is the normal entry point and honors the RunPod
  executor frozen in the compiled run.
- After showing the final plan and receiving the user's single explicit launch
  approval, an agent runs `kura run execute <run-id> --yes`. The flag carries
  that approval through the non-interactive launch gate; it must not cause a
  second user prompt.
- It is the only way to start a training run, and the way to follow one: if
  the session or the controller was lost while the run goes on, run
  `kura run execute <run-id>` again; it follows the job and collects it, and
  never starts a second Pod. Only a run that has ended without completing (its
  launch failed, or it failed or was stopped) starts again, as a new run from
  its settings: `kura run new --from <run-id> --slug <words>`.
- The compiled `compute.capacity` applies: by default the run waits for its
  GPU (no Pod, no billing while waiting); `mode: immediate` fails at once.
- `--max-lease 12h`: the Pod deletes itself this long after Kura first reaches it, whatever the local controller does.
  If Kura warns that training looks longer than the lease, tell the user the
  estimate and ask before running `kura run lease <run-id> <duration>`, which
  shows the change and its price; a longer lease is a billing decision, and
  Kura never extends it on its own. A render Pod's lease runs from the
  Pod's start and changes the same way.
- `--unattended-wait auto`: after training, if the outputs were not collected,
  the Pod deletes itself after the longer of 2 hours and the job time
  (from remote job start, including model download). Collecting the outputs marks the
  Pod so the timer leaves it to the controller;
  a download in progress marks it too, so the timer waits for it.
- An explicit `kura run reconcile` records a Pod that no longer exists (it
  deleted itself or was deleted elsewhere) as `interrupted` with
  `pod_missing_at`; whatever the Pod held is gone. Automatic observation never
  does. Before relaunching such a run, confirm in the RunPod console that the
  Pod is really gone.
- Kura records its intent before creating a Pod. If a launch dies, or RunPod
  does not confirm a create, Kura never creates again on its own: launch and
  stop refuse until `kura run reconcile` has looked for the Pod by name. A Pod
  it finds is recorded as `interrupted` and still bills until
  `kura run stop <run-id>` deletes it (with any duplicate); finding none
  records the launch as failed. Tell the user which one happened.
- Pod-side deletion (shared by training and render Pods) uses the Pod-scoped `RUNPOD_API_KEY` from the init process
  and calls `podTerminate`, then `podStop`, with a non-Python User-Agent. Do not
  rely on the Pod's preinstalled `runpodctl`; its syntax follows the version
  RunPod ships.
- `--job-timeout 0`: wait until remote exit.
- `runpod.storage_mode: upload`: no Network Volume by default. Manifest-v2
  runs require it: their inputs arrive only through the verified
  selected-file transfer. A Pod-side verification failure writes
  `realizations/<id>.runpod-input.json` with `status: failed`, never starts the
  trainer, and the normal download-then-stop path still runs. Launch pins the
  proven manifest in `realizations/<id>.transfer-manifest.json`; the Pod
  trusts only that pin. If the stage changes after launch, the controller
  refuses before uploading and stops the unused Pod at once.
  If stage or launch says "stage it again", the compile or staged files
  changed; rerun the stage rather than editing any staged file.
- Treat configured GPU candidates as workspace policy, not durable skill
  knowledge. Inspect current availability and price before selecting one.
  After that choice, compile the run and inspect the compiled resource plan
  before approval.
- Kura places a Pod only on hosts whose driver supports the image's CUDA
  version, and the stock and price it shows already apply that filter. Do not
  add a CUDA setting. An image Kura does not know gets only hosts with the
  newest CUDA Kura has seen; the plan warns, and fewer GPUs is the expected
  effect.
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
  When the run's training state is managed (`kura run plan` shows
  `saved: yes`: recovery is enabled and the backend can resume this
  architecture and mode), a completed trainer must also leave a durable
  training state. The downloaded snapshot holds everything the Pod had, so a
  missing state, or outputs that cannot be published, cannot be fetched by
  collecting again: Kura records the run as `recovery_required` with the reason
  (as on Docker), keeps the downloaded files, deletes the Pod, and notifies;
  `kura run download` exits 3 for it. A failed or stopped trainer may have
  ended before its first save; with no state it completes as failed with no
  error, and has nothing to resume from. A run whose state is not managed
  saves none and is never held for it.
- The SSH job exports the frozen command's `env` (Kura's own variables win);
  an SSH session does not inherit the Pod's create-time environment.
- Terminal finalization reuses only checkpoints recorded by the periodic mirror
  or protected training-state bytes whose size and SHA-256 match the post-exit
  remote manifest. It downloads missing or changed files, verifies a second
  remote inventory and the complete staged snapshot, then publishes atomically.
  Do not replace this with a broad output exclusion or treat periodic mirror
  metadata alone as completion.
- If download/completion is uncertain, leave the Pod running and print/notify recovery steps.
- Do not add unbounded keep-alive flags. Use bounded leases only.
- `max-lease` is a billing safety fuse, not output preservation. Do not set it shorter than expected training unless loss of container-disk outputs is acceptable.
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

The job runner launches, follows, collects, and deletes the Pod; `kura run
execute` only confirms billing and follows. When the command ended, run
`kura run execute <run-id>` again to follow; it never launches a second Pod.
The runner deletes a Pod whose job never started (nothing can be collected
there), and after three failed collections it marks the run
`recovery_required` and notifies; then collect and stop by hand with the
commands below. A job started before the runner existed is still followed
in-process by `kura run execute`.

```sh
kura run execute <run-id>
kura doctor runpod
kura run reconcile <run-id>
kura run download <run-id> --force
kura run pull <run-id> --since-step 1000
kura run stop <run-id>
```

`kura run stop` refuses while the Pod holds work not collected yet (outputs of
an ended job, or checkpoints of one still training) and says how to collect
it. Tell the user what would be lost; add `--yes` only on their instruction.

## Resume on a replacement Pod

Resume is available only from a training-state artifact that completed local
publication before the source Pod disappeared. Use this sequence:

1. Confirm the source run's recoverable artifact and that the old Pod is no
   longer an active billing resource with `kura run plan <source-run>`
   and `kura doctor runpod` as applicable.
2. Create the derived draft with `kura run resume <source-run>
   --additional-steps <N> --executor runpod --gpu <gpu>`. Kura selects the
   latest valid state unless the user explicitly names an older artifact.
3. Compile with `kura run compile <derived-run>`, then show `kura
   run plan <derived-run>`. State the restoration level, restored and missing
   components, source/target logical steps, selected artifact size, GPU, price,
   and capacity policy. A GPU or executor change is a new cost decision and
   requires approval of this plan.
4. After approval, use `kura run execute <derived-run> --yes`. The new
   Pod must receive and verify only the selected protected artifact; it must
   not depend on the old Pod or a Network Volume.
5. Confirm the requested additional optimizer updates, new logical state
   publication, remote exit, and local download before the replacement Pod is
   stopped. Finish with `kura doctor runpod` when lifecycle state is
   uncertain.

Kura's disposable-Pod smoke validated this transport and cleanup path for the
three supported backend families. It does not establish bit-for-bit numerical equivalence; use the backend's
restoration contract and matching numerical evidence for that judgment.

After a long unattended capacity wait, run `kura doctor runpod` to
confirm that no unrecorded Pod remains before retrying or leaving RunPod.

If RunPod fails before receiving a request with an OS-level permission error,
the current agent process may lack network access. Use
`.kura/reference/external-access.md` for the agent-specific setup. Do not classify that as
a RunPod outage or add a Kura-side network bypass.

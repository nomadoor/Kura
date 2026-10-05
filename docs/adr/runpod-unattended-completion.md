# A RunPod run without its controller waits as long as the training took

Status: accepted owner decision.

Date: 2026-10-02

Updated: 2026-10-05 — an uncollected Pod stops instead of deleting itself, so
its outputs survive until a controller returns; relay storage becomes optional.

## Context

Today only the local Kura controller collects a RunPod run's outputs and then
deletes the Pod. If the controller is gone when training ends, the Pod keeps
billing until its maximum lease, 12 hours by default. The controller can be
gone because of a closed session, a sleeping PC, or a stopped job runner. This
has happened.

The Pod cannot simply delete itself when training ends. Its disk is
disposable, so deleting it before collection destroys the trained outputs.
What is lost grows with the training time: deleting after a 30-minute run
loses little, while losing a 24-hour run is severe.

## Decision

**Outputs live where a stop keeps them.** A training Pod writes its outputs
to its volume disk (`/workspace`), which survives a stop and is deleted only
with the Pod. Its container disk, which a stop erases, holds nothing Kura
needs after training. A RunPod run therefore always has a volume disk large
enough for its outputs.

**With the controller present, nothing changes.** The controller waits for the
remote exit, downloads, verifies, and then terminates the Pod. No upload or
other step is added to this path.

**After training ends, an uncollected Pod waits, then stops.**

- The wait starts when training ends and lasts the longer of 2 hours and the
  time the training took. The training time is measured from the remote job
  start, so it includes input transfer and model download.
- When the wait ends without collection, the Pod stops itself instead of
  deleting itself. The GPU is released, so only the volume disk is billed, and
  the outputs stay on it.
- A controller that starts collecting marks the Pod, so the timer waits for an
  in-progress download instead of stopping the Pod under it.
- The maximum lease still runs from the remote job start and ends billing for
  compute. It stops the Pod the same way, so reaching it no longer destroys
  outputs either.
- The billing confirmation shows the wait, the lease, and the volume disk
  before launch, and the user can change them (`--unattended-wait`,
  `--max-lease`).

**A returning controller finishes the job.** When a controller finds a run
whose Pod stopped before collection, it starts the Pod again, downloads and
verifies the outputs, and then terminates it. RunPod may start it with no GPU
when the original machine is busy; collection needs none. Terminating is the
only step that deletes the outputs, and it happens only after they are
verified locally.

**Relay storage stays optional.** For a user who expects to be away longer
than they are willing to pay for a stopped volume, a later option can upload
outputs to temporary relay storage before stopping. It is never on by default
and never runs when the controller is present.

## Consequences

- A run left unattended costs at most about two hours of idle GPU time, then
  only volume storage until a controller returns. Its outputs are no longer
  lost when nobody collects them in time.
- Every RunPod run now pays for a volume disk while it runs and while stopped.
  `kura init` and the plan size it from the expected outputs; the previous
  default of no volume disk is no longer valid for training.
- The Pod-side helper (`container_scripts/pod_self_delete.sh`) changes with
  this decision: the unattended wait and the maximum lease call `podStop` only,
  and `podTerminate` is reserved for the controller after it has verified the
  downloaded outputs. The terminate-then-stop order described below is the
  behavior this decision replaces.
- Before this ships, real smokes must show that a Pod can stop itself from
  inside, that a stopped Pod keeps `/workspace`, and that a Pod restarted with
  no GPU accepts SSH for the download.
- The remote job script gains a post-exit timer. With the controller present,
  transfer, training, collection, and Pod stop are unchanged, so existing RunPod
  evidence carries over through a behavior-preserving identity migration; the
  unattended path is proven by its own real smokes.
- Implementing this showed that the previous Pod-side maximum lease had never
  worked: it called `runpodctl pod delete`, which the `runpodctl` RunPod
  preinstalls does not understand, and Python's default User-Agent is rejected
  by RunPod's edge. Pod-side deletion now reads the Pod-scoped
  `RUNPOD_API_KEY` from the init process and calls `podTerminate` (then
  `podStop`) with its own User-Agent; the maximum lease uses the same path.
- RunPod's `terminateAfter` create field was tried as a RunPod-side deadline;
  it was accepted at creation but not enforced, so Kura does not rely on it.
- Render session Pods had the same broken lease guards; they now use the same
  self-delete (`container_scripts/pod_self_delete.sh`).
- A Pod that deleted itself is a normal outcome, so `kura run reconcile` records
  a missing Pod as `interrupted` instead of failing.

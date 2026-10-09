# A RunPod run without its controller waits as long as the training took

Status: accepted owner decision.

Date: 2026-10-02

Updated: 2026-10-10 — a training Pod arms its maximum lease when the Pod
starts, like a render session Pod, instead of when Kura first reaches it.

Updated: 2026-10-08 — a person or agent can move the lease deadline of a running
Pod, and Kura warns when training looks unlikely to finish inside it; see
"Changing the lease".

Updated: 2026-10-07 — the maximum lease starts when Kura first reaches the Pod,
not when the job starts.

Updated: 2026-10-05 — keeping outputs past the wait by stopping the Pod, or by
writing them to other storage, was evaluated and not adopted; see "Alternatives
measured".

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

**After training ends, the Pod waits for collection, then deletes itself.**

- By default the wait is the longer of 2 hours and the time the training took.
  `--unattended-wait <duration>` sets it explicitly, and `0` disables the
  post-training timer.
- The training time is measured from the remote job start, so it includes
  input transfer and model download.
- The maximum lease starts when the Pod starts: the start command Kura gives
  every training Pod sets the deadline and starts the guard before anything
  else, and the lease still ends everything (amended 2026-10-10: it used to
  start when Kura first reached the Pod over SSH, so a Pod whose controller or
  job runner stopped before that contact had no timer; amended 2026-10-07: a
  Pod that failed before its job started used to have no timer at all). The
  guard Kura starts again over SSH never moves a deadline already set, and the
  run records the deadline the Pod holds.
  The wait therefore never outlasts the lease time remaining when training
  ends: the lease can end the Pod before the wait finishes, so a run expected
  to take longer than the lease needs a lease covering training plus the wait.
- The billing confirmation shows the wait and the lease before launch, and the
  user can change both (`--unattended-wait`, `--max-lease`).
- A controller that starts collecting marks the Pod, so the timer waits for an
  in-progress download instead of deleting the Pod under it.

**Optional relay storage: not adopted (2026-10-08)**

The job runner (`files-only-state-and-job-runner.md`) collects outputs when the
command that launched a run is gone, and the Pod waits at least two hours for
it, so only a PC that stays off longer loses outputs; the user accepts that.
The relay below was weighed against that remaining case and found too complex
for its return. The original design is kept for reference:

- When it is enabled, the Pod uploads its outputs to a relay destination
  after training and deletes itself immediately.
- Destinations are pluggable. A private Hugging Face repository comes first,
  using a write-scoped token; other storage follows.
- The controller downloads from the relay, verifies the files, and then
  deletes the relay copy. The relay is temporary transfer storage, not an
  archive.

## Changing the lease (2026-10-07)

The Pod keeps its lease deadline in a file and deletes itself once that time
passes. `kura run lease <run-id> <duration>` sets the deadline to that long from
now, after showing the current and the new deadline and taking the same
confirmation a launch takes; it is recorded in the run. Kura never moves the
deadline on its own: a longer lease is a billing decision.

A render session Pod keeps its deadline in the same file (2026-10-08), so the
same command changes it. Its lease starts when the Pod starts, as a training
Pod's does (2026-10-10): the Pod sets the deadline itself from the same start
command guard, and a second guard started over SSH never moves a deadline
already set.

While it follows a job, Kura estimates the time left from the training
progress. When training plus collection looks unlikely to finish before the
deadline, it warns once in the run's log and by notification, naming the
command that extends the lease.

## Consequences

- With the default wait, a short run left unattended now costs at most about
  two hours of idle Pod time instead of twelve.
- A long run keeps its outputs for at least as long as it took to produce
  them, unless the maximum lease expires first.
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

## Alternatives measured (2026-10-05)

The loss that prompted this review came from the controller, not the Pod: an
agent host ended `kura run execute` after two hours. That is answered by a
controller that outlives the agent (`files-only-state-and-job-runner.md`) and,
until it ships, by `kura run execute` resuming collection of a running run.
Two ways to keep outputs past the wait were tried and set aside:

- **Stop instead of delete.** A stopped Pod keeps its volume disk, but that
  disk belongs to the physical machine, and the API starts a stopped Pod only
  when that machine has a free GPU again (`podResume` with `gpuCount: 0` is
  ignored; starting without a GPU is offered only in the RunPod console). A
  real smoke was refused for that reason. Keeping the whole workspace on the
  volume also bills about $1 a day for 150 GB while stopped.
- **Upload outputs elsewhere, then delete.** RunPod network volumes survive
  deletion but pin the Pod to one Secure Cloud datacenter, which removes most
  of the GPU types and prices Kura uses. A user-owned S3-compatible bucket
  written through controller-presigned URLs avoids credentials in the Pod but
  adds an account, an upload on the billed Pod, and new failure paths; it
  stays a candidate for users who leave their machine off for longer than the
  wait.


# A RunPod run without its controller waits as long as the training took

Status: accepted owner decision.

Date: 2026-10-02

## Context

Today only the local Kura controller collects a RunPod run's outputs and then
deletes the Pod. If the controller is gone when training ends, the Pod keeps
billing until its maximum lease, 12 hours by default. The controller can be
gone because of a closed session, a sleeping PC, or a stopped `kura ui`. This
has happened.

The Pod cannot simply delete itself when training ends. Its disk is
disposable, so deleting it before collection destroys the trained outputs.
What is lost grows with the training time: deleting after a 30-minute run
loses little, while losing a 24-hour run is severe.

## Decision

**After training ends, the Pod waits for collection, then deletes itself.**

- The wait is the longer of 2 hours and the time the training took.
- The training time is measured from the remote job start, so it includes
  input transfer and model download.
- The maximum lease runs from the remote job start and still ends everything.
  The wait therefore never outlasts the lease time remaining when training
  ends, and a run expected to take longer than the lease needs a longer lease.
- The billing confirmation shows the wait and the lease before launch, and the
  user can change both (`--unattended-wait`, `--max-lease`).
- A controller that starts collecting marks the Pod, so the timer waits for an
  in-progress download instead of deleting the Pod under it.

**Optional relay storage, to be added later**

- When it is enabled, the Pod uploads its outputs to a relay destination
  after training and deletes itself immediately.
- Destinations are pluggable. A private Hugging Face repository comes first,
  using a write-scoped token; other storage follows.
- The controller downloads from the relay, verifies the files, and then
  deletes the relay copy. The relay is temporary transfer storage, not an
  archive.

## Consequences

- A short run left unattended now costs at most about two hours of idle Pod
  time instead of twelve.
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

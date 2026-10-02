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
- The maximum lease still applies.
- The plan shows the wait, and the user can change it before approval.

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
  them.
- The remote job script gains a post-exit timer. Because it changes RunPod
  lifecycle behavior, it needs a real RunPod smoke and a behavior-changing
  executor identity record.

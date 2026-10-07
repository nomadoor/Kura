# ADR: Record intent before every external effect; status is a projection; readers never write

Status: accepted owner decision.

Date: 2026-10-04

Updated: 2026-10-08 — the status projection runs in shadow mode first.

Updated: 2026-10-07 — the job runner deletes a Pod of its own launch whose job
never started, and continues a confirmed RunPod launch that created nothing yet.

Refines `files-only-state-and-job-runner.md` for the runner and `kura ui`.

## Context

`files-only-state-and-job-runner.md` makes the run files the only state and
lets one runner per workspace control launched runs. It says what may run, not
how a run's records stay true when a process dies between two steps. Comparing
Kura with a durable agent harness that commits every step before showing it
found these gaps in today's code:

- A RunPod Pod is created before anything records it. The realization, and
  the phase saying creation was requested, are written after the API call
  returns. A crash in between leaves a billed Pod that `kura run reconcile`
  cannot find, because it looks Pods up only by the id the realization holds.
  Only the Pod-side maximum lease stops it. The attempt loop also retries a
  create that failed transiently, so a create that timed out but succeeded can
  start a second Pod within one launch. Docker containers have the first gap,
  without the bill. Starting the remote job over SSH is recorded after it
  starts, too.
- `status.json` is changed step by step. `CONTEXT.md` calls it derived, but
  some of its fields, such as training progress, the capacity wait, and the
  stop time, exist nowhere else, so it cannot be rebuilt.
- The launch, observation, and stop records say what they are only through
  their file names.
- Commands that only show runs, such as `kura monitor` and `kura run status`,
  reconcile while they read, so a viewer writes `status.json`. With a runner,
  that is a second writer.

## Decision

**1. Intent before effect.** Before Kura starts anything outside the
workspace that can cost money or outlive the command, it records the intent
together with the deterministic name the effect will carry, then acts, then
records the outcome. This covers creating a Pod on both RunPod launch paths
(training and session), creating a Docker container, and starting the remote
job on a Pod. A local render run calls an existing ComfyUI endpoint and
creates nothing, so only decisions 3 to 7 apply to it.

- Intent found without an outcome is resolved by **discovery, never by acting
  again**. For a Pod, discovery lists the account's Pods by the deterministic
  name. That name contains the run id and the realization id, which is unique
  to the intent, so only a Pod Kura created for this intent carries it; any
  other Pod is never a candidate and is never touched. RunPod names are not
  unique in general, and Kura had no list call, so this needs a list operation
  proven by its own real smoke. For a container discovery lists by the
  realization label, and for a remote job it checks the job's marker on the
  Pod.
- Nothing found means the effect never happened only when the listing covered
  every Pod of the account; a listing that fails or may be partial leaves the
  intent unresolved, and the user is told so.
- One match found by a live launch is adopted and the launch continues.
- Recovery after a crash, and any case with more than one match, never adopts a
  Pod into a running job and never deletes one: it records the run as
  `interrupted` with every matching Pod's id, and `kura run stop` deletes them
  all. The user decides whether to stop or keep a Pod that is still billing.
  One exception (2026-10-07): when the job runner finds a Pod of its own launch
  whose records show the remote job never started (no remote-job intent, or an
  intent whose pid file the Pod does not have), it deletes the Pod, records the
  run as interrupted, and notifies. Nothing on that Pod can be collected, and
  waiting for a person only bills.
- A create that fails with a timeout or a transient error ends the attempt
  loop and goes to discovery. Only an explicit refusal, such as no capacity,
  may try the next GPU candidate.
- RunPod has no idempotency key for creation, so the Pod-side maximum lease
  remains the backstop.

**2. Steps that create are never retried blindly.** Reading state, verified
downloads, and stops that accept "already gone" may be retried. A step that
creates something or starts a job is resolved by discovery. When discovery
cannot tell what happened, Kura records the step as `interrupted`, keeps a
live Pod for collection, and hands the run to the user. Nothing relaunches on
its own.

**3. A launch request is claimed by an exclusive file.** A runner claims a
request by creating its claim file exclusively, so a claimed request is never
launched again whether or not any lock is still held; the realization cites
the claim. This replaces "records the claim in the realization" in
`files-only-state-and-job-runner.md`. Stopping a run is likewise a stop
request file that the runner, or the CLI when no runner is running, carries
out and records. A claim with no realization and no create intent after it means
the runner died before acting; nothing external exists, so the next runner, or
`kura run reconcile` when none is running, records the request as not
launched, and the request can be written again. A RunPod request is the
exception (2026-10-07): its billing was confirmed, and with no create intent no
Pod exists, so the next runner continues that launch, including a wait for
capacity. Claims rely on exclusive create on a local file system,
which is where the runner ADR and `windows-execution-model.md` place the
workspace; network shares are not supported.

**4. Writers are fenced by an epoch.** The runner file carries an epoch that
grows each time a runner starts. Claims, realizations, and `status.json`
record the epoch that wrote them, and a status update applies only if the
`last_realization` and epoch it was based on are still current. A CLI writer
records the current runner epoch, or 0 when there is no runner file; CLI
writers are serialized by the run operation locks as today. The advisory lock
shows who is alive; the epoch keeps a replaced writer from overwriting newer
facts.

**5. Status is a projection.** Every field of `status.json` has a record it
comes from: progress is written as observations, and the capacity wait and
the stop each write a record. One function builds a run's status from those
records, and `status.json` is its output, kept for fast reading. Deleting it
and projecting again gives the same result, and a test holds Kura to that.

The projection starts in shadow mode (2026-10-08): `status.json` is still
written step by step, and after each write the lifecycle fields the
projection covers are compared with it; a difference is appended to the run's
`logs/status-shadow.jsonl` and changes nothing else. Status is written from the
projection only once real runs show no differences.

**6. Records say what they are.** The launch, observation, publication, exit,
and stop records, and `status.json`, carry `kind` and `schema_version`.
Readers choose by `kind`, read every version they know, and never rewrite a
record to migrate it. A record whose kind or version a reader does not know
makes the projection report the run as having unreadable records, never as
if the record were absent. Appending to a log first closes a line that a
crash left unfinished.

**7. Readers never write.** Commands and views that only show runs,
`kura monitor`, `kura run status`, run listings, and `kura ui`, read the files
and never reconcile. Reconciling belongs to the runner for runner-controlled
runs, and to `kura run reconcile` and to a command that is following the run
it launched for the rest. A viewer shows when each run was last observed or updated, so a
stale view is visible rather than silent.

## Consequences

- The RunPod and Docker launch paths change order. Without a crash they make
  the same requests as before, so existing evidence carries over through a
  behavior-preserving identity migration; discovery is new behavior and is
  proven by its own real smokes.
- Run directories gain claim and stop request files, status gains the epoch,
  and records gain `kind` and `schema_version`. Older runs stay readable: a
  record without `kind` is read by its file name, as today.
- Until the runner exists, viewers keep reconciling as they do now. Decision 7
  applies from the runner's first release, together with the "last observed"
  display.
- Interrupted creates hand runs to the user more often than an optimistic
  retry would. That is the intended trade: an extra question costs less than a
  second billed Pod.

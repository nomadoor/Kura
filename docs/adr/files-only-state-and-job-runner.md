# Files are the only state; a stateless job runner may run

Status: accepted owner decision.

Date: 2026-10-02

## Context

Since its first release Kura has listed "a database, queue, daemon, or hidden
lifecycle state" as a non-goal, and AGENTS.md repeats it. The purpose was never
the absence of a process. The purposes were these:

- **One source of truth.** No second store can disagree with the run files.
- **Recovery.** Whatever crashes, the files are enough to reconcile a run.
- **Shared view.** Users, agents, and Git all read the same facts.
- **Nothing extra to install.**

The planned Web UI needs two things that the old wording forbids:

- training that keeps running when the chat or browser closes;
- a dashboard of parallel work.

Both need a process that keeps watching runs, and today the agent's own
session plays that role through `kura run execute`.

## Decision

The rule becomes: **the run files are the only authoritative state.** A
long-running process is allowed only if all of these hold:

- It holds no state of its own. Everything it acts on or produces is written
  to run files before anything relies on it.
- It can crash at any point. A replacement continues from the files alone.
- The CLI alone can inspect and recover every run, whether or not the process
  is running.

The job runner lives inside the `kura ui` server process. That process runs
where Kura runs: on Linux directly, and inside WSL2 on Windows. A browser
reaches it on `localhost`.

Launching is decoupled through a **launch request**. Approving a plan, from a
UI control or in chat, writes a launch request file into the run, and the job
runner picks it up. `kura run execute` keeps working directly without the
runner.

Kura records no separate approval. A launched run was approved, and a plan
that was not approved stays unlaunched. The reason for not approving shows up
as the next, revised run.

Stopping `kura ui` while it controls a RunPod run warns first. Until the
server returns, nobody collects the outputs or stops the Pod.
`runpod-unattended-completion.md` bounds what that costs.

## Consequences

- This supersedes the "database, queue, daemon" non-goal in
  `end-to-end-run-contract.md` and the matching AGENTS.md Core Model
  sentence. The other non-goals stand.
- The `kura-core` and `monitor-tui` skills must state the new rule instead of
  "no daemon".
- Two writers can act on the same run: the job runner and the CLI. They
  coordinate through the existing run operation locks. No in-memory
  coordination is allowed.

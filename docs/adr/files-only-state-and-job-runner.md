# Files are the only state; an independent job runner controls launched runs

Status: accepted owner decision.

Date: 2026-10-02

Updated: 2026-10-03 — the job runner is its own process, and every launch goes
through it.

## Context

Since its first release Kura has listed "a database, queue, daemon, or hidden
lifecycle state" as a non-goal, and AGENTS.md repeats it. The purpose was never
the absence of a process. The purposes were these:

- **One source of truth.** No second store can disagree with the run files.
- **Recovery.** Whatever crashes, the files are enough to reconcile a run.
- **Shared view.** Users, agents, and Git all read the same facts.
- **Nothing extra to install.**

The planned Web UI needs two things that the old wording forbids:

- training that keeps running when the chat or the browser closes;
- a view of parallel work.

Both need a process that keeps watching runs. Today the agent's own session
plays that role through `kura run execute`, so a lost session leaves a run
without its controller.

The first version of this decision put that process inside the `kura ui`
server, which would tie training to the UI instead.

## Decision

The rule becomes: **the run files are the only authoritative state.** A
long-running process is allowed only if all of these hold:

- It holds no state of its own. Everything it acts on or produces is written
  to run files before anything relies on it.
- It can crash at any point. A replacement continues from the files alone.
- The CLI alone can inspect and recover every run, whether or not the process
  is running.

**The job runner is its own process, `kura runner`.**

- It runs where Kura runs: on Linux directly, and inside WSL2 on Windows.
- One runner serves one workspace. It holds an advisory lock on a file in the
  workspace, which the operating system releases when the process dies, and
  records its process ID and Kura version in that file.
- A run is **runner-controlled** when its latest realization says so. That is
  a file fact, so a replacement runner knows which runs to continue.
- It exits on its own when no launch request is pending and no
  runner-controlled run is unfinished. It makes this check while still
  holding the lock, so a request written before the check is never lost.
- Anything that writes a launch request writes it first, then starts the
  runner if none holds the lock. If a runner holds the lock, the writer
  watches until the request is claimed; when the lock is released with the
  request still pending, because that runner made its final check just
  before the request was written, the writer starts a new runner.
  `kura runner start` starts it by hand.
- A runner launches only requests written by its own Kura version. A request
  from another version stays pending, and the runner and `kura runner status`
  report it. A later writer replaces that request after saying so.
- On start, the runner reconciles every unfinished runner-controlled run,
  then continues them. This is how runs recover after a crash, a reboot, or a
  sleep. `kura run reconcile` stays the CLI path that needs no runner.
- `kura runner stop` detaches the runner. Containers and Pods keep running,
  and the runs stay unfinished until a runner starts again. For a RunPod run it
  warns first; `runpod-unattended-completion.md` bounds what an absent
  controller costs.

**Every launch goes through a launch request.** This covers training and
render runs alike.

- A launch request is a file in the run. It names the run lock it launches
  and records everything the launch needs that the run lock does not hold:
  launch options such as the maximum lease, the unattended wait, the image,
  and notifications, and the Kura version that wrote it. The realization
  cites the request it came from.
- Confirmations happen before the request is written, never in the runner.
  The RunPod billing confirmation that `kura-decision-model.md` requires is
  shown by the writer: `kura run execute` at the terminal or through
  `--yes`, and the UI in its approval control. The request records that the
  confirmation was given. The runner never prompts.
- The runner claims a request under a per-run launch lock, and records the
  claim in the realization it starts. A claimed request is never launched
  again, and a run has at most one pending request.
- `kura run execute` and the other launching commands, such as
  `kura render launch`, write a launch request, make sure a runner is running,
  and follow the run until it finishes, returning its result as before.
  Interrupting them stops the following, not the run.
- The `kura ui` server never controls runs. It writes launch requests and
  shows the workspace, so stopping it never stops training.

**Approval.** A launch request is written only after the approval that the
run requires today, unchanged: in the UI by a click on its approval control,
and in an agent session through the conversational approval `AGENTS.md`
requires, before the agent runs the launching command.

Kura records no separate approval. A launched run was approved, and a plan
that was not approved stays unlaunched. The reason for not approving shows up
as the next, revised run.

## Consequences

- This supersedes the "database, queue, daemon" non-goal in
  `end-to-end-run-contract.md`, and its file roles gain the launch request.
  The AGENTS.md Core Model sentence and the `kura-core` and `monitor-tui`
  skills must state the new rule. `runpod-unattended-completion.md` names the
  runner instead of `kura ui` as the absent controller.
- The runner and the CLI can both act on one run. Launching takes the per-run
  launch lock; other operations keep the existing run operation locks. No
  in-memory coordination is allowed.
- Losing an agent session no longer leaves a run without a controller. The
  rule that an agent keeps `kura run execute` tracked until it returns stays,
  because the agent still owes the user the result.
- Upgrading Kura can crash an active runner, because the installed files
  change under it. The crash rule makes that recoverable: the next start runs
  the new version and reconciles. A writer that finds a runner of another
  version says so before writing its request; the user can run
  `kura runner stop`, which is safe, and the next start runs the new version.
- Whether the runner keeps the WSL2 VM running after every Windows-side
  process exits is part of the open verification in
  `windows-execution-model.md`.

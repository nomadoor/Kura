# Files are the only state; an independent job runner controls launched runs

Status: accepted owner decision.

Date: 2026-10-02

Updated: 2026-10-03 — the job runner is its own process, and every launch goes
through it. How records stay true across crashes, and who may write them, is
refined in `run-records-and-external-effects.md` (2026-10-04).

Updated: 2026-10-07 — how the runner starts, runs launches, reads secrets,
queues local training, and is followed, stopped, and woken; see "Runner
mechanics".

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
  records its process ID and Kura version in a file beside it ("Runner
  mechanics").
- A run is **runner-controlled** when its latest realization says so. That is
  a file fact, so a replacement runner knows which runs to continue.
- It exits on its own when no launch request is pending and no
  runner-controlled run is unfinished. It makes this check while still
  holding the lock, so a request written before the check is never lost.
- Anything that writes a launch request writes it first, then starts the
  runner if none holds the lock. If a runner holds the lock, the writer
  watches until the request is claimed; when the lock is released with the
  request still pending, because that runner made its final check just
  before the request was written, the writer starts a new runner ("Runner
  mechanics" bounds how often).
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
- The runner claims a request under a per-run launch lock by creating its
  claim file exclusively, and the realization it starts cites the claim
  (`run-records-and-external-effects.md`). A claimed request is never launched
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

## Runner mechanics (2026-10-07)

**Files.** The runner's lock file only carries the advisory lock. A separate
`runner.json` records its process ID, Kura version, Python environment, epoch,
and start time; it is replaced atomically, which would break a lock held on the
same file. These files live in `.kura/runner/`, which Kura owns; refreshing
shipped files never touches it.

**One child per launch.** The runner supervises; it does not run launches
itself. For each claimed request it starts one child process that runs the
existing launch and follow code for that run, and writes its own output to
that run's logs. A child that dies is replaced by one that follows the run from its
records: it follows the recorded container or Pod and never calls the launch
path again. A child holds its run's controller
lock for its whole life, so a runner that replaces a dead one never starts a
second follower beside a child that outlived it; the new follower waits for
that lock. The runner starts children with the
same Python environment it runs in, never a `kura` found on `PATH`.

**Detached start, no secrets in the environment.** The runner starts in a new
session, with no terminal, so it survives the terminal, the agent, or the SSH
session that started it. It inherits the starting environment except every
name Kura treats as a secret (`user-secrets.md`). Each child reads secrets when
it starts, with the precedence `user-secrets.md` sets; because the runner's
environment carries no secrets, they come from Kura's secrets files, and a
value exported in the shell that wrote the request does not reach a runner
launch. A child that reaches a step needing a missing secret records the run
as failed; it is not retried. Nothing in a launch path may prompt; a prompt
that reads end-of-input is answered no.

**Local training is queued.** The runner starts local Docker training runs up
to `runner.local_slots` in `workspace.yaml` at once (default 1) and keeps later
requests pending, in the order they were written, which their ids record, so a
replacement runner keeps the order. The status projection shows a pending request as queued and names what
it waits for. RunPod runs start at once. Local renders are not counted, because
they use an existing ComfyUI endpoint.

**No heartbeat.** The lock shows whether the runner is alive. A runner that
holds its lock but has stopped working is not detected; viewers show when each
run was last observed, so a stall is visible.

**Following.** `kura run execute` writes its request under the per-run launch
lock, so a run has at most one pending request, then follows by reading the
records and the log. It returns when the run is finished: the exit is
recorded, outputs are published, and post-training input verification is
done. Its exit code is
the run's: zero for success, nonzero for failure, interruption, or a run that
needs recovery, with the next step printed. Run again on the same run, it
follows a pending or running launch and only prints the result of a finished
one; it never launches a finished run again. When it sees the runner's lock
released while its run is unfinished, it starts a runner, at most once per
command, and reports the runner's exit from its log if that start fails too.

**Waking the runner.** A runner that dies, or that a reboot or sleep ends,
leaves no mark. Then any `kura` command, including commands that only show
runs, starts a runner when a runner-controlled run is unfinished and no runner
holds the lock; showing a run still never writes it. After a reboot, the first
`kura` command therefore resumes control. `kura runner stop` leaves a mark
that it was stopped on purpose; until `kura runner start` or a launching
command removes it, no command wakes the runner, and `kura run reconcile`
reconciles runs itself, as it does with no runner. `kura run reconcile`,
`kura runner stop`, and `kura runner status` never wake the runner. A RunPod run that nobody looks at
within its unattended wait (`runpod-unattended-completion.md`) loses its
uncollected outputs, as before.

**Stopping.** While a runner holds the lock, `kura run stop` writes a stop
request; the runner hands it to that run's child, which stops the container or
Pod the way `kura run stop` does today and records the stop. With no runner,
the CLI stops the run directly, as today.

**Viewers.** From the runner's first release, commands that show runs no
longer reconcile runner-controlled runs (`run-records-and-external-effects.md`,
decision 7).

**Interrupted work.** A training run whose child died with its container or
Pod gone is recorded as interrupted. Continuing it is a new decision made with
`kura run resume` from its saved training state; the runner never resumes on
its own. A render run whose child died is recorded as interrupted with the
images it already wrote; it is not continued from a later case, and its RunPod
Pod, which holds nothing those images lack, is deleted.

**Servers.** Kura can run on a Linux or WSL host that the user operates over
SSH. The workspace, runner, Docker, GPU, secrets, and agent are all on that
host; the user's own machine only provides a terminal, or a browser over a
forwarded port. A Linux host that kills a user's processes at logout stops the
runner; `kura doctor` reports that setting and names the command that exempts
the user (`loginctl enable-linger`).

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
- On the owner's PC a detached heartbeat in WSL kept running after every
  Windows-side window closed, with Docker Desktop running (2026-10-08,
  `docs/smoke-evidence/2026-10-08-wsl-runner-lifetime.yaml`). The runner is
  detached the same way, so it is expected to survive too; the runner itself,
  the case without Docker Desktop, and Windows sleep or sign-out are not
  verified.

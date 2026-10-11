# Kura development guide

This checkout is for developing Kura. To use Kura, install it with
`uv tool install` and create a workspace with `kura init`; the usage rules live
in `src/kura/shipped/AGENTS.md`, which `kura init` writes into every workspace.

This guide is for everyone who changes Kura, human or agent. Kura is meant to
grow with more contributors, and a mistake found before it is written costs far
less than one rewritten later, so read the principles before the code.

## What kind of session is this?

**A usage session in this checkout** (training, rendering, or preparing
datasets with the repository itself as the workspace) follows
`src/kura/shipped/AGENTS.md`, reads the skills it names from `.agents/skills/`
or `.claude/skills/`, and reads `src/kura/shipped/<path>` wherever shipped text
says `.kura/<path>`. Here `kura ...` runs as `uv run kura ...`, and the user's
own knowledge layer is the ignored root `knowledge/` directory. It does not
modify Kura's source, tests, or checks, and does not change Git state unless the
user separately authorizes Kura maintenance; the need to edit Kura is a finding
to report.

**A development session** (only when the user explicitly asks to change Kura
itself: code, tests, docs, skills, or release work) follows the rest of this
file. Kura's canonical terms are in `CONTEXT.md`.

## What Kura is

- **A tool, not a policy.** Kura runs trainers (AI-Toolkit, Musubi Tuner,
  sd-scripts) and ComfyUI renders reproducibly and safely. What to train and how
  is the user's decision; Kura measures, records, and stops irreversible
  accidents. It does not judge content or second-guess an approved plan.
- **Files are the only state.** A run's files say everything about it: intent in
  `run.yaml`, frozen inputs in `resolved/`, facts in append-only `realizations/`,
  and `status.json` as a projection of them. No hidden database, queue, or second
  record of truth. The job runner is the one long-running process
  (`docs/adr/files-only-state-and-job-runner.md`).
- **Agent-first.** Agents drive Kura through the CLI and the shipped skills. The
  CLI and its errors must be clear enough that an agent with no prior context
  does the right thing.
- **Works for anyone, any time.** The same run behaves the same on every executor,
  on every supported OS, for a user who has never seen the code.
- **Minimal.** Every feature and every mechanism must earn its place.

## Principles

**1. The simplest mechanism that works.** Before adding code to prevent a
problem, ask whether it is needed at all, whether an existing mechanism already
covers it, and what the simplest effective answer is. Code enforces only what
would otherwise cause an irreversible accident: billing that does not stop, data
lost, a secret leaked, a record falsified. Everything else is a clear
instruction, a clear error, or nothing. This is about how much mechanism Kura
has, never about how carefully code is written: code that is written meets
every rule below.

**2. One decision, one owner.** Each decision Kura makes (whether a run keeps
training state, what a stopped run ends as, which GPU a run asked for, what
counts as a secret, what a checkpoint's step is) lives in one function, and
every executor, backend, and command calls it. Before writing a condition, look
for the place that already decides it. A second copy always drifts, because a
fix lands in only one of them.

**3. Design before code.** A decision is any condition that chooses behavior
from an input. Any change to what Kura does, a one-line fix inside an owner
function included, starts from a short written design the maintainer approves
(see "How a change is made"). Only a change that leaves behavior the same needs
none: wording, typos, comments, pins, generated mirrors. New maintainer
decisions about behavior, naming, information architecture, or design rules go
into an ADR first (criteria in `docs/adr/README.md`).

**4. Tests first, and across executors.** A behavior change or bug fix starts
with a test that fails for the right reason. When the behavior exists on Docker
and RunPod (or for training and render), the test runs the same input through
both and asserts the same outcome, and asserts that both call the shared owner.
Fix the cause, not the symptom: ask why the bug was possible before patching
where it showed.

**5. Records are a contract.** Run records, status fields, and frozen manifests
are read by later versions of Kura. Change their meaning only with a reader for
the old form; never rewrite a record to migrate it. Source identity changes are
declared in `docs/adapter-source-identity-migrations.yaml`.

**6. Verified by someone without your context.** Reviews and acceptance checks
are done by an agent (or person) that has not seen your reasoning: give a
reviewer the diff and the relevant ADRs, not your explanation. Before a
milestone or release, a context-free agent installs Kura and does a user task
from the shipped docs alone, and reports where it stumbled.

**7. Promises are the checklist.** Kura's promises to its users are listed once,
as P1–P11 in `docs/adr/promises-and-verification.md`. A design names the
promises it touches; breaking one is a bug.

## Boundaries

- Backends compile native configuration and container command specs; they never
  launch. Executors launch, observe, collect, and stop; they never decide
  backend policy. Training runs in Docker locally or on RunPod, never directly on
  the host; renders call a ComfyUI endpoint.
- Every surface a user or agent authors is closed: a key with no consumer is
  refused where the file is loaded, never accepted and ignored.
- Record intent before any external effect (a Pod, a container, a remote job),
  and keep viewers read-only (`docs/adr/run-records-and-external-effects.md`).
- Secrets come only from the environment and the secrets files; nothing writes
  one into a workspace file, a log, an image, or a record, and no agent handles a
  value.

## Do

- Keep a change to one coherent behavior; touch as few files as it needs.
- Reuse existing helpers; delete code a change makes obsolete.
- Keep README, docs, CLI help, and skills consistent with the behavior you
  change; check current CLI output instead of remembered flags.
- At each milestone, have a context-free agent audit the whole codebase (not a
  diff) for decisions made in more than one place and for mechanism that guards
  against something unlikely.
- Before a model family is called supported, run it for one step on both local
  Docker and RunPod.

## Do not

- Patch a symptom without finding why it was possible.
- Copy a decision into a second place, or write a condition another function
  already owns.
- Add a guard, abstraction, option, or compatibility branch nothing needs yet.
- Add a dependency without a reason the existing ones cannot meet.
- Hide a behavior change inside a refactor, or a refactor inside a fix.
- Make Kura refuse something a user may reasonably decide, unless it is an
  irreversible accident.

## How a change is made

The unit of work is one design: one decision, or a set that must change
together. Each design gets one branch and one pull request. This section
replaces any delegation given earlier in a conversation or in memory.

1. **Name the decisions.** Say which decisions the change touches and find
   every place that makes them today (search the code, not your memory).
2. **Write the design.** Five to fifteen lines: what decides it, the owner
   function, the copies that go away, the promises it touches, the behavior
   that changes, the tests that show it, and why this is the simplest form (the accident it prevents, the
   simpler option considered, and why that falls short). If it needs more, split it, or it needs an ADR. Show it to the
   maintainer and wait. Until it is approved, do not implement it; write other
   designs or stop, and never take silence for approval. An approved design is
   permission to commit, push, and open its pull request. A contributor posts
   the design as an issue or as the first section of a draft pull request and
   starts once the maintainer approves it there.
3. **Implement on its branch.** Tests first, across executors (principle 4).
   When review or testing finds another copy of the same decision, add it to
   the written design and fix it on this branch, not in a follow-up pull
   request; if that changes behavior the maintainer did not approve, show the
   updated design again before pushing. A different decision gets its own
   design.
4. **Review before the pull request.** Run the release gate. A change that
   touches training state, Resume, saving, checkpoint naming, step accounting,
   model or dataset handoff, or an image or trainer pin also runs the
   conformance scenario (Validation, below) and reports its table. For a change
   to behavior or records, also give a reviewer with none of your context the
   design and the whole branch diff (principle 6), and fix what it finds on
   the same branch.
5. **Open one pull request**, ready for review. Its body states the design,
   `Owner:` (the function), `Copies removed:` (or none), `Behavior changed:`,
   the promises touched, validation, and risks. CodeRabbit reviews it there.
   The maintainer merges.

Each review, the context-free one and the pull request's, is one round:
Critical and Major findings are fixed, and only the fixed part is checked
again. Minor and Nit findings, and pre-existing issues a review happens to
find, go to `docs/backlog.md` and are fixed later in one batch pull request.

A change with no design (step 2's exceptions) still goes on a branch and into a
pull request; pushing it needs the maintainer's go-ahead. Findings from an
audit or an acceptance test are grouped by the decision they touch before
anything is fixed; each group follows the steps above.

## Working in this checkout

```sh
git status --short --branch
git log --oneline -5
```

- Use `uv` for Python commands. Identify the relevant tests before editing.
- Stage the paths you changed by name. Never `git add -A`, `git add .`,
  `git stash`, `git reset --hard`, or `git checkout .`: they sweep up or discard
  the maintainer's uncommitted work, such as a local `.claude/settings.json`.
- Answer a question before editing anything. When a request conflicts with a
  rule here, say so and confirm before overriding it.
- Never run these unless the user asks for that run: real smokes, model
  downloads, Docker image builds or publishes, anything that creates a RunPod
  Pod, and `kura cleanup`, `kura fix-permissions`, or `kura fix-links` with
  `--yes`. Show the cost before anything that bills. Read a script before
  running it, even in a dry or preview mode.

Layout: production code `src/kura/`; tests `tests/`; Docker skeletons
`docker/`; docs and ADRs `docs/`; mechanical checks `scripts/check_*.py`;
shipped usage content (usage `AGENTS.md`, skills, knowledge, references)
`src/kura/shipped/`, which is read in a workspace and so names `kura ...`
commands and `.kura/...` paths only. Development skills live in `dev/skills/`;
`.agents/skills/` and `.claude/skills/` are generated mirrors: edit the sources,
then run `uv run python scripts/sync_agent_skills.py --write`.

Skills for development: `kura-core` for run records, paths, and surfaces;
`training-backends` for adapter work; `backend-upgrade-audit` for pinned trainer
updates; `monitor-tui` for the monitor; `release-check` before a pull request or
a milestone. Add a usage skill only when the change touches its domain.
Repository language and branch names are in `docs/agents/workflow.md`.
Workspace configuration keys are in `docs/workspace-config.md`.

Validation: focused tests first, then before a pull request

```sh
uv run python scripts/check_release.py
```

Stage new files before running it: the secrets and artifact checks read tracked
files.

The conformance scenario trains real models on local Docker, so it runs only
when step 4 or a release requires it and the maintainer has asked for the run.
It needs a separate workspace (`kura init`, with `docker.hf_cache` set to the
cache that holds the models); without `--yes` it only prints its plan:

```sh
uv run python scripts/real_smoke.py conformance --workspace <workspace> [--backend <name>] --yes
```

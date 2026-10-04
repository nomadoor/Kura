# Repository Guidelines

This checkout is for developing Kura. To use Kura, install it with
`uv tool install` and create a workspace with `kura init`; the usage rules live
in `src/kura/shipped/AGENTS.md`, which `kura init` writes into every workspace.

## Agent skills

Repository workflow configuration is recorded under `docs/agents/`. Start with
`docs/agents/workflow.md`, then use `issue-tracker.md`, `labels.md`, and
`domain.md` when the task touches those concerns. Kura's canonical terms are
defined in the root `CONTEXT.md`.

## What kind of session is this?

**A usage session in this checkout** (training, rendering, or preparing
datasets with the repository itself as the workspace) follows
`src/kura/shipped/AGENTS.md`, reads the skills it names from `.agents/skills/`
or `.claude/skills/`, and reads `src/kura/shipped/<path>` wherever shipped text
says `.kura/<path>` (knowledge, reference docs). Here `kura ...` runs as
`uv run kura ...`, and the user's own knowledge layer is the ignored root
`knowledge/` directory. It does
not modify Kura's source, tests, or checks, and does not change Git state
unless the user separately authorizes Kura maintenance; the need to edit Kura is
a finding to report. In supported Claude Code sessions, repository permissions
ask before direct Edit/Write operations in `src/`, `tests/`, `scripts/`, and
`docker/` as defense in depth.

**A development session** (only when the user explicitly asks to change Kura
itself: code, tests, docs, skills, or release work) follows the rest of this
file, starting with the `kura-core` skill. Changes to the usage rules go into
`src/kura/shipped/AGENTS.md`.

## Developing Kura

Before changing code, inspect:

```sh
git status --short --branch
git log --oneline -5
```

Use `uv` for Python commands when available, and identify the relevant tests before editing. Preserve unrelated user changes.

Git hygiene in a shared checkout:

- Stage the paths you changed by name. Do not use `git add -A`, `git add .`,
  `git stash`, `git reset --hard`, or `git checkout .`: they sweep up or discard
  the owner's uncommitted work, such as a local `.claude/settings.json`.
- Answer a question before editing anything, and when a request conflicts with
  a rule here, say so and confirm before overriding the rule.

Never run these unless the user asks for that run: real smokes, model
downloads, Docker image builds or publishes, anything that creates a RunPod
Pod, and `kura cleanup` without `--dry-run`. Read a script before running it,
even in a dry or preview mode.

New owner decisions that change behavior, information architecture, naming, writing rules, or design rules must be reflected in an ADR before implementation.
Before writing an ADR, apply the criteria in `docs/adr/README.md`.

Keep backend adapters and executors separate. Backends compile native configuration and container-native command specifications; they do not launch runs. Executors launch, reconcile, and stop runs.

Layout:

- Production code: `src/kura/`
- Tests: `tests/`
- Docker skeletons: `docker/`
- Authored examples: `examples/`
- Authored docs: `docs/`
- Shipped usage content (usage `AGENTS.md`, skills, model-family knowledge,
  regrets, workflow samples, reference docs): `src/kura/shipped/`. Text there is
  read in a workspace, so it names `kura ...` commands and workspace paths
  (`.kura/...` for shipped files); a release check refuses repository-only
  references.
- Development skills (canonical): `dev/skills/`
- Skill mirrors for agents in this checkout: `.agents/skills/` and
  `.claude/skills/` (generated from both; do not edit directly)
- Mechanical checks: `scripts/check_*.py`

Edit development skills under `dev/skills/` and usage skills under
`src/kura/shipped/skills/`, then run
`uv run python scripts/sync_agent_skills.py --write`. The release check rejects
missing skill metadata or any drift in either skill mirror.

For local workspace configuration keys, see `docs/workspace-config.md`.

Skills for development sessions: start with `kura-core`; use
`training-backends` for adapter work, `backend-upgrade-audit` for pinned trainer
updates, and `release-check` for broad handoff. Add the usage skills (shipped in `src/kura/shipped/skills/`) only when the
change touches their operational domain.

Validation — run focused tests for behavior changes; for broad changes:

```sh
uv run python -m unittest discover -s tests
uv run python scripts/check_python.py
uv run python scripts/check_no_artifacts.py
uv run python scripts/check_secrets.py
```

Before a broad handoff or push, prefer the combined gate:

```sh
uv run python scripts/check_release.py
```

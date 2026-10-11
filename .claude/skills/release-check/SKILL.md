---
name: release-check
description: Pull-request, release, and milestone quality gate for Kura. Use when preparing a pull request, a release, or a larger handoff, or when closing a milestone; covers the release gate, the conformance run, real smokes, the review round, the context-free acceptance test, and the whole-codebase audit.
---

# Release Check

## Before a pull request

```sh
git status --short --branch
uv run python scripts/check_release.py
```

Stage new files first: the secrets and artifact checks read tracked files. Run
targeted commands (`uv run kura ... --help`, a focused test module) for the
surface you changed.

An execution path (local Docker, RunPod SSH, RunPod session, ComfyUI render) is
not done until a real smoke has run through it end to end: unit tests mock the
executor and container seams where environment-contract bugs live. At least one
smoke per path uses real-sized data (several images at 1024 px or more): tiny
one-image smokes miss what only real data exercises. Real smokes
need the maintainer's request, and anything that bills needs the cost shown first.

A change that touches training state, Resume, saving, checkpoint naming, step
accounting, model or dataset handoff, or an image or trainer pin runs the
conformance scenario (`docs/adr/promises-and-verification.md`) on local Docker
before its pull request, once the maintainer has asked for the run:

```sh
uv run python scripts/real_smoke.py conformance --workspace <workspace>        # plan only
uv run python scripts/real_smoke.py conformance --workspace <workspace> --yes  # launch
```

The workspace is a separate one (`kura init`), never this checkout; set its
`docker.hf_cache` to the cache that already holds the models, which the plan
lists with their cache state. `--backend` limits it to the backends the change
touches. Put the pass/fail table in the pull request; a FAIL is a bug in the
promise it names, or a deviation the ADR already accepts.

Before publishing, check `git status --ignored --short` so datasets, runs,
downloads, caches, checkpoints, and experiments stay out of the commit.

## Review round

A pull request gets one review round. Critical and Major findings are fixed on
the branch, and only the fixed part is checked again, not a new full round.
Minor and Nit findings, and pre-existing issues the review happens to find,
are added to `docs/backlog.md` in its one-line format and fixed later in one
batch pull request. Never grow a pull request to fix a pre-existing issue.

## Before a milestone closes or a release

**Conformance.** Run the conformance scenario on every backend on local Docker,
then the RunPod pass on one backend (`--runpod --backend <name>`): it prints
the cost ceiling and launches nothing until the maintainer approves that cost
and the command is run again with `--yes`. Record both tables as smoke
evidence under `docs/smoke-evidence/`.

Two checks by agents that have none of your context, each given only what is
written below; report their findings to the maintainer with your assessment.

**Acceptance test.** Install Kura from the branch into a fresh tool environment
and create a fresh workspace with `kura init`; give the agent only that
workspace and a user task (prepare a small dataset, plan a run, run a short
local training, report the result), never the repository or this session. It
works from the shipped `AGENTS.md`, skills, CLI help, and errors alone, and
reports every place it hesitated, guessed, or was misled. RunPod tasks need the
maintainer's approval with the cost shown.

**Whole-codebase audit.** Give the agent the repository and the root
`AGENTS.md`, not a diff. It looks for decisions made in more than one place
(between executors, between training and render, between commands), for
mechanism that guards against something unlikely, and for records or status
fields without a single owner, and reports each with the places involved and a
concrete input on which the copies disagree.

Fix the findings by design, not one by one: group them by the decision they
touch, and take each group through "How a change is made" in `AGENTS.md`.

## Handoff and pull requests

- Pull request rules: `AGENTS.md`, "How a change is made".
- Report the tests and checks run, skipped external checks, and whether RunPod
  has live Pods when relevant; do not hide a dirty worktree.
- No agent attribution, local paths, secrets, or dataset details.

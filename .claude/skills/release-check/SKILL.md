---
name: release-check
description: Pre-push, pre-release, and milestone quality gate for Kura. Use when preparing a commit, PR, release, or larger handoff, or when closing a milestone; covers the release gate, real smokes, the context-free acceptance test, and the whole-codebase audit.
---

# Release Check

## Before a push

```sh
git status --short --branch
uv run python scripts/check_release.py
```

Stage new files first: the secrets and artifact checks read tracked files. Run
targeted commands (`uv run kura ... --help`, a focused test module) for the
surface you changed.

An execution path (local Docker, RunPod SSH, RunPod session, ComfyUI render) is
not done until a real smoke has run through it end to end: unit tests mock the
executor and container seams where environment-contract bugs live. Real smokes
need the owner's request, and anything that bills needs the cost shown first.

Before publishing, check `git status --ignored --short` so datasets, runs,
downloads, caches, checkpoints, and experiments stay out of the commit.

## Before a milestone closes or a release

Two checks by agents that have none of your context, each given only what is
written below; report their findings to the owner with your assessment.

**Acceptance test.** Install Kura from the branch into a fresh tool environment
and create a fresh workspace with `kura init`; give the agent only that
workspace and a user task (prepare a small dataset, plan a run, run a short
local training, report the result), never the repository or this session. It
works from the shipped `AGENTS.md`, skills, CLI help, and errors alone, and
reports every place it hesitated, guessed, or was misled. RunPod tasks need the
owner's approval with the cost shown.

**Whole-codebase audit.** Give the agent the repository and the root
`AGENTS.md`, not a diff. It looks for decisions made in more than one place
(between executors, between training and render, between commands), for
mechanism that guards against something unlikely, and for records or status
fields without a single owner, and reports each with the places involved and a
concrete input on which the copies disagree.

## Handoff and pull requests

- Report the tests and checks run, skipped external checks, and whether RunPod
  has live Pods when relevant; do not hide a dirty worktree.
- A PR is about the product change: summary, validation, risks. No agent
  attribution, local paths, secrets, or dataset details.

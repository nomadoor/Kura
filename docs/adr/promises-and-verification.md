# Kura's promises and how each is verified

Status: proposed.

Date: 2026-10-11

## Context

From 2026-10-08 to 2026-10-11 most fixes followed one pattern: a change was
checked on one backend, one dataset shape, or one executor, and the same
mechanism then failed on another (AI-Toolkit's default pruning, Musubi's unset
save cadence, the sd-scripts Resume step counter across epochs). Real smokes
were assembled by hand for each change, Resume equivalence had only ever run on
a one-item dataset, and claims about trainers were written before reading the
pinned sources. Reviews ran in rounds: each round's fixes drew a new round.

The cause is that Kura's promises to its users lived nowhere as a list. Each
change re-derived what to check, so the check followed whatever was in view.

## Decision

### 1. One list of promises

These follow from what `AGENTS.md` says Kura is (it runs trainers reproducibly
and safely, it stops irreversible accidents, it is agent-first, it works for
anyone on any supported OS). A change that can affect one names it in its
design; a violation is a bug. How Kura is built (one owner per decision,
simplest mechanism) stays in `AGENTS.md`; it is not a promise to users.

| # | Promise | Verified by |
|---|---|---|
| P1 | A run trains exactly the optimizer steps it declares (a fresh run its recipe steps; a Resume up to its target step). | conformance |
| P2 | A published checkpoint or training state is recorded, and named where Kura names it, at the optimizer step it contains. | conformance |
| P3 | A Resume continues the same training from the saved step: its step count and scheduler equal an uninterrupted run's; on the one-item conformance dataset its learned state does too when two uninterrupted runs equal each other (when they differ, the trainer is nondeterministic and an exact Resume is reported as not checkable, not as a failure). | conformance |
| P4 | A compiled run runs as its locks say: the trainer receives exactly the frozen dataset, models, image, and command. | py tests + conformance |
| P5 | The same run behaves the same on every executor and every supported OS: same outputs, states, and final status recorded. | CI + conformance (RunPod pass) |
| P6 | Billing stops: every RunPod Pod Kura creates has its maximum lease armed when it starts, and none is left billing after its run ends. | py tests + conformance (RunPod pass) |
| P7 | No data is lost: datasets are never rewritten, outputs are collected before a Pod is deleted, and nothing a user made is deleted without their yes. | py tests + conformance |
| P8 | Kura does not start a run its disk cannot hold: the launch estimate is at least the run's real peak. | conformance |
| P9 | A record never states something that did not happen, and records written by an earlier Kura stay readable. | py tests |
| P10 | No secret value reaches a workspace file, a log, an image, or a record. | py tests + release gate |
| P11 | Every refusal and error says what to do next, clearly enough for an agent with no other context. | acceptance test |

Known deviation, accepted: AI-Toolkit writes outputs under `outputs/<run-id>/`
on Docker and under `outputs/` on RunPod; both are recorded correctly in
`status.outputs` (P5 holds for records, not for the folder layout).

### 2. One conformance run instead of hand-built smokes

`scripts/real_smoke.py` (the existing real-smoke harness, extended; no second
harness) runs a fixed scenario on every backend with one small, pinned model
per backend, and checks P1–P4, P7, P8 mechanically, printing one pass/fail
table:

- datasets: one item, and several items with buckets so that a run crosses
  epochs;
- runs: a fresh run with a save cadence; a split run and its Resume to a target
  that is not a multiple of the steps per epoch; uninterrupted controls (two on
  the one-item dataset);
- checks: steps trained, saved steps and names, published state steps,
  Resume against the controls, disk peak against the estimate.

It runs on local Docker without billing. A RunPod pass runs the same scenario
on one backend, with the cost ceiling shown first, before a release.

Required: before a pull request that touches training state, Resume, saving,
checkpoint naming, step accounting, model or dataset handoff, or an image or
trainer pin; and before every release. Other changes do not run it.

### 3. Each test layer has one job

- **py tests**: Kura's own decisions, with trainers faked: one owner per
  decision (parity tests), record compatibility, refusals, messages. Every
  change.
- **CI**: the py tests on Windows, macOS, and Linux. Every pull request.
- **conformance**: real trainers against the promises (section 2).
- **acceptance test**: before a release, an agent with no context installs
  Kura, creates a workspace, and does a user task from the shipped docs alone,
  reporting every hesitation (P11, and anything the others missed).

A trainer fact that Kura code depends on (how it counts steps, saves, prunes,
resumes, or names files) cites the pinned source (file and line) in a comment;
the conformance run is what proves it.

### 4. One review round

A pull request gets one review. Critical and Major findings block and are
fixed; the fix gets a check of the fixed part only, not a new full round.
Minor and Nit findings go to `docs/backlog.md` and are fixed later in one
batch pull request. A pull request never grows to fix pre-existing issues the
review happens to find; those go to the backlog with their scope.

## Alternatives considered

- **Test every model, backend, and executor combination.** Not affordable and
  not needed: the promises are about Kura's handling, which one small model per
  backend exercises.
- **Keep hand-built smokes per change.** This is what let each fix be checked
  only where it was made.
- **Run GPU conformance in CI.** No GPU runners; the maintainer's machine runs
  it, and its result is recorded as smoke evidence.
- **Review until no finding remains.** Each round's fixes drew new findings;
  the severity gate and the backlog stop that loop without dropping findings.

## Consequences

- A change states which promises it touches; the conformance run, not memory,
  decides whether they still hold.
- The release gate gains one manual step: the conformance run's table.
- Fixes land in fewer pull requests; minor findings wait in one backlog.

# Kura decision model: three layers, one gate

Status: accepted owner decision.

Date: 2026-07-03

## Mission (owner statement)

> Every user — light or expert — should spend their time on datasets and
> parameter tuning, not on setup. Kura does not just provide a place to run
> things: it proposes settings likely to work, stops clear failures, and
> points out likely failures.
>
> A baseline built from many people's knowledge; an update line built from
> each person's own experiments.

## The three layers

1. **CLI (code)** — Docker/RunPod, paths, downloads, disk, inspect, plan,
   launch. It prepares the workplace mechanically, looks at data, and emits
   facts. It does not judge.
2. **Knowledge / files** — the essence of Kura being file-first. What was
   trained, when, with which dataset and settings, and what came of it:
   `run.yaml`, `resolved/`, `notes.md`, knowledge cards, regrets. Successes
   feed the cards; regrets feed the regret list. Both feed the next decision.
3. **Skill / agent** — an agent could train without Kura; Kura's skills give
   it the flow, the reading of plans, the use of past runs, and the judgment
   procedure.

Short form: **the CLI measures, the files remember, the skill judges, the
user decides.**

## Credo

- Code measures.
- Code stops only irreversible accidents.
- The agent judges.
- The user approves once, before launch.
- Last look is not a gate; it is a regret reminder.

"Kura does not decide what good training is" means: Kura core/CLI never
auto-judges training quality, never silently blocks on quality, and never
silently rewrites settings. The Kura agent/skill *should* propose promising
configurations and explain suspicious signals. Core does not decide; the
agent proposes; the user decides.

## Rules derived from the credo

### One approval gate, with a recorded contingency envelope

The only mandatory approval is the `kura run plan` review before launch.
After approval, nothing may change silently. A relaunch adjustment (e.g.
"if OOM, enable gradient checkpointing and relaunch") may skip a second
plan review **only if** that contingency was explicitly recorded in
`run.yaml` and shown in the approved plan. Anything outside the recorded
envelope returns to plan approval.

### Pre-empt criterion (when Kura may duplicate a trainer check)

Kura's pre-checks are not a mirror of the trainer's validation. They are a
device that moves failure to before money/time is spent and into language a
human can read. Kura may pre-empt a trainer check only when the trainer's
own failure would be:

1. **expensive-late** — it fails after large downloads or after Pod billing
   started (e.g. "0 images" after a 33 GiB download), or
2. **cryptic** — it fails deep in a container stack trace that a light user
   cannot read.

If the trainer stops immediately, before downloads, with a clear message,
Kura does not duplicate that check. This criterion bounds preflight
complexity by principle rather than accumulation.

### Preflight is a consolidation, not a new subsystem

Kura already has scattered preflight behavior (`docker_preflight`, disk
preflight, checkpoint safety, the download gate, dataset shape validation).
The work is to unify them under one name and one report structure
(check / severity / fact / affected file), shown in `kura run plan` and
enforced at launch. Severities are `error` / `warning` / `info` only.
There is no "quality error": code never blocks on training quality.

### validate / inspect split

- `kura dataset validate` — genuine pass/fail on structure: missing
  `dataset.yaml`, broken JSON, missing required fields.
- `kura dataset inspect` — measurable facts, no verdicts: image count,
  resolution distribution, caption statistics (empty / duplicates / trigger
  word occurrences), pair integrity, video fps/length distribution.
  Inspect output is the agent's raw material, produced **before** parameter
  proposal so the agent never proposes without material.

### Last look (regret reminder)

Immediately before presenting the plan for approval, the agent checks the
regret list and may attach a short note. Hard constraints so it never
becomes a second gate:

- Last look does not modify the plan.
- Last look does not modify `run.yaml`.
- Last look does not return a launch verdict.
- Last look returns only a few lines of note.

Regret entries are `trigger -> reminder`, never `trigger -> block`. Example
tone: "A past run with these conditions was regretted: forced low-VRAM mode
with heavy block swap. Continue if intentional."

### VLM / heavy content inspection

Looking at images/videos with a VLM (blur, watermarks, absent subject, low
quality) is agent work performed on request. It never enters Kura core, is
never a gate, and is never required: it is heavy, non-deterministic, and
error-prone in exactly the way quality gates fail.

## The run flow

| # | Step | Layer |
| --- | --- | --- |
| 1 | Place the dataset | user |
| 2 | `dataset inspect` (+ `validate` if needed) | code: facts |
| 3 | Parameter proposal (skill + cards + inspect facts) | agent: judgment |
| 4 | Draft `run.yaml` | free; not a gate |
| 5 | RunPod only: draft `kura run plan` — live GPU stock/price, select immediate/wait intent | code facts + agent/user choice; not a gate |
| 6 | `compile` — fail-fast checks, path resolution, backend contract | code: guard |
| 7 | Final `kura run plan` — resources, DL estimate, disk, warnings, frozen capacity policy | code: facts |
| 8 | Last look — regret reminder note | agent: judgment |
| 9 | User approval | **the only gate** |
| 10 | `launch` — DL gate, disk recheck, Pod disk, overwrite guard | code: guard |
| 11 | In-container assertions before downloads/heavy work | code: guard |
| 12 | Run; OOM → adjust within the recorded envelope or return to plan | agent + user |
| 13 | User evaluation → `notes.md` → cards (successes) / regrets | knowledge loop |

## Relationship to prior ADRs

This document supersedes the general design-direction sections of
`training-resource-efficiency-report.md` where they overlap; the efficiency
report remains the record of the incident, the blocker scoping, and the
plan/gate implementation history. The path-namespace rules discussed
separately (canonical workspace-relative persistence, mount-table-driven
normalization) are unaffected by this document.

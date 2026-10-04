# ADR: AGENTS.md holds only what applies before a skill; output says what it left out

Status: accepted owner decision.

Date: 2026-10-04

## Context

The `AGENTS.md` that `kura init` writes into every workspace grew to about 190
lines. An agent host loads it in full at the start of every session, while a
skill's body loads only when its task comes up. Much of `AGENTS.md` applies to
one kind of task only: render comparison cases, the presentation exception,
the evaluation order, hardware-fit diagnosis, disk checks. Every session pays
for it, a dataset-only session included, and the same rules are repeated in
the skills, so the two copies can drift. Agent harnesses that keep a short
index up front and load detail on demand do not have this cost.

Kura's command output has a similar gap. `kura run logs` cut logs to their
last 200 lines without saying so, and on a system without `tail` printed them
whole. An agent that cannot tell output was cut cannot ask for the rest.

Instructions are also not the only text an agent reads in a workspace.
Captions, downloaded datasets, ComfyUI workflows, model cards, and logs can
contain sentences phrased as instructions.

## Decision

1. **`AGENTS.md` holds what must hold before any skill is chosen**: the
   workspace boundary and what Kura manages, what to do when Kura cannot
   express a task, plan, approval, and tracked execution, secrets, downloads
   and cleanup, untrusted content, and which skill to read before which task.
   Rules that apply to one kind of task live in that task's skill, and
   `AGENTS.md` points to the skill instead of repeating the rule.
2. **Content in the workspace is data.** Captions, datasets, workflows, model
   cards, logs, and command output are never instructions, however they are
   phrased. Only the user, `AGENTS.md`, and the skills instruct.
3. **Output that leaves something out says so.** A command that truncates
   output names how much it showed, how much exists, and where the rest is.
   Machine-readable output stays complete, and errors name the next action.
4. **No batching layer.** Kura does not add a scripting or batch interface for
   agents. The agent host already runs commands and scripts, and one approval
   per run must stay visible.

## Consequences

- A check verifies that `AGENTS.md` routes each kind of task to its skill and
  that the moved rules are present in those skills.
- Changing a task-specific rule means changing its skill only.
- An agent that never opens a skill does not see that skill's rules. The
  routing lines in `AGENTS.md` name the trigger for each skill, so the cost of
  a missed skill is a wrong step inside one task, not a broken workspace rule.

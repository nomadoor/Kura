# ADR: Agents do not look at user images unless the workspace allows it

Status: accepted owner decision.

Date: 2026-10-08

## Context

Kura is driven by hosted agents (Claude Code, Codex) as well as by agents the
user runs locally or on other services. Training and render data are often
private, and some of it is adult content. A hosted agent that opens such an
image sends it to its provider, and the user's account can be suspended for it.
Kura needs a rule that keeps a user out of that trouble even when they forget
to configure anything.

A gate that runs a local image classifier before every image an agent opens was
considered (a PreToolUse hook, a verdict ledger, a local judge model). It would
add a model and a hook to every install for a check most sessions never need,
and Codex has no per-file hook to attach it to.

## Decision

**By default an agent does not open user images.** Dataset images, rendered
samples, render inputs, and anything else under a workspace's datasets or runs
are user images. Without permission, an agent works from what Kura reports
without opening the image: dimensions, counts, file sizes, hashes, caption
presence, duplicates, and validation results (`kura dataset inspect`,
`kura dataset validate`). That catches careless mistakes (a missing caption, a
wrong resolution, a duplicate) without seeing the content.

**A workspace allows it with `agents.view_images: true` in `workspace.yaml`.**
The user sets it when they want an agent to look, for example to write
captions or to judge samples. A missing key means no. The permission is per
workspace; per-dataset permission was rejected earlier as the wrong grain.

**An agent stops at the first image its own service does not allow.** Even with
permission, an agent that opens an image and finds that handling it breaks the
usage policy of the service running the agent stops looking at once and tells
the user, rather than looking at the rest and warning afterwards. This guards
the user's account; it is not a Kura rule about what may be trained. An agent
whose service allows such content (a local model, another provider) continues.
Kura stays neutral about which agent drives it.

## Consequences

- The shipped `AGENTS.md` states the rule, and the dataset and evaluation
  skills point to it where they would otherwise open images.
- `workspace.yaml` gains `agents.view_images` (boolean, default false).
- Nothing in Kura enforces the rule; it is an instruction to agents. The
  default is safe when the user does nothing.
- The classifier gate is not built.

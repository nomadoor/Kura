# The UI hosts the user's own agent; UI parts are a fixed catalog delivered as MCP Apps

Status: accepted owner decision.

Date: 2026-10-02

## Context

Kura's Web UI is chat-centered. The user and an agent work through a training
cycle, and the UI places interactive parts where direct manipulation beats
typing:

- parameter forms;
- image grids and comparisons;
- loss charts;
- choice cards;
- approval.

Kura could embed its own model and its own UI framework. Two things argue
against that:

- Users already pay for and trust coding agents such as Claude Code and
  Codex.
- Kura's interactions are predictable enough that free-form generated UI adds
  nothing.

## Decision

**Agent**

- The Kura UI hosts the user's own coding agent over the Agent Client
  Protocol (ACP), starting with Claude Code and Codex.
- The same agent talks in front and does the work behind it.
- Billing and authentication stay with the user's agent subscription. Kura
  embeds no model and holds no model API key.

**Kura operations for agents**

- Kura exposes its operations as an MCP server.
- These are the same operations the CLI performs, so there is still one
  execution path. The UI adds no capability that the CLI and an agent lack.

**UI parts**

- UI parts form a fixed catalog that Kura owns, delivered as **MCP Apps**.
  Parts appear in two ways:
  - **From run state.** For example, a compiled run that waits for launch
    shows its plan form, and a finished render shows its image grid.
  - **On an agent's request.** An agent may request a catalog part, for
    example comparison candidates. Agents never author UI.
- Because MCP Apps is a host-neutral standard, the same parts can render in
  the Kura UI and inside other MCP Apps hosts, such as Claude Desktop.
- Parameter forms are not written per backend. They are generated from each
  backend adapter's declared configuration surface. Each field carries
  presentation metadata: a user-facing label, and whether it is always shown
  or kept under details.
- Fields that decide what is learned are shown by default. Execution
  accommodations stay under details.

## Consequences

**Open verification before building**

- Whether MCP Apps UI returned by a tool reaches a UI that hosts the agent
  over ACP.
- Whether a Windows MCP Apps host, such as Claude Desktop, can reach a Kura
  MCP server running inside WSL2.
- Whether charts and image grids are practical inside MCP Apps.

If the first check fails, the Kura UI renders the same catalog from run state
and agent requests directly. The catalog and the file-driven trigger do not
change.

**Other consequences**

- Backend adapters gain presentation metadata for their configuration
  surface (`kura run capabilities`).
- The Web UI still owns what does not fit in one conversation: the dashboard
  of parallel work, the library of artifacts and their links, and the current
  position of each experiment.

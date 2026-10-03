# The UI is a shared workspace for the user and their own agent

Status: accepted owner decision.

Date: 2026-10-02

Updated: 2026-10-03 — Kura draws its UI itself instead of delivering parts as
MCP Apps, ships no MCP server, and keeps the UI to threads and a Library.

## Context

Kura is an experiment harness for the cycle of dataset, training, evaluation,
and the next experiment, not a settings GUI for a trainer. A UI that grows one
screen per feature turns into the trainer GUIs Kura is meant to replace, and
the set of tasks keeps growing: image, video, audio, and music; generation and
editing; single files and pairs.

Users already pay for and trust coding agents such as Claude Code and Codex.
Without an agent, Kura is tedious: the agent explains, proposes, fixes errors,
and connects one experiment to the next. The UI is therefore built for the
user and the agent together.

The first version of this decision delivered the parts as MCP Apps through a
Kura MCP server, so they could also appear inside other agent hosts. That is
withdrawn. A probe observed on 2026-10-03 that it does not reach where users
work:

- The Claude Desktop Code tab (Claude Code 2.1.286) did not offer MCP Apps to
  servers, and the terminal Claude Code does not render them.
- The Claude Desktop chat rendered MCP Apps only after the user registered an
  MCP server by hand, and Desktop rewrote its configuration file while it ran,
  which lost such edits.
- The Codex desktop app renders MCP Apps, with open rendering bugs reported.

## Decision

**Agent**

- The Kura UI hosts the user's own coding agent over the Agent Client
  Protocol (ACP), starting with Claude Code and Codex. Billing and
  authentication stay with the user's agent subscription; Kura embeds no model
  and holds no model API key.
- Agents operate Kura through the `kura` CLI, inside the UI and outside it.
  Kura ships no MCP server, and does not build its own UI inside Claude Code,
  Codex, or Claude Desktop chats.
- Launching follows `files-only-state-and-job-runner.md`: closing the UI or
  its agent never stops a run. A plan waiting in a thread shows its approval
  control; the agent waits for the click, or launches through the CLI when the
  user approves in the conversation.

**Two places: threads and the Library**

- **Threads.** A thread is a conversation with the agent, and most features
  live in it as widgets. A thread is not bound to one task: a user may stop a
  training midway, move on to comparing results, or start another training in
  the same thread.
- **Library.** Datasets, trained adapters, and generated images and videos,
  linked by the facts in their run files: an adapter leads to its dataset
  revision, recipe, checkpoints, and evaluations, and a generated image leads
  back to the adapter and settings that made it.
- A sidebar holds a compact widget of active runs at the top and the list of
  threads below it. Each active run shows its state there and opens the thread
  that launched it. There is no separate dashboard screen.
- Kura stores no transcript; the agent host keeps the conversation. A thread
  is a session of the hosted agent plus a small record in the workspace: its
  title, agent, and the widget requests made in it. The UI gives the hosted
  agent's session its thread identity, and a launch request records the thread
  it came from, whether written by a click or by the agent's `kura` command.
  The sidebar finds a run's thread from run files.
  A run launched outside the UI has no thread, and opening it starts a new
  thread about that run.

**Widgets**

- Widgets form a fixed catalog that Kura owns and draws. Agents never author
  UI.
- What a thread shows follows its state. A widget appears from run state (a
  compiled plan shows its form; a running training shows loss, progress, and
  cost; a finished render shows its image grid) or on the agent's request,
  made through a `kura` command that appends the request to the thread's
  record. A request names the runs or datasets it shows and changes only what
  the UI displays.
- Widgets read run files, so they show the current state without waiting for
  the agent, and an old widget in a long thread shows its run as it is now.
- For the run a thread is currently on, a progress map shows every milestone
  of that kind of run, from start to goal, and where the run stands. It is
  derived from run files, and it changes or disappears when the thread moves
  on.
- A widget can expand to fill the screen for work that needs room, such as
  curating hundreds of images. The Library uses the same widgets.

**Suggestions**

- A new thread opens with an input box and a few suggestions taken from the
  workspace: recommended starting points for a new user, and continuations of
  past work for others ("retrain No.0003 with other parameters", "compare the
  last two").
- Typing narrows them as a search over supported models, what can be done
  with them, and the workspace's own assets (`kre…` offers training and
  generating with Krea 2). What can be done with each model comes from shipped
  knowledge, not from a task taxonomy in core.
- Suggestions are found locally and immediately, never by waiting for a
  generating model. Phrasings are prepared ahead of time, Helpfeel style, and
  matched as the user types. A small local ranking or decision model may order
  them; no hosted model or model API key is involved.
- During a thread, the next steps that follow from run state are offered the
  same way, and the agent adds suggestions that need judgment.

**One catalog for every task kind**

- Widgets are not written per task or per backend. Forms come from each
  backend adapter's declared configuration surface, with presentation
  metadata: a user-facing label, and whether a field is always shown or kept
  under details. Fields that decide what is learned are shown by default;
  execution accommodations stay under details.
- Dataset and result widgets come from the roles of an item's files and their
  media types: an image is shown as an image, video and audio as players, text
  as text, and a pair side by side. Each backend adapter declares the roles it
  consumes and their media types, so a new task kind adds a declaration, not a
  screen. Core still has no common task taxonomy.
- The UI reviews and selects. Its dataset edits go through the same
  `kura dataset` commands an agent uses, with the user as author.
  Transformations such as trimming video, resampling audio, or cropping faces
  are done by the agent with tools, and their results follow
  `dataset-revisions.md`.

## Consequences

- Backend adapters gain presentation metadata for their configuration
  surface (`kura run capabilities`) and declare the file roles they consume
  with their media types.
- CLI commands the UI reads gain machine-readable output.
- The workspace gains thread records, and the launch request gains the thread
  that wrote it. `kura` gains a command an agent uses to request a widget.
- Shipped knowledge gains the recommended starting points for new users.
- Whether an ACP-hosted agent's session can be shown next to the widgets in
  one page is verified when the agent hosting is built.
- If MCP Apps reaches the Code tab later, an MCP view of the same catalog can
  be reconsidered as an addition.

# Kura is installed; users work in a separate workspace

Status: accepted owner decision.

Date: 2026-10-02

## Context

Today a user clones the Kura repository and works inside it. As a result, one
directory holds four kinds of content:

- Kura's source;
- its agent instructions;
- the maintainer's personal knowledge;
- the user's datasets, runs, and cache.

This has three costs:

- **Setup is long.** It is clone, `uv sync`, then `kura init` inside the repo.
- **Personal knowledge ships.** The maintainer's preferences reached every
  user.
- **Usage needs guards against editing the source.** Usage sessions need
  permission rules so that an agent does not patch Kura's source to get past
  a refusal, and those same rules slow down real development.

The Web UI would also read and write user data. It should not do that inside
the source tree.

## Decision

**Installation and workspace**

- Kura is installed as a tool (`uv tool install`). The Kura repository is for
  development only.
- A user creates a **workspace** with `kura init` in any directory. The
  workspace holds `workspace.yaml`, datasets, runs, cache, and the user's own
  knowledge.
- `kura init` also runs the environment checks that `kura doctor` performs
  and reports only what is missing.
- The setup path is: install uv, then install Kura with
  `uv tool install <kura source>` (a Git source until a package is
  published), then `kura init`. Windows users do all three inside WSL2.
- A one-line installer may wrap these steps later.

**Agent instructions in the workspace**

- `kura init` writes the usage agent instructions into the workspace: a usage
  AGENTS.md, the usage skills in each agent's expected location, and the
  shipped knowledge.
- Agents read only what is in their working directory, so these must be
  files there.
- Kura records which version wrote them and refreshes them when the installed
  version changes. Users do not edit these files. Their own material lives in
  separate files that Kura never overwrites.

**Two knowledge layers**

- **Shipped knowledge:** family cards and general regrets, curated by the
  maintainer and versioned with Kura.
- **User knowledge:** preferences, the user's own regrets, and the user's own
  cards. It lives in the workspace.
- Agents read both. User knowledge wins when they conflict.
- Moving a user finding into shipped knowledge is a maintainer decision, made
  with a `source:` line.

## Consequences

**What changes in Kura**

- Five places assume that the repository is the workspace, and each changes:
  - the README setup;
  - workspace discovery from the current directory;
  - `.gitignore` treating `runs/`, `datasets/`, and `cache/` as part of the
    source tree;
  - Dockerfile paths and build context in `workspace.yaml`, together with
    `kura init` writing Dockerfiles into the workspace (users pull pinned
    images and never build);
  - skills, AGENTS.md, and knowledge being found by repository-relative paths.
- The repository `.claude/settings.json` ask rules exist only to protect usage
  sessions that run inside the source tree. They become unnecessary once
  usage happens in a separate workspace.

**Migration**

- The maintainer's current runs, datasets, and cache move into a new
  workspace.
- Run records are not rewritten. Absolute paths inside old records stay as
  historical facts.

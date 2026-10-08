# ADR: Secrets live in one user-level file, and agents never handle them

Status: accepted owner decision (2026-10-04).

## Context

Kura needs a few credentials: the RunPod API key, a Hugging Face token for
gated models, and optional notification and object-store keys. Every command
used to load them from the workspace's `.env.local`, which `kura init` created
as a template.

Two things changed. Kura is now installed once as a tool and used from many
workspaces, and these credentials belong to the person, not to one workspace.
And most work happens through an AI agent that runs commands as the user. A
secret pasted into the chat is kept in the conversation and sent to the model
provider. A secret file inside the workspace sits where the agent reads and
searches.

No single convention for local API tokens exists. The options were:

- **A file in each workspace** (the previous design): simple, but the same key
  is entered again per workspace, and the file sits in the agent's working
  directory.
- **The OS keychain**: no plaintext file, but WSL, the main Windows path,
  usually has no keychain service, it adds dependencies, and an agent running
  as the user can read the keychain as easily as a file.
- **A password manager injecting values at run time** (`op run` and similar):
  strong, but every user would need the service.
- **One user-level file outside every workspace** (chosen).

## Decision

1. **Location.** Secrets live in `secrets.env` in Kura's user configuration
   directory (`~/.config/kura/secrets.env` on Linux and WSL, the platform's
   per-user configuration directory elsewhere). It is created readable only by
   the user where the file system supports that. WSL and native Windows have
   separate stores.
2. **Override.** A workspace `.env.local` still works and wins over the user
   file, for a workspace that needs a different account. `kura init` no longer
   creates it.
3. **Precedence.** The process environment, then the workspace `.env.local`,
   then the user file. An empty value (`NAME=`) counts as unset, so a leftover
   empty template line never hides a value set elsewhere.
4. **Entry.** `kura secrets set NAME` reads the value with hidden input in the
   user's own terminal and refuses without one, so it cannot run inside an
   agent's shell. `--workspace` writes the workspace `.env.local` instead.
   `kura doctor secrets` lists which names are set and where, never values.
5. **Agents never handle values.** An agent never asks for a secret in the
   chat, never reads, prints, or writes a secret file, and never runs
   `kura secrets set`. When a secret is missing, it tells the user to run
   `kura secrets set NAME` in their own terminal and waits.
6. **Checks.** No check ever scans a secret file. `kura check secrets` loads
   the secrets like every other command and reports a line that holds a value
   Kura has as a secret, or a fixed token shape (`hf_…`, `rpa_…`, `Bearer …`),
   by `path:line` only. It does not guess from names: a secret Kura does not
   hold, pasted without a token shape, is not found.
7. **Names and values.** `kura.secrets.is_secret_name` decides whether a name
   holds a secret, by whole words (`TOKEN`, `SECRET`, `PASSWORD`, `API KEY`,
   `ACCESS KEY`, `PRIVATE KEY`, split at `_`, `-`, and case changes); every
   refusal of a secret name in a configuration or command uses it.
   `kura.secrets.secret_values` decides which values are secrets: those of
   such names and of the key names `workspace.yaml` chooses. Output redaction
   and the check above use it. A value the user picks as a plain word, such as
   an ntfy topic, is not one: hiding it would rewrite matching text in every
   record.

## Consequences

- A user enters each key once per machine; new workspaces need nothing.
- An existing `.env.local` keeps working unchanged.
- The owner of the account can still read the plaintext file; the design
  protects against leaks through the agent, logs, run records, and shared
  workspaces, not against the account owner.
- An OS keychain or password-manager references can be added later behind
  `kura secrets set` without changing the agent rule or the precedence.

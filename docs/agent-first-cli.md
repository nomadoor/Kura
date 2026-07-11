# Agent-first, CLI-complete

Kura is designed for an AI agent to do most of the authoring and operation.
The agent is not part of the runtime.

> **Agent-first, not agent-dependent.** The agent authors and judges; files
> carry the decisions; the CLI executes without conversational state.

## Responsibility at a glance

| Owner | Responsibility | Source of truth |
| --- | --- | --- |
| 🤖 Agent | Inspect facts, propose a backend/model/config, explain trade-offs, diagnose results | Decisions written to workspace files |
| 🧑 User | Choose intent, quality/cost trade-offs, approve the plan, evaluate outputs | `run.yaml`, approval, `notes.md` |
| 📄 Files | Carry intent, compiled inputs, runtime facts, and evaluation | Workspace artifacts |
| ⚙️ Kura CLI | Validate Kura-owned structure, compile, plan, execute, monitor, recover | Deterministic file and provider operations |
| 🧰 Backend | Interpret its native model/task/config and train | AI-Toolkit or Musubi native configuration |

## What remains when the agent is gone

An agent-created run is an ordinary Kura run. It does not contain a hidden
conversation reference, agent session ID, or private UI state.

| Stage | Files required | Works without an agent |
| --- | --- | --- |
| Draft | `run.yaml`, dataset files, `workspace.yaml`, secrets | compile and inspect |
| Compiled | `resolved/`, dataset files, `workspace.yaml`, secrets | plan and execute |
| Running | compiled inputs plus realization/provider facts | status, logs, watch, reconcile, stop, recover outputs |
| Completed | immutable inputs, realizations, logs, outputs | render, compare, inspect, reproduce |

The normal commands are the same whether an agent or a human invokes them:

```sh
uv run kura run compile <run-id>
uv run kura run plan <run-id>
uv run kura run execute <run-id>
uv run kura run watch <run-id>
```

Manual authoring is supported, but it is not the primary UX. The important
guarantee is that the files an agent produces are complete, readable CLI input.

## What Kura does not normalize

Kura gives all backends the same run lifecycle. It does not give them a fake
shared model or task language.

| Kept common | Kept backend-native |
| --- | --- |
| run identity and lifecycle | AI-Toolkit `model.arch` and native config |
| natural-language intent | Musubi architecture, selector, scripts, and flags |
| dataset identities and observations | backend-owned model-role labels |
| executor choice and runtime facts | backend-specific recipe values whose meanings differ |
| artifact provenance and evaluation links | unknown-model/native-command escape hatches |

Changing a run from AI-Toolkit to Musubi is therefore a new projection by an
agent or human, not a guaranteed mechanical conversion. The natural-language
intent and prior observations provide the migration context.

## Reproducibility contract

- Agent decisions that affect execution must be written to `run.yaml` or
  frozen under `resolved/`.
- Agent-created dataset mappings must be frozen; conversation-only mappings are
  not valid execution input.
- `resolved/` is immutable after compile.
- Runtime attempts and external identities are append-only realization facts.
- Secrets come only from `.env.local` or environment variables and are never
  frozen into run artifacts.
- Model and image provenance records the strongest identity actually observed;
  Kura does not invent a content hash it could not observe.

## Failure behavior without an agent

The CLI does not replace the agent with a hidden policy engine. It reports
facts and concrete adapter errors. It blocks malformed Kura files, missing
explicit inputs, unsafe paths, immutable-input changes, and adapter-declared
requirements. Model suitability and output quality remain judgments.

This keeps automated replay and CI possible while preserving the intended UX:
the agent handles unfamiliar trainer details, and the user spends time on the
dataset, plan, and result.

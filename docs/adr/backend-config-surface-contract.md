# ADR: Declared backend configuration surfaces

Status: accepted owner direction; implementation in progress.

Date: 2026-08-13

## Context

Kura is consumed by agents that know general training practice and Kura's
public files, but should not need to inspect adapter source.  The previous
opaque-boundary decision correctly kept backend meaning out of core, but it did
not require adapters to enumerate their authored input.  As a result,
sd-scripts rejected unknown `backend.config` keys while Musubi Tuner and the
outer AI-Toolkit wrapper silently ignored them.

A plausible substitution such as `optimizer` for `optimizer_type` is more
dangerous than an obvious typo: training still runs, the output remains
plausible, and `run.yaml` appears to record a choice that never reached the
native command.

This ADR refines, rather than replaces,
`run-envelope-and-backend-boundaries.md`.  Backend semantics remain adapter
local.  The rejected design was a core-owned catalog of upstream model and task
support; this decision requires each adapter to describe only Kura's authored
configuration surface.

## Decision

Every registered adapter owns a machine-readable surface contract.  The
registry requires the contract, so a newly registered backend cannot omit it.
The contract divides authored input into:

- common recipe fields whose meaning Kura has proved identical;
- adapter-owned `backend.config` fields that Kura recognizes and routes to a
  direct adapter consumer;
- named escape hatches whose contents Kura records but does not fully validate.

The common recipe remains `steps` and `seed`.  Similar vocabulary is not enough
to move learning rate, optimizer, scheduler, rank, precision, or batching into
the common envelope.  Adapters may expose consistent spellings for those
concepts without claiming identical cross-backend semantics.

The run-plan and compile paths validate top-level `backend.config` membership
and selector applicability before presenting an approval target or compiling
model, dataset, or command artifacts. An unknown key, or a known key that is
not consumed by the selected architecture/mode, is an error. The error names
the selected backend and points to the capability command. Known-but-unowned
concepts carry a reason and, where one exists, the supported placement or
reviewed escape path. Suggestions are an explicit semantic mapping, not
spelling-distance guesses: a plausible but wrong concept is more dangerous
than no suggestion.
Validation data and capability output come from the same contract. Value and
combination validation remains adapter-local; surface membership does not
claim a uniform cross-backend type system.

`kura run capabilities <backend>` is the public, source-free discovery path.
Its JSON form is stable machine-readable input for an agent. Always-applicable
and selector-conditional fields are separate, and each conditional field names
its accepted architecture/mode clauses. Escape hatches are visibly marked
unvalidated; accepting their contents is never presented as first-class Kura
support.

AI-Toolkit's old `backend.config.config` name is removed.  Ordinary training
controls are exposed as adapter-owned fields and translated into AI-Toolkit's
nested process YAML.  The remaining raw nested override is named
`backend.config.native_config`, making its unvalidated status explicit.  Kura
does not accept both spellings because merge precedence would make the record
ambiguous.

Explicit `command`, native `extra_args`, AI-Toolkit `native_config`, and
Musubi's native dataset configuration are escape hatches where applicable.
Their presence is
frozen in `manifest.lock.yaml`; capability output states that Kura does not
validate their inner vocabulary.

## Enforcement

The registry type makes a surface declaration mandatory.  Registry-driven
tests exercise every adapter with plausible wrong names and an arbitrary
sentinel. Every conditional declaration is exercised through the shared
validator, with regression cases for Musubi architecture and sd-scripts mode
boundaries. Tests also register a hypothetical adapter without a declaration
and require failure. Adapter-specific compile tests prove that supported
ordinary values reach native YAML or argv rather than merely appearing in a
static list.

The adapter source identity scope advances to `selected-adapter-v2` and hashes
the selected surface, selector conditions, semantic corrections, unavailable
concepts, and central validator. This is a behavior change for previously
unknown or inapplicable inputs. Historical evidence is migrated only for the
named, evidence-scoped valid configurations whose generated native config and
runtime command remain unchanged.

Dataset manifest and `items.jsonl` vocabulary are not changed here.  The audit
found permissive fields, including nested native dataset configuration; that
surface requires a separate decision because its observational and
backend-native roles are different.

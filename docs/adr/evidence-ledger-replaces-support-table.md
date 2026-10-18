# The evidence ledger and family cards replace the support table

Status: accepted owner decision.

Date: 2026-10-18

## Context

Until now Kura answered "does this model train?" with a support table,
`docs/backend-support.md`, and kept it honest with three mechanisms:

- every table row that claimed support had to cite records in the evidence
  ledger (`docs/backend-smoke-evidence.yaml`);
- each record carried the adapter's source identity, and RunPod records also an
  executor source identity; when either changed, a record in
  `docs/adapter-source-identity-migrations.yaml` had to declare the change
  behavior-preserving, or the cited evidence stopped counting;
- upgrade plans under `docs/backend-validation/`, with their own checker,
  generated test matrices that the table pointed to.

In practice this cost a migration record for nearly every adapter or executor
change, including changes that never touched what a recorded run did. The
table drifted from what had really run, and the checks that guarded it covered
only some backends, so a corrupted musubi-tuner migration chain passed the
release gate unnoticed.

## Decision

The support table, the identity-migration ledger, the executor source
identity, and the backend-validation plans and checker are retired. The plans
are kept unchanged under `docs/archive/backend-validation/` as history and are
no longer checked.

What replaces them:

- **The evidence ledger** stays as append-only history of real runs. Each
  record states when it ran, the adapter source identity, and the trainer image
  digest; its artifact names the Kura commit or version. `scripts/check_smoke_evidence.py`
  checks only the ledger's own shape: schema, unique ids, required fields, that
  each artifact file exists, and render workflow digests. It neither requires
  nor rejects the `executor_source` field older records carry. Records are
  never rewritten.
- **The shipped family cards** (`src/kura/shipped/knowledge/model-families/`)
  are where users and agents learn what has run, through `Recorded:` lines that
  name their run. A family has run only where a card or an evidence record says
  so; a family Kura only compiles is called untested or not listed.
- **"Did this change invalidate recorded runs?"** is a review question, not a
  gate. A reviewer answers it from the ledger: what each record ran with, and
  whether the change alters that path. A change that does names the records it
  makes stale in its pull request.

The adapter source identity itself stays: runs record it, and Resume refuses
training state saved under a different one.

## Consequences

- An adapter or executor change no longer needs a migration record.
- Upgrades state their old and new pins and inventory the upstream delta in the
  pull request, or in a `docs/smoke-evidence/` note it links, and record one
  real run per family they call runnable (`backend-upgrade-audit`).
- Nothing mechanical tells whether old evidence still applies to today's code;
  the ledger's dates, identities, and images are what a reader judges by.

# Audit: dataset authored surfaces

Status: measured; separate follow-up required.

Date: 2026-08-13

## Scope decision

Dataset vocabulary is not included in the backend configuration surface
change. It combines workspace observations, authored sample relationships, and
trainer-native dataset configuration; treating those as one key registry would
repeat the vocabulary-unification mistake this work avoids.

## Measurements

- `dataset.yaml` validation checks required files, the manifest shape used by
  Kura, item counts, paths, captions, and hashes, but does not reject unknown
  top-level metadata keys.
- `items.jsonl` preserves additional authored fields. Observation code consumes
  explicit target/condition relationships and several layout aliases; this is
  intentionally more extensible than a trainer flag surface.
- Musubi `backend.config.dataset_config` copies native `general` and dataset
  entries into TOML, with path and duplicate-dataset safeguards but without a
  closed upstream-key vocabulary.
- sd-scripts `dataset_config` is different: Kura owns closed key sets for its
  general, dataset, and subset tables and rejects unknown keys.
- AI-Toolkit's dataset projection starts from Kura's declared `datasets[]`.
  Raw `native_config.datasets` is rejected because it would replace the locked
  dataset identities with a second execution input.

## Follow-up boundary

A follow-up should separately decide:

1. which `dataset.yaml` and `items.jsonl` fields are Kura-owned observations;
2. how extension metadata is named and made visible without rejecting useful
   authored relationships;
3. whether Musubi native dataset TOML remains an explicitly unverified escape
   hatch or gains adapter-owned subcontracts by dataset form.

Until then, the capability command marks Musubi `dataset_config` as unverified.
This is visible debt, not a claim that its inner keys are validated.

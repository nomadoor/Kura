# Authored input is read in one form

Status: accepted.

Date: 2026-10-14

## Context

Kura was meant to be minimal, but it grew helpful input handling: aliases,
case folding, strings read as numbers or booleans, `value or default`
substitutions, two key names for one thing, and guesses from names. An audit
of every authored-input path on 2026-10-14 found about 80 such places. Each is
a second interpretation of one input, and the second interpretation drifts
from the first:

- `architecture: flux` was mapped to `flux1` by the sd-scripts adapter, while
  the shared applicability check compared the raw value and refused
  flux1-only fields.
- Musubi Tuner replaced a written `0` with its own default
  (`max_data_loader_n_workers: 0` became 2).
- YAML `false` turned a Musubi feature on, because the code read a falsy value
  as "not written" and filled the default.
- Kura counted `save_every_n_steps: 2.5` as 2 for checkpoint names and disk,
  while the trainer received 2.5.

The same decision also differed by backend: sd-scripts type-checked values
that Musubi Tuner and AI-Toolkit passed unchecked, and Musubi Tuner and
sd-scripts filled omitted values with hand-written defaults.

Most authored values are only forwarded. Of about 150 top-level backend
settings and about 100 native dataset settings, Kura decides from roughly ten
per backend. The specification is issue #208.

## Decision

### Three rules

Every authored value falls under one of three rules.

1. **Values Kura only forwards or records pass through.** A value Kura hands
   to the trainer without deciding from it (learning rate, optimizer, rank,
   and similar), or only records (intent, notes), reaches the trainer or the
   record exactly as written. Kura does not coerce, type-check, or replace it;
   the trainer reports its own errors.
2. **Values that would not take effect are refused.** An unknown key, or a key
   the selected architecture or mode does not consume, is refused where the
   file is loaded. Otherwise the record would claim a choice that never
   reached the trainer, whether the trainer errors on it or silently ignores
   it. This is the closed-surface rule of
   `backend-config-surface-contract.md`, applied without gaps.
3. **Values Kura decides from are checked strictly.** A value Kura itself
   reads to choose behavior (selectors, model selection and model files, save
   cadence and retention, steps, the Resume scheduler check, disk and safety
   settings, durations, render inputs, dataset files) has one accepted form.
   Anything else is refused with an error that names the accepted form, before
   any later check can give a misleading message. When such a value is also
   forwarded, Kura forwards exactly the value it checked.

One meaning has one spelling. Kura never adds an alias, folds case, or reads
a second shape to make an error go away; the error names the right value
instead. This is not about what a user trains with: Kura still refuses no
recipe a user may reasonably choose.

### One YAML reader and writer

`kura.yamlio` reads and writes every YAML file Kura reads or writes, and no
other Kura code imports PyYAML or another YAML library
(`scripts/check_architecture.py`). Reading follows the YAML 1.2 core schema, so
`1e-4` is a number, only `true` and `false` are booleans, and `on`, `yes`, or a
bare date stay strings. A mapping key may appear only once, and the YAML 1.1
merge key `<<` is refused, so one file has one reading. Writing
quotes any string that a YAML 1.2 or a YAML 1.1 reader would turn into another
type, so a file Kura writes reads back the same in Kura and in a trainer's
PyYAML loader. Container scripts run inside trainer images without Kura and
keep their own loader.

This is reading the file correctly, not a lenient interpretation: `1e-4` is a
number under the current YAML specification, and PyYAML's default YAML 1.1
reading turned it into a string.

### No training defaults of Kura's own

Kura adds no training defaults. A value that changes training quality, and
that the trainer would otherwise take from a generic class default, is
required: an omitted one is refused, naming the field. AI-Toolkit keeps
filling unset values from the pinned upstream UI baseline
(`upstream-training-baseline.md`), which is upstream's recommendation, not
Kura's. Kura-owned constants a contract needs (a path the handoff owns, the
network module) stay, as that ADR allows.

### Past failures are knowledge, not refusals

An observation from a past run, such as an out-of-memory failure on one GPU
with one configuration, goes into the shipped knowledge cards for the agent to
weigh when planning. It is not enforced as a refusal in code: no two runs are
the same, the failure shows itself early, and the user decides. Code refuses a
combination only when the pinned trainer source itself cannot run it, citing
that source.

### Compatibility

Authored input keeps no compatibility: aliases, synonyms, alternate shapes,
retired spellings, and their translation code are removed. Readers that keep
Kura's own existing records viewable stay where they are small. Recompiling an
old `run.yaml` is not promised. One such reader is `load_yaml_1_1`: a Resume
recipe fingerprint recorded before this change hashed values such as `1e-6` as
the strings YAML 1.1 gave, so it is compared against the parent's frozen
manifest read that way, and the manifest read today must equal the derived
run's recipe.
Another is `load_run_yaml`: once a run is compiled its run.yaml is a record,
which an older Kura may have accepted with a repeated key or a `<<` merge, so
the readers that scan or copy runs read it that way (the last value of a
repeated key wins); a draft and every compile read strictly.

## Consequences

- `backend-config-surface-contract.md` keeps membership and applicability;
  pass-through and strictness are added here.
- `upstream-training-baseline.md` no longer leaves Musubi Tuner and sd-scripts
  hand-written defaults as debt to migrate: they are removed, and
  quality-relevant values without an upstream baseline become required.
- An empty string in `workspace.yaml` no longer means "unset"; an unset
  setting is omitted, and `kura workspace migrate` removes empty strings.
- Promises touched (`promises-and-verification.md`): P1, P2, P4, P8, P9, P11.

## Alternatives considered

- **Keep aliases and normalize once in the shared layer.** This was the
  2026-09 fix for Musubi (`SelectorNormalization`). It is opt-in per backend,
  so sd-scripts was missed and the bug returned. Removing the aliases removes
  the second interpretation entirely.
- **Leave every error to the trainer.** Trainers that ignore unknown keys
  would let a record claim a choice that never took effect, and Kura cannot
  name checkpoints or budget disk from a value it does not understand.
- **Kura-maintained default tables for Musubi Tuner and sd-scripts.** This is
  the most mechanism and keeps Kura choosing values; requiring the few
  quality-relevant values keeps the choice with the agent and the user.

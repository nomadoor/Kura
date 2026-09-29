# ADR: Unset trainer settings come from the pinned upstream baseline

Status: accepted owner decision (2026-09-29).

Date: 2026-09-29

## Context

The 2026-09-29 real-smoke campaign
(`docs/smoke-evidence/2026-09-29-real-smoke-campaign.yaml`) ran every
AI-Toolkit image selector through Kura. The dataset handoff was correct in
every run: transfers verified and input postflights matched. Still, several
runs failed because of the native configuration Kura generated:

| Setting | AI-Toolkit class default | AI-Toolkit UI job | Kura emitted | Result |
| --- | --- | --- | --- | --- |
| `train.noise_scheduler` | `ddpm` | `flowmatch`, overridden per architecture | nothing, so `ddpm` | FLUX.2 Klein and Krea 2 stopped at step 1 |
| `train.dtype` | `fp32` | `bf16` | nothing unless `mixed_precision` was set, so `fp32` | FLUX.1 ran out of memory on a 44 GiB A40 |
| `datasets[].cache_latents_to_disk` | `false` | `false` | always `true` | pinned Flex.2 with controls cannot run (it reads pixel tensors) |

The configuration AI-Toolkit users consider normal is the one the UI builds.
The UI starts from `ui/src/app/jobs/new/jobConfig.ts` and applies the
architecture entry in `extensions_built_in/diffusion_models/ui.tsx`. The
defaults on AI-Toolkit's Python config classes are not that configuration.
Kura sat between the two and produced a third configuration that matched
neither.

`backend-config-surface-contract.md` rejected a core-owned catalog of upstream
model support. Hand-writing per-model scheduler, precision, and caching values
would bring that catalog back, and it would drift every time the pin moves.

## Decision

An unset trainer setting takes its value from the **pinned upstream baseline**.
That baseline is the configuration the pinned trainer's own recommended entry
point builds for the selected architecture. Kura extracts it mechanically from
the pinned source and records it. Kura does not author these values.

1. **Baseline artifact.** Each adapter that needs one keeps a machine-readable
   baseline in the repository. It maps each architecture to the native values
   the upstream entry point sets. The file records the upstream commit and the
   SHA-256 of every source file it was extracted from. For AI-Toolkit, those
   sources are `jobConfig.ts` (the base job) and `ui.tsx` (the per-architecture
   entries, applied on top of the base).
2. **Fill order.** A value the user or agent authored always wins. For a value
   that was not authored, the adapter emits the baseline value. Compile records
   which values came from the baseline, and `kura run plan` shows them as
   baseline-derived. The record therefore separates "the run chose this" from
   "upstream recommends this".
3. **Unknown architecture refuses.** If a selector has no baseline entry, Kura
   cannot know values such as the noise scheduler. Compile stops and names the
   settings that must be authored. It never falls back to the class defaults.
4. **Kura-owned constants must match the baseline or be justified.** A value
   Kura forces for its own contract stays, with a comment and a test that name
   the contract; an example is a path the handoff owns. Any other forced value
   is removed. This removes AI-Toolkit's unconditional `cache_latents_to_disk:
   true`, because the dataset contract does not need it.
5. **Known-broken combinations refuse at compile.** When pinned upstream code
   is known to fail on a combination, the adapter refuses that combination
   before launch and cites the source line. Pinned Flex.2 reading
   `batch.tensor` while latents are cached is one such combination. The
   refusal is keyed to the upstream commit, and a pin upgrade re-audits it.
6. **Extraction fails closed.** The extractor parses only the shapes it
   understands. An entry it cannot fully parse, a changed source file, or a
   missing architecture fails regeneration, and the old baseline is not kept
   silently. The UI source is not a stable API. This brittleness is accepted as
   a bridge until upstream exposes a supported configuration API.

## Implementation notes

- `scripts/extract_ai_toolkit_baseline.py` runs
  `scripts/ai_toolkit_baseline_extract.js` with Node inside the pinned image,
  with no network. It transpiles the UI modules with the image's own
  TypeScript and evaluates them the way the UI loader does. It then repeats the
  UI's architecture switch (base job, restore the start arch's unselected
  defaults, apply the new arch's selected defaults). The result is
  `src/kura/backends/ai_toolkit_baseline.json`.
- The baseline fills only listed native keys: `train` scheduler, timestep type,
  precision, optimizer and its parameters, learning rate, loss, and
  text-encoder caching; `model` quantization, `low_vram`, and `model_kwargs`
  (missing subkeys only); and each dataset's `cache_latents_to_disk`. Paths,
  steps, seed, and sampling stay Kura-owned.
- Several UI entries can share one arch (`zimage`, `zimage:turbo`). The entry
  whose model path equals `model.base` wins, then the entry named like the
  authored selector, then the entry named after the arch, then a sole entry.
  Anything else is ambiguous and treated as unknown.
- One UI name differs from the pinned arch: the `sd15` entry builds a Stable
  Diffusion 1.5 job, which the registry selects as `sd1`. The extractor records
  this as a named alias.
- For an unknown or ambiguous arch, compile requires
  `native_config.train.noise_scheduler` and `mixed_precision`, the two values
  observed to decide whether a run trains at all.
- With the forced `cache_latents_to_disk: true` removed, latent caching follows
  the baseline. It is on only where the pinned UI enables it (MiniMax-H3 and its
  Ref2VA variant, LTX-2.3, LTX-2.5, YuE2). The `flex2` entry records `false`, so
  the Flex.2 refusal described above has no reachable input today. It becomes
  necessary if caching becomes an authored option.
- Several UI entries can share one model path (`krea2` and `krea2:o_edit`). A
  path match therefore narrows the candidates rather than deciding alone.
  `model_kwargs.edit` is never taken from the baseline, because the typed
  `model_edit` field owns edit mode.
- Existing AI-Toolkit optimizer evidence predates the baseline. The generated
  configuration changed, so it is not migrated and needs a re-smoke.

## Enforcement

- The release gate checks that each baseline's recorded upstream commit equals
  the adapter's pinned upstream commit. The baseline file is part of the
  adapter source identity, so evidence recorded against a different baseline
  needs a re-smoke or a declared behavior-preserving migration.
- Regenerating a baseline requires the pinned image, so it belongs to the
  `backend-upgrade-audit` flow rather than CI. The regeneration script shows
  the diff; the upgrade review reads that diff.
- Adapter tests prove three things: an unset value takes the baseline value; an
  authored value overrides it; an architecture without an entry refuses.
- The real-smoke harness stops re-deriving UI defaults. It relies on the
  adapter baseline, so a smoke exercises what a user gets with no native
  overrides.

## Scope

This decision applies to AI-Toolkit first, the backend where the gap was
observed. Musubi and sd-scripts adapters already emit per-architecture defaults
by hand. Those hand-written values are existing debt under this decision. They
move to extracted baselines when a pin upgrade or a failure touches them; this
ADR does not require an immediate rewrite.

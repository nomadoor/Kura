# Backlog

Findings a review round deferred (`AGENTS.md`, "How a change is made"): Minor
and Nit findings, and pre-existing issues a review happened to find. They are
fixed later in one batch pull request, which removes the lines it fixes.

One line per item:

```text
- YYYY-MM-DD | area | finding | source (pull request or review) | promise (P1–P11) or -
```

A rejected item keeps its line, starts its finding with `REJECTED:`, and says
why, so it is not raised again.

## Open

- 2026-10-11 | AI-Toolkit outputs | AI-Toolkit writes outputs under `outputs/<run-id>/` on Docker and under `outputs/` on RunPod; both are recorded correctly in `status.outputs`. Accepted deviation for the folder layout, kept here until the layouts are made the same or the deviation is documented for users. Making them the same needs one owner for when a file moves and what happens when the target exists (a reconcile of an already-published run must not move files; a relaunch in the same directory must not collide); see the review of the backlog harness pull request | `docs/adr/promises-and-verification.md` | P5

- 2026-10-11 | checkpoint counts | `expected_checkpoints` counts a Resume's saves as `additional_steps // cadence`, while every trainer now saves on logical multiples (`peak_checkpoints` uses `end // c - start // c`); the plan warning, guard, preflight line, and monitor can be off by one save. Two formulas for one decision | review of the step-labels pull request | P2
- 2026-10-11 | images | Nothing checks at run time that a Musubi Tuner or sd-scripts image carries the step patch; an `images.*` override with an unpatched build trains a Resume to the logical target as extra steps. The state runner could compare the `-stepNNNNNNNN-state` directory step with the scheduler step | review of the step-labels pull request | P1
- 2026-10-11 | AI-Toolkit | After an OOM-skipped iteration (upstream tolerates three), the next step save fails Kura's name check with a message that blames the missing patch rather than the skipped update | review of the step-labels pull request | P11
- 2026-10-11 | AI-Toolkit patch | The patch header says sampling and logging skip the first step on a save; only sampling does (harmless; changing it changes the image build, so it waits for the next image rebuild) | review of the step-labels pull request | -
- 2026-10-11 | state marker | `kura-state-info.json` `logical_step` became `last_epoch // num_processes` under the same `schema_version: 1`; a multi-GPU state still pending from the old runner would publish at an inflated step | review of the step-labels pull request; CodeRabbit on that pull request suggested a schema 2 marker and checking the marker on Resume | P9
- 2026-10-11 | state runner | The Musubi Tuner runner path is written in three places (the contract, `musubi_command.py`, `registry.py`); a drift would make new runs read as compiled before the runner | review of the step-labels pull request | -
- 2026-10-11 | plan | For a Resume compiled before the logical-step change (`process_local` frozen in its lock), the plan and disk preflight show the configured cadence and count saves over the logical range, while its frozen command carries the cadence capped at the steps it adds | CodeRabbit on the step-labels pull request | P8
- 2026-10-14 | save cadence | `_retention_value` and `checkpoint_save_cadence` still `int()` the cadence and treat 0 or less as unset; harmless once the registry validates it, but a second reader of the same decision (keep only for old records) | review of the selectors and decided-values pull request | P2
- 2026-10-14 | AI-Toolkit conditions | A conditional field (`bypass_guidance_embedding`) whose arch is written only in `native_config.model.arch` is refused as "not applicable for model_arch=None"; the message should say to move the arch to `model_arch` | review of the selectors and decided-values pull request | P11
- 2026-10-14 | sd-scripts records | `training_state_contract_sd_scripts` reads an old run.yaml written with an alias (`architecture: flux`) as unsupported, which changes state publication for such a run still in flight across the upgrade | review of the selectors and decided-values pull request | P9
- 2026-10-14 | HF token record | The RunPod launch records `HF_TOKEN` presence from `os.environ` directly (`runpod.py` launch records) instead of `declared_secret`, a second reader of the same decision | review of the run and workspace one-form pull request | P9
- 2026-10-14 | old records | Viewing or planning an old frozen run whose `safety.*_gb` is a string or whose capacity duration is `2H` now fails in `_configured_gib` and `_parse_duration_seconds`; record readers need the old form | review of the run and workspace one-form pull request | P9
- 2026-10-14 | resume copy | Resuming an old Musubi FLUX.2 run that set `allow_a40_*` copies the removed keys, so the draft does not compile until they are deleted | review of the run and workspace one-form pull request | P11
- 2026-10-14 | leftovers | `kura run new --executor docker --gpu X` writes a draft compile refuses | review of the run and workspace one-form pull request | -
- 2026-10-15 | argv booleans | A non-boolean forwarded field written as `false` (for example `learning_rate: false`) renders as no flag, so the trainer's generic default applies although the field is required; refuse true/false on a non-boolean argv field or forward it as text | review of the forwarded values pull request | P11
- 2026-10-15 | owned flags | sd-scripts `OWNED_FLAGS` and `MUSUBI_OWNED_FLAGS` still list field flags (`--network_dim`, `--timestep_sampling`, ...) that the registry's extra_args check now owns, with a different scope; keep only adapter-owned flags there | review of the forwarded values pull request | -
- 2026-10-15 | required clauses | `required=` clauses for Musubi `timestep_sampling` and the MiniMax-H3 guidance values repeat their `_when` condition tuples by hand; derive them from the field's condition | review of the forwarded values pull request | -
- 2026-10-15 | Krea 2 fp8 | `fp8_base` without `fp8_scaled` on Krea 2 now fails inside the trainer (krea2_train_network.py:78-79) after downloads and caching; refuse it at compile citing that line | review of the forwarded values pull request | P6
- 2026-10-15 | tests | No test asserts that Musubi `fp8` is refused with the "use fp8_base" reason | review of the forwarded values pull request | -
- 2026-10-15 | Musubi flow values | `timestep_sampling` / `discrete_flow_shift` are not declared for qwen_image, zimage, flux_kontext, hunyuan_video(_1_5), framepack, kandinsky5, or ideogram4, so those runs take the generic `sigma` / `1.0` unless extra_args sets them; check each architecture's upstream default and declare or require them | review of the forwarded values pull request (pre-existing) | P11
- 2026-10-16 | ComfyUI prepare | The RunPod render copies the merged `comfyui_model_registry.json` to the Pod, and `docker/comfyui/kura_comfy_prepare.py` resolves it again with its own `repo_id`/`direct_url`/`file` fallbacks and its own required-model scan; send the resolved `comfyui_models.json` and have the image read only that at the next image rebuild | review of the model selection pull request | -
- 2026-10-16 | FLUX.2 base | The `model.base` Klein-variant conflict check misses spellings such as `flux-2-klein-9b.safetensors` or `flux2-klein-9b_fp8`; either drop the check (base is a label) or widen the pattern | review of the model selection pull request | P11
- 2026-10-16 | resume copy | A Resume or `--from` copy of an older Musubi run that relied on the automatic bundle, a bundle alias, `model_bundle: none`, or a bundle-filled `model_downloads` fails at compile (safely); the refusal should say to name the bundle or Fork from Weight | review of the model selection pull request | P9
- 2026-10-16 | resume copy | A Resume of an older sd-scripts run whose frozen dataset settings hold a `null` or a subset without `dataset_id` is refused when the derived run compiles; the refusal should say to fix the settings in a Fork from Weight | review of the dataset files pull request | P9
- 2026-10-16 | run ids | A run ID ending in `-state` is read as a training-state directory by `is_training_state_output`, so its weights are never published on either executor | review of the backlog harness pull request (pre-existing) | P5

## Rejected

- 2026-10-14 | render compile | REJECTED: a legacy promptset render whose workflow binds no checkpoint still requires its `inputs.checkpoint` file to exist; moot, since compile now refuses every new promptset render, and launching one an older Kura compiled stages a checkpoint only when its workflow binds `lora` or `model_patch` or inserts a LoRA | review of the render checkpoint pull request | P11
- 2026-10-11 | capabilities | REJECTED: declare field types in `kura run capabilities`; typed fields would break runs that give `learning_rate` as a string | deferred before this backlog existed; source not recorded | -

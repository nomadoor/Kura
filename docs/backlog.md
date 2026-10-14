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

- 2026-10-11 | AI-Toolkit outputs | AI-Toolkit writes outputs under `outputs/<run-id>/` on Docker and under `outputs/` on RunPod; both are recorded correctly in `status.outputs`. Accepted deviation for the folder layout, kept here until the layouts are made the same or the deviation is documented for users | `docs/adr/promises-and-verification.md` | P5
- 2026-10-11 | RunPod wording | The wording of the maximum lease and of the unattended wait does not tell the two apart | deferred before this backlog existed; source not recorded | P11
- 2026-10-11 | progress | Progress text shows `unknown/100` | deferred before this backlog existed; source not recorded | P11
- 2026-10-11 | job runner | The runner's status wording after an idle stop | deferred before this backlog existed; source not recorded | P11
- 2026-10-11 | doctor | `kura doctor disk` output is too long | deferred before this backlog existed; source not recorded | P11
- 2026-10-11 | plan JSON | `cadence_steps` mixes integers and the text "trainer default" | deferred before this backlog existed; source not recorded | -

- 2026-10-11 | conformance | P8 can only fail on the checkpoint count: the byte estimate is 1 GiB per checkpoint against small LoRAs, no scenario run exercises retention or pruning, and the 0.5 s sampling would miss a short peak | review of the promises-and-verification pull request | P8
- 2026-10-11 | conformance | `_sample_peak` can raise `FileNotFoundError` when a staging folder disappears during `rglob`, which stops the harness while `kura run execute` keeps running | review of the promises-and-verification pull request | -
- 2026-10-11 | conformance | With `--runpod` and no `--yes`, the harness still creates datasets and compiles a run before printing the cost, and `--yes` launches whatever the ceiling (even unknown) | review of the promises-and-verification pull request | P6
- 2026-10-11 | conformance | The AI-Toolkit conformance model (`hf-internal-testing/tiny-stable-diffusion-pipe`) has no pinned revision | review of the promises-and-verification pull request | P4
- 2026-10-11 | conformance | A Resume that fails to run shows `-` for P3 instead of FAIL (P1 still fails and the exit code is 1) | review of the promises-and-verification pull request | P3
- 2026-10-11 | checkpoint counts | `expected_checkpoints` counts a Resume's saves as `additional_steps // cadence`, while every trainer now saves on logical multiples (`peak_checkpoints` uses `end // c - start // c`); the plan warning, guard, preflight line, and monitor can be off by one save. Two formulas for one decision | review of the step-labels pull request | P2
- 2026-10-11 | images | Nothing checks at run time that a Musubi Tuner or sd-scripts image carries the step patch; an `images.*` override with an unpatched build trains a Resume to the logical target as extra steps. The state runner could compare the `-stepNNNNNNNN-state` directory step with the scheduler step | review of the step-labels pull request | P1
- 2026-10-11 | AI-Toolkit | After an OOM-skipped iteration (upstream tolerates three), the next step save fails Kura's name check with a message that blames the missing patch rather than the skipped update | review of the step-labels pull request | P11
- 2026-10-11 | AI-Toolkit patch | The patch header says sampling and logging skip the first step on a save; only sampling does (harmless; changing it changes the image build) | review of the step-labels pull request | -
- 2026-10-11 | state marker | `kura-state-info.json` `logical_step` became `last_epoch // num_processes` under the same `schema_version: 1`; a multi-GPU state still pending from the old runner would publish at an inflated step | review of the step-labels pull request; CodeRabbit on that pull request suggested a schema 2 marker and checking the marker on Resume | P9
- 2026-10-11 | state runner | The Musubi Tuner runner path is written in three places (the contract, `musubi_command.py`, `registry.py`); a drift would make new runs read as compiled before the runner | review of the step-labels pull request | -
- 2026-10-11 | images | `kura image build` builds from the editable install's checkout, not from the current worktree, so a worktree's Docker changes are silently left out | found while building the step-label images | -
- 2026-10-11 | plan | For a Resume compiled before the logical-step change (`process_local` frozen in its lock), the plan and disk preflight show the configured cadence and count saves over the logical range, while its frozen command carries the cadence capped at the steps it adds | CodeRabbit on the step-labels pull request | P8

## Rejected

- 2026-10-11 | capabilities | REJECTED: declare field types in `kura run capabilities`; typed fields would break runs that give `learning_rate` as a string | deferred before this backlog existed; source not recorded | -
- 2026-10-14 | YAML reader | `kura.yamlio` keeps the last value of a duplicate mapping key; YAML 1.2 requires unique keys, so a repeated setting should be refused instead of one copy silently winning | CodeRabbit on the authored-input foundation pull request | P9
- 2026-10-14 | YAML reader | `kura.yamlio` still accepts the YAML 1.1 merge key `<<: *anchor`, a second way to write a mapping | CodeRabbit on the authored-input foundation pull request | -
- 2026-10-14 | release gate | The one-YAML-reader check looks only for `import yaml`; another YAML library (for example `ruamel.yaml`) would pass it | CodeRabbit on the authored-input foundation pull request | -
- 2026-10-14 | workspace migrate | `kura workspace migrate` drops an empty `docker.mounts[].source` string but keeps the rest of that mount, which validation then refuses for its missing source | CodeRabbit on the authored-input foundation pull request | P11
- 2026-10-14 | conformance | A conformance workspace on a Windows drive (`/mnt/d`, drvfs) leaves the staged dataset directory unreadable after `rename`, so `ensure_dataset` fails; the harness should refuse a non-ext4 workspace up front or say where to put it | found running conformance for the authored-input foundation pull request | P11
- 2026-10-14 | save cadence | `_retention_value` and `checkpoint_save_cadence` still `int()` the cadence and treat 0 or less as unset; harmless once the registry validates it, but a second reader of the same decision (keep only for old records) | review of the selectors and decided-values pull request | P2
- 2026-10-14 | AI-Toolkit conditions | A conditional field (`bypass_guidance_embedding`) whose arch is written only in `native_config.model.arch` is refused as "not applicable for model_arch=None"; the message should say to move the arch to `model_arch` | review of the selectors and decided-values pull request | P11
- 2026-10-14 | sd-scripts records | `training_state_contract_sd_scripts` reads an old run.yaml written with an alias (`architecture: flux`) as unsupported, which changes state publication for such a run still in flight across the upgrade | review of the selectors and decided-values pull request | P9
- 2026-10-14 | guidance | AGENTS.md and the kura-core skill say values Kura only forwards pass through untouched, but leave out values Kura only records, which the ADR also covers | CodeRabbit on the authored-input foundation pull request | -
- 2026-10-14 | job runner | With an invalid workspace.yaml and a pending local request, the runner logs "local launches wait" every poll, and `follow` tells the user another local run holds the slot | review of the run and workspace one-form pull request | P11
- 2026-10-14 | render compute | A render run.yaml's `compute.gpu` is not validated (a string passes, and it is not refused on a local render); `requested_gpu_types` still reads one string for old records | review of the run and workspace one-form pull request | -
- 2026-10-14 | HF token record | The RunPod launch records `HF_TOKEN` presence from `os.environ` directly (`runpod.py` launch records) instead of `declared_secret`, a second reader of the same decision | review of the run and workspace one-form pull request | P9
- 2026-10-14 | workspace migrate | `comfyui.runpod.cloud_type` and a `storage.host_drive` written as `D:\\` are refused but not migrated | review of the run and workspace one-form pull request | P11
- 2026-10-14 | old records | Viewing or planning an old frozen run whose `safety.*_gb` is a string or whose capacity duration is `2H` now fails in `_configured_gib` and `_parse_duration_seconds`; record readers need the old form | review of the run and workspace one-form pull request | P9
- 2026-10-14 | resume copy | Resuming an old Musubi FLUX.2 run that set `allow_a40_*` copies the removed keys, so the draft does not compile until they are deleted | review of the run and workspace one-form pull request | P11
- 2026-10-14 | leftovers | `kura run new --executor docker --gpu X` writes a draft compile refuses; `_validate_musubi_resource_flags` keeps an unused `architecture` argument and `_musubi_flux2_model_version` import | review of the run and workspace one-form pull request | -

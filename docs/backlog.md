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

- 2026-10-17 | one owner | Which AI-Toolkit archs take `model_edit` is decided twice: the registry condition (`registry.py`, krea2 and moss_image) and the `AI_TOOLKIT_PROFILE_REQUIREMENTS` keys; derive the condition from the requirement keys | review of the AI-Toolkit 0.13.30 upgrade pull request | -
- 2026-10-17 | tests | No test covers: moss_image with references and no `model_edit` (refused), `moss_image:base_edit` without `model_edit` (no `edit` in model_kwargs), `model_edit` refused for flux3, ming_image, and iris3b, ming_image with a reference (refused), and moss_image with an `OpenMOSS-Team/MOSS-Image-Base` base (conflict naming `moss_image:base`) | review of the AI-Toolkit 0.13.30 upgrade pull request | -
- 2026-10-17 | AI-Toolkit baseline | The pinned UI fills `model.assistant_lora_path` (a training adapter) for `qwen_image_2:turbo`, `ming_image`, and the MiniMax-H3 entries, but the baseline fills only the keys docs/adr/upstream-training-baseline.md lists, so a Kura run of those entries trains without the adapter unless the author writes it. Decide in the ADR whether the baseline fills it; filling it changes what the MiniMax-H3 evidence trained | CodeRabbit on the AI-Toolkit 0.13.30 upgrade pull request | P11
- 2026-10-17 | docker queries | `docker info --format {{.DockerRootDir}}` is asked in three places (doctor's storage summary, doctor's docker checks, and `docker_data_location`), and doctor's `_docker_run` has no timeout | review of the image-build free-space pull request | -
- 2026-10-17 | tests | No test covers the doctor build-cache advisory at its 10 GiB boundary, or `kura image build` stopping on a workspace.yaml that still holds the retired `docker.build_cache_limit_gb` | review of the image-build free-space pull request | -
- 2026-10-17 | ComfyUI specs | A RunPod render compiled before 2026-10-16 may hold a model spec with both repo and url or an empty target_name; the image refuses it on the Pod (saying to compile again) instead of the launch refusing it before a Pod exists | review of the ComfyUI prepare pull request | P11
- 2026-10-17 | runner epoch | A follower of an older Kura replaced by a new runner still writes `epoch` beside `runner_epoch` once, because the old reader knows only `epoch`; update Kura with the runner stopped, or say so in the upgrade notes | review of the sd-scripts acceptance fixes pull request | P9
- 2026-10-17 | HF cache check | `hf_cached_file` strips a ref file's trailing whitespace and huggingface_hub does not, so a hand-edited ref could read as cached on the host and not in the container | review of the sd-scripts acceptance fixes pull request | -
- 2026-10-17 | tests | No test covers a CLI writer without `KURA_RUNNER_EPOCH` rewriting an old `epoch` to `runner_epoch`, or a host-side parity test of `hf_cached_file` against `try_to_load_from_cache` where huggingface_hub is installed | review of the sd-scripts acceptance fixes pull request | -
- 2026-10-17 | RunPod recovery | RunPod collection does not bring back `recovery/` (only outputs, checkpoints, samples, metrics), so an Anima native recovery copy made on a failed RunPod run stays on the Pod | review of the sd-scripts acceptance fixes pull request | P7

## Rejected

- 2026-10-17 | frozen keys | REJECTED: `validate_backend_config(frozen=True)` accepts every retired `backend.config` key on every backend, though each was only ever written for one; scoping it per backend would keep a second list of the same keys beside `RETIRED_TRAIN_RUN_KEYS`, and accepting a key no Kura wrote for that backend changes nothing a manifest compiles to | review of the disk-cache-every-backend pull request | -
- 2026-10-14 | render compile | REJECTED: a legacy promptset render whose workflow binds no checkpoint still requires its `inputs.checkpoint` file to exist; moot, since compile now refuses every new promptset render, and launching one an older Kura compiled stages a checkpoint only when its workflow binds `lora` or `model_patch` or inserts a LoRA | review of the render checkpoint pull request | P11
- 2026-10-11 | capabilities | REJECTED: declare field types in `kura run capabilities`; typed fields would break runs that give `learning_rate` as a string | deferred before this backlog existed; source not recorded | -
- 2026-10-11 | state marker | REJECTED: no run recorded in the workspace trained on more than one GPU (`runpod.gpu_count: 1`; no realization records a larger count), so no schema-1 marker holds a multi-process step; a multi-GPU run (which `accelerate launch` starts on its own when a Pod has several GPUs) writes the current rule from its first save (was: `kura-state-info.json` `logical_step` became `last_epoch // num_processes` under the same `schema_version: 1`; a multi-GPU state still pending from the old runner would publish at an inflated step) | review of the step-labels pull request; CodeRabbit on that pull request suggested a schema 2 marker and checking the marker on Resume | P9
- 2026-10-11 | plan | REJECTED: every Resume compiled before the change in the workspace has already completed, so no plan or launch disk preflight reads such a run again; its frozen command is unaffected (was: For a Resume compiled before the logical-step change (`process_local` frozen in its lock), the plan and disk preflight show the configured cadence and count saves over the logical range, while its frozen command carries the cadence capped at the steps it adds) | CodeRabbit on the step-labels pull request | P8

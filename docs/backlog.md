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

- 2026-10-11 | AI-Toolkit patch | The patch header says sampling and logging skip the first step on a save; only sampling does (harmless; changing it changes the image build, so it waits for the next image rebuild) | review of the step-labels pull request | -
- 2026-10-16 | ComfyUI prepare | The RunPod render copies the merged `comfyui_model_registry.json` to the Pod, and `docker/comfyui/kura_comfy_prepare.py` resolves it again with its own `repo_id`/`direct_url`/`file` fallbacks and its own required-model scan; send the resolved `comfyui_models.json` and have the image read only that at the next image rebuild | review of the model selection pull request | -

## Rejected

- 2026-10-17 | frozen keys | REJECTED: `validate_backend_config(frozen=True)` accepts every retired `backend.config` key on every backend, though each was only ever written for one; scoping it per backend would keep a second list of the same keys beside `RETIRED_TRAIN_RUN_KEYS`, and accepting a key no Kura wrote for that backend changes nothing a manifest compiles to | review of the disk-cache-every-backend pull request | -
- 2026-10-14 | render compile | REJECTED: a legacy promptset render whose workflow binds no checkpoint still requires its `inputs.checkpoint` file to exist; moot, since compile now refuses every new promptset render, and launching one an older Kura compiled stages a checkpoint only when its workflow binds `lora` or `model_patch` or inserts a LoRA | review of the render checkpoint pull request | P11
- 2026-10-11 | capabilities | REJECTED: declare field types in `kura run capabilities`; typed fields would break runs that give `learning_rate` as a string | deferred before this backlog existed; source not recorded | -
- 2026-10-11 | state marker | REJECTED: no run recorded in the workspace trained on more than one GPU (`runpod.gpu_count: 1`; no realization records a larger count), so no schema-1 marker holds a multi-process step; a multi-GPU run (which `accelerate launch` starts on its own when a Pod has several GPUs) writes the current rule from its first save (was: `kura-state-info.json` `logical_step` became `last_epoch // num_processes` under the same `schema_version: 1`; a multi-GPU state still pending from the old runner would publish at an inflated step) | review of the step-labels pull request; CodeRabbit on that pull request suggested a schema 2 marker and checking the marker on Resume | P9
- 2026-10-11 | plan | REJECTED: every Resume compiled before the change in the workspace has already completed, so no plan or launch disk preflight reads such a run again; its frozen command is unaffected (was: For a Resume compiled before the logical-step change (`process_local` frozen in its lock), the plan and disk preflight show the configured cadence and count saves over the logical range, while its frozen command carries the cadence capped at the steps it adds) | CodeRabbit on the step-labels pull request | P8

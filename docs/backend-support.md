# Backend support

The baseline support snapshot was taken for Kura 0.2.0 on 2026-08-04. Kura
0.3.0 carried that evidence forward through reviewed, evidence-scoped identity
migrations, and a 2026-08-26 Anima LLLite smoke additionally verified caption
dropout with text-encoder disk caching. Identity transitions are recorded as
`behavior_changed: false` only for the evidence IDs whose execution contracts
remain unchanged in
[adapter-source-identity-migrations.yaml](adapter-source-identity-migrations.yaml).
`scripts/check_smoke_evidence.py` fails if evidence is ever carried across a
transition that is not recorded that way.

This page answers three questions: which upstream version Kura uses, whether
Kura has an adapter, and how far that path has been tested. It intentionally
does not record personal run IDs, hardware inventories, or experiment history.
See [musubi-adapters.md](musubi-adapters.md) for Musubi mechanics.
Machine-readable historical observations live in
[backend-smoke-evidence.yaml](backend-smoke-evidence.yaml); they are
identity-bound evidence, not a second capability registry. When an execution
contract changes, old observations stay in that ledger with an explicit
invalidation instead of being silently promoted to the new contract.
Per-upgrade inventories and evidence cells live under
[backend-validation/](backend-validation/) and follow
[backend-validation.md](backend-validation.md). A plan remains `in_progress`
while Kura cannot faithfully validate or compile an upstream-supported
contract, the pinned image cannot start its entrypoint, or a changed executor
lifecycle lacks runtime evidence.

Support is measured per execution contract: entrypoint, required model roles,
dataset shape, cache behavior, and output/recovery behavior. A change to one of
those requires new evidence; substituting weights within the same contract does
not require an exhaustive matrix. Kura support means an upstream-supported
contract has a tested Kura validation/compile projection and its pinned image
starts the native entrypoint. Real optimizer-step smoke and operational
recovery are higher evidence levels, displayed separately rather than required
for every supported upstream mode. None of these establishes output quality.

## Versions

| Backend | Version used by Kura | Identity |
| --- | --- | --- |
| AI-Toolkit | Docker `0.13.18` plus pinned MiniMax-H3 finite-gradient patch | Kura image `nomadoor/kura-ai-toolkit@sha256:9aa6861b0f54f24f0ebad07b6018b431e8c2403d27eed9233595951b466dbc3a`; base `ostris/aitoolkit:0.13.18@sha256:9bc99d51efc5b6c38a951b3bf8547bda0f9db58abeb75573548d449f82b34bcc`; embedded commit `31ddc709c35d3d3b820c636745397561f806b246`; upstream patch commit `d1985f9bf380b6ce1c409b7875e2d367df486e19`; both commits recorded in `/opt/kura-runtime.json` at build time |
| Musubi Tuner | Git tag `v0.3.5` | commit `4e7c7149249e7715e9168920feb4c420423abba7` |
| sd-scripts | Git tag `v0.11.1` | commit `6721028c79ee85a78b3a06dfd8954dae310a1cce` |

Mutable `latest` is not a supported default.

## 2026-09-21 upgrade audit

The AI-Toolkit upgrade comparison is the exact embedded-source range from
Docker `0.10.22` (`a4bbe167ce03521bf9052d2349f01b2997d67ac7`) to Docker
`0.13.18` (`31ddc709c35d3d3b820c636745397561f806b246`). Kura applies
the exact upstream finite-gradient change from commit
`d1985f9bf380b6ce1c409b7875e2d367df486e19` because the pinned release predates
that fix. The patch is checksum-verified and fails closed if it cannot be
applied or detected as already present. The Musubi comparison is tag `v0.3.4`
to tag `v0.3.5`. The complete contract inventory, including image-only,
conditioned, audio, and loss-recipe distinctions omitted by this summary table,
is recorded in
[the active validation plan](backend-validation/2026-09-21-training-backends.yaml),
with a generated [human-readable test matrix](backend-validation/2026-09-21-training-backends.md).
The completed MiniMax-H3 smokes prove only their exact AI-Toolkit base-video
and image contracts and Musubi silent T2VA-guidance and plain one-frame image
contracts.

| Upstream delta | Kura status | Required action | Current evidence |
| --- | --- | --- | --- |
| AI-Toolkit Qwen-Image 2.1 T2I | Generic image projection | Keep explicit `model_arch: qwen_image_2`; real smoke remains separate | Exact registry diff, compile fixture, and GPU image import passed |
| AI-Toolkit Qwen-Image 2.1 Edit | Manifest image projection plus typed control roles | Author ordered per-sample `control` references in `items.jsonl`; real smoke remains separate | Compile fixture and GPU image import passed |
| AI-Toolkit Anima | Generic image projection | Real smoke remains separate | Exact registry diff, compile fixture, and GPU image import passed |
| AI-Toolkit Mage-Flow Base | Generic image projection | Real smoke remains separate | Exact registry diff, compile fixture, and GPU image import passed |
| AI-Toolkit Mage-Flow Edit | Manifest image projection plus typed control roles | Author ordered per-sample `control` references in `items.jsonl`; real smoke remains separate | Compile fixture and GPU image import passed |
| AI-Toolkit LTX-2.5 | Manifest-backed typed video projection | Real training smoke remains separate | Typed `num_frames`, `fps`, `do_i2v`, and `do_audio` compile fixture; in a real container the embedded-audio preflight rejected a silent video and accepted one with audio before model acquisition (`ai-toolkit-embedded-audio-preflight-docker-2026-09-29`); GPU image import passed |
| AI-Toolkit MiniMax-H3 | Generic typed video projection | Keep gradient checkpointing disabled for the pinned runtime; validate quality separately from this bounded smoke | Patched image published by digest; base one-step A40 smoke passed with all 208 `lora_B` tensors finite and non-zero; Ref2VA and VSA/Fast remain compile/import-only |
| AI-Toolkit MiniMax-H3 Ref2VA | Generic typed video projection plus typed control path | Real smoke remains separate from the base target | `minimax_h3_ref2va` compile fixture and GPU image import passed |
| AI-Toolkit MiniMax-H3 VSA/Fast | Generic typed video projection | Keep evidence distinct from the base real smoke | `minimax_h3_vsa` compile fixture and GPU image import passed |
| AI-Toolkit YuE2 | Outside current contract | Keep explicitly unsupported until Kura owns an audio-training dataset and artifact contract | Added upstream audio model; no Kura audio-training path |
| AI-Toolkit Qwen2.5-Omni | Outside current contract | Keep explicitly unsupported until Kura owns an LLM-training contract | Added upstream LLM model; no Kura LLM-training path |
| Musubi MiniMax-H3 | Built-in adapter for T2VA, FL2VA, Ref2VA, one-frame, guidance, training-adapter, and teacher-matching contracts | T2VA guidance loss and plain one-frame image guidance loss each passed one A40 optimizer step; every other mode remains separate | Typed directory/JSONL datasets, timed controls, ordered references, mode-specific commands, and all three image entrypoints pass validation. Evidence: `musubi-minimax-h3-runpod-2026-09-22`, `musubi-minimax-h3-image-runpod-2026-09-22` |
| Musubi Krea 2 ConvRot INT8 | Typed built-in execution accommodation | Run a real optimizer smoke separately before claiming runtime verification | `convrot_int8` and `convrot_int8_bwd` compile; FP8 and Turbo incompatibilities are rejected |
| Musubi Krea 2 checkpoint CPU offload | Typed built-in execution accommodation | Run a real optimizer smoke separately before claiming runtime verification | `gradient_checkpointing_cpu_offload` compiles and requires gradient checkpointing |
| Musubi Krea 2 Turbo LoRA composition | Sampling-only enhancement | No training-path change while Kura disables training-time sampling | Source/release-note audit only |
| Musubi audio sidecars and JSONL-relative paths | Typed for MiniMax-H3; not shared across other Kura video adapters | Keep non-H3 shared support explicitly unclaimed | MiniMax-H3 validates target/reference audio paths and reference limits; optimizer smoke remains separate |

The GPU image import check loads all eight newly registered AI-Toolkit diffusion
classes: Qwen-Image 2.1, LTX-2.5, Anima, both Mage-Flow classes, and all three
MiniMax-H3 classes. Import smoke proves that the pinned image contains the
implementations; it is not compile or optimizer-step evidence.

For the ordinary `backend.config.model_arch` field, Kura checks the selector
against the registry observed in the pinned AI-Toolkit image before compiling.
For Stable Diffusion 1.x the upstream selector is `sd1`, not `sd15`; Kura does
not silently rewrite one to the other. This check establishes selector
recognition only, not training support or output quality. An advanced custom
image can use `backend.config.native_config.model.arch` as an explicitly
unvalidated escape hatch; its selector and runtime compatibility remain the
author's responsibility.

## Status

| Mark | Meaning |
| --- | --- |
| ✅ | All execution scopes claimed in Verified scope, output materialization, and executor cleanup are verified |
| 🔥 | At least one listed execution contract passed a real optimizer step; other listed contracts keep their own evidence level |
| 🧪 | Upstream support, Kura validation/compile projection, and the pinned-image entrypoint are verified; a real optimizer step is additional confidence, not the support boundary |
| 🔧 | Adapter compilation is covered, but current pinned-image entrypoint evidence is incomplete or historical |
| 🧩 | Native configuration can be expressed; no real smoke claim |
| 📋 | Upstream lists the family; Kura support is not established |
| ⚠️ | Only the stated subset is covered |
| ❌ | Outside the current Kura training contract |

Manifest-v2 RunPod runs use a verified selected-file transfer: stage archives
only the files the frozen handoff selected, proving each file's compile-time
SHA-256 while writing; launch re-proves the staged archive against the compile
before a Pod exists; the Pod verifies every file and the archive, publishes the
verified tree, and builds the frozen views before model acquisition. A real
A40 smoke proved this path end to end
(`ai-toolkit-selected-file-transfer-runpod-main-2026-09-29`, on the current RunPod executor source); older RunPod evidence
below records the execution contract verified at that time.

## Support matrix

AI-Toolkit now fills unset training settings from the pinned UI baseline
(`docs/adr/upstream-training-baseline.md`). This changed the native
configuration Kura generates. AI-Toolkit optimizer evidence recorded before
that change describes the earlier configuration and needs a re-smoke on the
current adapter before it supports a current claim.

| Backend | Model family | Adapter | Status | Verified scope |
| --- | --- | --- | --- | --- |
| AI-Toolkit | SDXL | Generic native-config projection | 🔥 | A local Docker run on the baseline adapter passed, finishing one optimizer step with a finite loss, a published adapter, and a matched input postflight. The 2026-07-12 local and RunPod evidence predates the baseline and is historical. Evidence: `ai-toolkit-sdxl-2026-10-01-1458`, `ai-toolkit-sdxl-2026-10-01-1458`, `ai-toolkit-sdxl-2026-09-29-2323`, `ai-toolkit-sdxl-docker-2026-07-12`, `ai-toolkit-sdxl-runpod-2026-07-12` |
| AI-Toolkit | SD 1.5 | Generic native-config projection | 🔥 | A local Docker run on the baseline adapter passed, finishing one optimizer step with a finite loss, a published adapter, and a matched input postflight. Earlier pinned-image local `sd1` path completed one optimizer step and Kura structural publication; non-root model-cache acquisition also passed. The final manifest-v2 handoff completed one step for root-level and `images/` layouts, over the RunPod selected-file transfer, with a read-only local-path model, and as a Resume. The earlier handoff, transfer, local-path, and Resume evidence predates the baseline. Evidence: `ai-toolkit-sd1-2026-09-29-2323`, `ai-toolkit-sd1-publication-docker-2026-09-23`, `ai-toolkit-manifest-v2-handoff-docker-2026-09-29`, `ai-toolkit-selected-file-transfer-runpod-main-2026-09-29`, `ai-toolkit-local-path-model-docker-2026-09-29`, `ai-toolkit-resume-manifest-v2-docker-2026-09-29` |
| AI-Toolkit | FLUX.1 / Kontext / Flex.2 / Chroma | Typed manifest projection | 🔥 | Fixed-source ordinary image+caption covers FLUX.1, Chroma, Chroma Radiance, and control-free Kontext. The typed paired-folder codec covers one-control Kontext and Flex.2 authored control alternatives; typed `generated_controls` covers the pinned Flex.2 derivations. Flex.2 requires `bypass_guidance_embedding: true` and records that the trainer randomly selects one control per step. FLUX.1, control-pair Kontext, control-pair Flex.2, and Chroma each passed on a RunPod A40 on the baseline adapter over the selected-file transfer, finishing one optimizer step with a finite loss, a published adapter, and a matched input postflight, and stopping the Pod. Evidence: `ai-toolkit-flux-2026-09-29-2323`, `ai-toolkit-flux-kontext-2026-09-29-2323`, `ai-toolkit-flex2-2026-09-29-2323`, `ai-toolkit-chroma-2026-09-29-2323` |
| AI-Toolkit | Qwen Image | Typed manifest projection | 🔥 | Fixed-source ordinary image+caption T2I is first-class. Typed paired folders cover original Edit with one control and Edit Plus with one to three controls. Qwen-Image T2I and control-pair Qwen-Image-Edit each passed on a RunPod A100 80GB PCIe (the A40 container's 50 GB host RAM is too small for the load) over the selected-file transfer, finishing one optimizer step with a finite loss, a published adapter, and a matched input postflight, and stopping the Pod. Edit Plus has no optimizer smoke. Evidence: `ai-toolkit-qwen-image-2026-09-30-0040`, `ai-toolkit-qwen-image-edit-2026-09-30-0040` |
| AI-Toolkit | Qwen-Image 2.1 | Typed manifest projection | 🔥 | T2I and ordered multi-control Edit compile fixtures pass; the control slot count must be uniform across samples. Pinned-image class import passes. T2I passed on a RunPod A40 on the baseline adapter over the selected-file transfer, finishing one optimizer step with a finite loss, a published adapter, and a matched input postflight, and stopping the Pod. Evidence: `ai-toolkit-qwen-image-2-2026-10-01-1458`, `ai-toolkit-qwen-image-2-2026-10-01-1458`, `ai-toolkit-qwen-image-2-2026-09-29-2323` |
| AI-Toolkit | Anima | Generic native-config projection | 🔥 | Compile fixture and pinned-image class import pass. The base model passed on a RunPod A40 on the baseline adapter over the selected-file transfer, finishing one optimizer step with a finite loss, a published adapter, and a matched input postflight, and stopping the Pod. Evidence: `ai-toolkit-anima-2026-09-29-2323` |
| AI-Toolkit | Mage-Flow / Mage-Flow Edit | Typed manifest projection | 🧪 | Base image and ordered one-or-more-control Edit compile fixtures pass; pinned-image class imports pass; no real smoke. |
| AI-Toolkit | HiDream | Typed manifest projection | 🔥 | Fixed-source ordinary image+caption covers HiDream I1 and O1; typed paired folders cover E1 with exactly one control. HiDream I1 Full passed on a RunPod A100 80GB PCIe (the A40 container's 50 GB host RAM is too small for the load) over the selected-file transfer, finishing one optimizer step with a finite loss, a published adapter, and a matched input postflight, and stopping the Pod. O1 and E1 have no optimizer smoke. Evidence: `ai-toolkit-hidream-2026-09-30-0052` |
| AI-Toolkit | FLUX.2 / Krea 2 | Typed manifest projection | 🔥 | Fixed-source ordinary image+caption covers FLUX.2, Klein 4B/9B, and Krea 2 with edit mode false. Ordered image controls are typed for FLUX.2/Klein and Krea 2; Krea 2 requires typed `model_edit: true`. Musubi evidence does not apply. FLUX.2 Klein base 4B and Krea 2 Raw each passed on a RunPod A40 on the baseline adapter over the selected-file transfer, finishing one optimizer step with a finite loss, a published adapter, and a matched input postflight, and stopping the Pod. Evidence: `ai-toolkit-flux2-klein-4b-2026-10-02-2027`, `ai-toolkit-flux2-klein-4b-2026-09-29-2323`, `ai-toolkit-krea2-2026-09-29-2323` |
| AI-Toolkit | Z-Image | Typed manifest projection | ⚠️ | Fixed-source ordinary image+caption path covers Z-Image. Z-Image L2P uses the same media path only when its required tokenizer/text-encoder companion is declared through typed `extras_name_or_path`; the first-class profile refuses an undeclared companion. Z-Image passed on a RunPod A40 on the baseline adapter over the selected-file transfer, finishing one optimizer step with a finite loss, a published adapter, and a matched input postflight, and stopping the Pod; Z-Image L2P has no optimizer smoke. Evidence: `ai-toolkit-zimage-2026-09-29-2323` |
| AI-Toolkit | Wan 2.1 / 2.2 | Native override only | ⚠️ | No first-class video dataset projection |
| AI-Toolkit | LTX-2 / LTX-2.3 | — | 📋 | Not re-audited under the new typed video projection |
| AI-Toolkit | LTX-2.5 | Generic config plus typed video-dataset projection | 🧪 | Compile fixture and pinned-image class import pass; no real smoke |
| AI-Toolkit | MiniMax-H3 | Generic config plus typed video/control-dataset projection | 🔥 | Base video and plain image-only paths each passed one RunPod optimizer step on an NVIDIA A40 with finite, non-zero 208/208 saved `lora_B` tensors, durable step-1 state, output recovery, and Pod shutdown. Ref2VA now has typed same-stem image/video reference folders with uniform slot-kind/count checks and the pinned image-before-video effective order frozen in identity; it remains compile-only. First-frame video, joint audio, Ref2VA, and VSA/Fast do not inherit the base runtime claim. Evidence: `ai-toolkit-minimax-h3-runpod-2026-09-21`, `ai-toolkit-minimax-h3-image-runpod-2026-09-22` |
| AI-Toolkit | ACE-Step | — | ❌ | Audio is outside the current training contract |
| AI-Toolkit | YuE2 | — | ❌ | Audio is outside the current training contract |
| AI-Toolkit | Qwen2.5-Omni | — | ❌ | LLM training is outside the current training contract |
| AI-Toolkit | Other image families | Native override only | ⚠️ | Model-specific review required |
| Musubi Tuner | FLUX.2 | Built-in | 🔥 | dev; Klein/base 4B and 9B; reference-image path compiles. The Klein base 4B selector completed a RunPod A40 run over the selected-file transfer and finished one optimizer step with a finite loss, a published adapter, a matched input postflight, and a stopped Pod. Evidence: `musubi-flux2-klein-4b-2026-09-29` |
| Musubi Tuner | MiniMax-H3 | Built-in | 🔥 | Typed T2VA, FL2VA, Ref2VA, one-frame, timed-control, ordered-reference, guidance, training-adapter, and asymmetric teacher-matching contracts compile; official bundles resolve; all three v0.3.5 entrypoints pass image smoke. T2VA guidance loss and plain one-frame image guidance loss are optimizer/lifecycle verified on A40; other supported modes are marked compile/image verified with no optimizer observation in the generated matrix. Evidence: `musubi-minimax-h3-runpod-2026-09-22`, `musubi-minimax-h3-image-runpod-2026-09-22`, `musubi-minimax-h3-guidance-cache-runpod-2026-09-29` |
| Musubi Tuner | Wan 2.1 / 2.2 | Built-in | ✅ | T2V/I2V, Fun Control, dual-DiT, and Single Frame covered. Evidence: `musubi-wan-t2v-1.3b-docker-2026-07-12`, `musubi-wan-t2v-1.3b-runpod-2026-07-12`, `musubi-wan-video-jsonl-docker-2026-09-29` |
| Musubi Tuner | Krea 2 | Built-in | 🔥 | Broader Krea validation remains separate. A RunPod A40 run over the selected-file transfer completed one optimizer step with a finite loss, a published adapter, a matched input postflight, and a stopped Pod. Evidence: `musubi-krea2-2026-09-29` |
| Musubi Tuner | Qwen-Image | Built-in | 🔥 | Original, Edit, 2509, 2511, and Layered compile paths covered. The original model version completed a RunPod A40 run over the selected-file transfer and finished one optimizer step with a finite loss, a published adapter, a matched input postflight, and a stopped Pod. Evidence: `musubi-qwen-image-2026-09-29` |
| Musubi Tuner | Z-Image | Built-in | 🔥 | A RunPod A40 run over the selected-file transfer completed one optimizer step with a finite loss, a published adapter, a matched input postflight, and a stopped Pod. Evidence: `musubi-zimage-2026-10-01-1458`, `musubi-zimage-2026-10-01-1458`, `musubi-zimage-2026-09-29` |
| Musubi Tuner | FLUX.1 Kontext | Built-in | 🔥 | Paired/control dataset path covered. A RunPod A40 run over the selected-file transfer completed one optimizer step with a finite loss, a published adapter, a matched input postflight, and a stopped Pod. Evidence: `musubi-flux-kontext-2026-09-29` |
| Musubi Tuner | Ideogram 4 | Built-in | 🔥 | A RunPod A40 run over the selected-file transfer completed one optimizer step with a finite loss, a published adapter, a matched input postflight, and a stopped Pod. Evidence: `musubi-ideogram4-2026-09-29` |
| Musubi Tuner | HiDream-O1-Image | Built-in | 🔥 | T2I and I2I compile paths covered. The dev T2I selector completed a RunPod A40 run over the selected-file transfer and finished one optimizer step with a finite loss, a published adapter, a matched input postflight, and a stopped Pod. Evidence: `musubi-hidream-o1-2026-09-29` |
| Musubi Tuner | HunyuanVideo | Built-in | 🔥 | A one-frame video smoke passed on a RunPod A40 over the selected-file transfer, finishing one optimizer step with a finite loss, a published adapter, a matched input postflight, and a stopped Pod. Evidence: `musubi-hunyuan-video-2026-09-29` |
| Musubi Tuner | HunyuanVideo 1.5 | Built-in | 🔥 | T2V and I2V compile paths covered. A one-frame T2V video smoke passed on a RunPod A40 over the selected-file transfer, finishing one optimizer step with a finite loss, a published adapter, a matched input postflight, and a stopped Pod. Evidence: `musubi-hunyuan-video-1-5-2026-09-29` |
| Musubi Tuner | FramePack | Built-in | 🔥 | Normal, F1, and Single Frame compile paths covered. A 37-frame head-extraction video smoke from a 30 fps source passed on a RunPod A40 over the selected-file transfer, finishing one optimizer step with a finite loss, a published adapter, and a matched input postflight, and stopping the Pod; a source below 30 fps cannot fill the window with the pinned loader. Evidence: `musubi-framepack-2026-09-30-0201`, `musubi-framepack-video-docker-2026-07-12` |
| Musubi Tuner | Kandinsky 5 | Built-in | ⚠️ | Lite real-smoked; Pro remains capacity-dependent. The Lite T2V model passed a one-frame video smoke on a RunPod A40 over the selected-file transfer, finishing one optimizer step with a finite loss, a published adapter, a matched input postflight, and a stopped Pod. Evidence: `musubi-kandinsky5-lite-2026-09-29` |
| sd-scripts | Stable Diffusion 1.5 LoRA | Built-in | 🔥 | Two uninterrupted 100-step controls and a 50+50 Resume run completed with identical learned weights, optimizer, scheduler, and normalized train state in the recorded one-item case. The newer Kura post-exit publication gate still needs a real container smoke. The final manifest-v2 handoff completed one step for a nested multi-concept dataset (per-group subsets and repeats), with a read-only local-path checkpoint, and as a Resume. Evidence: `sd-scripts-sd15-resume-equivalence-docker-2026-08-27`, `sd-scripts-manifest-v2-multiconcept-docker-2026-09-29` |
| sd-scripts | SDXL LoRA | Built-in | 🔥 | Compiles through the manifest `ordinary-image-lora` profile and `dreambooth-image-subset` codec (frozen view, native TOML, trainer command). A local Docker run completed one optimizer step under the current durable-state contract, with a finite loss, a published adapter, and a matched input postflight. The earlier optimizer smoke is historical only. Evidence: `sd-scripts-sdxl-2026-10-01-1458`, `sd-scripts-sdxl-2026-10-01-1458`, `sd-scripts-sdxl-2026-09-29` |
| sd-scripts | FLUX.1 LoRA | Built-in | 🔥 | Compiles through the manifest `ordinary-image-lora` profile and `dreambooth-image-subset` codec (frozen view, native TOML, trainer command). A RunPod A40 run over the selected-file transfer completed one optimizer step under the current durable-state contract, with a finite loss, a published adapter, a matched input postflight, and a stopped Pod. The earlier optimizer smoke is historical only. Evidence: `sd-scripts-flux1-2026-09-29` |
| sd-scripts | Anima LoRA | Built-in | 🔥 | Compiles through the manifest `ordinary-image-lora` profile and `dreambooth-image-subset` codec (frozen view, native TOML, trainer command). A local Docker run completed one optimizer step under the current durable-state contract, with a finite loss, a published adapter, and a matched input postflight. The earlier optimizer smoke is historical only. Evidence: `sd-scripts-anima-lora-2026-09-29` |
| sd-scripts | Anima ControlNet-LLLite | Built-in | 🔥 | Typed manifest target/control projection and exact native TOML verification are implemented. A local Docker run with a target/control pair completed one optimizer step under the current durable-state contract, with a finite loss, a published LLLite output, and a matched input postflight. Earlier optimizer smokes are historical only. Evidence: `sd-scripts-anima-lllite-2026-09-29` |
| sd-scripts | Other upstream families and modes | Explicit command only | ⚠️ | No built-in selector or support claim in the initial milestone |

Musubi `v0.3.5` and Kura's FLUX.2 VAE compatibility patch passed the full image
entrypoint smoke on 2026-09-21: all 39 registered scripts existed and completed
their `--help` path, including the three MiniMax-H3 scripts. Variant coverage
means Kura selects the correct scripts, model roles, dataset shape, and flags;
it does not mean every checkpoint has been trained.

AI-Toolkit owns model acquisition and model-specific configuration. Kura keeps
one generic native-config projection rather than duplicating AI-Toolkit's model
catalog. SDXL is the verified default path; SD 1.5 also verifies that this
projection is not SDXL-specific. Other families remain explicit configurations
until representative tests promote them.

AI-Toolkit video runs keep dataset source paths under Kura ownership and use
the typed `backend.config.dataset_config` mapping for `num_frames`, `fps`,
`do_i2v`, and `do_audio`. `do_i2v` conditions on the selected target video's
first frame. `do_audio` selects only embedded target-video audio and runs the
pinned trainer's own video/audio loader before model acquisition; authored
audio sidecars are rejected. In pinned `toolkit/dataloader_mixins.py:740-838`,
a video with no audio stream leaves `audio_tensor` as `None`; no zero-filled
audio is synthesized. A real-container smoke confirmed that preflight rejects
such a video before acquisition
(`ai-toolkit-embedded-audio-preflight-docker-2026-09-29`). Short videos are
stretched by the pinned loader rather than silently skipped. Paired image/edit runs use manifest control roles
projected into ordered run-owned folders. The protected native `datasets` list
cannot bypass Kura's dataset source contract.

MiniMax-H3 uses a checksum-pinned upstream finite-gradient patch on top of the
official `0.13.18` image. For Kura-managed LoRA training-state runs, publication
also inspects the saved safetensors payload and requires at least one finite,
non-zero `lora_B` tensor after an optimizer step. A successful process exit and
checkpoint file alone are therefore not accepted as optimizer-step evidence.

The pinned MiniMax-H3 runtime rejects `gradient_checkpointing: true`. Two A40
probes reached backward but failed PyTorch's non-reentrant checkpoint
recomputation check because forward and recomputation saved different tensor
counts. The successful one-step smoke therefore establishes only the explicit
non-checkpointed contract; Kura fails the incompatible setting at compile time
instead of spending GPU time on a known-bad path.

Real smoke validates execution, not LoRA quality. Quality still requires a
meaningful training run followed by generation and human evaluation.

All five sd-scripts Tier 1 paths have now completed a real optimizer step through
local Docker. Both Anima output forms also completed their managed-ComfyUI load
and render contracts. These are execution-compatibility results, not claims
about output quality from the bounded one-step recipes.

# Backend support matrix

Snapshot date: 2026-07-11.

This is the operational support summary for Kura's training backends. It keeps
three different facts separate:

1. the upstream version Kura actually pins;
2. whether Kura has an adapter or native-config projection for the model;
3. what has completed a real run through Kura.

An upstream model name or a successful compile is not, by itself, a claim that
training works or that the resulting LoRA has useful quality. Detailed audit
notes and upstream links are in
[upstream-model-support-audit.md](upstream-model-support-audit.md).

## Status legend

| Mark | Level | Meaning |
| --- | --- | --- |
| ✅ | Operational | Real run, output recovery, and disposable-compute cleanup verified. |
| 🧪 | Real smoke | A real model completed at least one optimizer step through Kura. |
| 🔧 | Image smoke | Required entrypoints exist in the pinned image and start. |
| 🧩 | Expressible | Kura can freeze the native config or command; no real-run claim. |
| 📋 | Listed | The pinned upstream advertises it; Kura support is not established. |
| ⚠️ | Limited | Only the stated subset or explicit native override is supported. |
| ❌ | Out of scope | The current Kura train-run contract does not support this workflow. |

| Backend | Version | Adapter coverage | Best verified path |
| --- | --- | --- | --- |
| AI-Toolkit | `0.10.22` | ⚠️ | ✅ SDXL local + RunPod |
| Musubi Tuner | `v0.3.4` | ✅ 12/12 | ✅ Wan local + RunPod; 🧪 all families |

`🧪 Real smoke` and `✅ Operational` validate the execution path, not training
quality. Quality requires generation and human evaluation from a meaningful
training run.

## Versions in use

| Backend | Kura pin | Local image | RunPod image | Verified upstream identity |
| --- | --- | --- | --- | --- |
| AI-Toolkit | Docker `0.10.22` | `nomadoor/kura-ai-toolkit:dev`, built from `ostris/aitoolkit:0.10.22@sha256:5a810f50de920aaa3439487959ae392bf0d1458345baddee24a7bf33787c0438` | `ostris/aitoolkit:0.10.22` | embedded commit `a4bbe167ce03521bf9052d2349f01b2997d67ac7` |
| Musubi Tuner | Git tag `v0.3.4` | `nomadoor/kura-musubi-tuner:dev` | `nomadoor/kura-musubi-tuner:dev` | commit `30c658c4f4b0bf05038b3346eff9670259b10fc7` |

The AI-Toolkit digest is pinned in `docker/ai-toolkit/Dockerfile`. Mutable
`latest` is not the supported default.

## AI-Toolkit

AI-Toolkit owns base-model and companion-model acquisition. Kura generates its
native YAML and passes backend-specific overrides; it does not duplicate
AI-Toolkit's model loader as a Kura model registry.

| Model family | Status | Kura path | Real hardware evidence | Notes |
| --- | --- | --- | --- | --- |
| SDXL | ✅ | Native-config projection | RTX 4070 Ti local; RTX A5000 RunPod | LoRA, config, optimizer recovery, and Pod cleanup verified. |
| SD 1.5 | 🧩 | Native-config projection | — | Generic image-folder contract; no current real smoke. |
| FLUX.1 / Kontext / Flex / Chroma | 🧩 | Native config + overrides | — | Model-specific defaults are not promoted as verified. |
| Qwen Image | ⚠️ | Native config + overrides | — | T2I expressible; edit/control requires an explicit native dataset config. |
| HiDream | 🧩 | Native config + overrides | — | No AI-Toolkit real smoke through Kura. |
| FLUX.2 / Krea 2 | 🧩 | Native config + overrides | — | Musubi evidence does not transfer to this backend. |
| Z-Image | ⚠️ | Native config + overrides | — | Companion artifacts differ by variant. |
| Wan 2.1 / 2.2 | ⚠️ | Native override required | — | No first-class video projection. |
| LTX-2 / LTX-2.3 | 📋 | Native override required | — | No first-class Kura video projection. |
| ACE-Step | ❌ | — | — | Audio is outside the current train-run dataset contract. |
| Other image families | ⚠️ | Native override only | — | Model-specific review required. |

The default generated AI-Toolkit recipe is operationally verified for SDXL.
Other families remain explicit, reviewable backend configurations until they
gain representative evidence.

## Musubi Tuner

Kura has a built-in adapter for every top-level architecture in the pinned
Musubi Tuner `v0.3.4` release. The image smoke checks all 36 expected cache and
training entrypoints.

| Architecture | Adapter | Variant compile coverage | Evidence | Real hardware evidence |
| --- | --- | --- | --- | --- |
| FLUX.2 | ✅ | dev; Klein/base 4B and 9B; reference images | 🧪 | Representative real smoke recorded |
| Wan 2.1 / 2.2 | ✅ | 2.1 T2V/I2V/Fun Control; 2.2 dual-DiT T2V/I2V; Single Frame | ✅ | 1.3B local + RTX A6000 RunPod; Single Frame 14B on RTX 4070 Ti |
| Krea 2 | ✅ | Standard LoRA path | 🧪 | Historical Kura smoke; broader validation remains separate |
| Qwen-Image | ✅ | Original; Edit; Edit-2509; Edit-2511; Layered | 🧪 | Original path on RunPod A40 |
| Z-Image | ✅ | Standard LoRA path | 🧪 | Representative real smoke recorded |
| FLUX.1 Kontext | ✅ | Paired/control data path | 🧪 | Representative paired-data smoke recorded |
| Ideogram 4 | ✅ | Standard LoRA path | 🧪 | Representative real smoke recorded |
| HiDream-O1-Image | ✅ | T2I; I2I control/reference | 🧪 | T2I representative smoke recorded |
| HunyuanVideo | ✅ | Standard LoRA path | 🧪 | Representative real smoke recorded |
| HunyuanVideo 1.5 | ✅ | T2V; I2V image-encoder path | 🧪 | T2V representative smoke recorded |
| FramePack | ✅ | Normal; F1; Single Frame | 🧪 | Normal representative smoke recorded |
| Kandinsky 5 | ✅ | Lite/Pro T2V; Pro I2V | ⚠️ | Lite T2V recorded; Pro remains capacity-dependent |

Variant compile coverage means that Kura selects the correct cache scripts,
training script, mandatory model roles, and variant flags. It does not mean
that every checkpoint in the variant family was downloaded and trained.

## Shared execution evidence

The current local and RunPod paths use the same Hugging Face cache contract:

```text
HF_HOME=/workspace/cache/huggingface
HF_HUB_CACHE=/workspace/cache/huggingface/hub
```

Verified acceptance runs include:

- AI-Toolkit SDXL local and RunPod one-step training;
- Musubi Wan 2.1 1.3B local and RunPod one-step training;
- Musubi Wan 2.1 Single Frame 14B local one-step training after an empty-cache
  30.5 GiB acquisition;
- output validation and recovery for both RunPod backends;
- zero remaining RunPod Pods and Network Volumes after recovery;
- zero cgroup OOM kills in the latest AI-Toolkit and Musubi RunPod acceptance
  runs.

## Updating this matrix

Update a backend pin and this matrix together. A new upstream family starts at
`Listed`; it reaches `Expressible` only after Kura can compile its real native
contract. Promote it to `Real smoke` or `Operational` only with run artifacts
that identify the image/revision, dataset shape, hardware, output, and cleanup
result. Do not add a second global model registry merely to mirror upstream
model names.

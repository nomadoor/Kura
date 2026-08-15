# Backend support

Support snapshot taken for Kura 0.2.0 on 2026-08-04, and still current for
0.3.0. No new smokes were run for 0.3.0; the snapshot carries forward because
adapter behaviour did not change. The 0.3.0 adapter source identity moved from
`selected-adapter-v1` to `selected-adapter-v2` when the backend configuration
surfaces were declared, and each backend records that transition as
`behavior_changed: false` in
[adapter-source-identity-migrations.yaml](adapter-source-identity-migrations.yaml).
`scripts/check_smoke_evidence.py` fails if evidence is ever carried across a
transition that is not recorded that way.

This page answers three questions: which upstream version Kura uses, whether
Kura has an adapter, and how far that path has been tested. It intentionally
does not record personal run IDs, hardware inventories, or experiment history.
See [musubi-adapters.md](musubi-adapters.md) for Musubi mechanics.
Machine-readable historical observations live in
[backend-smoke-evidence.yaml](backend-smoke-evidence.yaml); they are
identity-bound evidence, not a second capability registry.

Support is measured per execution contract: entrypoint, required model roles,
dataset shape, cache behavior, and output/recovery behavior. A change to one of
those requires new evidence; substituting weights within the same contract does
not require an exhaustive matrix. “Upstream listed”, expressible configuration,
image smoke, real optimizer-step smoke, and operational recovery are distinct
claims. Only real smoke plus the stated recovery scope supports an unqualified
claim that a path works. None of these establishes output quality.

## Versions

| Backend | Version used by Kura | Identity |
| --- | --- | --- |
| AI-Toolkit | Docker `0.10.22` | `ostris/aitoolkit:0.10.22`; embedded commit `a4bbe167ce03521bf9052d2349f01b2997d67ac7` |
| Musubi Tuner | Git tag `v0.3.4` | commit `30c658c4f4b0bf05038b3346eff9670259b10fc7` |
| sd-scripts | Git tag `v0.11.1` | commit `6721028c79ee85a78b3a06dfd8954dae310a1cce` |

Mutable `latest` is not a supported default.

## Status

| Mark | Meaning |
| --- | --- |
| ✅ | All execution scopes claimed in Verified scope, output materialization, and executor cleanup are verified |
| 🧪 | At least one real one-step training smoke passed |
| 🔧 | Adapter compiles and image entrypoints start |
| 🧩 | Native configuration can be expressed; no real smoke claim |
| 📋 | Upstream lists the family; Kura support is not established |
| ⚠️ | Only the stated subset is covered |
| ❌ | Outside the current Kura training contract |

## Support matrix

| Backend | Model family | Adapter | Status | Verified scope |
| --- | --- | --- | --- | --- |
| AI-Toolkit | SDXL | Generic native-config projection | ✅ | Local and RunPod one-step paths verified. Evidence: `ai-toolkit-sdxl-docker-2026-07-12`, `ai-toolkit-sdxl-runpod-2026-07-12` |
| AI-Toolkit | SD 1.5 | Generic native-config projection | 🧪 | Local one-step path verified |
| AI-Toolkit | FLUX.1 / Kontext / Flex / Chroma | Generic native-config projection | 🧩 | Model-specific defaults not verified |
| AI-Toolkit | Qwen Image | Generic native-config projection | ⚠️ | T2I expressible; edit/control needs explicit dataset config |
| AI-Toolkit | HiDream | Generic native-config projection | 🧩 | No current real smoke |
| AI-Toolkit | FLUX.2 / Krea 2 | Generic native-config projection | 🧩 | Musubi evidence does not apply to this backend |
| AI-Toolkit | Z-Image | Generic native-config projection | ⚠️ | Companion artifacts vary by variant |
| AI-Toolkit | Wan 2.1 / 2.2 | Native override only | ⚠️ | No first-class video dataset projection |
| AI-Toolkit | LTX-2 / LTX-2.3 | — | 📋 | No first-class video dataset projection |
| AI-Toolkit | ACE-Step | — | ❌ | Audio is outside the current training contract |
| AI-Toolkit | Other image families | Native override only | ⚠️ | Model-specific review required |
| Musubi Tuner | FLUX.2 | Built-in | 🧪 | dev; Klein/base 4B and 9B; reference-image path compiles |
| Musubi Tuner | Wan 2.1 / 2.2 | Built-in | ✅ | T2V/I2V, Fun Control, dual-DiT, and Single Frame covered. Evidence: `musubi-wan-t2v-1.3b-docker-2026-07-12`, `musubi-wan-t2v-1.3b-runpod-2026-07-12` |
| Musubi Tuner | Krea 2 | Built-in | 🧪 | Broader Krea validation remains separate |
| Musubi Tuner | Qwen-Image | Built-in | 🧪 | Original, Edit, 2509, 2511, and Layered compile paths covered |
| Musubi Tuner | Z-Image | Built-in | 🧪 | — |
| Musubi Tuner | FLUX.1 Kontext | Built-in | 🧪 | Paired/control dataset path covered |
| Musubi Tuner | Ideogram 4 | Built-in | 🧪 | — |
| Musubi Tuner | HiDream-O1-Image | Built-in | 🧪 | T2I and I2I compile paths covered |
| Musubi Tuner | HunyuanVideo | Built-in | 🧪 | — |
| Musubi Tuner | HunyuanVideo 1.5 | Built-in | 🧪 | T2V and I2V compile paths covered |
| Musubi Tuner | FramePack | Built-in | 🧪 | Normal, F1, and Single Frame compile paths covered. Evidence: `musubi-framepack-video-docker-2026-07-12` |
| Musubi Tuner | Kandinsky 5 | Built-in | ⚠️ | Lite real-smoked; Pro remains capacity-dependent |
| sd-scripts | Stable Diffusion 1.5 LoRA | Built-in | 🧪 | Local one-step LoRA passed with the v2 symlink compatibility path; its adapter identity is connected to the current tree by reviewed behavior-preserving migrations. Evidence: `sd-scripts-sd15-docker-v2-2026-08-01` |
| sd-scripts | SDXL LoRA | Built-in | 🧪 | Local one-step LoRA passed through `sdxl_train_network.py`; its adapter identity is connected to the current tree by reviewed behavior-preserving migrations. Evidence: `sd-scripts-sdxl-docker-2026-08-01` |
| sd-scripts | FLUX.1 LoRA | Built-in | 🧪 | Local one-step LoRA passed through `flux_train_network.py` on a 12 GiB-class GPU with fp8 base and 16-block swapping; its adapter identity is connected to the current tree by reviewed behavior-preserving migrations. Evidence: `sd-scripts-flux1-docker-2026-08-01` |
| sd-scripts | Anima LoRA | Built-in | 🧪 | Local training published every retained step checkpoint plus the final alias as validated ComfyUI LoRAs; the prior compatibility smoke also completed a managed-ComfyUI render. Evidence: `sd-scripts-anima-checkpoint-publication-docker-2026-08-03` |
| sd-scripts | Anima ControlNet-LLLite | Built-in | 🧪 | Local one-step training produced a v2 model patch and managed ComfyUI loaded it through `ModelPatchLoader` and `AnimaLLLiteApply`. Evidence: `sd-scripts-anima-lllite-docker-2026-08-03` |
| sd-scripts | Other upstream families and modes | Explicit command only | ⚠️ | No built-in selector or support claim in the initial milestone |

Musubi `v0.3.4` has no missing top-level Kura adapter. All 36 expected cache
and training entrypoints pass image smoke. Variant coverage means Kura selects
the correct scripts, model roles, dataset shape, and flags; it does not mean
every checkpoint has been trained.

AI-Toolkit owns model acquisition and model-specific configuration. Kura keeps
one generic native-config projection rather than duplicating AI-Toolkit's model
catalog. SDXL is the verified default path; SD 1.5 also verifies that this
projection is not SDXL-specific. Other families remain explicit configurations
until representative tests promote them.

Real smoke validates execution, not LoRA quality. Quality still requires a
meaningful training run followed by generation and human evaluation.

All five sd-scripts Tier 1 paths have now completed a real optimizer step through
local Docker. Both Anima output forms also completed their managed-ComfyUI load
and render contracts. These are execution-compatibility results, not claims
about output quality from the bounded one-step recipes.

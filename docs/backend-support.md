# Backend support

Snapshot: 2026-07-11.

This page answers three questions: which upstream version Kura uses, whether
Kura has an adapter, and how far that path has been tested. It intentionally
does not record personal run IDs, hardware inventories, or experiment history.
See [upstream-model-support-audit.md](upstream-model-support-audit.md) for the
detailed audit and [musubi-adapters.md](musubi-adapters.md) for Musubi mechanics.
Machine-readable historical observations live in
[backend-smoke-evidence.yaml](backend-smoke-evidence.yaml); they are
identity-bound evidence, not a second capability registry.

## Versions

| Backend | Version used by Kura | Identity |
| --- | --- | --- |
| AI-Toolkit | Docker `0.10.22` | `ostris/aitoolkit:0.10.22`; embedded commit `a4bbe167ce03521bf9052d2349f01b2997d67ac7` |
| Musubi Tuner | Git tag `v0.3.4` | commit `30c658c4f4b0bf05038b3346eff9670259b10fc7` |

Mutable `latest` is not a supported default.

## Status

| Mark | Meaning |
| --- | --- |
| ✅ | Local and RunPod execution, output recovery, and cleanup verified |
| 🧪 | At least one real one-step training smoke passed |
| 🔧 | Adapter compiles and image entrypoints start |
| 🧩 | Native configuration can be expressed; no real smoke claim |
| 📋 | Upstream lists the family; Kura support is not established |
| ⚠️ | Only the stated subset is covered |
| ❌ | Outside the current Kura training contract |

## Support matrix

| Backend | Model family | Adapter | Status | Notes |
| --- | --- | --- | --- | --- |
| AI-Toolkit | SDXL | Generic native-config projection | ✅ | Local and RunPod one-step paths verified |
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
| Musubi Tuner | Wan 2.1 / 2.2 | Built-in | ✅ | T2V/I2V, Fun Control, dual-DiT, and Single Frame covered |
| Musubi Tuner | Krea 2 | Built-in | 🧪 | Broader Krea validation remains separate |
| Musubi Tuner | Qwen-Image | Built-in | 🧪 | Original, Edit, 2509, 2511, and Layered compile paths covered |
| Musubi Tuner | Z-Image | Built-in | 🧪 | — |
| Musubi Tuner | FLUX.1 Kontext | Built-in | 🧪 | Paired/control dataset path covered |
| Musubi Tuner | Ideogram 4 | Built-in | 🧪 | — |
| Musubi Tuner | HiDream-O1-Image | Built-in | 🧪 | T2I and I2I compile paths covered |
| Musubi Tuner | HunyuanVideo | Built-in | 🧪 | — |
| Musubi Tuner | HunyuanVideo 1.5 | Built-in | 🧪 | T2V and I2V compile paths covered |
| Musubi Tuner | FramePack | Built-in | 🧪 | Normal, F1, and Single Frame compile paths covered |
| Musubi Tuner | Kandinsky 5 | Built-in | ⚠️ | Lite real-smoked; Pro remains capacity-dependent |

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

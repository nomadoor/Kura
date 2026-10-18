# sd-scripts 37a1cbbc -> v0.12.0: upstream delta and Kura impact

The inventory the `backend-upgrade-audit` skill asks for, for the pin change
of the sd-scripts image. The real runs on the first v0.12.0 image (patches
0001 and 0002, `sha256:93a887f4...`) are in
`2026-10-18-sd-scripts-v0.12.0-conformance.yaml` and
`2026-10-18-sd-scripts-v0.12.0-real-smokes.yaml`; the runs on the pinned image,
rebuilt with patch 0003, are in
`2026-10-19-sd-scripts-v0.12.0-flux-prefix-conformance.yaml` and
`2026-10-19-sd-scripts-v0.12.0-flux-prefix-real-smokes.yaml`, all next to this
note. The 93a887f4 image was never pinned on main.

## Identities

| | Tag | Commit | Date |
| --- | --- | --- | --- |
| Old pin | none (between v0.11.1 and v0.12.0) | 37a1cbbc5725ed2a3575506e7bd2001c9908ac92 | 2026-07-23 |
| New pin | v0.12.0 (also the upstream main HEAD) | 690ea7f96c23182352ec63def76d431c6120bd2f | 2026-09-24 |

37a1cbbc is an ancestor of v0.12.0; the range has 40 commits. The upstream
`dev` branch has 27 newer, unreleased commits (MOSS-Image LoRA, ConvRot INT8,
multi-line captions, a Text Encoder cache caption fix, scheduler kwargs); they
are not adopted here and need their own audit.

Old image: `nomadoor/kura-sd-scripts@sha256:4810a8ce5edf6959ae3d1fbd3a75e975ff3d3fdb330f8775fd33542bbc116f06`
(CUDA 12.8, base `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`). The new
digest is in `src/kura/images.py` (`PINNED_IMAGES["sd-scripts"]`) and in the
evidence files.

## Delta, 37a1cbbc..v0.12.0

58 files, +4370/-4109. None of the files Kura runs or patches changed:
`train_network.py`, `sdxl_train_network.py`, `flux_train_network.py`,
`anima_train_network.py`, `anima_train_control_net_lllite.py`,
`networks/convert_anima_lora_to_comfy.py`, `networks/lora*.py`,
`networks/control_net_lllite*.py`, `library/train_util.py`,
`library/config_util.py`, `library/anima_*.py`, `library/flux_train_utils.py`,
`library/strategy_anima.py`, `library/optimizer.py`,
`library/custom_train_functions.py`.

| Upstream change (PR) | Class |
| --- | --- |
| OFTv2 and BOFT network modules (PR 2357, PR 2427) | new network modules |
| opencv optional through a stub (PR 2430) | deps; real opencv still installs on Linux x86_64 |
| requirements markers for Windows ARM64 (PR 2431-2433) | deps |
| local regression harness under `tests/local` (PR 2434) | tests only |
| SD2 and Lumina tokenizer sources (PR 2435) | runtime Hub fetch by the trainer |
| transformers 4.54.1 -> 5.5.4, diffusers 0.32.1 -> 0.40.0, accelerate 1.6.0 -> 1.15.0, huggingface-hub 0.34.3 -> 1.32.0; ftfy-normalizing CLIP tokenizer wrapper (PR 2436) | deps; tokenization meant to be id-identical |
| transformers pinned to 5.17.0; `CLIPTextModel` wrapper keeps `text_model.` keys and `lora_te_text_model_*` names; T5 uses SDPA (PR 2437) | deps; FLUX.1 T5 outputs differ slightly from the old image |
| `dpmsolver` sampler fix; `lms` needs scipy at startup when sampling (PR 2438) | sampling only |
| `gen_img_diffusers.py` removed (PR 2439) | unused by Kura |
| changelog (PR 2440) | docs |

Already in the old pin, so not part of this delta: Anima weight renaming,
the caption_separator shuffle fix, the FineTuning wildcard fix, and the
per-subset timestep offset with `--show_timesteps_offset`.

CLI arguments, compared by AST over every `add_argument` at both refs: the
only removals are the six `gen_img_diffusers.py` flags, the only additions are
two flags of the `tests/local` harness. No training flag was added, removed,
renamed, or re-defaulted. Save format, Resume state file names
(Accelerate 1.15), and the dataset TOML schema are unchanged.

Kura's patches against v0.12.0 (`git apply --check`): 0001 (symlink name)
applies with hunk offsets +1/+2; 0002 (logical Resume step) applies with no
offset, `train_network.py` being identical. Both are still needed. Patch 0003
(below) is new with this upgrade.

Adapter surface: the 96 flags Kura emits or owns and the 33 dataset TOML keys
resolve the same at both refs (`--network_multiplier`, `dataset_id`, and
`group` are Kura-owned or staging keys, as before). Every entrypoint in
`ENTRYPOINTS` and the image probe exists and is byte-identical.

## Kura impact matrix

| Upstream delta | Kura status | Required action | Evidence |
| --- | --- | --- | --- |
| Pin 37a1cbbc -> v0.12.0 | built-in: sd15, sdxl, flux1 LoRA; anima LoRA and LLLite | bump the ref, build, publish, re-pin the digest | image build; conformance; smokes |
| Patch 0001 (symlink name) | built-in | none | `git apply --check` in the image build |
| Patch 0002 (logical Resume step) | built-in, `native_progress: logical` | header names the new commit | image build; conformance P2/P3 |
| transformers 5.17 / diffusers 0.40 / accelerate 1.15 / hub 1.32 | runtime of every path | image build | image build; conformance (sd15); smokes below |
| CLIPTextModel wrapper | sd15, sdxl, flux1 | none; Kura's sdxl/flux1 smokes are U-Net only | conformance (sd15 trains the TE) |
| ftfy CLIP tokenizer | sd15, sdxl, flux1 captions | none; id identity not checked by Kura | source only |
| T5 SDPA (FLUX.1) | flux1 LoRA | none; a flux1 run is not numerically identical to one on the old image | FLUX.1 diagnostic run (runs, not identity) |
| safetensors 0.4.5 -> 0.8.0 key order | flux1 LoRA from a checkpoint with the `model.diffusion_model.` prefix and other components | patch 0003 strips the prefix from every key in `load_flow_model` (FLUX.1 and Chroma branches); remove it when upstream does the same | FLUX.1 smoke failed without 0003, passed with it |
| Anima Qwen3 under transformers 5.17 | anima LoRA, LLLite | none | Anima LoRA smoke; LLLite not re-run |
| SD2 tokenizer source | sd15 with `--v2` via `extra_args` only | none | source |
| Lumina tokenizer source | unsupported (no Lumina selector) | none | source |
| OFTv2 / BOFT | escape hatch only (`backend.config.command`); Kura's LoRA output check would reject OFT keys | none now; a selector needs its own design | source |
| `gen_img_diffusers.py` removed | unused | none | grep |
| sampler fixes, `lms` scipy check | not a Kura field | none | source |
| opencv stub, ARM64 markers | Linux x86_64 keeps opencv | none | requirements.txt |
| training states saved by the old image | same-executor Resume refuses another image | none in code; resuming an old state needs `images.sd-scripts` set to the old digest | existing refusal |

## Which run covers what

- Conformance (sd15, `train_network.py`): fresh, split, and Resume runs on one
  and three items, with gradient accumulation; covers patch 0002, the
  Accelerate 1.15 state files, and the CLIP text-model wrapper on a trained
  text encoder.
- SDXL LoRA smoke (`sdxl_train_network.py`): SDXL loaders and the tokenizer
  wrapper under transformers 5.17.
- FLUX.1 LoRA smoke (`flux_train_network.py`) with the harness's
  `Comfy-Org/flux1-dev` `flux1-dev-fp8.safetensors`: FAILED on the first
  v0.12.0 image (93a887f4), passed on the pinned image with patch 0003.
  That file holds the DiT under `model.diffusion_model.` next to text encoders
  and a VAE. Upstream `flux_utils.load_flow_model` strips the prefix in a loop
  that stops at the first key without it; safetensors 0.8.0 returns
  `load_file` keys in file order (text encoders first in this file), where
  0.4.5 returned them sorted, so no DiT key is renamed and the model stays on
  the meta device. A file with a bare DiT (BFL format) is not affected.
  `analyze_checkpoint_state` reads keys through `safe_open(...).keys()`,
  which safetensors 0.8.0 still returns sorted (checked on this file in the
  image), so only the loop in `load_flow_model` needed the patch. Upstream
  `dev` (0e9cdee) has the same loop.
- FLUX.1 diagnostic run with a prefix-free DiT (the FLUX.1 Kontext dev file):
  passed one step, so the T5-XXL tokenizer and encoder work under
  transformers 5.17. protobuf is in the image through tensorboard.
- Anima LoRA smoke (`anima_train_network.py`): Qwen3 under transformers 5.17 and
  the Anima publish conversion.
- Chroma: Kura has no Chroma selector, so 0003's Chroma hunk is not
  exercised by any run; it is the same loop and the same fix.
- Not re-run on the new image: Anima LLLite. It shares the Anima loaders but has
  its own entrypoint, dataset shape, and output, so it is untested on the new
  image.
- No RunPod run on the new image yet.

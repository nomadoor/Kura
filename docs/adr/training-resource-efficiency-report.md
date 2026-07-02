# Training resource efficiency report

Status: investigation report, release-blocking concerns.

Date: 2026-07-02

## Summary

Kura's current training defaults and smoke recipes are too resource-rich. The
problem is not limited to FLUX.1 Kontext. Across backends, Kura exposes useful
low-memory knobs, but it does not yet treat model artifact size, runtime
precision, VRAM class, and download cost as one planned training decision.

This is a release blocker for the current branch. Kura's value is that an AI
agent can make reasonable training trade-offs for the user's environment,
instead of making the user hand-pick every low-memory flag. The current behavior
does not meet that bar: it can silently select or encourage large full-precision
artifacts, then only discover the cost after launch.

## Owner concern to carry forward

FLUX Kontext に限らず、全てのモデルにおいて、現在、富豪的すぎる。
一般的に fp8 でなんら問題ないし、最近は int8 を使用するようになっている。
low vram モードに関しても、あまりに遅くなるなら許可がいるが、大したことないならユーザーの環境しだいでは適応したほうがよい。この当たりをあまりユーザーに選択させず、AI エージェントが最適化するのが Kura の得意であるわけで、ここは丁寧に設計しよう。

## Observed incident

A local Docker Musubi FLUX.1 Kontext real smoke was launched after a clean disk
doctor and an approved run plan. The run downloaded the full BFL Kontext DiT and
AE, then began downloading the FP16 T5 text encoder. The user stopped the run
before validation or training because the download size was unexpectedly large.

After the stop, the generated Docker container was removed, root-owned cache
files were repaired with `kura fix-permissions`, and `kura cleanup cache --yes`
deleted the downloaded Hugging Face/model cache. A final disk doctor reported no
root-owned Kura files and no warnings.

The incident exposed a planning problem:

- The run plan did not show the estimated model download size.
- The smoke recipe used `t5xxl_fp16.safetensors` and did not enable `fp8_t5`.
- The plan displayed `fp8_base`, `fp8_scaled`, gradient checkpointing, and block
  swap, but those runtime flags did not reduce the already-selected artifact
  download size.

## Current facts

### Musubi model handling

Kura's Musubi backend requires either explicit `model_paths`,
`model_downloads`, or a known Kura model bundle. This is a sound architecture for
reproducibility: Kura can record provenance in `model-bundle.lock.yaml`, estimate
disk writes, reuse a stable workspace cache, and keep container execution
file-first.

However, Kura does not yet distinguish between:

- model family, such as FLUX.1 Kontext or FLUX.2 klein
- artifact profile, such as full, fp8, int8, quantized text encoder, or local
  existing paths
- runtime profile, such as default, 24GB, 12GB, low-vram, or remote-large
- output precision, such as saving LoRA as bf16

As a result, a run can look optimized at runtime while still selecting wasteful
artifacts.

Upstream Musubi FLUX Kontext documentation expects explicit DiT, AE, T5, and
CLIP model paths. It also documents memory-saving options such as fp8 for the
DiT, fp8 T5, fp8 scaled mode, gradient checkpointing, and checkpointing CPU
offload:

- https://github.com/kohya-ss/musubi-tuner/blob/main/docs/flux_kontext.md

Therefore, Kura should keep explicit model preparation for Musubi, but it must
make the artifact choice visible and resource-aware.

### Musubi smoke recipes

`scripts/musubi_real_smoke.py` is a developer acceptance test, but it is still a
high-risk source of copied recipes and release confidence. The current specs mix
capacity-conscious choices with full artifacts. The FLUX.1 Kontext spec is the
clearest problem:

- downloads `black-forest-labs/FLUX.1-Kontext-dev/flux1-kontext-dev.safetensors`
- downloads `black-forest-labs/FLUX.1-Kontext-dev/ae.safetensors`
- downloads `comfyanonymous/flux_text_encoders/t5xxl_fp16.safetensors`
- sets `fp8_base` and `fp8_scaled`
- does not set `fp8_t5`

The smoke may prove that the adapter works, but it does not prove the practical
default path a local user should run.

### AI-Toolkit backend defaults

`src/kura/backends/ai_toolkit.py` currently defaults to:

- `train.gradient_checkpointing: false`
- `model.quantize: false`
- `model.quantize_te: false`
- `model.low_vram: false`

This conflicts with the spirit of upstream AI-Toolkit examples for large FLUX
and Kontext training, which use quantization and gradient checkpointing as
normal consumer-GPU settings, with low-vram available when needed:

- https://raw.githubusercontent.com/ostris/ai-toolkit/main/config/examples/train_lora_flux_24gb.yaml
- https://raw.githubusercontent.com/ostris/ai-toolkit/main/config/examples/train_lora_flux_kontext_24gb.yaml

Saving the final LoRA as bf16 is fine. That is separate from how aggressively
Kura should optimize model loading and training memory.

### Run plan visibility

`src/kura/run_commands/plan.py` already has `_estimate_musubi_download_bytes()`
and local Docker launch preflight accounts for estimated Hugging Face writes.
But `kura run plan` does not include or render that estimate.

This means the command the user must inspect before launch can omit the most
important fact: how many GiB will be downloaded, and which files are responsible.

## Problems to fix before release

1. `kura run plan` must show model download estimates.

   Show total bytes, per-file estimates, unknown-size files, and whether the
   estimate is based on explicit downloads, known bundles, or existing paths.
   A model download above a configured threshold should require explicit
   approval, not just a generic disk preflight warning.

2. Kura needs a resource/artifact profile, not just scattered flags.

   The schema should express intent such as `auto`, `local-12gb`, `local-24gb`,
   `remote-24gb`, `remote-large`, or equivalent. Kura can then choose artifact
   variants and runtime flags consistently. This should include DiT precision,
   text encoder precision, quantization, block swap/offload, gradient
   checkpointing, batch, accumulation, and resolution.

3. Full-precision artifacts must not be the silent default for large models.

   If fp8/int8 artifacts are normal and quality-acceptable for training, Kura
   should prefer them by default for local runs. If a low-vram setting has a
   serious speed or quality trade-off, Kura should surface that trade-off and
   require approval before applying it.

4. Musubi real smoke must separate adapter proof from user-practical proof.

   Heavy acceptance recipes should be labeled as such. Local default smoke
   recipes should use the smallest practical artifact profile and the same
   memory-saving path Kura would recommend to a user.

5. AI-Toolkit defaults need an audit.

   Kura should not override upstream consumer-GPU practice with heavier defaults.
   At minimum, quantization and gradient checkpointing should be considered
   default-on for large image/video models unless the user or a profile says
   otherwise.

6. Plan output needs VRAM-risk warnings.

   Disk fit is not enough. The plan should warn when a known-heavy architecture
   lacks expected memory-saving settings, or when selected model artifacts are
   unlikely to fit the configured/local GPU class.

7. Documentation must stop presenting "passed smoke" as enough capacity proof.

   The Musubi smoke docs should record artifact profile, expected VRAM class,
   download size, and whether the result represents a practical local default or
   a heavy developer acceptance test.

## Design direction

Kura should treat training configuration and compute selection as one plan:

- dataset size and resolution
- effective batch and accumulation
- base model family and artifact variant
- runtime precision and quantization
- text encoder strategy
- optimizer and checkpointing
- block swap/offload strategy
- local disk and remote cost
- expected VRAM class

The user should still be able to override every important trade-off. But when
the user has not asked for a special configuration, Kura should choose a
reasonable efficient profile from the environment and show what it chose.

## Immediate next implementation candidates

1. Add model download estimates to `kura run plan`.
2. Add warning or approval gates for large downloads and missing low-memory
   flags on known-heavy architectures.
3. Change FLUX.1 Kontext smoke to an efficient profile, including fp8 T5 where
   compatible, or clearly mark the current recipe as heavy.
4. Audit AI-Toolkit generated defaults against upstream examples.
5. Draft a schema change for resource/artifact profiles before adding more
   model bundles.


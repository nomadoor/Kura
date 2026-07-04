# Training resource efficiency report

Status: investigation report, scoped release blockers.

Date: 2026-07-02

## Summary

Kura's current training defaults and smoke recipes are too resource-rich. The
problem is not limited to FLUX.1 Kontext. Across backends, Kura exposes useful
low-memory knobs, but it does not yet treat model artifact size, runtime
precision, VRAM class, and download cost as one planned training decision.

Parts of this are release blockers for the current branch. Kura's value is that
an AI agent can make reasonable training trade-offs for the user's environment,
instead of making the user hand-pick every low-memory flag. The current behavior
does not meet that bar: it can silently select or encourage large
full-precision artifacts, then only discover the cost after launch.

The immediate release blockers are the places where Kura can still surprise the
user: missing plan visibility, missing approval gates for large downloads, and
heavy smoke/default recipes that look like practical local paths. The broader
resource profile schema should be designed carefully in a separate ADR rather
than rushed as part of the emergency fix.

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
- runtime strategy, such as gradient checkpointing, quantization, block swap,
  CPU offload, or text encoder precision
- hardware fit, such as detected or requested VRAM class
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
high-risk source of copied recipes and release confidence. The specs must not
mix capacity-conscious choices with full artifacts in a way that looks like a
practical default. The FLUX.1 Kontext incident recipe was the clearest problem:

- downloads `black-forest-labs/FLUX.1-Kontext-dev/flux1-kontext-dev.safetensors`
- downloads `black-forest-labs/FLUX.1-Kontext-dev/ae.safetensors`
- downloads `comfyanonymous/flux_text_encoders/t5xxl_fp16.safetensors`
- sets `fp8_base` and `fp8_scaled`
- does not set `fp8_t5`

The smoke may prove that the adapter works, but it does not prove the practical
default path a local user should run.

### AI-Toolkit backend defaults

`src/kura/backends/ai_toolkit.py` previously defaulted to:

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

The owner decision supersedes automatic default changes: code must provide facts
and guard irreversible accidents, while the agent/skill decides trade-offs.
AI-Toolkit therefore keeps gradient checkpointing, model quantization, text
encoder quantization, and `low_vram` opt-in through explicit backend overrides.
The audit conclusion is not "turn these on in code"; it is "surface enough facts
in `kura run plan` for the skill to propose them when the user's hardware and
run intent justify the trade-off."

### Run plan visibility

`src/kura/run_commands/plan.py` already has `_estimate_musubi_download_bytes()`
and local Docker launch preflight accounts for estimated Hugging Face writes.
But `kura run plan` does not include or render that estimate.

This means the command the user must inspect before launch can omit the most
important fact: how many GiB will be downloaded, and which files are responsible.

`kura run plan` should also include a factual `Resources` section for the agent
skill: detected local GPU/VRAM when available, executor and requested RunPod GPU
types, model architecture/base/artifact filenames, and memory-related flags.
Unset values should be shown as `(not set)`. This section must not recommend,
warn, or automatically change settings; it is an instrument panel for the skill.

## Fix order

The fixes should not be treated as one large schema change. The right order is:

1. Visibility.

   Add model download estimates to `kura run plan`. This is behavior-preserving
   and would have prevented the incident by showing the large download before
   launch.

2. Gates.

   Add approval gates or hard warnings for large downloads, unknown-size model
   files, and known-heavy configurations that lack expected low-memory settings.

3. Defaults and smoke recipes.

   Fix clearly wrong default recipes, especially FLUX.1 Kontext smoke using
   FP16 T5 without `fp8_t5`. Audit AI-Toolkit defaults, but document any
   behavior change because recompiling an existing `run.yaml` may produce a
   different native config.

4. Schema.

   Design resource/artifact profiles as a separate ADR. This should happen
   before adding more known model bundles, because every new bundle otherwise
   hardcodes another artifact choice.

## Release blockers

The following items block the current release:

1. `kura run plan` must show model download estimates.

   Show total bytes, per-file estimates, unknown-size files, and whether the
   estimate is based on explicit downloads, known bundles, existing paths, or
   cache hits. The estimate should subtract files already present in the
   workspace cache when Kura can prove they are already available. The plan
   should also expose factual resource inputs without recommendations so the
   skill can decide what to propose.

2. Large downloads need an approval gate.

   A model download above a configured threshold should require explicit
   approval, not just a generic disk preflight warning. Unknown-size files must
   be shown honestly as unknown rather than omitted. Disk fit is not enough:
   the gate should also cover known-heavy configurations that lack expected
   low-memory settings, or selected model artifacts that are unlikely to fit the
   configured/local GPU class.

3. Musubi real smoke must separate adapter proof from user-practical proof.

   Heavy acceptance recipes should be labeled as such. Local default smoke
   recipes should use the smallest practical artifact profile and the same
   memory-saving path Kura would recommend to a user. At minimum, FLUX.1 Kontext
   smoke must stop presenting FP16 T5 without `fp8_t5` as the normal local path.
   The Musubi smoke docs should record artifact profile, expected VRAM class,
   download size, and whether each result represents a practical local default
   or a heavy developer acceptance test.

4. AI-Toolkit defaults need an audit.

   Kura should not override upstream consumer-GPU practice with heavier defaults,
   but code must not bake in model-family trade-off decisions either. The audit
   result is to keep defaults stable and opt-in while making plan facts complete
   enough for the skill to propose quantization, checkpointing, or low-vram
   settings with the required user approval.

## Important follow-up, but not an emergency blocker

Kura needs a resource/artifact profile, not just scattered flags.

This follow-up belongs in the agent skill, not in an in-code automatic resolver.
Code must expose the facts and freeze explicit user/agent decisions; it must not
silently choose quality, precision, batch, rank, or resolution trade-offs. If a
future schema is needed for recording accepted decisions, it should not be a
single enum such as `local-12gb`, `local-24gb`, or `remote-large`. That mixes
location and VRAM class into one name and creates combination pressure. The
recorded axes may include:

- `vram_class` or `auto`
- `artifact_precision`
- `speed_tolerance`
- explicit override fields for model roles when needed

Any accepted `auto`-like decision must resolve before compile and freeze concrete
values into `resolved/manifest.lock.yaml` and backend lock files. This follows
Kura's file-first rule: the same locked run must not change meaning just because
it is replayed on another machine.

Status update (2026-07-02): this decision procedure is now implemented as the
`training-parameter-planning` skill. Knowledge lives beside it as small
per-architecture cards (`knowledge/<architecture>.md`) plus
`knowledge/user-preferences.md`; file location expresses precedence
(preferences outrank baselines), and every value carries a `source:` line
(owner / run id / upstream / agent) so evidence stays orthogonal to
precedence. Run `notes.md` remains the primary evidence record; cards cite
run ids instead of duplicating them. The single mandatory approval is the
`kura run plan` review before launch.

The skill implements the existing safety principle as a decision ladder:
first propose adjustments that do not change meaning; then propose speed-only
trade-offs and report them clearly; finally, require explicit user approval for
quality-affecting changes such as resolution, batch size, rank, or precision.
Block swap, CPU offload, gradient checkpointing, quantization, and low-vram
settings are recommendations the skill can make from plan facts, not defaults
the backend should silently apply.

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
2. Add approval gates for large downloads and missing low-memory
   flags on known-heavy architectures.
3. Keep FLUX.1 Kontext smoke on an efficient profile, including fp8 T5 where
   compatible, or clearly mark any heavier recipe as heavy.
4. Audit AI-Toolkit generated defaults against upstream examples.
5. Draft a separate resource/artifact profile ADR before adding more model
   bundles.

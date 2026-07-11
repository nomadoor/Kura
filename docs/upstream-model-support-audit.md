# Upstream trainer and model-support audit

Snapshot date: 2026-07-11.

This audit separates upstream capability from what Kura can safely claim. A
trainer containing a model class is not, by itself, proof that Kura's generated
configuration, dataset projection, runtime image, and recovery path work for
that model.

## Audited upstream revisions

| Backend | Audited upstream | Release state | Kura image before this audit |
| --- | --- | --- | --- |
| AI-Toolkit | source head `96a3a0611176d2d4f4f319ad5840b4effa191b70`; official Docker `0.10.22` contains `a4bbe167ce03521bf9052d2349f01b2997d67ac7` | no GitHub releases; versioned official Docker images | local image contained `dba092fc15b915c33d1c2221815906a9af4807c3` (`0.10.16` generation); RunPod followed mutable `latest` |
| Musubi Tuner | `v0.3.4`, commit `30c658c4f4b0bf05038b3346eff9670259b10fc7` | latest stable GitHub release | local and published `nomadoor/kura-musubi-tuner:dev` already contain this exact commit |

Primary sources:

- [AI-Toolkit repository](https://github.com/ostris/ai-toolkit)
- [AI-Toolkit model choices](https://github.com/ostris/ai-toolkit/blob/main/ui/src/app/jobs/new/options.ts)
- [AI-Toolkit official Docker tags](https://hub.docker.com/r/ostris/aitoolkit/tags)
- [Musubi Tuner repository and support overview](https://github.com/kohya-ss/musubi-tuner)
- [Musubi Tuner v0.3.4](https://github.com/kohya-ss/musubi-tuner/releases/tag/v0.3.4)

## Confidence vocabulary

Kura uses these levels when describing support:

1. **Upstream listed**: the audited trainer advertises the architecture.
2. **Kura expressible**: Kura can freeze a native configuration or command for
   it. This does not establish that Kura's defaults are appropriate.
3. **Image smoke**: the pinned image contains the expected entrypoints and they
   start successfully.
4. **Real smoke**: a real model and dataset completed at least one optimizer
   step through Kura's normal executor.
5. **Operationally verified**: output recovery and cleanup were also observed
   on the named executor and hardware class.

Only the last two levels justify an unqualified user-facing claim that a model
works. Quality still requires render and human evaluation; a one-step smoke
does not establish useful LoRA quality.

## AI-Toolkit

The audited AI-Toolkit UI exposes the following families. Some rows combine
variants that use the same model class or workflow shape.

| Family | Upstream choices in the audited image | Kura status |
| --- | --- | --- |
| Stable Diffusion | SD 1.5, SDXL | SDXL operationally verified in local Docker and RunPod; SD 1.5 only expressible |
| FLUX / Flex / Chroma | FLUX.1, FLUX.1 Kontext, Flex.1, Flex.2, Chroma, Zeta Chroma | expressible through `model_arch` and native overrides; not model-by-model verified |
| Wan video | Wan 2.1 T2V/I2V, Wan 2.2 T2V/I2V/TI2V | upstream listed; Kura's simple image-folder projection is not a sufficient video contract |
| Qwen Image | Qwen-Image, 2512, Edit, Edit-2509, Edit-2511 | text-to-image form is expressible; edit/control dataset forms need explicit native dataset configuration |
| HiDream | HiDream I1, E1, O1 | expressible; not AI-Toolkit real-smoked through Kura |
| FLUX.2 / Krea | FLUX.2 dev, klein 4B/9B, Krea 2 Raw/Turbo and edit variants | upstream listed and expressible; not yet verified through Kura's AI-Toolkit path |
| Z-Image | Turbo adapter, Base, De-Turbo, L2P | upstream listed; variants require different companion artifacts and config, so no family-wide support claim |
| Instruction and image models | OmniGen2, ERNIE-Image, Nucleus-Image, Ideogram 4, PRXPixel, Boogu Image/Edit | upstream listed; model-specific dataset and config review required |
| Video | LTX-2, LTX-2.3 | upstream listed; Kura has no first-class AI-Toolkit video dataset projection |
| Audio | ACE-Step 1.5 and XL | upstream listed; outside Kura's current train-run dataset contract |

AI-Toolkit deliberately owns repository download and companion-model
resolution. Kura should not duplicate those loaders. Kura's current compiler
is a generic native-config projection with an override escape hatch, not a
registry proving every upstream model. The default generated recipe is only
known to be sound for the tested SDXL path. Large, video, audio, edit, control,
and pixel-space models require model-specific planning and acceptance before
being promoted to verified status.

## Musubi Tuner

The audited `v0.3.4` release contains architecture documentation and the three
cache/train entrypoint families expected by every current Kura built-in
adapter:

| Architecture | Upstream v0.3.4 | Kura built-in adapter | Current evidence |
| --- | --- | --- | --- |
| FLUX.2 | yes | yes | real smoke recorded |
| Wan 2.1/2.2 | yes | yes | Wan 2.1 1.3B operationally verified local and RunPod |
| Krea 2 | yes | yes | historical real smoke recorded; broader Krea validation remains separate |
| Qwen-Image | yes | yes | real smoke recorded on RunPod A40 |
| Z-Image | yes | yes | real smoke recorded |
| FLUX.1 Kontext | yes | yes | real smoke recorded with paired/control data |
| Ideogram 4 | yes | yes | real smoke recorded |
| HiDream-O1-Image | yes | yes | real smoke recorded |
| HunyuanVideo | yes | yes | real smoke recorded |
| HunyuanVideo 1.5 | yes | yes | real smoke recorded |
| FramePack | yes | yes | real smoke recorded |
| Kandinsky 5 | yes | yes | Lite T2V real smoke recorded; Pro remains capacity-dependent |

No top-level upstream `v0.3.4` architecture is missing from Kura's built-in
adapter list. That statement does not imply that every execution variant was
already covered. Variant-level review found these distinct paths:

| Architecture | Variant paths | Kura evidence after audit |
| --- | --- | --- |
| FLUX.2 | dev; klein/base 4B; klein/base 9B; optional reference images | common scripts supported; klein representative real-smoked; dev contract corrected for Mistral 3 and official AE, compile-tested |
| Wan | 2.1 T2V/I2V/Fun Control; 2.2 low/high-noise T2V/I2V; Single Frame | 2.1 T2V real-smoked; I2V CLIP requirement, dual-DiT, and Single Frame cache/train paths compile-tested after audit fixes |
| Qwen-Image | original; Edit; Edit-2509; Edit-2511; Layered | original real-smoked; every `model_version` reaches latent cache, text cache, and training in compile tests; dataset-specific control/multiple-target data remains user intent |
| HunyuanVideo 1.5 | T2V; I2V | T2V real-smoked; I2V image-encoder cache/train path compile-tested |
| FramePack | normal; F1; Single Frame | normal real-smoked; Single Frame cache/train path added and compile-tested; F1 already represented by its distinct flag |
| HiDream-O1 | T2I; I2I control/reference | T2I real-smoked; I2I task, control dataset, and optional conv-network args compile-tested |
| Kandinsky 5 | Lite/Pro T2V; Pro I2V | Lite T2V real-smoked; Pro/I2V task selection compile-tested but capacity-dependent |
| Krea 2, Z-Image, FLUX.1 Kontext, Ideogram 4, HunyuanVideo | documented LoRA path does not change top-level cache/train scripts | representative real smoke recorded for each |

The audit unit is an execution contract, not every checkpoint. A new real smoke
is required when scripts, mandatory model roles, dataset shape, cache behavior,
or output behavior change. Merely substituting weights within the same contract
does not require a complete matrix rerun.

## Image findings and policy

Before this audit, three different meanings of freshness were mixed:

- AI-Toolkit local builds inherited mutable `latest`.
- AI-Toolkit RunPod defaults also named `latest`, independently of the local
  image that had been real-smoked.
- Musubi local builds cloned `main`, although the published image happened to
  contain the latest stable release.
- `kura image build ai-toolkit --ref` passed an `AI_TOOLKIT_REF` build argument
  that the Dockerfile did not consume.

The corrected policy is:

- pin AI-Toolkit local builds to the audited official image digest and use its
  version tag for the RunPod default;
- make the AI-Toolkit `--ref` override mean an explicit upstream Docker image
  reference;
- pin Musubi builds to the audited stable tag by default while retaining an
  explicit git-ref override for development;
- record the actual upstream image or git revision in image inspection output;
- update a pin only with image smoke, focused adapter/config checks, and at
  least one representative real smoke before calling it verified.

## Follow-up order

Validation completed during this audit:

- built the pinned AI-Toolkit 0.10.22 image as
  `nomadoor/kura-ai-toolkit:dev` and confirmed its embedded upstream commit is
  `a4bbe167ce03521bf9052d2349f01b2997d67ac7`;
- started the image's `run.py --help` path successfully;
- completed a real SDXL one-step local Docker run on an RTX 4070 Ti through
  `kura run execute`, including cache reuse, LoRA output, optimizer output,
  realization records, and host-user file ownership;
- completed a real Wan 2.1 Single Frame 14B one-step local Docker run on an
  RTX 4070 Ti after an empty-cache 30.5 GiB acquisition; AI-Toolkit and
  Kura-managed Musubi downloads now share
  `HF_HUB_CACHE=/workspace/cache/huggingface/hub`, while `cache/models` only
  contains stable Kura links;
- completed a real AI-Toolkit 0.10.22 SDXL one-step RunPod run on an RTX A5000;
  the backend acquired 14.2 GB through the shared cache environment, produced
  and recovered the LoRA/config/optimizer artifacts, and the disposable Pod
  was stopped;
- completed a real Musubi Wan 2.1 1.3B one-step RunPod run on an RTX A6000;
  Kura preflighted and acquired 13.5 GiB through the same hub cache, validated
  both LoRA outputs, recovered them locally, and stopped the disposable Pod;
- confirmed after both remote runs that RunPod had zero Pods and zero Network
  Volumes, with no cgroup OOM kills in either run;
- ran `kura doctor musubi` against the v0.3.4 image: all 36 expected adapter
  scripts existed and completed their help smoke;
- passed Kura's release gate with 305 tests and all mechanical checks after the
  variant adapter and shared Hugging Face cache corrections.

Remaining follow-up order:

1. Publish the pinned images only as part of the normal reviewed release flow;
   a successful local build does not authorize a registry push.
2. Keep model requirements run-scoped in
   `resolved/model-requirements.lock.yaml`; do not create a second global model
   registry or duplicate upstream model lists in production code.
3. Real-smoke only the newly added Musubi execution paths whose cache or train
   contract differs from the existing representative evidence.
4. Return to Krea 2 only after deciding whether its first supported path should
   be AI-Toolkit, Musubi, or both, with separate evidence for each backend.

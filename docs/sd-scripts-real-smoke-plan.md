# sd-scripts Tier 1 real-smoke plan

The initial backend is not complete until every Tier 1 row has one real
optimizer step. Use `scripts/sd_scripts_real_smoke.py`; it creates a one-item
synthetic dataset and runs only through Kura.

| Selector | Native contract | Required model roles | Additional acceptance |
| --- | --- | --- | --- |
| `sd15` | `train_network.py` + `networks.lora` | `base` | valid ComfyUI LoRA |
| `sdxl` | `sdxl_train_network.py` + `networks.lora` | `base` | valid ComfyUI LoRA |
| `flux1` | `flux_train_network.py` + `networks.lora_flux` | `dit`, `clip_l`, `t5xxl`, `ae` | valid ComfyUI LoRA |
| `anima-lora` | `anima_train_network.py` + conversion | `dit`, `qwen3`, `vae` | managed-ComfyUI LoRA render |
| `anima-lllite` | `anima_train_control_net_lllite.py` | `dit`, `qwen3`, `vae` | v2 metadata and managed-ComfyUI model-patch render |

The local 12 GiB smoke profiles are intentionally bounded and are not quality
presets. SDXL uses 512px, rank 8/alpha 4, U-Net-only training, and latent/text
cache. FLUX uses 512px, rank 16/alpha 1, `fp8_base`, block swap 16, U-Net-only
training, `flux_shift`, guidance 1.0, and raw prediction. Anima LoRA follows the
upstream rank 8/alpha 1/LR 1e-4/sigmoid example with the image-only 2D VAE.
Anima LLLite follows the upstream command example at LR 5e-5, shift 3.0, and
the v2 32/64/1/64/self-attn-Q capacity settings, with SDPA attention selected
explicitly because the pinned upstream training path rejects an unset attention
mode at forward time. Its separate user-facing
lineart baseline is LR 1e-4 at 1024px; it is not inferred from this one-step
smoke.

The models file maps each role either to a container-visible path or an
immutable Hugging Face download mapping:

```yaml
dit:
  repo: circlestone-labs/Anima
  filename: split_files/diffusion_models/anima-base-v1.0.safetensors
  revision: f7382c4bf9d7ffe4ceea593a0adbb470c56dd79b
qwen3:
  repo: circlestone-labs/Anima
  filename: split_files/text_encoders/qwen_3_06b_base.safetensors
  revision: f7382c4bf9d7ffe4ceea593a0adbb470c56dd79b
vae:
  repo: circlestone-labs/Anima
  filename: split_files/vae/qwen_image_vae.safetensors
  revision: f7382c4bf9d7ffe4ceea593a0adbb470c56dd79b
```

Prepare and display the exact plan without launching:

```sh
uv run python scripts/sd_scripts_real_smoke.py anima-lora \
  --models docs/sd-scripts-anima-models.yaml --executor docker --gpu "NVIDIA RTX 4070 Ti"
```

After the user approves the displayed plan, reuse the printed run ID:

```sh
uv run python scripts/sd_scripts_real_smoke.py anima-lora \
  --run-id <run-id> --launch --yes
```

The harness reruns `doctor disk`, `doctor sd-scripts`, compile, and plan before
launch. Validation requires completion, exactly one recorded step, the expected
entrypoint, a safetensors output, unchanged shared-dataset identities, and
the terminal Docker-container state. LLLite additionally reads the real
artifact header and requires `lllite.version=2`.

All five rows use local Docker; live RunPod execution is outside this milestone.
After the Anima runs, use the authored fixtures in
`examples/sd-scripts-anima-smoke/` against
Kura's managed ComfyUI image. Do not mutate a user's separately managed
ComfyUI. The Anima evidence must identify the loaded base-v1.0 DiT and prove
that the produced LoRA or `anima-preview/control-net-lllite`-labelled patch was
actually applied to that same base during the render. Add evidence to
`docs/backend-smoke-evidence.yaml` only after the run,
artifact, adapter source, image identity, terminal container state, and render result
have actually been observed.

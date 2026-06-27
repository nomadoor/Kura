from pathlib import Path


TARGET = Path("/opt/musubi-tuner/src/musubi_tuner/flux_2/flux2_utils.py")


OLD = """    logger.info(f"Loading state dict from {ckpt_path}")
    sd = load_split_weights(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)
    info = ae.load_state_dict(sd, strict=True, assign=True)
"""


NEW = """    logger.info(f"Loading state dict from {ckpt_path}")
    sd = load_split_weights(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)
    if any(".down_blocks." in key or ".up_blocks." in key or key.endswith("conv_norm_out.weight") for key in sd):
        logger.info("Converting Diffusers-layout Flux2 VAE state dict")
        from musubi_tuner.ideogram4.ideogram4_autoencoder import convert_diffusers_state_dict

        sd = convert_diffusers_state_dict(sd)
    info = ae.load_state_dict(sd, strict=True, assign=True)
"""


def main() -> None:
    text = TARGET.read_text()
    if NEW in text:
        return
    if OLD not in text:
        raise SystemExit(f"expected load_ae block not found in {TARGET}")
    TARGET.write_text(text.replace(OLD, NEW))


if __name__ == "__main__":
    main()

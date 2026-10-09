"""Container images Kura runs, pinned by digest, and how a workspace overrides them.

Users pull these images and never build them. Each digest is the one the
backend's smoke evidence ran with; changing one is a runtime update that needs
its own evidence. A workspace may name another image with `images.<name>`.
Building images is a development task done from an editable Kura checkout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PINNED_IMAGES: dict[str, str] = {
    "ai-toolkit": "nomadoor/kura-ai-toolkit@sha256:9aa6861b0f54f24f0ebad07b6018b431e8c2403d27eed9233595951b466dbc3a",
    "musubi-tuner": "nomadoor/kura-musubi-tuner@sha256:21294bb1fcd9d7253181a8eb386d8a924837c136a3dcbb03b55a3aeadd8a18a1",
    "sd-scripts": "nomadoor/kura-sd-scripts@sha256:66cb9a2fe9b1841db5b9fcd4de27528efb6f9a7223726b8b78a75f544d07da02",
    "comfyui": "nomadoor/kura-comfyui@sha256:280fefddccc40ea06d9b96f8106124d7297bdd4d0ed0d3bf7b28a902a4b1d097",
}

# The build argument that selects each image's upstream source, and the value
# the pinned image was built with. AI-Toolkit builds on the upstream image.
BUILD_SOURCES: dict[str, tuple[str, str]] = {
    "ai-toolkit": ("AI_TOOLKIT_IMAGE", "ostris/aitoolkit:0.13.18@sha256:9bc99d51efc5b6c38a951b3bf8547bda0f9db58abeb75573548d449f82b34bcc"),
    "musubi-tuner": ("MUSUBI_TUNER_REF", "v0.3.5"),
    "sd-scripts": ("SD_SCRIPTS_REF", "37a1cbbc5725ed2a3575506e7bd2001c9908ac92"),
    "comfyui": ("COMFYUI_REF", "0f42ba51463174fb255f2c4605ae0e0b441fe6d7"),
}

# The CUDA version each image Kura has pinned was built with, by digest, so a
# run compiled against an earlier image keeps that image's requirement. Add the
# new digest here whenever an image is re-pinned.
IMAGE_CUDA_VERSIONS: dict[str, str] = {
    "sha256:9aa6861b0f54f24f0ebad07b6018b431e8c2403d27eed9233595951b466dbc3a": "13.0",
    "sha256:de5d31f26dde97a45457b4fed243f1c5778d3e398f6bbe85b4ff4d74d1a5c211": "12.8",
    "sha256:a3b2cee58a00807c1a1f897f086869821d8dad20b1a090baf96c157772cb8901": "12.8",
    "sha256:4607399fc1b9bcde0ea416ba43eb28069eafb1b80a1914e9906762dba8d24f5a": "12.8",
    "sha256:21294bb1fcd9d7253181a8eb386d8a924837c136a3dcbb03b55a3aeadd8a18a1": "13.0",
    "sha256:66cb9a2fe9b1841db5b9fcd4de27528efb6f9a7223726b8b78a75f544d07da02": "12.8",
    "sha256:280fefddccc40ea06d9b96f8106124d7297bdd4d0ed0d3bf7b28a902a4b1d097": "13.0",
}

# The newest CUDA version seen on RunPod hosts (2026-10-07). A newer driver
# runs an older image, so an image whose version Kura does not know asks for
# this one: it finds fewer hosts, but none whose driver is older than Kura saw.
NEWEST_KNOWN_CUDA = "13.2"

# The tag a development build gets on the local Docker host.
DEVELOPMENT_TAG = "kura-{name}:dev"


def image_names() -> tuple[str, ...]:
    return tuple(PINNED_IMAGES)


def effective_image(config: dict[str, Any], name: str) -> dict[str, str]:
    """Return the image a run uses: the workspace override, else the pinned digest."""

    if name not in PINNED_IMAGES:
        raise ValueError(f"unknown Kura image {name!r}; expected one of: {', '.join(PINNED_IMAGES)}")
    overrides = config.get("images") if isinstance(config.get("images"), dict) else {}
    override = overrides.get(name)
    if override is None:
        return {"name": name, "reference": PINNED_IMAGES[name], "origin": "pinned"}
    if not isinstance(override, str) or not override.strip():
        raise ValueError(f"images.{name} must name an image; delete the line to use the pinned image")
    return {"name": name, "reference": override.strip(), "origin": "override"}


def image_cuda_version(reference: str) -> str | None:
    """The CUDA version an image was built with, when Kura recorded it."""

    _, separator, digest = reference.partition("@")
    return IMAGE_CUDA_VERSIONS.get(digest) if separator else None


def runpod_min_cuda_version(reference: str) -> str:
    """The oldest host CUDA version that runs this image; the newest known when Kura does not know it."""

    return image_cuda_version(reference) or NEWEST_KNOWN_CUDA


def is_mutable(reference: str) -> bool:
    return "@sha256:" not in reference


def mutable_override_warning(image: dict[str, Any]) -> str | None:
    warnings = launch_image_warnings({**image, "frozen": False})
    return warnings[0] if warnings else None


def launch_image(config: dict[str, Any], name: str, env_lock: Any) -> dict[str, Any]:
    """The image a launch uses: the one frozen at compile, else the current one.

    `current` is what `workspace.yaml` selects now, so a plan can say when a
    compiled run would need recompiling to follow it.
    """

    current = effective_image(config, name)
    frozen = env_lock.get("selected_image") if isinstance(env_lock, dict) else None
    if isinstance(frozen, str) and frozen:
        origin = env_lock.get("image_origin")
        return {
            "name": name, "reference": frozen, "origin": origin if isinstance(origin, str) else "compiled",
            "frozen": True, "current": current["reference"],
        }
    return {**current, "frozen": False, "current": current["reference"]}


def launch_image_warnings(image: dict[str, Any]) -> list[str]:
    warnings = []
    if image["origin"] != "pinned" and is_mutable(image["reference"]):
        source = f"images.{image['name']}" if image["origin"] == "override" else "the image frozen at compile"
        warnings.append(f"{source} uses mutable tag {image['reference']}; pin a digest before relying on reproducible behavior")
    if image.get("frozen") and image.get("current") != image["reference"]:
        warnings.append(
            f"workspace.yaml now selects {image['current']}, but this run keeps {image['reference']} from compile; recompile to use the current image"
        )
    return warnings


def development_checkout() -> Path | None:
    """The checkout of an editable install, which alone may build images."""

    from kura.install_source import kura_provenance

    source = kura_provenance()["kura_source"]
    if source.get("kind") != "editable" or not isinstance(source.get("path"), str):
        return None
    checkout = Path(source["path"])
    return checkout if (checkout / "docker").is_dir() else None

"""Environment variables a user sets for Kura.

`kura secrets set` accepts these names, and a test keeps the list in step with
the variables Kura actually reads. Variables Kura sets for its own
containers or reads from the host system are listed separately.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UserVariable:
    name: str
    purpose: str
    aliases: tuple[str, ...] = ()


USER_VARIABLES: tuple[UserVariable, ...] = (
    UserVariable("RUNPOD_API_KEY", "Required for RunPod training and renders. Get it from the RunPod console."),
    UserVariable(
        "HF_TOKEN",
        "Only for models that need authenticated Hugging Face access (gated or private).",
        aliases=("HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"),
    ),
    UserVariable("KURA_NTFY_TOPIC", "Optional: ntfy topic for completion and failure notifications; pick a long, unguessable name."),
    UserVariable("KURA_NTFY_SERVER", "Optional: your own ntfy server URL; ntfy.sh is used when empty."),
    UserVariable("KURA_NTFY_TOKEN", "Optional: access token for a protected ntfy topic."),
    UserVariable("KURA_NTFY_PRIORITY", "Optional: ntfy priority from 1 to 5; 4 when empty."),
    UserVariable("KURA_NOTIFY", "Optional: notification channels to use by default, for example ntfy."),
    UserVariable("R2_ACCESS_KEY_ID", "Only for runpod.storage_mode object_staging: the object store access key."),
    UserVariable("R2_SECRET_ACCESS_KEY", "Only for runpod.storage_mode object_staging: the object store secret key."),
)

# Read by Kura but never set by a user: values Kura passes to its own remote
# jobs, and facts of the host system.
INTERNAL_VARIABLES = frozenset({
    "KURA_EXIT_CODE", "KURA_REMOTE_NOTIFY_NTFY", "KURA_RUNPOD_UPLOAD_CODE", "KURA_RUN_ID", "KURA_WORKSPACE",
    "TMPDIR", "WSL_DISTRO_NAME",
    # Set by the job runner for its followers, so their status writes carry its epoch.
    "KURA_RUNNER_EPOCH",
    # Development switches for the status projection's shadow mode.
    "KURA_STATUS_SHADOW", "KURA_STATUS_SHADOW_COLLECT",
    # The login name `kura doctor` uses when it checks whether logout ends the runner.
    "USER", "LOGNAME",
    # Cache locations `kura doctor disk` reports when the host sets them.
    "KURA_CACHE_DIR", "HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME", "XDG_CACHE_HOME",
})

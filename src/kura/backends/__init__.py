"""Backend adapters compile intent; they never execute commands."""

from __future__ import annotations

from kura.backends.ai_toolkit import command_ai_toolkit, compile_ai_toolkit
from kura.backends.musubi_command import command_musubi_tuner, compile_musubi_tuner
from kura.backends.musubi_models import MUSUBI_ADAPTER_SCRIPTS, _safetensors_validator_code, musubi_model_download_specs
from kura.backends.sd_scripts import command_sd_scripts, compile_sd_scripts, display_sd_scripts
from kura.backends.sd_scripts_models import sd_scripts_model_download_specs
from kura.backends.registry import BACKENDS, BackendAdapter, BackendSurface, backend_capabilities, backend_names, get_backend, validate_backend_config

__all__ = [
    "MUSUBI_ADAPTER_SCRIPTS",
    "_safetensors_validator_code",
    "command_ai_toolkit",
    "command_musubi_tuner",
    "command_sd_scripts",
    "compile_ai_toolkit",
    "compile_musubi_tuner",
    "compile_sd_scripts",
    "display_sd_scripts",
    "musubi_model_download_specs",
    "sd_scripts_model_download_specs",
    "BACKENDS",
    "BackendAdapter",
    "BackendSurface",
    "backend_capabilities",
    "backend_names",
    "get_backend",
    "validate_backend_config",
]

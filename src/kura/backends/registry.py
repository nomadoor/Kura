"""Single registry for backend adapter ownership and dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from kura.backends.ai_toolkit import command_ai_toolkit, compile_ai_toolkit, display_ai_toolkit, requirements_ai_toolkit
from kura.backends.musubi_command import command_musubi_tuner, compile_musubi_tuner, display_musubi_tuner
from kura.backends.musubi_models import requirements_musubi
from kura.backends.musubi_models import musubi_model_download_specs
from kura.backends.musubi_datasets import validate_musubi_dataset_layout


Compile = Callable[[dict[str, Any], Path, Path | None, bool], dict[str, Any]]


@dataclass(frozen=True)
class BackendAdapter:
    name: str
    image_name: str
    compile: Compile
    command: Callable[[dict[str, Any]], dict[str, Any]]
    display: Callable[[dict[str, Any]], dict[str, Any]]
    requirements: Callable[..., list[dict[str, Any]]]
    download_specs: Callable[..., tuple[list[dict[str, Any]], dict[str, str]]] | None = None
    validate_dataset: Callable[[dict[str, Any], Path], None] | None = None
    runpod_template_compatible: bool = False
    default_ports: tuple[str, ...] = ("22/tcp",)


def _compile_ai(run: dict[str, Any], resolved: Path, workspace: Path | None, strict: bool) -> dict[str, Any]:
    del workspace, strict
    return compile_ai_toolkit(run, resolved / "ai-toolkit")


def _compile_musubi(run: dict[str, Any], resolved: Path, workspace: Path | None, strict: bool) -> dict[str, Any]:
    return compile_musubi_tuner(run, resolved / "musubi", workspace=workspace, strict=strict)


BACKENDS: dict[str, BackendAdapter] = {
    "ai-toolkit": BackendAdapter("ai-toolkit", "ai-toolkit", _compile_ai, command_ai_toolkit, display_ai_toolkit, requirements_ai_toolkit, default_ports=("8675/http", "22/tcp")),
    "musubi-tuner": BackendAdapter("musubi-tuner", "musubi-tuner", _compile_musubi, command_musubi_tuner, display_musubi_tuner, requirements_musubi, musubi_model_download_specs, validate_musubi_dataset_layout),
}


def backend_names() -> tuple[str, ...]:
    return tuple(BACKENDS)


def get_backend(name: Any) -> BackendAdapter:
    if not isinstance(name, str) or name not in BACKENDS:
        raise ValueError(f"unsupported backend: {name}")
    return BACKENDS[name]

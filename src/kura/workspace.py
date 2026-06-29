"""Workspace discovery and local configuration helpers."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


def dump_yaml(path: Path, value: Any) -> None:
    """
    Serialize a value to YAML and write it to a file.
    
    Parameters:
    	path (Path): The destination file path.
    	value (Any): The value to serialize.
    """
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    """
    Load a YAML mapping from a file.
    
    Parameters:
    	path (Path): The YAML file to read.
    
    Returns:
    	dict[str, Any]: The parsed mapping.
    
    Raises:
    	ValueError: If the file does not contain a YAML mapping.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def workspace(start: Path | None = None) -> Path:
    """
    Find the workspace root by searching for workspace.yaml.
    
    Parameters:
    	start (Path | None): The directory to start searching from.
    
    Returns:
    	Path: The first directory at or above the start directory that contains workspace.yaml, or the resolved start directory when none is found.
    """
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "workspace.yaml").is_file():
            return candidate
    return current


def require_workspace() -> Path:
    """
    Locate the current Kura workspace root.
    
    Returns:
    	Path: The workspace root directory.
    
    Raises:
    	ValueError: If workspace.yaml is missing from the workspace root.
    """
    root = workspace()
    if not (root / "workspace.yaml").is_file():
        raise ValueError("workspace.yaml was not found; run `kura init` or execute this command from inside a Kura workspace")
    return root


def workspace_config() -> dict[str, Any]:
    """
    Load the current workspace configuration.
    
    Returns:
        dict[str, Any]: The parsed contents of <workspace_root>/workspace.yaml.
    """
    return load_yaml(require_workspace() / "workspace.yaml")


def parse_env_file_line(line: str) -> tuple[str, str] | None:
    """
    Parse a single environment file line into a key-value pair.
    
    Parameters:
    	line (str): A line from an environment file.
    
    Returns:
    	tuple[str, str] | None: The parsed environment variable name and value, or None if the line is blank, commented, or invalid.
    """
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    if text.startswith("export "):
        text = text[len("export "):].lstrip()
    if "=" not in text:
        return None
    key, value = text.split("=", 1)
    key = key.strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return key, value


def load_env_local(path: Path | None = None) -> None:
    """
    Load environment variables from a local `.env` file.
    
    Parameters:
    	path (Path | None): Path to the environment file to load.
    
    """
    env_path = path or (workspace() / ".env.local")
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        parsed = parse_env_file_line(line)
        if parsed is None:
            continue
        key, value = parsed
        os.environ.setdefault(key, value)


def run_path(run_id: str) -> Path:
    """
    Build the path to a run directory within the workspace.
    
    Parameters:
    	run_id (str): The run identifier.
    
    Returns:
    	Path: The workspace run directory for the given identifier.
    """
    return require_workspace() / "runs" / run_id


def workspace_relative_path(value: str) -> Path:
    """
    Resolve a path relative to the current workspace.
    
    Parameters:
    	value (str): A path string to resolve.
    
    Returns:
    	Path: The resolved absolute path.
    """
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = require_workspace() / path
    return path.resolve()

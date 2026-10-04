"""Where Kura's secrets live, how commands load them, and `kura secrets set`.

Secrets live in one user-level file outside every workspace; a workspace
`.env.local` overrides it. The process environment wins over both, and an
empty value counts as unset. Values are entered only through hidden input in
the user's own terminal, never through an agent; see
`docs/adr/user-secrets.md`.
"""

from __future__ import annotations

import argparse
import getpass
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

import platformdirs
import yaml

from kura.environment import USER_VARIABLES
from kura.fsio import atomic_write_bytes
from kura.workspace import parse_env_file_line, workspace

ENVIRONMENT = "environment"
WORKSPACE_FILE = ".env.local"
USER_FILE = "user secrets file"

# Where each name loaded by `load_secrets` came from, for `kura doctor secrets`.
_loaded_from: dict[str, str] = {}


def user_secrets_path() -> Path:
    return Path(platformdirs.user_config_dir("kura", appauthor=False, roaming=False)) / "secrets.env"


def workspace_secrets_path() -> Path:
    return workspace() / ".env.local"


def missing(name: str, purpose: str) -> str:
    """The message for a secret a command needs but cannot find."""
    return f"{name} is not set ({purpose}); run `kura secrets set {name}` in your own terminal"


class MissingSecret(ValueError):
    """A command needs a secret that is not set."""

    def __init__(self, name: str, purpose: str) -> None:
        super().__init__(missing(name, purpose))
        self.name = name


def _read(path: Path) -> dict[str, str]:
    """Non-empty values in a secrets file, first occurrence winning."""
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return values
    for line in lines:
        parsed = parse_env_file_line(line)
        if parsed is not None and parsed[1] and parsed[0] not in values:
            values[parsed[0]] = parsed[1]
    return values


def load_secrets() -> None:
    """Fill the environment from `.env.local`, then the user file; set names win."""
    for source, path in ((WORKSPACE_FILE, workspace_secrets_path()), (USER_FILE, user_secrets_path())):
        for name, value in _read(path).items():
            if os.environ.get(name):
                continue
            os.environ[name] = value
            _loaded_from[name] = source


def known_names() -> list[str]:
    """Names `kura secrets set` accepts: declared variables, aliases, and names workspace.yaml configures."""
    names = [variable.name for variable in USER_VARIABLES]
    names += [alias for variable in USER_VARIABLES for alias in variable.aliases]
    for configured in _configured_names():
        if configured not in names:
            names.append(configured)
    return names


def _configured_names() -> list[str]:
    """Secret names workspace.yaml chooses: the RunPod key and the object-store keys."""
    try:
        config = yaml.safe_load((workspace() / "workspace.yaml").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return []
    runpod = config.get("runpod") if isinstance(config, dict) else None
    if not isinstance(runpod, dict):
        return []
    store = runpod.get("object_store") if isinstance(runpod.get("object_store"), dict) else {}
    candidates = (runpod.get("api_key_env"), store.get("access_key_env"), store.get("secret_key_env"))
    return [name for name in candidates if isinstance(name, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)]


def sources() -> dict[str, str]:
    """Where each known secret comes from, or "unset"; never values."""
    result = {}
    for name in known_names():
        if name in _loaded_from and os.environ.get(name):
            result[name] = _loaded_from[name]
        elif os.environ.get(name):
            result[name] = ENVIRONMENT
        else:
            result[name] = "unset"
    return result


@lru_cache(maxsize=1)
def _resolved_user_secrets_path() -> Path:
    return user_secrets_path().resolve()


def is_secret_file(path: Path) -> bool:
    """Whether `path` is, or links to, the user secrets file."""
    try:
        return path.resolve() == _resolved_user_secrets_path()
    except OSError:
        return False


def _line(name: str, value: str) -> str:
    plain = not re.search(r"[\s#'\"]", value)
    if plain:
        return f"{name}={value}"
    if "'" not in value:
        return f"{name}='{value}'"
    if '"' not in value:
        return f'{name}="{value}"'
    raise ValueError("a value containing both quote characters cannot be stored")


def write_secret(path: Path, name: str, value: str) -> None:
    """Set `name` in a secrets file, keeping every other line; readable only by the user.

    A symlinked file is updated where it points, so a synced copy stays the one read.
    """
    path = path.resolve() if path.is_symlink() else path
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        lines = ["# Kura secrets. Set values with `kura secrets set NAME`; never share this file.", ""]
    entry = _line(name, value)
    replaced = False
    kept = []
    for line in lines:
        parsed = parse_env_file_line(line)
        if parsed is not None and parsed[0] == name:
            if not replaced:
                kept.append(entry)
                replaced = True
            continue
        kept.append(line)
    if not replaced:
        kept.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, ("\n".join(kept) + "\n").encode("utf-8"))
    if os.name != "nt":
        os.chmod(path, 0o600)


def cmd_secrets_set(args: argparse.Namespace) -> int:
    name = args.name
    if name not in known_names():
        print(f"cannot set {name}: Kura reads only these secrets: {', '.join(known_names())}", file=sys.stderr)
        return 1
    if args.stdin:
        value = sys.stdin.read()
        value = value[:-1] if value.endswith("\n") else value
    else:
        if not sys.stdin.isatty():
            print(
                "kura secrets set needs your own terminal: it reads the value with hidden input, so it does not run "
                "inside an AI agent or a non-console shell (on Windows, use Windows Terminal or PowerShell). "
                "Nothing was written. Never paste a secret into an agent chat.",
                file=sys.stderr,
            )
            return 1
        value = getpass.getpass(f"{name} (input is hidden): ")
    value = value.strip()
    if not value or len(value.splitlines()) != 1:
        print(f"cannot set {name}: the value must be one non-empty line; nothing was written", file=sys.stderr)
        return 1
    path = workspace_secrets_path() if args.workspace else user_secrets_path()
    if args.workspace and not (path.parent / "workspace.yaml").is_file():
        print("cannot set a workspace secret outside a Kura workspace; nothing was written", file=sys.stderr)
        return 1
    try:
        write_secret(path, name, value)
    except (OSError, ValueError) as exc:
        print(f"cannot set {name}: {exc}", file=sys.stderr)
        return 1
    print(f"set {name} in {path}")
    if os.environ.get(name):
        print(f"note: {name} is also set in this shell's environment, which takes precedence")
    elif not args.workspace and _read(workspace_secrets_path()).get(name):
        print(f"note: this workspace's .env.local also sets {name}, which takes precedence here")
    return 0

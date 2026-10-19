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
from typing import Any

import platformdirs

from kura.environment import USER_VARIABLES
from kura.fsio import atomic_write_bytes, file_lock
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


# A name holds a secret when its words include one of these, in order. Words are split at
# `_`, `-`, and lower-to-upper case changes, so `TOKENIZERS_PARALLELISM` is not one and
# `apiKey` is. Every check of a name uses this rule.
SECRET_NAME_WORDS = (("TOKEN",), ("SECRET",), ("PASSWORD",), ("API", "KEY"), ("ACCESS", "KEY"), ("PRIVATE", "KEY"))


def is_secret_name(name: str) -> bool:
    split = re.sub(r"([A-Z])([A-Z][a-z])", r"\1_\2", re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name))
    words = re.split(r"[^A-Za-z0-9]+", split.upper())
    return any(
        tuple(words[index:index + len(sequence)]) == sequence
        for sequence in SECRET_NAME_WORDS
        for index in range(len(words))
    )


# A secret value is at least this long. Shorter values under a secret-looking name are
# settings such as HF_HUB_DISABLE_IMPLICIT_TOKEN=true; hiding them would rewrite records.
MIN_SECRET_VALUE_LENGTH = 8


def secret_values() -> list[str]:
    """The values Kura holds as secrets, longest first: variables whose names hold a secret.
    Output hides them and `kura check secrets` looks for them."""
    values = {value for name, value in os.environ.items() if holds_secret(name, value)}
    return sorted(values, key=len, reverse=True)


def holds_secret(name: str, value: str) -> bool:
    """Whether a value written under a name is a secret: a secret name, and a long enough value."""
    return len(value) >= MIN_SECRET_VALUE_LENGTH and is_secret_name(name)


_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")


def secret_flags(arguments: list[str]) -> list[str]:
    """The names of the arguments that hold a secret: `--flag value`, `--flag=value`,
    `--flag value` written as one item, and `NAME=value` (as `env` takes it). Never a value.

    A bare flag, or a short value such as sd-scripts' `--max_token_length 225`, is a setting.
    """
    found = []
    for index, argument in enumerate(arguments):
        if argument.startswith("--"):
            name, equals, value = argument[2:].partition("=")
            if not equals:
                name, _, value = re.sub(r"\s", " ", name).partition(" ")
            if not equals and not value:
                following = arguments[index + 1] if index + 1 < len(arguments) else ""
                value = "" if following.startswith("--") else following
            # The label is only the name's leading word characters, so no input puts a value in it.
            label = "--" + re.match(r"[A-Za-z0-9_-]*", name).group()
        elif _ASSIGNMENT.match(argument):
            name, _, value = argument.partition("=")
            label = name
        else:
            continue
        if holds_secret(name, value.strip()):
            found.append(label)
    return found


def undeclared_secret_refusal(where: str, names: list[str]) -> str:
    """The one refusal of a secret written into a run file or a container's env."""
    declared = ", ".join(f"`{name}`" for name in CONTAINER_SECRETS)
    return (
        f"{where} must not contain secrets ({', '.join(sorted(names))}): "
        f"a value written there is kept in the run's records, so remove it. Kura sends a trainer only "
        f"the secrets it declares ({declared}), from Kura's secrets files, which the user fills with "
        f"`kura secrets set NAME`; no other secret, and nothing exported in a shell, reaches a container"
    )


def redact_text(text: str, values: list[str] | None = None) -> str:
    """`text` with every secret value replaced by `***`."""
    redacted = text
    for value in secret_values() if values is None else values:
        redacted = redacted.replace(value, "***")
    return redacted


def redact(value: Any, values: list[str] | None = None) -> Any:
    """`value` with every secret value replaced, and every text under a secret-named key hidden."""
    # The secret values are looked up once per record, not once per string in it.
    values = secret_values() if values is None else values
    if isinstance(value, dict):
        return {key: "***" if isinstance(key, str) and is_secret_name(key) and isinstance(item, str) else redact(item, values) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item, values) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item, values) for item in value)
    if isinstance(value, str):
        return redact_text(value, values)
    return value


def declared_secret(name: str) -> str | None:
    """A declared variable's value from the environment, read under its one name.

    Every place that sends or records a secret reads it here, so every executor sees the same value.
    """
    if name not in {variable.name for variable in USER_VARIABLES}:
        raise ValueError(f"{name} is not a declared variable")
    return os.environ.get(name) or None


# The declared secrets a training or render container gets, under the names it reads them by.
# The Hugging Face libraries read the token as `HF_TOKEN`, the one name it is sent under.
CONTAINER_SECRETS = ("HF_TOKEN",)


def container_secret_env() -> dict[str, str]:
    """The set ones of CONTAINER_SECRETS and their values.

    Every executor sends a container exactly these, so a trainer sees the same variables
    locally and on RunPod.
    """
    return {name: value for name in CONTAINER_SECRETS if (value := declared_secret(name))}


def known_names() -> list[str]:
    """Names `kura secrets set` accepts: the declared variables."""
    return [variable.name for variable in USER_VARIABLES]


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path.with_name(f".{path.name}.lock")):
        _write_locked(path, name, value)


def _write_locked(path: Path, name: str, value: str) -> None:
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

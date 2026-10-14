"""The one reader and writer of every YAML file Kura reads or writes.

Reading follows the YAML 1.2 core schema: `1e-4` is a number, only true/false
are booleans, and `on`, `yes`, or a bare date stay strings. Writing quotes any
string that a YAML 1.2 or a YAML 1.1 reader would turn into another type, so a
file Kura writes reads back the same here and in a trainer's PyYAML loader.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from yaml import YAMLError

__all__ = ["YAMLError", "dump_yaml", "load_yaml", "load_yaml_mapping", "load_yaml_text"]

_BOOL = "tag:yaml.org,2002:bool"
_INT = "tag:yaml.org,2002:int"
_FLOAT = "tag:yaml.org,2002:float"
# YAML 1.2 core schema, section 10.3.2.
_CORE = (
    (_BOOL, re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF")),
    (_INT, re.compile(r"^(?:[-+]?[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+)$"), list("-+0123456789")),
    (
        _FLOAT,
        re.compile(r"^(?:[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$"),
        list("-+0123456789."),
    ),
)


class _Loader(yaml.SafeLoader):
    pass


_Loader.yaml_implicit_resolvers = {
    first: [(tag, regexp) for tag, regexp in resolvers if tag not in {_BOOL, _INT, _FLOAT, "tag:yaml.org,2002:timestamp"}]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
for _tag, _regexp, _first in _CORE:
    _Loader.add_implicit_resolver(_tag, _regexp, _first)


def _construct_int(loader: _Loader, node: yaml.ScalarNode) -> int:
    value = loader.construct_scalar(node)
    if value.startswith("0o"):
        return int(value[2:], 8)
    if value.startswith("0x"):
        return int(value[2:], 16)
    return int(value)


_Loader.add_constructor(_INT, _construct_int)


class _Dumper(yaml.SafeDumper):
    pass


# The YAML 1.1 resolvers stay so a 1.1 reader never retypes a string; the 1.2 ones are added.
_Dumper.yaml_implicit_resolvers = {first: list(resolvers) for first, resolvers in yaml.SafeDumper.yaml_implicit_resolvers.items()}
for _tag, _regexp, _first in _CORE:
    _Dumper.add_implicit_resolver(_tag, _regexp, _first)


def load_yaml_text(text: str) -> Any:
    return yaml.load(text, Loader=_Loader)  # noqa: S506 - _Loader is a SafeLoader


def load_yaml(path: Path) -> Any:
    return load_yaml_text(path.read_text(encoding="utf-8"))


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    data = load_yaml(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def dump_yaml(value: Any) -> str:
    return yaml.dump(value, Dumper=_Dumper, allow_unicode=True, sort_keys=False)

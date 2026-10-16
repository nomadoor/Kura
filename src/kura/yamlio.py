"""The one reader and writer of every YAML file Kura reads or writes.

Reading follows the YAML 1.2 core schema: `1e-4` is a number, only true/false
are booleans, and `on`, `yes`, or a bare date stay strings. A mapping key may
appear once, and the YAML 1.1 merge key `<<` is refused. Writing quotes any
string that a YAML 1.2 or a YAML 1.1 reader would turn into another type, so a
file Kura writes reads back the same here and in a trainer's PyYAML loader.
"""

from __future__ import annotations

import re
from collections.abc import Hashable
from pathlib import Path
from typing import Any

import yaml
from yaml import YAMLError
from yaml.constructor import ConstructorError

__all__ = ["YAMLError", "dump_yaml", "load_run_yaml", "load_yaml", "load_yaml_1_1", "load_yaml_mapping", "load_yaml_text"]

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


class _RecordLoader(yaml.SafeLoader):
    """YAML 1.2 core as Kura read it before 2026-10-16: a repeated key keeps its last value, `<<` merges."""


class _Loader(_RecordLoader):
    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                raise ConstructorError(
                    "while constructing a mapping", node.start_mark,
                    "the YAML 1.1 merge key `<<` is not read; write the keys out", key_node.start_mark,
                )
            key = self.construct_object(key_node, deep=True)
            if isinstance(key, Hashable):
                if key in seen:
                    raise ConstructorError(
                        "while constructing a mapping", node.start_mark, f"found duplicate key {key!r}", key_node.start_mark,
                    )
                seen.add(key)
        return super().construct_mapping(node, deep=deep)


_RecordLoader.yaml_implicit_resolvers = {
    first: [(tag, regexp) for tag, regexp in resolvers if tag not in {_BOOL, _INT, _FLOAT, "tag:yaml.org,2002:timestamp"}]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
for _tag, _regexp, _first in _CORE:
    _RecordLoader.add_implicit_resolver(_tag, _regexp, _first)


def _construct_int(loader: _RecordLoader, node: yaml.ScalarNode) -> int:
    value = loader.construct_scalar(node)
    if not _CORE[1][1].match(value):
        # Only an explicit `!!int` tag reaches here with another form; PyYAML reads it as before.
        return yaml.SafeLoader.construct_yaml_int(loader, node)
    if value.startswith("0o"):
        return int(value[2:], 8)
    if value.startswith("0x"):
        return int(value[2:], 16)
    return int(value)


_RecordLoader.add_constructor(_INT, _construct_int)


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


def load_run_yaml(path: Path) -> Any:
    """A run.yaml as a reader that scans runs sees it. Once the run is compiled
    (`resolved/manifest.lock.yaml` exists) the file is a record, which an older Kura
    may have accepted with a repeated key or a `<<` merge, so it is read as that Kura
    read it; a draft is read strictly, as compile reads it."""
    if (path.parent / "resolved" / "manifest.lock.yaml").is_file():
        return yaml.load(path.read_text(encoding="utf-8"), Loader=_RecordLoader)  # noqa: S506 - a SafeLoader
    return load_yaml(path)


def load_yaml_1_1(path: Path) -> Any:
    """A record as Kura read it before 2026-10-14 (YAML 1.1), only to compare a digest recorded then."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def dump_yaml(value: Any) -> str:
    return yaml.dump(value, Dumper=_Dumper, allow_unicode=True, sort_keys=False)

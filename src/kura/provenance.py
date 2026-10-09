"""Small provenance observations shared by compile and executors."""

from __future__ import annotations

import ast
import hashlib
import io
import json
from pathlib import Path
from typing import Any


def artifact_pinning(identity: dict[str, Any], *, observable: bool) -> dict[str, Any]:
    if isinstance(identity.get("sha256"), str):
        return {"strength": "content-hash", "observation": "observed"}
    revision = identity.get("revision")
    if isinstance(revision, str) and len(revision) >= 40 and all(char in "0123456789abcdefABCDEF" for char in revision):
        return {"strength": "immutable-revision", "observation": "observed"}
    if revision:
        return {"strength": "mutable-reference", "observation": "observed", "detail": "revision is not proven immutable"}
    if identity.get("kind") == "path":
        return {"strength": "external-unobserved", "observation": "not-observed" if observable else "not-observable", "detail": "Kura did not hash the external model path during compile"}
    return {"strength": "mutable-reference", "observation": "not-observed" if observable else "not-observable", "detail": "no immutable revision or content hash was observed"}


def _hash_source_parts(parts: list[tuple[str, bytes]], backend_name: str, *, scope: str) -> dict[str, str]:
    hasher = hashlib.sha256()
    for label, payload in parts:
        hasher.update(label.encode("utf-8") + b"\0")
        hasher.update(payload + b"\0")
    return {"kind": "source-tree-sha256", "value": hasher.hexdigest(), "backend": backend_name, "scope": scope}


_PACKAGE_ROOT = Path(__file__).resolve().parent
# The registry dispatches to every adapter, so an import of it is not followed
# and the walk never enters its table: one adapter's identity does not take in
# the others. The selected adapter's own entry in the table is the start point.
_DISPATCH_MODULES = frozenset({"kura.backends", "kura.backends.registry"})
_REGISTRY_MODULE = "kura.backends.registry"
_REGISTRY_TABLE = "BACKENDS"


class _SourceModule:
    """One parsed kura module: its top-level names, imports, and other statements."""

    def __init__(self, name: str) -> None:
        parts = name.split(".")[1:]
        candidate = _PACKAGE_ROOT.joinpath(*parts).with_suffix(".py") if parts else None
        self.path = candidate if candidate is not None and candidate.is_file() else _PACKAGE_ROOT.joinpath(*parts, "__init__.py")
        self.name = name
        self.relative = self.path.relative_to(_PACKAGE_ROOT).as_posix()
        # Split lines the way the parser numbers them (str.splitlines also splits on form feeds).
        self.lines = io.StringIO(self.path.read_bytes().decode("utf-8"), newline="").readlines()
        tree = ast.parse("".join(self.lines), filename=str(self.path))
        self.definitions: dict[str, list[ast.stmt]] = {}
        self.imports: dict[str, tuple[str, ast.ImportFrom]] = {}
        # Top-level statements that define no name and import nothing run
        # whenever the module is imported, so they belong to every reach of it.
        self.statements: list[ast.stmt] = []
        for index, node in enumerate(tree.body):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.definitions.setdefault(node.name, []).append(node)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        self.definitions.setdefault(target.id, []).append(node)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    self.imports[alias.asname or alias.name] = (alias.name, node)
            elif isinstance(node, ast.Import):
                # Checked when the module is reached; never hashed.
                self.statements.append(node)
            elif not (index == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)):
                self.statements.append(node)

    def location(self, node: ast.AST) -> str:
        return f"{self.relative}:{getattr(node, 'lineno', '?')}"

    def source(self, node: ast.AST) -> bytes:
        decorators = getattr(node, "decorator_list", [])
        start = min([node.lineno, *(item.lineno for item in decorators)])
        return "".join(self.lines[start - 1:node.end_lineno]).encode("utf-8")

    def symbol(self, name: str) -> ast.stmt:
        matches = self.definitions.get(name, [])
        if len(matches) != 1:
            raise ValueError(f"source identity dependency {name!r} was not found exactly once in {self.relative}")
        return matches[0]


def _kura_import_target(module: _SourceModule, node: ast.ImportFrom, name: str) -> tuple[str, str] | None:
    """Resolve one imported name to the kura symbol it names, or None when the walk stops there."""
    if node.level:
        raise ValueError(f"source identity cannot follow a relative import at {module.location(node)}")
    target = node.module or ""
    if (target != "kura" and not target.startswith("kura.")) or target in _DISPATCH_MODULES:
        return None
    parts = [*target.split(".")[1:], name]
    if _PACKAGE_ROOT.joinpath(*parts).with_suffix(".py").is_file() or _PACKAGE_ROOT.joinpath(*parts, "__init__.py").is_file():
        raise ValueError(
            f"source identity cannot follow module import {target}.{name} at {module.location(node)}; "
            "import the names the code uses"
        )
    return target, name


def _literal_argument(call: ast.Call) -> str | None:
    if len(call.args) == 1 and not call.keywords and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
        return call.args[0].value
    return None


def _is_module_file_path(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Path"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "__file__"
    )


def _references(module: _SourceModule, node: ast.AST) -> tuple[list[tuple[str, str]], list[Path]]:
    """Return the kura symbols and the whole files one reached node uses."""
    symbols: list[tuple[str, str]] = []
    files: list[Path] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Import):
            for alias in child.names:
                if alias.name == "kura" or alias.name.startswith("kura."):
                    raise ValueError(
                        f"source identity cannot follow module import {alias.name} at {module.location(child)}; "
                        "import the names the code uses"
                    )
        elif isinstance(child, ast.ImportFrom):
            for alias in child.names:
                target = _kura_import_target(module, child, alias.name)
                if target is not None:
                    symbols.append(target)
        elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            if child.id in module.definitions:
                symbols.append((module.name, child.id))
            elif child.id in module.imports:
                name, statement = module.imports[child.id]
                target = _kura_import_target(module, statement, name)
                if target is not None:
                    symbols.append(target)
        elif isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "script_source":
            literal = _literal_argument(child)
            if literal is None:
                raise ValueError(f"source identity needs a literal script_source name at {module.location(child)}")
            files.append(_PACKAGE_ROOT / "container_scripts" / literal)
        elif (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "with_name"
            and _is_module_file_path(child.func.value)
        ):
            literal = _literal_argument(child)
            if literal is not None:
                files.append(module.path.with_name(literal))
    return symbols, files


def _adapter_source_parts(backend_name: str) -> tuple[list[tuple[str, bytes]], set[tuple[str, str]]]:
    """Walk the import graph from the selected adapter's registry entry.

    Same-module names and ``from kura.X import name`` (at module level or inside
    a function) are followed symbol by symbol; container scripts and data files
    named by literal are hashed whole; each reached module adds its top-level
    statements that define nothing.
    """
    modules: dict[str, _SourceModule] = {}

    def load(name: str) -> _SourceModule:
        if name not in modules:
            modules[name] = _SourceModule(name)
        return modules[name]

    registry = load(_REGISTRY_MODULE)
    table = registry.symbol(_REGISTRY_TABLE)
    entries = table.value if isinstance(table, (ast.Assign, ast.AnnAssign)) else None
    entry = next(
        (value for key, value in zip(entries.keys, entries.values) if isinstance(key, ast.Constant) and key.value == backend_name),
        None,
    ) if isinstance(entries, ast.Dict) else None
    if entry is None:
        raise ValueError(f"unsupported backend for source identity: {backend_name}")
    parts: dict[str, bytes] = {f"{registry.relative}:{_REGISTRY_TABLE}[{backend_name}]": registry.source(entry)}
    files: set[Path] = set()
    reached: set[tuple[str, str]] = set()
    pending: list[tuple[_SourceModule, ast.AST]] = [(registry, entry)]
    while pending:
        module, node = pending.pop()
        symbols, node_files = _references(module, node)
        files.update(node_files)
        for key in symbols:
            if key in reached or key == (_REGISTRY_MODULE, _REGISTRY_TABLE):
                continue
            reached.add(key)
            target = load(key[0])
            module_label = f"{target.relative}:<module>"
            if module_label not in parts:
                parts[module_label] = b"\0".join(target.source(item) for item in target.statements if not isinstance(item, ast.Import))
                pending.extend((target, item) for item in target.statements)
            definition = target.symbol(key[1])
            parts[f"{target.relative}:{key[1]}"] = target.source(definition)
            pending.append((target, definition))
    missing = sorted(path.name for path in files if not path.is_file())
    if missing:
        raise ValueError("source identity input is missing: " + ", ".join(missing))
    parts.update((path.relative_to(_PACKAGE_ROOT).as_posix(), path.read_bytes()) for path in files)
    return sorted(parts.items()), reached


# Source files whose behavior an executor's transport and lifecycle evidence
# depends on, including the shared modules they consume (handoff inventory,
# media suffixes, event and file durability, executor selection). RunPod
# evidence proves these, not the adapter, so it is bound to this identity
# separately. A change to a shared module therefore needs a re-smoke or a
# declared behavior-preserving executor migration.
EXECUTOR_SOURCE_FILES: dict[str, tuple[str, ...]] = {
    "runpod": (
        "container_scripts/pod_self_delete.sh",
        "container_scripts/runpod_input_verify.py",
        "dataset_handoff.py",
        "dataset_transfer.py",
        "executors/common.py",
        "executors/runpod.py",
        "fsio.py",
        "media_types.py",
        "run_commands/launch.py",
        "run_commands/runpod_ssh.py",
        "run_envelope.py",
    ),
}


def executor_source_identity(executor: str, *, read: Any = None) -> dict[str, str]:
    """Hash the executor transport and lifecycle sources.

    ``read`` maps a package-relative path to its bytes; it defaults to the
    installed package and lets tooling hash a historical tree.
    """
    files = EXECUTOR_SOURCE_FILES.get(executor)
    if files is None:
        raise ValueError(f"unsupported executor for source identity: {executor}")
    package_root = Path(__file__).resolve().parent
    reader = read or (lambda relative: (package_root / relative).read_bytes())
    parts = [(relative, reader(relative)) for relative in files]
    identity = _hash_source_parts(parts, executor, scope="executor-v1")
    identity.pop("backend")
    identity["executor"] = executor
    return identity


def adapter_source_identity(backend_name: str) -> dict[str, str]:
    """Hash the selected adapter's registry entry and the kura code it reaches."""
    from dataclasses import fields

    from kura.backends.registry import _GENERAL_ML_ALIASES, _GENERAL_UNAVAILABLE, get_backend

    adapter = get_backend(backend_name)
    parts, reached = _adapter_source_parts(backend_name)
    unreached = sorted(
        f"{field.name}={value.__module__}.{value.__qualname__}"
        for field in fields(adapter)
        if callable(value := getattr(adapter, field.name))
        and (value.__module__, value.__qualname__) not in reached
    )
    if unreached:
        raise ValueError(f"source identity for {backend_name} does not reach its registered callables: " + ", ".join(unreached))
    # Surface membership changes which authored intent reaches the adapter, so
    # the surface the registry holds at run time belongs to the identity too.
    surface = adapter.surface
    parts.append((
        "backend-surface-contract.json",
        json.dumps(
            {
                "fields": sorted(surface.fields),
                "escape_hatches": sorted(surface.escape_hatches),
                "conditions": [
                    {
                        "field": item.field,
                        "when_any": [
                            {selector: list(allowed) for selector, allowed in clause}
                            for clause in item.when_any
                        ],
                    }
                    for item in surface.conditions
                ],
                "selector_defaults": dict(surface.selector_defaults),
                "selector_normalizations": [
                    {
                        "field": item.field,
                        "aliases": list(item.aliases),
                        "rule": item.rule,
                        "value_aliases": dict(item.value_aliases),
                    }
                    for item in surface.selector_normalizations
                ],
                "nested_config_fields": surface.nested_config_fields or {},
                **({"config_value_choices": {field: list(values) for field, values in surface.config_value_choices}}
                   if surface.config_value_choices else {}),
                "aliases": {
                    key: value for key, value in _GENERAL_ML_ALIASES.items()
                    if value in surface.fields | surface.escape_hatches
                },
                "unavailable": {**_GENERAL_UNAVAILABLE, **dict(surface.unavailable)},
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
    ))
    return _hash_source_parts(parts, backend_name, scope="selected-adapter-v3")


def image_reference_identity(reference: str, observed_id: str | None = None) -> dict[str, Any]:
    if observed_id and observed_id.startswith("sha256:"):
        return {"reference": reference, "pinning": {"strength": "content-hash", "observation": "observed", "value": observed_id}}
    if "@sha256:" in reference:
        return {"reference": reference, "pinning": {"strength": "content-hash", "observation": "observed", "value": reference.split("@", 1)[1]}}
    return {
        "reference": reference,
        "pinning": {
            "strength": "mutable-reference",
            "observation": "not-observed",
            "detail": "runtime image digest was not observed",
        },
    }


def training_runtime_contract(
    adapter_source: dict[str, Any],
    local_image_identity: dict[str, Any],
    remote_image_identity: dict[str, Any],
) -> str:
    """Identify the declared local/remote runtime pair for portability checks."""

    payload = {
        "schema": "training-runtime-pair-v1",
        "adapter_source": adapter_source,
        "local_image_identity": local_image_identity,
        "remote_image_identity": remote_image_identity,
    }
    return "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

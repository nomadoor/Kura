#!/usr/bin/env python3
"""Validate canonical project skills and sync the Claude compatibility mirror."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / ".agents" / "skills"
CLAUDE_MIRROR = ROOT / ".claude" / "skills"
SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
# These constraints come from the skill-creator packaging contract. Keep them
# aligned with its references/openai_yaml.md when that contract changes.
SHORT_DESCRIPTION_MIN = 25
SHORT_DESCRIPTION_MAX = 64


def _files(root: Path) -> dict[Path, bytes]:
    if not root.is_dir():
        return {}
    return {
        path.relative_to(root): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }


def _symlinks(root: Path) -> list[Path]:
    if root.is_symlink():
        return [root]
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*") if path.is_symlink())


def _frontmatter(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    pieces = text.split("---", 2)
    if len(pieces) != 3 or pieces[0].strip():
        raise ValueError(f"{path}: SKILL.md must start with YAML frontmatter")
    value = yaml.safe_load(pieces[1])
    if not isinstance(value, dict):
        raise ValueError(f"{path}: frontmatter must be a mapping")
    return value


def validate_skills(root: Path) -> list[str]:
    errors: list[str] = []
    if root.is_symlink():
        return [f"canonical skill directory must not be a symlink: {root}"]
    if not root.is_dir():
        return [f"canonical skill directory does not exist: {root}"]
    for path in _symlinks(root):
        errors.append(f"{path}: skill trees must not contain symlinks")
    skill_dirs = sorted(path for path in root.iterdir() if path.is_dir() and not path.is_symlink())
    if not skill_dirs:
        return [f"canonical skill directory contains no skills: {root}"]
    for skill_dir in skill_dirs:
        skill_path = skill_dir / "SKILL.md"
        if skill_path.is_symlink():
            continue
        if not skill_path.is_file():
            errors.append(f"{skill_dir}: missing SKILL.md")
            continue
        try:
            frontmatter = _frontmatter(skill_path)
        except (OSError, UnicodeError, yaml.YAMLError, ValueError) as exc:
            errors.append(str(exc))
            continue
        # skill-creator deliberately defines this as a closed frontmatter
        # surface. Supporting another key requires an explicit contract update.
        if set(frontmatter) != {"name", "description"}:
            errors.append(f"{skill_path}: frontmatter must contain only name and description")
        name = frontmatter.get("name")
        description = frontmatter.get("description")
        if name != skill_dir.name or not isinstance(name, str) or not SKILL_NAME.fullmatch(name):
            errors.append(f"{skill_path}: name must match the lowercase hyphenated directory name")
        if not isinstance(description, str) or not description.strip():
            errors.append(f"{skill_path}: description must be a non-empty string")
        if len(skill_path.read_text(encoding="utf-8").splitlines()) > 500:
            errors.append(f"{skill_path}: keep SKILL.md at or below 500 lines")

        metadata_path = skill_dir / "agents" / "openai.yaml"
        if metadata_path.is_symlink():
            continue
        if not metadata_path.is_file():
            errors.append(f"{skill_dir}: missing agents/openai.yaml")
            continue
        try:
            metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            errors.append(f"{metadata_path}: cannot load metadata: {exc}")
            continue
        interface = metadata.get("interface") if isinstance(metadata, dict) else None
        if not isinstance(interface, dict):
            errors.append(f"{metadata_path}: interface must be a mapping")
            continue
        for key in ("display_name", "short_description", "default_prompt"):
            if not isinstance(interface.get(key), str) or not interface[key].strip():
                errors.append(f"{metadata_path}: interface.{key} must be a non-empty string")
        short_description = interface.get("short_description")
        if isinstance(short_description, str) and not SHORT_DESCRIPTION_MIN <= len(short_description) <= SHORT_DESCRIPTION_MAX:
            errors.append(
                f"{metadata_path}: short_description must be "
                f"{SHORT_DESCRIPTION_MIN}-{SHORT_DESCRIPTION_MAX} characters"
            )
        default_prompt = interface.get("default_prompt")
        if isinstance(default_prompt, str) and isinstance(name, str) and f"${name}" not in default_prompt:
            errors.append(f"{metadata_path}: default_prompt must mention ${name}")
    return errors


def compare_trees(canonical: Path, mirror: Path) -> list[str]:
    expected = _files(canonical)
    actual = _files(mirror)
    errors = [f"canonical skill tree contains symlink {path}" for path in _symlinks(canonical)]
    errors.extend(f"Claude skill mirror contains symlink {path}" for path in _symlinks(mirror))
    for path in sorted(expected.keys() - actual.keys()):
        errors.append(f"Claude skill mirror is missing {path}")
    for path in sorted(actual.keys() - expected.keys()):
        errors.append(f"Claude skill mirror has extra file {path}")
    for path in sorted(expected.keys() & actual.keys()):
        if expected[path] != actual[path]:
            errors.append(f"Claude skill mirror differs at {path}")
    return errors


def sync_mirror(canonical: Path, mirror: Path) -> None:
    """Update the physical mirror without ever deleting the live tree first."""
    if canonical.is_symlink():
        raise ValueError(f"canonical skill directory must not be a symlink: {canonical}")
    if mirror.is_symlink():
        raise ValueError(f"Claude skill mirror must not be a symlink: {mirror}")
    mirror.mkdir(parents=True, exist_ok=True)
    for path in sorted(_symlinks(mirror), key=lambda item: len(item.parts), reverse=True):
        path.unlink()
    expected = _files(canonical)
    for relative in sorted(expected):
        source = canonical / relative
        target = mirror / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    extras = sorted(_files(mirror).keys() - expected.keys(), reverse=True)
    for relative in extras:
        (mirror / relative).unlink()
    for directory in sorted(
        (path for path in mirror.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="replace the Claude compatibility mirror from canonical .agents/skills")
    args = parser.parse_args()

    errors = validate_skills(CANONICAL)
    if errors:
        print("Skill validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    if args.write:
        sync_mirror(CANONICAL, CLAUDE_MIRROR)
    errors = compare_trees(CANONICAL, CLAUDE_MIRROR)
    if errors:
        print("Skill mirror check failed; run `uv run python scripts/sync_agent_skills.py --write`:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    print(f"skills valid and synchronized: {len([path for path in CANONICAL.iterdir() if path.is_dir()])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

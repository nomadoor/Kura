"""Which Kura produced a run: its version and where it was installed from.

The source comes from the installed distribution's standard direct-URL
metadata (PEP 610). A Git install names its URL and commit; an editable
install names its checkout and that checkout's current commit. Anything else
is recorded as unknown rather than guessed.
"""

from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from urllib.request import url2pathname

from kura import __version__


def describe_install(direct_url: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(direct_url, dict) or not isinstance(direct_url.get("url"), str):
        return {"kind": "unknown"}
    url = direct_url["url"]
    vcs = direct_url.get("vcs_info")
    if isinstance(vcs, dict) and vcs.get("vcs") == "git":
        source: dict[str, Any] = {"kind": "git", "url": _without_credentials(url), "commit": vcs.get("commit_id")}
        if vcs.get("requested_revision"):
            source["requested_revision"] = vcs["requested_revision"]
        return source
    dir_info = direct_url.get("dir_info")
    if isinstance(dir_info, dict) and dir_info.get("editable") and url.startswith("file:"):
        checkout = Path(url2pathname(urlsplit(url).path))
        commit, dirty = _checkout_state(checkout)
        return {"kind": "editable", "path": str(checkout), "commit": commit, "dirty": dirty}
    return {"kind": "unknown"}


def kura_provenance() -> dict[str, Any]:
    """The facts every compiled run and realization records about Kura itself."""

    return {"kura_version": __version__, "kura_source": dict(_installed_source())}


# Cached for the process: every writer today is a short-lived CLI command. A
# long-running process must not reuse this for a checkout that changes under it.
@lru_cache(maxsize=1)
def _installed_source() -> dict[str, Any]:
    try:
        raw = metadata.distribution("kura").read_text("direct_url.json")
    except (metadata.PackageNotFoundError, OSError):
        return {"kind": "unknown"}
    try:
        direct_url = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        direct_url = None
    return describe_install(direct_url)


def _without_credentials(url: str) -> str:
    """Drop a password or token from a URL; a bare user name such as ``git@`` stays."""

    parts = urlsplit(url)
    if parts.password is None and not (parts.username and parts.scheme in {"http", "https"}):
        return url
    host = parts.netloc.rsplit("@", 1)[1]
    user = parts.username if parts.password is None and parts.scheme not in {"http", "https"} else None
    netloc = f"{user}@{host}" if user else host
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _checkout_state(checkout: Path) -> tuple[str | None, bool | None]:
    commit = _git(checkout, "rev-parse", "HEAD")
    if not commit:
        return None, None
    changes = _git(checkout, "status", "--porcelain", "--untracked-files=no")
    return commit, None if changes is None else bool(changes)


def _git(checkout: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(checkout), *args], check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def kura_continuity(source_env: dict[str, Any]) -> dict[str, Any]:
    """Compare the Kura that compiled a Resume source run with the one compiling now.

    ``source_env`` is the source run's env lock. A difference is reported,
    never refused: continuity evidence may not carry over, and the user
    decides whether that matters.
    """

    target = kura_provenance()
    source = {key: source_env[key] for key in ("kura_version", "kura_source") if key in source_env}
    source_version = source.get("kura_version")
    source_install = source.get("kura_source")
    if isinstance(source_version, str) and source_version != target["kura_version"]:
        status = "differs"
    elif not isinstance(source_install, dict) or source_install.get("kind") in (None, "unknown"):
        status = "source-unrecorded"
    elif target["kura_source"].get("kind") == "unknown":
        status = "unverified"
    else:
        same = all(source_install.get(field) == target["kura_source"].get(field) for field in ("kind", "url", "commit", "dirty"))
        status = "same" if same and not target["kura_source"].get("dirty") else "differs"
    return {"status": status, "source": source, "target": target}


def kura_continuity_warning(continuity: Any) -> str | None:
    """One line for a Resume compiled by a different Kura than its source run."""

    if not isinstance(continuity, dict) or continuity.get("status") != "differs":
        return None
    source, target = continuity.get("source"), continuity.get("target")
    kinds = {_install(item).get("kind") for item in (source, target)}
    return (
        f"{_describe(source, len(kinds) > 1)} -> {_describe(target, len(kinds) > 1)}; "
        "Resume continuity evidence may not carry over"
    )


def _install(provenance: Any) -> dict[str, Any]:
    install = provenance.get("kura_source") if isinstance(provenance, dict) else None
    return install if isinstance(install, dict) else {}


def _describe(provenance: Any, with_kind: bool) -> str:
    version = (provenance.get("kura_version") if isinstance(provenance, dict) else None) or "unknown version"
    install = _install(provenance)
    details = [str(install.get("kind") or "unknown")] if with_kind else []
    if isinstance(install.get("commit"), str) and install["commit"]:
        details.append(install["commit"][:12])
    label = f"{version} ({' '.join(details)})" if details else str(version)
    return f"{label}, uncommitted changes" if install.get("dirty") else label

"""Notification helpers for foreground Kura commands."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from typing import Any

from kura.secrets import bearer_request, redact_text


def safe_error(exc: BaseException | str) -> str:
    return redact_text(str(exc))


NOTIFICATION_CHANNELS = ("desktop", "ntfy")


def notification_channels(raw: Any) -> list[str]:
    """The channels a run notifies: `--notify`, else KURA_NOTIFY, else the ones this host can use.

    An explicit choice is `none` or a comma-separated list of desktop and ntfy, spelled exactly.
    """
    explicit = raw if raw is not None else os.environ.get("KURA_NOTIFY") or None
    if explicit is not None:
        values = explicit.split(",") if isinstance(explicit, str) else explicit
        if values == ["none"]:
            return []
        if not isinstance(values, (list, tuple)) or not values or any(value not in NOTIFICATION_CHANNELS for value in values):
            raise ValueError(f"notification channels {explicit!r} must be none or a comma-separated list of desktop and ntfy")
        return list(values)
    channels: list[str] = []
    if shutil.which("notify-send"):
        channels.append("desktop")
    if os.environ.get("KURA_NTFY_TOPIC"):
        channels.append("ntfy")
    return channels


def notify(channels: Any, *, subject: str, body: str, priority: str | None = None) -> None:
    try:
        selected = notification_channels(channels)
    except ValueError as exc:  # notification must never break run lifecycle
        print(f"warning: notification skipped: {exc}", file=sys.stderr)
        return
    if not selected:
        return
    # Every notification leaves the machine through here, so none carries a secret.
    subject, body = redact_text(subject), redact_text(body)
    for channel in selected:
        try:
            if channel == "desktop":
                if shutil.which("notify-send"):
                    subprocess.run(["notify-send", subject, body], check=False)
                continue
            if channel == "ntfy":
                send_ntfy_notification(subject, body, priority=priority)
        except Exception as exc:  # notification must never break run lifecycle
            print(f"warning: notification failed ({channel}): {safe_error(exc)}", file=sys.stderr)


def send_ntfy_notification(subject: str, body: str, priority: str | None = None) -> None:
    topic = os.environ.get("KURA_NTFY_TOPIC")
    if not topic:
        raise ValueError("ntfy notification requires KURA_NTFY_TOPIC")
    server = os.environ.get("KURA_NTFY_SERVER", "https://ntfy.sh").rstrip("/")
    parsed = urllib.parse.urlparse(server)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("KURA_NTFY_SERVER must be an absolute http:// or https:// URL")
    token = os.environ.get("KURA_NTFY_TOKEN")
    priority = priority or os.environ.get("KURA_NTFY_PRIORITY", "4")
    url = f"{server}/{topic.lstrip('/')}"
    request = bearer_request(url, token, method="POST", data=body.encode("utf-8"),
                             headers={"Title": subject, "Tags": "rocket", "Priority": priority})
    with urllib.request.urlopen(request, timeout=20) as response:
        response.read()

"""Notification helpers for foreground Kura commands."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from typing import Any

from kura.executors import _redact_secret_text


def safe_error(exc: BaseException | str) -> str:
    return _redact_secret_text(str(exc))


def notification_channels(raw: Any) -> list[str]:
    explicit = raw
    if explicit in (None, "", False):
        explicit = os.environ.get("KURA_NOTIFY")
    if explicit not in (None, "", False):
        if isinstance(explicit, str):
            values = [part.strip().lower() for part in explicit.split(",") if part.strip()]
            if any(value in ("none", "off", "false", "0") for value in values):
                return []
            return values
        if isinstance(explicit, (list, tuple)):
            values = [str(part).strip().lower() for part in explicit if str(part).strip()]
            if any(value in ("none", "off", "false", "0") for value in values):
                return []
            return values
    channels: list[str] = []
    if shutil.which("notify-send"):
        channels.append("desktop")
    if os.environ.get("KURA_NTFY_TOPIC"):
        channels.append("ntfy")
    return channels


def notify(channels: Any, *, subject: str, body: str, priority: str | None = None) -> None:
    selected = notification_channels(channels)
    if not selected:
        return
    for channel in selected:
        try:
            if channel == "desktop":
                if shutil.which("notify-send"):
                    subprocess.run(["notify-send", subject, body], check=False)
                continue
            if channel == "ntfy":
                send_ntfy_notification(subject, body, priority=priority)
                continue
            print(f"warning: unknown notification channel: {channel}", file=sys.stderr)
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
    headers = {"Title": subject, "Tags": "rocket", "Priority": priority}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=body.encode("utf-8"), method="POST", headers=headers)
    with urllib.request.urlopen(request, timeout=20) as response:
        response.read()

"""Notification helpers for foreground Kura commands."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import urllib.request
from typing import Any

from kura.executors import _redact_secret_text


def safe_error(exc: BaseException | str) -> str:
    return _redact_secret_text(str(exc))


def format_duration(seconds: int) -> str:
    seconds = max(0, int(seconds))
    if seconds >= 86400 and seconds % 86400 == 0:
        return f"{seconds // 86400}d"
    if seconds >= 3600:
        hours, remainder = divmod(seconds, 3600)
        minutes = remainder // 60
        return f"{hours}h{minutes:02d}m" if minutes else f"{hours}h"
    if seconds >= 60:
        minutes, remainder = divmod(seconds, 60)
        return f"{minutes}m{remainder:02d}s" if remainder else f"{minutes}m"
    return f"{seconds}s"


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
            return [str(part).strip().lower() for part in explicit if str(part).strip()]
    channels: list[str] = []
    if shutil.which("notify-send"):
        channels.append("desktop")
    if os.environ.get("KURA_NTFY_TOPIC"):
        channels.append("ntfy")
    return channels


def notify(channels: Any, *, subject: str, body: str) -> None:
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
                send_ntfy_notification(subject, body)
                continue
            print(f"warning: unknown notification channel: {channel}", file=sys.stderr)
        except Exception as exc:  # notification must never break run lifecycle
            print(f"warning: notification failed ({channel}): {safe_error(exc)}", file=sys.stderr)


def send_ntfy_notification(subject: str, body: str) -> None:
    topic = os.environ.get("KURA_NTFY_TOPIC")
    if not topic:
        raise ValueError("ntfy notification requires KURA_NTFY_TOPIC")
    server = os.environ.get("KURA_NTFY_SERVER", "https://ntfy.sh").rstrip("/")
    token = os.environ.get("KURA_NTFY_TOKEN")
    priority = os.environ.get("KURA_NTFY_PRIORITY", "4")
    url = f"{server}/{topic.lstrip('/')}"
    headers = {"Title": subject, "Tags": "rocket", "Priority": priority}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=body.encode("utf-8"), method="POST", headers=headers)
    with urllib.request.urlopen(request, timeout=20) as response:
        response.read()


def sleep_with_completion_reminders(*, delay_sec: int, interval_sec: int, channels: Any, subject: str, body: str) -> None:
    remaining = max(0, int(delay_sec))
    interval = max(0, int(interval_sec))
    elapsed = 0
    while remaining > 0:
        chunk = remaining if interval <= 0 else min(interval, remaining)
        time.sleep(chunk)
        elapsed += chunk
        remaining -= chunk
        if interval > 0 and remaining > 0:
            notify(
                channels,
                subject=f"{subject} (reminder)",
                body=f"{body}\nReminder: {format_duration(elapsed)} since completion. Pod stops in about {format_duration(remaining)}.",
            )

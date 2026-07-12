"""Interactive read-only Textual monitor for Kura runs."""

from __future__ import annotations

import base64
import math
import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.table import Table
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Static

from kura.monitor import ACTIVE_STATES, DRAFT_STATE, RunDataset, RunSummary, _collect_one_run, _collect_run_ids, _format_seconds_per_iter, collect_run_summaries, loss_sparkline


FG = "#c5cdf0"
FG_MUTED = "#8089b3"
MUTED = "#565f89"
ACCENT = "#7aa2f7"
RUN = "#9ece6a"
STALE = "#e0af68"
DONE = "#7dcfff"
FAIL = "#f7768e"
QUEUE = "#6a719c"
LOSS = "#bb9af7"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
AWARE_MIN = datetime.min.replace(tzinfo=timezone.utc)
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
DATASET_STATS_TTL = 30.0
_DATASET_STATS_CACHE: dict[str, tuple[float, DatasetStats]] = {}
_RECENT_DATASET_FILES_CACHE: dict[tuple[str, int], tuple[float, list[str]]] = {}


@dataclass(frozen=True)
class PathTarget:
    label: str
    path: Path


@dataclass(frozen=True)
class DatasetStats:
    items: int | None = None
    images: int = 0
    captions: int = 0
    files: int = 0
    bytes: int = 0
    has_manifest: bool = False
    has_items: bool = False


@dataclass(frozen=True)
class HostMetrics:
    gpu_name: str | None = None
    gpu_util: int | None = None
    vram_used_mb: int | None = None
    vram_total_mb: int | None = None
    gpu_temp_c: int | None = None
    power_draw_w: float | None = None
    power_limit_w: float | None = None
    ram_used_bytes: int | None = None
    ram_total_bytes: int | None = None


class KuraMonitorApp(App[None]):
    """Foreground-only read projection for Kura workspace state."""

    CSS = """
    /* design tokens ---------------------------------------------------------
       background: level0 app, level1 pane, level2 hover, level3 selected/focus, bar
       spacing: grid gap 1, pane padding 1, section gap 1, row height 1 / padding 0 1
       text: strong/value fg, label fg-muted, secondary muted, link accent, state colors only for state
    ---------------------------------------------------------------------- */
    Screen {
        background: #0f1016;
        color: #c5cdf0;
    }
    #frame {
        background: #0f1016;
        height: 1fr;
        layout: grid;
        grid-size: 1 3;
        grid-rows: 1 1fr 1;
        grid-gutter: $gap-y 0;
    }
    #status, #keybar {
        background: #13141e;
        height: 1;
        padding: 0 2;
    }
    #keybar {
        color: #8089b3;
    }
    #grid, #watch-row {
        background: #0f1016;
        height: 1fr;
        layout: grid;
    }
    .pane {
        background: #1a1b26;
        border: none;
        /* 1 row / 2 cols: aspect-corrected so top and left insets look equal */
        padding: 1 2;
    }
    .pane:focus {
        background: #1d2030;
    }
    #nav {
        height: 1fr;
        layout: grid;
        grid-size: 1 3;
        grid-rows: 1 1 1fr;
        grid-gutter: $gap-y 0;
    }
    #detail {
        height: 1fr;
        overflow-y: auto;
    }
    #side {
        height: 1fr;
        layout: grid;
        grid-size: 1 3;
        grid-rows: 5fr 3fr 7fr;
        grid-gutter: $gap-y 0;
    }
    #grid {
        grid-size: 3 1;
        grid-columns: 28 1fr 36;
        grid-gutter: 0 $gap-x;
    }
    #watch-row {
        grid-size: 1 1;
        grid-columns: 1fr;
    }
    #loss {
        height: 100%;
        overflow-y: auto;
    }
    #datasets {
        height: 100%;
        overflow-y: auto;
    }
    #compute {
        height: 100%;
        overflow-y: auto;
    }
    #watch-main {
        height: 1fr;
        overflow-y: auto;
    }
    VerticalScroll {
        scrollbar-size-vertical: 1;
        scrollbar-size-horizontal: 1;
    }
    #nav-body {
        height: 1fr;
        layout: vertical;
    }
    #nav-body.datasets-mode {
        layout: vertical;
    }
    #nav-active {
        height: auto;
        margin-bottom: 1;
        overflow-y: auto;
    }
    #nav-history {
        height: 1fr;
        overflow-y: auto;
    }
    #nav-body.datasets-mode #nav-active {
        height: 1fr;
        margin-bottom: 0;
    }
    #nav-body.datasets-mode #nav-history {
        display: none;
    }
    .tabs {
        height: 1;
    }
    PaneTitle, SectionLabel {
        height: 1;
        color: #8089b3;
    }
    TabButton {
        height: 1;
        width: auto;
        min-width: 8;
        padding: 0 1;
        color: #8089b3;
    }
    TabButton:hover {
        background: #222637;
    }
    TabButton.selected {
        background: #7aa2f7;
        color: #10131f;
        text-style: bold;
    }
    RunRow, EmptyActiveRow, DatasetRow, PathRow, UrlRow {
        height: 1;
        padding: 0;
        background: transparent;
    }
    RunRow:hover, EmptyActiveRow:hover, DatasetRow:hover, PathRow:hover, UrlRow:hover {
        background: #222637;
    }
    RunRow.selected, EmptyActiveRow.selected, DatasetRow.selected, PathRow.selected {
        background: #283041;
    }
    .section-gap {
        height: 1;
    }
    """

    BINDINGS = [("q", "quit", "quit"), ("?", "help", "help")]

    def __init__(self, workspace: Path, *, interval: float = 2.0, stale_after: float = 90.0, initial_run_id: str | None = None, limit: int = 30, include_drafts: bool = False):
        super().__init__()
        self.workspace = Path(workspace)
        self.interval = interval
        self.stale_after = stale_after
        self.initial_run_id = initial_run_id
        self.limit = limit
        self.include_drafts = include_drafts
        self.hidden_draft_count = 0
        self.remote_metrics_interval = 30.0
        self._remote_metrics_cache: dict[str, tuple[float, HostMetrics]] = {}
        self._host_metrics_cache: tuple[float, HostMetrics] | None = None
        self.host_metrics_interval = 5.0
        self._summary_cache: dict[str, tuple[tuple[tuple[str, int, int], ...], RunSummary]] = {}

    def get_css_variables(self) -> dict[str, str]:
        # terminal cells are ~1:2 (w:h); a horizontal gap needs 2 cols to
        # visually match a 1-row vertical gap.
        return {**super().get_css_variables(), "gap-y": "1", "gap-x": "2"}

    def on_mount(self) -> None:
        if self.initial_run_id:
            self.push_screen(MonitorScreen())
            self.push_screen(WatchScreen(self.initial_run_id))
        else:
            self.push_screen(MonitorScreen())

    def action_help(self) -> None:
        self.notify("↑↓ select · click row · Enter/w watch · Esc back · Tab path · r/d tabs · o open · y copy · q quit", timeout=8)

    def collect_summaries_cached(self) -> list[RunSummary]:
        run_ids = _collect_run_ids(self.workspace)
        live_ids = set(run_ids)
        for stale_id in set(self._summary_cache) - live_ids:
            self._summary_cache.pop(stale_id, None)
        summaries: list[RunSummary] = []
        for run_id in run_ids:
            run_dir = self.workspace / "runs" / run_id
            fingerprint = _run_fingerprint(run_dir)
            cached = self._summary_cache.get(run_id)
            if cached and cached[0] == fingerprint and (cached[1].state or "").lower() not in ACTIVE_STATES:
                summaries.append(cached[1])
                continue
            summary = _collect_one_run(self.workspace, run_dir, run_id, loss_tail=80, stale_after=self.stale_after)
            self._summary_cache[run_id] = (fingerprint, summary)
            summaries.append(summary)
        self.hidden_draft_count = 0
        if self.include_drafts:
            return summaries
        visible: list[RunSummary] = []
        for summary in summaries:
            if (summary.state or "").lower() == DRAFT_STATE:
                if summary.id == self.initial_run_id:
                    visible.append(summary)
                    continue
                self.hidden_draft_count += 1
                continue
            visible.append(summary)
        return visible

    def metrics_for(self, summary: RunSummary | None) -> HostMetrics:
        if summary and summary.executor_info.kind == "remote":
            return self._remote_metrics(summary)
        now = time.monotonic()
        if self._host_metrics_cache and now - self._host_metrics_cache[0] < self.host_metrics_interval:
            return self._host_metrics_cache[1]
        metrics = _host_metrics()
        self._host_metrics_cache = (now, metrics)
        return metrics

    def _remote_metrics(self, summary: RunSummary) -> HostMetrics:
        if (summary.state or "").lower() not in ACTIVE_STATES:
            return HostMetrics()
        pod_id = summary.executor_info.pod.id if summary.executor_info.pod else None
        cache_key = pod_id or summary.id
        now = time.monotonic()
        cached = self._remote_metrics_cache.get(cache_key)
        if cached and now - cached[0] < self.remote_metrics_interval:
            return cached[1]
        metrics = _runpod_host_metrics(self.workspace, summary)
        self._remote_metrics_cache[cache_key] = (now, metrics)
        return metrics


class PaneTitle(Static):
    pass


class SectionLabel(Static):
    def __init__(self, label: str):
        super().__init__(label.upper())


class TabButton(Static):
    class Selected(Message):
        def __init__(self, tab: str) -> None:
            super().__init__()
            self.tab = tab

    def __init__(self, label: str, tab: str, *, selected: bool = False, id: str | None = None):
        super().__init__(label, id=id)
        self.tab = tab
        if selected:
            self.add_class("selected")

    def on_click(self) -> None:
        self.post_message(self.Selected(self.tab))


class RunRow(Static):
    class Selected(Message):
        def __init__(self, run_id: str | None, lane: str) -> None:
            super().__init__()
            self.run_id = run_id
            self.lane = lane

    def __init__(self, summary: RunSummary | None, *, lane: str, selected: bool = False):
        self.summary = summary
        self.lane = lane
        super().__init__(self.render_row())
        if selected:
            self.add_class("selected")

    def render_row(self) -> Text:
        if self.summary is None:
            return Text("  none", style=MUTED)
        summary = self.summary
        dot = _state_dot(summary)
        loc = "☁" if summary.executor_info.kind == "remote" else "⌂"
        loc_style = DONE if summary.executor_info.kind == "remote" else MUTED
        try:
            width = max(int(self.size.width or 0), 0)
        except RuntimeError:
            width = 0
        middle_width = 5 if (summary.state or "").lower() in ACTIVE_STATES and (width == 0 or width >= 18) else 0
        name_width = max((width - middle_width - 5) if width else 13, 6)
        name = _fit_plain(_short_run_label(summary), name_width).ljust(name_width)
        activity_percent = _activity_percent(summary.activity)
        if middle_width == 0:
            middle = None
        elif activity_percent and (summary.state or "").lower() in ACTIVE_STATES:
            middle = (_fit_plain(activity_percent, middle_width).rjust(middle_width), FG_MUTED)
        elif summary.activity and (summary.state or "").lower() in ACTIVE_STATES:
            middle = (_fit_plain(summary.activity, middle_width).rjust(middle_width), FG_MUTED)
        else:
            middle = (" " * middle_width, FG_MUTED)
        if middle is None:
            return Text.assemble((loc, loc_style), (" "), (name, FG), (" "), (dot, _state_style(summary)))
        return Text.assemble((loc, loc_style), (" "), (name, FG), (" "), middle, (" "), (dot, _state_style(summary)))

    def on_click(self) -> None:
        self.post_message(self.Selected(self.summary.id if self.summary else None, self.lane))


class EmptyActiveRow(Static):
    def __init__(self, *, selected: bool = False):
        super().__init__(Text("  none", style=MUTED))
        if selected:
            self.add_class("selected")

    def on_click(self) -> None:
        self.post_message(RunRow.Selected(None, "active"))


class DatasetRow(Static):
    class Activated(Message):
        def __init__(self, dataset: RunDataset, action: str) -> None:
            super().__init__()
            self.dataset = dataset
            self.path = dataset.path
            self.action = action

    def __init__(self, dataset: RunDataset, *, action: str = "open", selected: bool = False):
        self.dataset = dataset
        self.action = action
        role = (dataset.role or "-").upper()
        if action == "select":
            text = Text.assemble((dataset.id or "-", FG), ("  "), (role, FG_MUTED if dataset.role else MUTED))
        else:
            digest = _digest_short(dataset.digest)
            text = Text.assemble((role.ljust(6), STALE if dataset.role else FG_MUTED), (" "), (dataset.id or "-", ACCENT), (f"  {digest}" if digest else "", MUTED))
        super().__init__(text)
        if selected:
            self.add_class("selected")

    def on_click(self, event: events.Click) -> None:
        action = self.action if self.action == "select" else ("copy" if event.button == 3 else "open")
        self.post_message(self.Activated(self.dataset, action))


class PathRow(Static):
    class Activated(Message):
        def __init__(self, target: PathTarget, action: str) -> None:
            super().__init__()
            self.target = target
            self.action = action

    def __init__(self, label: str, *, path: Path, display: str | None = None, selected: bool = False):
        self.label = label
        self.path = path
        self.path_display = display
        self.target = PathTarget(label, path)
        super().__init__("")
        if selected:
            self.add_class("selected")

    def render(self) -> Text:
        label = self.label.ljust(5)
        # Keep the visible link inside the row width.  Textual will otherwise
        # clip styled Text in ways that can hide the whole path on narrow panes.
        width = max(int(self.size.width or 0), 0)
        max_len = max(width - len(label) - 2, 1) if width else len(self.path_display or str(self.path))
        display = self.path_display
        if display is None or len(display) > max_len:
            display = _compact_path(self.path, max_len=max_len)
        return Text.assemble((label, FG_MUTED), ("  "), (display, f"underline {ACCENT}"))

    def on_click(self, event: events.Click) -> None:
        self.post_message(self.Activated(self.target, "copy" if event.button == 3 else "open"))


class UrlRow(Static):
    def __init__(self, label: str, *, url: str, display: str):
        self.url = url
        super().__init__(Text.assemble((label.ljust(7), FG_MUTED), (" "), (display, f"underline {ACCENT}")))

    def on_click(self, event: events.Click) -> None:
        if event.button == 3:
            _copy_text(self.url, self.screen)
            self.notify("copied RunPod URL")
        else:
            if _open_url(self.url):
                self.notify("opened RunPod pod")
            else:
                self.notify("could not open RunPod URL", severity="warning")


class MetricGrid(Static):
    def update_summary(self, summary: RunSummary | None) -> None:
        if not summary:
            self.update(Text("no active run", style=MUTED))
            return
        table = Table.grid(padding=(0, 2))
        table.add_column(style=FG_MUTED, width=8)
        table.add_column()
        table.add_row("step", Text(f"{summary.progress.step or 0}/{summary.progress.total or '-'}", style="bold"))
        if summary.activity and (summary.state or "").lower() in ACTIVE_STATES and (summary.progress.step or 0) == 0:
            table.add_row("phase", Text(summary.activity, style=FG_MUTED))
        table.add_row("s/it", Text(_seconds_per_iter(summary), style="bold"))
        table.add_row("elapsed", Text(_elapsed(summary), style="bold"))
        table.add_row("loss", Text(f"{summary.latest_loss:.4g}" if summary.latest_loss is not None else "-", style=f"bold {LOSS}"))
        table.add_row("best", Text(f"{summary.best_loss:.4g}" if summary.best_loss is not None else "-", style="bold"))
        self.update(table)


class NavigatorPane(Grid):
    def compose(self) -> ComposeResult:
        yield PaneTitle("NAVIGATOR")
        with Horizontal(classes="tabs"):
            yield TabButton("Runs", "runs", selected=True, id="tab-runs")
            yield TabButton("Datasets", "datasets", id="tab-datasets")
        with Vertical(id="nav-body"):
            yield VerticalScroll(id="nav-active")
            yield VerticalScroll(id="nav-history")

    def update_state(
        self,
        *,
        active_tab: str,
        active: list[RunSummary],
        history: list[RunSummary],
        datasets: list[RunDataset],
        selected_run_id: str | None,
        selected_dataset_key: tuple[str | None, str | None, str | None, str] | None,
    ) -> None:
        self.query_one("#tab-runs", TabButton).set_class(active_tab == "runs", "selected")
        self.query_one("#tab-datasets", TabButton).set_class(active_tab == "datasets", "selected")
        active_container = self.query_one("#nav-active", VerticalScroll)
        history_container = self.query_one("#nav-history", VerticalScroll)
        nav_body = self.query_one("#nav-body", Vertical)
        nav_body.set_class(active_tab == "datasets", "datasets-mode")
        if active_tab == "datasets":
            _sync_dataset_section(active_container, "datasets", datasets, action="select", selected_dataset_key=selected_dataset_key)
            _sync_static_section(history_container, "")
            return
        _sync_run_section(active_container, "active", active, selected_run_id, show_none=True)
        _sync_run_section(history_container, "history", history, selected_run_id, show_none=False)


def _sync_run_section(container: VerticalScroll, label: str, summaries: list[RunSummary], selected_run_id: str | None, *, show_none: bool) -> None:
    signature = tuple(summary.id for summary in summaries) or (("__none__",) if show_none else ())
    if getattr(container, "_kura_signature", None) != signature:
        container.remove_children()
        if summaries:
            container.mount(SectionLabel(label))
            for summary in summaries:
                container.mount(RunRow(summary, lane=label, selected=summary.id == selected_run_id))
        elif show_none:
            container.mount(SectionLabel(label))
            container.mount(EmptyActiveRow(selected=selected_run_id is None))
        else:
            container.mount(SectionLabel(label))
        setattr(container, "_kura_signature", signature)
        return

    for row in container.query(EmptyActiveRow):
        row.set_class(selected_run_id is None, "selected")
    for row in container.query(RunRow):
        if row.summary is None:
            row.set_class(selected_run_id is None, "selected")
            row.update(row.render_row())
            continue
        summary = next((item for item in summaries if item.id == row.summary.id), row.summary)
        row.summary = summary
        row.lane = label
        row.update(row.render_row())
        row.set_class(summary.id == selected_run_id, "selected")


def _sync_dataset_section(
    container: VerticalScroll,
    label: str,
    datasets: list[RunDataset],
    *,
    action: str,
    selected_dataset_key: tuple[str | None, str | None, str | None, str] | None = None,
) -> None:
    signature = tuple((dataset.role, dataset.id, dataset.digest, str(dataset.path) if dataset.path else "") for dataset in datasets)
    if getattr(container, "_kura_signature", None) == signature:
        for row in container.query(DatasetRow):
            row.set_class(_dataset_key(row.dataset) == selected_dataset_key, "selected")
        return
    container.remove_children()
    container.mount(SectionLabel(label))
    if not datasets:
        container.mount(Static(Text("  no datasets", style=MUTED)))
    for dataset in datasets:
        container.mount(DatasetRow(dataset, action=action, selected=_dataset_key(dataset) == selected_dataset_key))
    setattr(container, "_kura_signature", signature)


def _sync_static_section(container: VerticalScroll, text: str) -> None:
    signature = ("__static__", text)
    if getattr(container, "_kura_signature", None) == signature:
        return
    container.remove_children()
    if text:
        container.mount(Static(Text(text, style=MUTED)))
    setattr(container, "_kura_signature", signature)


class DetailPane(VerticalScroll):
    def update_summary(self, summary: RunSummary | None, selected_target: PathTarget | None) -> None:
        signature = (
            "summary",
            summary.id if summary else None,
            summary.state if summary else None,
            summary.progress.step if summary else None,
            summary.progress.total if summary else None,
            summary.progress.seconds_per_iter if summary else None,
            summary.last_updated if summary else None,
            summary.latest_loss if summary else None,
            summary.is_stale if summary else None,
            summary.activity if summary else None,
            summary.outputs_path if summary else None,
            selected_target,
        )
        if getattr(self, "_kura_signature", None) == signature:
            return
        setattr(self, "_kura_signature", signature)
        scroll_y = self.scroll_y
        self.remove_children()
        self.mount(PaneTitle("RUN" + (f" · {summary.id}" if summary else "")))
        if not summary:
            self.mount(Static(Text("no active run", style=f"bold {FG_MUTED}"), expand=False))
            self.mount(Static(Text("Nothing is running. Select HISTORY to inspect a run.", style=MUTED), expand=False))
            self._restore_scroll(scroll_y)
            return
        self.mount(Static(_run_headline(summary)))
        if summary.is_stale:
            self.mount(Static(Text(f"STALE · last local update {_age(summary.last_updated)}", style=f"bold {STALE}")))
        self.mount(Static("", classes="section-gap"))
        self.mount(Static(_progress_text(summary)))
        self.mount(Static("", classes="section-gap"))
        self.mount(SectionLabel("config"))
        self.mount(Static(_config_table(summary)))
        self.mount(Static("", classes="section-gap"))
        self.mount(SectionLabel("files"))
        for target in _base_path_targets(summary):
            self.mount(PathRow(target.label, path=target.path, display=_compact_path(target.path, max_len=52), selected=selected_target == target))
        self.mount(Static(Text("paths are links · o open · y copy", style=MUTED)))
        self.mount(Static("", classes="section-gap"))
        self.mount(SectionLabel("events"))
        self.mount(Static(_events_table(summary, lines=5)))
        self._restore_scroll(scroll_y)

    def update_dataset(self, dataset: RunDataset | None, selected_target: PathTarget | None) -> None:
        signature = ("dataset", _dataset_key(dataset) if dataset else None, selected_target)
        if getattr(self, "_kura_signature", None) == signature:
            return
        setattr(self, "_kura_signature", signature)
        scroll_y = self.scroll_y
        self.remove_children()
        self.mount(PaneTitle("DATASET" + (f" · {dataset.id}" if dataset and dataset.id else "")))
        if not dataset:
            self.mount(Static(Text("no dataset selected", style=f"bold {FG_MUTED}"), expand=False))
            self.mount(Static(Text("Select a dataset in NAVIGATOR to inspect it.", style=MUTED), expand=False))
            self._restore_scroll(scroll_y)
            return
        stats = _dataset_stats(dataset.path)
        self.mount(Static(Text.assemble((dataset.id or "-", f"bold {FG}"), ("  "), ((dataset.role or "-").upper(), STALE))))
        self.mount(Static("", classes="section-gap"))
        self.mount(SectionLabel("summary"))
        summary = Table.grid(padding=(0, 4))
        summary.add_column(style=FG_MUTED)
        summary.add_column()
        summary.add_column(style=FG_MUTED)
        summary.add_column()
        summary.add_row("items", _count_text(stats.items), "images", str(stats.images))
        summary.add_row("captions", str(stats.captions), "size", _format_bytes(stats.bytes))
        self.mount(Static(summary))
        self.mount(Static("", classes="section-gap"))
        self.mount(SectionLabel("metadata"))
        table = Table.grid(padding=(0, 3))
        table.add_column(style=FG_MUTED, no_wrap=True, width=8)
        table.add_column()
        table.add_row("role", dataset.role or "-")
        table.add_row("digest", dataset.digest or "-")
        table.add_row("path", _compact_path(dataset.path, max_len=56) if dataset.path else "-")
        self.mount(Static(table))
        if dataset.path:
            self.mount(Static("", classes="section-gap"))
            self.mount(SectionLabel("files"))
            target = PathTarget("dataset", dataset.path)
            self.mount(PathRow(target.label, path=target.path, display=_compact_path(target.path, max_len=52), selected=selected_target == target))
            self.mount(Static(Text("left click opens only here · right click copies", style=MUTED)))
            recent = _recent_dataset_files(dataset.path, limit=6)
            if recent:
                self.mount(Static("", classes="section-gap"))
                self.mount(SectionLabel("recent files"))
                self.mount(Static(Text("\n".join(recent), style=FG_MUTED)))
        self._restore_scroll(scroll_y)

    def _restore_scroll(self, y: float) -> None:
        if y:
            self.call_after_refresh(self.scroll_to, y=y, animate=False)


class LossPane(Vertical):
    def __init__(self):
        super().__init__(id="loss", classes="pane")
        self.metrics = MetricGrid()

    def compose(self) -> ComposeResult:
        yield PaneTitle("LOSS")
        yield Static(id="loss-chart")
        yield Static("", classes="section-gap")
        yield self.metrics

    def update_summary(self, summary: RunSummary | None) -> None:
        self.query_one(PaneTitle).update("LOSS")
        chart = self.query_one("#loss-chart", Static)
        if summary and not summary.losses and summary.activity and (summary.state or "").lower() in ACTIVE_STATES:
            chart.update(Text.assemble(("MODEL DOWNLOAD / STARTUP\n", FG_MUTED), (_fit_plain(summary.activity, 28), f"bold {ACCENT}")))
        else:
            chart.update(Text(_mini_loss_chart(summary.losses, width=26, height=3) if summary else "no active run", style=LOSS if summary else MUTED))
        self.metrics.update_summary(summary)

    def update_dataset(self, dataset: RunDataset | None) -> None:
        self.query_one(PaneTitle).update("SUMMARY")
        chart = self.query_one("#loss-chart", Static)
        if not dataset:
            chart.update(Text("no dataset selected", style=MUTED))
            self.metrics.update(Text(""))
            return
        stats = _dataset_stats(dataset.path)
        chart.update(Text.assemble((dataset.id or "-", f"bold {FG}"), ("\n"), (_compact_path(dataset.path, max_len=27) if dataset.path else "-", FG_MUTED)))
        table = Table.grid(padding=(0, 2))
        table.add_column(style=FG_MUTED, width=8)
        table.add_column()
        table.add_row("items", _count_text(stats.items))
        table.add_row("images", str(stats.images))
        table.add_row("captions", str(stats.captions))
        table.add_row("size", _format_bytes(stats.bytes))
        self.metrics.update(table)


class DatasetsPane(Vertical):
    def update_summary(self, summary: RunSummary | None) -> None:
        signature = ("summary", summary.id if summary else None, tuple((d.role, d.id, d.digest) for d in summary.datasets) if summary else ())
        if getattr(self, "_kura_signature", None) == signature:
            return
        setattr(self, "_kura_signature", signature)
        self.remove_children()
        self.mount(PaneTitle("DATASETS"))
        if not summary:
            self.mount(Static(Text("no active run", style=MUTED)))
            return
        if not summary.datasets:
            self.mount(Static(Text("no datasets", style=MUTED)))
        for dataset in summary.datasets:
            self.mount(DatasetRow(dataset, action="open"))
        self.mount(Static(Text("o open · y copy", style=MUTED)))

    def update_dataset(self, dataset: RunDataset | None, summaries: list[RunSummary]) -> None:
        signature = ("dataset", _dataset_key(dataset) if dataset else None, tuple(summary.id for summary in summaries))
        if getattr(self, "_kura_signature", None) == signature:
            return
        setattr(self, "_kura_signature", signature)
        self.remove_children()
        self.mount(PaneTitle("USED BY"))
        if not dataset:
            self.mount(Static(Text("no dataset selected", style=MUTED)))
            return
        runs = _runs_using_dataset(summaries, dataset)
        if not runs:
            self.mount(Static(Text("no runs found", style=MUTED)))
            return
        for summary in runs[:8]:
            self.mount(Static(Text.assemble((_short_run_label(summary), FG), ("  "), (_badge(summary), _badge_style(summary)))))


class ComputePane(Vertical):
    def update_summary(self, summary: RunSummary | None) -> None:
        self.remove_children()
        self.mount(PaneTitle("COMPUTE"))
        if not summary:
            self.mount(Static(Text("idle", style=MUTED)))
            self._mount_host_metrics()
            return
        info = summary.executor_info
        pod_status = "● pod up" if info.kind == "remote" and info.pod else ""
        self.mount(Static(Text.assemble(("☁ " if info.kind == "remote" else "⌂ ", DONE if info.kind == "remote" else MUTED), (info.provider or "local", f"bold {DONE}" if info.kind == "remote" else FG), ("   "), (pod_status, RUN))))
        if info.kind == "remote":
            table = Table.grid(padding=(0, 3))
            table.add_column(style=FG_MUTED, width=7)
            table.add_column()
            table.add_row("gpu", info.gpu or "-")
            self._add_metrics_rows(table, self.app_ref.metrics_for(summary), fallback_gpu=None, include_gpu_name=False)
            self.mount(Static(table))
            if info.pod:
                pod_label = f"{info.pod.id or '-'} · {_fmt_dt(info.pod.started)}–"
                pod_url = _runpod_pod_url(info.pod.id)
                if pod_url:
                    self.mount(UrlRow("pod", url=pod_url, display=_fit_plain(pod_label, 28)))
                else:
                    self.mount(Static(Text.assemble(("pod".ljust(7), FG_MUTED), (" "), (pod_label, FG))))
                pod_table = Table.grid(padding=(0, 3))
                pod_table.add_column(style=FG_MUTED, width=7)
                pod_table.add_column()
                pod_table.add_row("uptime", _duration_since(info.pod.started))
                pod_table.add_row("cost", f"{_money_per_hour(info.pod.cost_per_h)} · {_money(info.pod.cost_used)} used")
                self.mount(Static(pod_table))
        else:
            self._mount_host_metrics(fallback_gpu=info.gpu)

    @property
    def app_ref(self) -> KuraMonitorApp:
        return self.app  # type: ignore[return-value]

    def _mount_host_metrics(self, *, fallback_gpu: str | None = None) -> None:
        metrics = self.app_ref.metrics_for(None)
        table = Table.grid(padding=(0, 2))
        table.add_column(style=FG_MUTED, width=6)
        table.add_column()
        self._add_metrics_rows(table, metrics, fallback_gpu=fallback_gpu, include_gpu_name=True)
        self.mount(Static(table))

    def _add_metrics_rows(self, table: Table, metrics: HostMetrics, *, fallback_gpu: str | None, include_gpu_name: bool) -> None:
        if include_gpu_name:
            table.add_row("gpu", _fit_plain(metrics.gpu_name or fallback_gpu or "-", 30))
        if metrics.gpu_util is not None:
            table.add_row("load", f"{metrics.gpu_util}%")
        if metrics.vram_used_mb is not None and metrics.vram_total_mb:
            table.add_row("vram", f"{metrics.vram_used_mb}/{metrics.vram_total_mb} MiB · {_percent(metrics.vram_used_mb, metrics.vram_total_mb)}")
        if metrics.ram_used_bytes is not None and metrics.ram_total_bytes:
            table.add_row("ram", f"{_format_bytes(metrics.ram_used_bytes)}/{_format_bytes(metrics.ram_total_bytes)} · {_percent(metrics.ram_used_bytes, metrics.ram_total_bytes)}")
        if metrics.gpu_temp_c is not None:
            table.add_row("temp", f"{metrics.gpu_temp_c}°C")
        if metrics.power_draw_w is not None and metrics.power_limit_w is not None:
            table.add_row("power", f"{metrics.power_draw_w:.0f}/{metrics.power_limit_w:.0f} W")

    def update_dataset(self, dataset: RunDataset | None) -> None:
        signature = ("dataset", _dataset_key(dataset) if dataset else None)
        if getattr(self, "_kura_signature", None) == signature:
            return
        setattr(self, "_kura_signature", signature)
        self.remove_children()
        self.mount(PaneTitle("FILES"))
        if not dataset or not dataset.path:
            self.mount(Static(Text("no dataset selected", style=MUTED)))
            return
        files = [
            ("dir", dataset.path),
            ("yaml", dataset.path / "dataset.yaml"),
            ("items", dataset.path / "items.jsonl"),
            ("images", dataset.path / "images"),
        ]
        for label, path in files:
            if path.exists():
                self.mount(PathRow(label, path=path, display=_compact_path(path, max_len=22)))


class WatchPane(VerticalScroll):
    def update_summary(self, summary: RunSummary | None, selected_target: PathTarget | None) -> None:
        scroll_y = self.scroll_y
        self.remove_children()
        self.mount(PaneTitle("WATCH" + (f" · {summary.id}" if summary else "")))
        if not summary:
            self.mount(Static(Text("run not found", style=FAIL)))
            self._restore_scroll(scroll_y)
            return
        self.mount(Static(Text.assemble((summary.id, "bold"), ("  "), (_badge(summary), _badge_style(summary)))))
        overview = Table.grid(expand=True)
        overview.add_column(ratio=1)
        overview.add_column(ratio=1)
        left = Table.grid(padding=(0, 2))
        left.add_column(style=FG_MUTED)
        left.add_column()
        left.add_row("state", _badge(summary))
        left.add_row("progress", f"{summary.progress.step or 0}/{summary.progress.total or '-'}")
        left.add_row("executor", _executor_label(summary))
        right = Table.grid(padding=(0, 2))
        right.add_column(style=FG_MUTED)
        right.add_column()
        right.add_row("created", _fmt_dt(summary.created))
        right.add_row("finished", _fmt_dt(summary.finished))
        right.add_row("outputs", str(summary.outputs_path or "-"))
        overview.add_row(left, right)
        self.mount(Static(overview))
        self.mount(Static("", classes="section-gap"))
        self.mount(SectionLabel("files"))
        for target in _path_targets(summary):
            self.mount(PathRow(target.label, path=target.path, display=_compact_path(target.path, max_len=72), selected=selected_target == target))
        self.mount(Static("", classes="section-gap"))
        self.mount(SectionLabel("config"))
        self.mount(Static(_config_table(summary)))
        self.mount(Static("", classes="section-gap"))
        self.mount(SectionLabel("loss"))
        self.mount(Static(Text(_plot_loss(summary.losses, width=80, height=12) or "loss unavailable", style=LOSS)))
        self.mount(Static("", classes="section-gap"))
        self.mount(SectionLabel("events"))
        self.mount(Static(Text(_events_tail(summary, lines=12))))
        self._restore_scroll(scroll_y)

    def _restore_scroll(self, y: float) -> None:
        if y:
            self.call_after_refresh(self.scroll_to, y=y, animate=False)


class MonitorScreen(Screen[None]):
    """Three-column monitor board."""

    BINDINGS = [
        ("up", "cursor_up", "select up"),
        ("down", "cursor_down", "select down"),
        ("enter", "watch", "watch"),
        ("w", "watch", "watch"),
        ("a", "toggle_all", "all"),
        ("r", "runs", "runs"),
        ("d", "datasets", "datasets"),
        ("tab", "cycle_path", "path"),
        ("o", "open_path", "open"),
        ("y", "copy_path", "copy"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.summaries: list[RunSummary] = []
        self.selected_run_id: str | None = None
        self.selected_run_lane = "active"
        self._run_selection_initialized = False
        self.selected_dataset_key: tuple[str | None, str | None, str | None, str] | None = None
        self.tab = "runs"
        self.path_targets: list[PathTarget] = []
        self.path_index = 0

    @property
    def app_ref(self) -> KuraMonitorApp:
        return self.app  # type: ignore[return-value]

    def compose(self) -> ComposeResult:
        with Grid(id="frame"):
            yield Static(id="status")
            with Grid(id="grid"):
                yield NavigatorPane(id="nav", classes="pane")
                yield DetailPane(id="detail", classes="pane")
                with Grid(id="side"):
                    yield LossPane()
                    yield DatasetsPane(id="datasets", classes="pane")
                    yield ComputePane(id="compute", classes="pane")
            yield Static(id="keybar")

    def on_mount(self) -> None:
        self.refresh_data()
        self.set_interval(max(self.app_ref.interval, 0.2), self.refresh_data)

    @property
    def active_runs(self) -> list[RunSummary]:
        active = [item for item in self.summaries if (item.state or "").lower() in ACTIVE_STATES]
        active.sort(key=lambda item: _aware_datetime(item.last_updated or item.created) or AWARE_MIN, reverse=True)
        return active

    @property
    def history_runs(self) -> list[RunSummary]:
        active_ids = {item.id for item in self.active_runs}
        history = [item for item in self.summaries if item.id not in active_ids]
        history.sort(key=lambda item: _aware_datetime(item.last_updated or item.finished or item.created) or AWARE_MIN, reverse=True)
        return history[: max(self.app_ref.limit, 0)]

    @property
    def ordered_runs(self) -> list[RunSummary]:
        return self.active_runs + self.history_runs

    @property
    def current_run(self) -> RunSummary | None:
        if self.selected_run_id is None:
            return None
        current = next((item for item in self.summaries if item.id == self.selected_run_id), None)
        if self.selected_run_lane == "active" and current and (current.state or "").lower() not in ACTIVE_STATES:
            return None
        return current

    @property
    def current_dataset(self) -> RunDataset | None:
        if self.selected_dataset_key is None:
            return None
        return next((dataset for dataset in _all_datasets(self.summaries) if _dataset_key(dataset) == self.selected_dataset_key), None)

    def refresh_data(self) -> None:
        previous = self.selected_run_id
        self.summaries = self.app_ref.collect_summaries_cached()
        ids = {item.id for item in self.summaries}
        active = self.active_runs
        if not self._run_selection_initialized:
            self.selected_run_id = active[0].id if active else None
            self.selected_run_lane = "active"
            self._run_selection_initialized = True
        else:
            self.selected_run_id, self.selected_run_lane = _resolve_run_selection(previous, self.selected_run_lane, active, ids)
        self.update_view()

    def update_view(self) -> None:
        current = self.current_run
        suffix = None
        if self.app_ref.hidden_draft_count:
            suffix = Text(f"{self.app_ref.hidden_draft_count} draft run(s) hidden (--all to show)", style=MUTED)
        self.query_one("#status", Static).update(_status_bar(self.summaries, workspace=self.app_ref.workspace, width=max(self.size.width, 1), suffix=suffix))
        datasets = _all_datasets(self.summaries)
        dataset_keys = {_dataset_key(dataset) for dataset in datasets}
        if self.tab == "datasets" and self.selected_dataset_key not in dataset_keys:
            self.selected_dataset_key = _dataset_key(datasets[0]) if datasets else None
        self.query_one(NavigatorPane).update_state(
            active_tab=self.tab,
            active=self.active_runs,
            history=self.history_runs,
            datasets=datasets,
            selected_run_id=self.selected_run_id,
            selected_dataset_key=self.selected_dataset_key,
        )
        current_dataset = self.current_dataset
        self.path_targets = _dataset_path_targets(current_dataset) if self.tab == "datasets" else _path_targets(current)
        self.path_index = min(self.path_index, max(len(self.path_targets) - 1, 0))
        selected_target = self.path_targets[self.path_index] if self.path_targets else None
        if self.tab == "datasets":
            self.query_one(DetailPane).update_dataset(current_dataset, selected_target)
            self.query_one(LossPane).update_dataset(current_dataset)
            self.query_one(DatasetsPane).update_dataset(current_dataset, self.summaries)
            self.query_one(ComputePane).update_dataset(current_dataset)
        else:
            self.query_one(DetailPane).update_summary(current, selected_target)
            self.query_one(LossPane).update_summary(current)
            self.query_one(DatasetsPane).update_summary(current)
            self.query_one(ComputePane).update_summary(current)
        self.query_one("#keybar", Static).update(_keybar(self.path_targets, self.path_index))

    def on_tab_button_selected(self, message: TabButton.Selected) -> None:
        self.tab = message.tab
        self.path_index = 0
        self.update_view()

    def on_run_row_selected(self, message: RunRow.Selected) -> None:
        self.selected_run_id = message.run_id
        self.selected_run_lane = message.lane
        self.path_index = 0
        self.update_view()

    def on_dataset_row_activated(self, message: DatasetRow.Activated) -> None:
        if message.action == "select":
            self.selected_dataset_key = _dataset_key(message.dataset)
            self.path_index = 0
            self.update_view()
            return
        if message.path:
            self._select_path(message.path)
            self._activate_path(message.path, message.action)

    def on_path_row_activated(self, message: PathRow.Activated) -> None:
        self._select_path(message.target.path)
        self._activate_path(message.target.path, message.action)

    def _activate_path(self, path: Path, action: str) -> None:
        if action == "copy":
            _copy_text(str(path), self)
            self.notify(f"copied {path.name}")
        else:
            if _open_path(path):
                self.notify(f"opened {path.name}")
            else:
                self.notify(f"could not open {path.name}", severity="warning")

    def action_cursor_up(self) -> None:
        self._move_selection(-1)

    def action_cursor_down(self) -> None:
        self._move_selection(1)

    def action_runs(self) -> None:
        self.tab = "runs"
        self.path_index = 0
        self.update_view()

    def action_datasets(self) -> None:
        self.tab = "datasets"
        self.path_index = 0
        self.update_view()

    def action_watch(self) -> None:
        if self.current_run:
            self.app.push_screen(WatchScreen(self.current_run.id))

    def action_toggle_all(self) -> None:
        self.app_ref.include_drafts = not self.app_ref.include_drafts
        self.path_index = 0
        self._run_selection_initialized = False
        self.refresh_data()

    def action_cycle_path(self) -> None:
        if self.path_targets:
            self.path_index = (self.path_index + 1) % len(self.path_targets)
            self.update_view()

    def action_open_path(self) -> None:
        if self.path_targets:
            target = self.path_targets[self.path_index]
            if _open_path(target.path):
                self.notify(f"opened {target.label}")
            else:
                self.notify(f"could not open {target.label}", severity="warning")

    def action_copy_path(self) -> None:
        if self.path_targets:
            _copy_text(str(self.path_targets[self.path_index].path), self)
            self.notify(f"copied {self.path_targets[self.path_index].label}")

    def _move_selection(self, delta: int) -> None:
        runs = self.ordered_runs
        if not runs:
            self.selected_run_id = None
            self.selected_run_lane = "active"
            self.update_view()
            return
        if self.selected_run_id is None:
            index = -1 if delta > 0 else 0
        else:
            index = next((i for i, item in enumerate(runs) if item.id == self.selected_run_id), 0)
        index = min(max(index + delta, 0), len(runs) - 1)
        self.selected_run_id = runs[index].id
        self.selected_run_lane = "active" if runs[index].id in {item.id for item in self.active_runs} else "history"
        self.path_index = 0
        self.update_view()

    def _select_path(self, path: Path) -> None:
        for index, target in enumerate(self.path_targets):
            if target.path == path:
                self.path_index = index
                self.update_view()
                return


class WatchScreen(Screen[None]):
    """Single-run full-screen view."""

    BINDINGS = [("escape", "app.pop_screen", "back"), ("o", "open_path", "open"), ("y", "copy_path", "copy"), ("tab", "cycle_path", "path")]

    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id
        self.path_targets: list[PathTarget] = []
        self.path_index = 0

    @property
    def app_ref(self) -> KuraMonitorApp:
        return self.app  # type: ignore[return-value]

    def compose(self) -> ComposeResult:
        with Grid(id="frame"):
            yield Static(id="status")
            with Grid(id="watch-row"):
                yield WatchPane(id="watch-main", classes="pane")
            yield Static(id="keybar")

    def on_mount(self) -> None:
        self.refresh_data()
        self.set_interval(max(self.app_ref.interval, 0.2), self.refresh_data)

    def refresh_data(self) -> None:
        summaries = collect_run_summaries(self.app_ref.workspace, loss_tail=10_000, stale_after=self.app_ref.stale_after)
        summary = next((item for item in summaries if item.id == self.run_id), None)
        self.query_one("#status", Static).update(_watch_status_bar(summaries, self.run_id, workspace=self.app_ref.workspace, width=max(self.size.width, 1)))
        self.path_targets = _path_targets(summary)
        self.path_index = min(self.path_index, max(len(self.path_targets) - 1, 0))
        selected_target = self.path_targets[self.path_index] if self.path_targets else None
        self.query_one(WatchPane).update_summary(summary, selected_target)
        self.query_one("#keybar", Static).update(_keybar(self.path_targets, self.path_index, watch=True))

    def on_path_row_activated(self, message: PathRow.Activated) -> None:
        for index, target in enumerate(self.path_targets):
            if target.path == message.target.path:
                self.path_index = index
                self.refresh_data()
                break
        if message.action == "copy":
            _copy_text(str(message.target.path), self)
            self.notify(f"copied {message.target.path.name}")
        else:
            if _open_path(message.target.path):
                self.notify(f"opened {message.target.path.name}")
            else:
                self.notify(f"could not open {message.target.path.name}", severity="warning")

    def action_cycle_path(self) -> None:
        if self.path_targets:
            self.path_index = (self.path_index + 1) % len(self.path_targets)
            self.refresh_data()

    def action_open_path(self) -> None:
        if self.path_targets:
            target = self.path_targets[self.path_index]
            if _open_path(target.path):
                self.notify(f"opened {target.label}")
            else:
                self.notify(f"could not open {target.label}", severity="warning")

    def action_copy_path(self) -> None:
        if self.path_targets:
            _copy_text(str(self.path_targets[self.path_index].path), self)
            self.notify(f"copied {self.path_targets[self.path_index].label}")


def run_textual_monitor(workspace: Path, *, interval: float = 2.0, stale_after: float = 90.0, initial_run_id: str | None = None, limit: int = 30, include_drafts: bool = False) -> int:
    KuraMonitorApp(workspace, interval=interval, stale_after=stale_after, initial_run_id=initial_run_id, limit=limit, include_drafts=include_drafts).run()
    return 0


def _status_bar(summaries: list[RunSummary], *, workspace: Path | None = None, width: int | None = None, suffix: Text | None = None) -> Text:
    counts = {
        "running": sum(item.state == "running" for item in summaries),
        "queued": sum(item.state in {"queued", "staged", "launching"} for item in summaries),
        "done": sum(item.state == "completed" for item in summaries),
        "failed": sum(item.state in {"failed", "launch_failed", "interrupted"} for item in summaries),
    }
    left = Text.assemble(("▸ kura", f"bold {ACCENT}"), ("   "), (str(counts["running"]), f"bold {RUN}"), (" running   ", RUN), (str(counts["queued"]), f"bold {QUEUE}"), (" queued   ", QUEUE), (str(counts["done"]), f"bold {DONE}"), (" done   ", DONE), (str(counts["failed"]), f"bold {FAIL}"), (" failed", FAIL))
    if workspace is not None:
        left.append("   ")
        left.append("ws ", style=FG_MUTED)
        left.append(_compact_path(workspace.resolve(), max_len=36), style=MUTED)
    if suffix:
        left.append("   ")
        left.append_text(suffix)
    width = width or shutil.get_terminal_size((120, 24)).columns
    if len(left.plain) < width:
        left.append(" " * (width - len(left.plain)))
    return left


def _watch_status_bar(summaries: list[RunSummary], run_id: str, *, workspace: Path | None = None, width: int | None = None) -> Text:
    suffix = Text.assemble(("WATCH", f"bold {STALE}"), (" "), (_fit_plain(run_id, 22), FG_MUTED), ("  Esc", ACCENT))
    return _status_bar(summaries, workspace=workspace, width=width, suffix=suffix)


def _all_datasets(summaries: list[RunSummary]) -> list[RunDataset]:
    seen: set[tuple[str | None, str | None]] = set()
    datasets: list[RunDataset] = []
    for summary in summaries:
        for dataset in summary.datasets:
            key = (dataset.role, dataset.id)
            if key not in seen:
                seen.add(key)
                datasets.append(dataset)
    return datasets


def _resolve_run_selection(previous: str | None, lane: str, active: list[RunSummary], all_ids: set[str]) -> tuple[str | None, str]:
    if lane == "active":
        active_ids = {item.id for item in active}
        if previous in active_ids:
            return previous, "active"
        return (active[0].id if active else None), "active"
    if previous is not None and previous not in all_ids:
        return None, "active"
    return previous, lane


def _base_path_targets(summary: RunSummary) -> list[PathTarget]:
    targets: list[PathTarget] = []
    if summary.run_dir:
        targets.append(PathTarget("dir", summary.run_dir))
    if summary.outputs_path:
        targets.append(PathTarget("out", summary.outputs_path))
    return targets


def _path_targets(summary: RunSummary | None) -> list[PathTarget]:
    if not summary:
        return []
    targets = _base_path_targets(summary)
    for dataset in summary.datasets:
        if dataset.path:
            targets.append(PathTarget(f"dataset {dataset.role or dataset.id or ''}", dataset.path))
    return targets


def _dataset_path_targets(dataset: RunDataset | None) -> list[PathTarget]:
    if not dataset or not dataset.path:
        return []
    return [PathTarget("dataset", dataset.path)]


def _dataset_key(dataset: RunDataset) -> tuple[str | None, str | None, str | None, str]:
    return (dataset.role, dataset.id, dataset.digest, str(dataset.path) if dataset.path else "")


def _dataset_stats(path: Path | None) -> DatasetStats:
    if not path:
        return DatasetStats()
    cache_key = str(path)
    now = time.monotonic()
    cached = _DATASET_STATS_CACHE.get(cache_key)
    if cached and now - cached[0] < DATASET_STATS_TTL:
        return cached[1]
    items_path = path / "items.jsonl"
    items = _line_count(items_path) if items_path.exists() else None
    images = 0
    captions = 0
    files = 0
    bytes_total = 0
    try:
        for child in path.rglob("*"):
            if not child.is_file():
                continue
            files += 1
            try:
                bytes_total += child.stat().st_size
            except OSError:
                pass
            suffix = child.suffix.lower()
            if suffix in IMAGE_EXTS:
                images += 1
            elif suffix == ".txt":
                captions += 1
    except OSError:
        pass
    stats = DatasetStats(items=items, images=images, captions=captions, files=files, bytes=bytes_total, has_manifest=(path / "dataset.yaml").exists(), has_items=items_path.exists())
    _DATASET_STATS_CACHE[cache_key] = (now, stats)
    return stats


def _line_count(path: Path) -> int | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except (OSError, UnicodeDecodeError):
        return None


def _recent_dataset_files(path: Path | None, *, limit: int) -> list[str]:
    if not path:
        return []
    cache_key = (str(path), limit)
    now = time.monotonic()
    cached = _RECENT_DATASET_FILES_CACHE.get(cache_key)
    if cached and now - cached[0] < DATASET_STATS_TTL:
        return list(cached[1])
    files: list[Path] = []
    try:
        files = [child for child in path.rglob("*") if child.is_file()]
    except OSError:
        return []
    files.sort(key=lambda child: child.stat().st_mtime if child.exists() else 0, reverse=True)
    recent = [_compact_path(child.relative_to(path), max_len=48) for child in files[:limit]]
    _RECENT_DATASET_FILES_CACHE[cache_key] = (now, recent)
    return list(recent)


def _runs_using_dataset(summaries: list[RunSummary], dataset: RunDataset) -> list[RunSummary]:
    matches: list[RunSummary] = []
    key = _dataset_key(dataset)
    for summary in summaries:
        if any(_dataset_key(item) == key or (item.id == dataset.id and item.digest == dataset.digest) for item in summary.datasets):
            matches.append(summary)
    matches.sort(key=lambda item: _aware_datetime(item.last_updated or item.finished or item.created) or AWARE_MIN, reverse=True)
    return matches


def _count_text(value: int | None) -> str:
    return str(value) if value is not None else "-"


def _host_metrics() -> HostMetrics:
    gpu_metrics = HostMetrics()
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=1.0,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_metrics = _parse_nvidia_smi_csv(result.stdout)
    except (OSError, subprocess.TimeoutExpired):
        pass
    ram_total, ram_available = _host_memory()
    ram_used = ram_total - ram_available if ram_total is not None and ram_available is not None else None
    return HostMetrics(
        gpu_name=gpu_metrics.gpu_name,
        gpu_util=gpu_metrics.gpu_util,
        vram_used_mb=gpu_metrics.vram_used_mb,
        vram_total_mb=gpu_metrics.vram_total_mb,
        gpu_temp_c=gpu_metrics.gpu_temp_c,
        power_draw_w=gpu_metrics.power_draw_w,
        power_limit_w=gpu_metrics.power_limit_w,
        ram_used_bytes=ram_used,
        ram_total_bytes=ram_total,
    )


def _runpod_host_metrics(workspace: Path, summary: RunSummary) -> HostMetrics:
    """Best-effort live telemetry for a running RunPod pod.

    This is intentionally read-only: resolve SSH details through the existing
    RunPod CLI path, then run nvidia-smi and read /proc/meminfo inside the Pod.
    Failures return empty metrics so monitor rendering never depends on remote
    telemetry being available.
    """

    try:
        from kura.run_commands import _runpod_ssh_details, _ssh_base

        details = _runpod_ssh_details(workspace / "runs" / summary.id, timeout_sec=3, interval_sec=1)
        script = r"""
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits 2>/dev/null || true
printf '__KURA_MEM__ '
awk '/MemTotal:/ {total=$2} /MemAvailable:/ {available=$2} END {if (total) printf "%s,%s\n", total, available}' /proc/meminfo 2>/dev/null || true
""".strip()
        result = subprocess.run([*_ssh_base(details), script], text=True, capture_output=True, check=False, timeout=6.0)
    except Exception:
        return HostMetrics()
    if result.returncode != 0 or not result.stdout.strip():
        return HostMetrics()
    return _parse_remote_metrics_output(result.stdout)


def _parse_remote_metrics_output(output: str) -> HostMetrics:
    gpu_lines: list[str] = []
    ram_used: int | None = None
    ram_total: int | None = None
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("__KURA_MEM__"):
            raw = stripped.removeprefix("__KURA_MEM__").strip()
            parts = [part.strip() for part in raw.split(",", 1)]
            if len(parts) == 2:
                total_kib = _int_or_none(parts[0])
                available_kib = _int_or_none(parts[1])
                if total_kib is not None and available_kib is not None:
                    ram_total = total_kib * 1024
                    ram_used = max((total_kib - available_kib) * 1024, 0)
            continue
        gpu_lines.append(stripped)
    gpu_metrics = _parse_nvidia_smi_csv("\n".join(gpu_lines))
    return HostMetrics(
        gpu_name=gpu_metrics.gpu_name,
        gpu_util=gpu_metrics.gpu_util,
        vram_used_mb=gpu_metrics.vram_used_mb,
        vram_total_mb=gpu_metrics.vram_total_mb,
        gpu_temp_c=gpu_metrics.gpu_temp_c,
        power_draw_w=gpu_metrics.power_draw_w,
        power_limit_w=gpu_metrics.power_limit_w,
        ram_used_bytes=ram_used,
        ram_total_bytes=ram_total,
    )


def _parse_nvidia_smi_csv(output: str) -> HostMetrics:
    for line in output.splitlines():
        if not line.strip() or line.startswith("__KURA_"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            continue
        return HostMetrics(
            gpu_name=parts[0] or None,
            gpu_util=_int_or_none(parts[1]),
            vram_used_mb=_int_or_none(parts[2]),
            vram_total_mb=_int_or_none(parts[3]),
            gpu_temp_c=_int_or_none(parts[4]),
            power_draw_w=_float_or_none(parts[5]),
            power_limit_w=_float_or_none(parts[6]),
        )
    return HostMetrics()


def _host_memory() -> tuple[int | None, int | None]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            if key in {"MemTotal", "MemAvailable"}:
                values[key] = int(rest.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return None, None
    return values.get("MemTotal"), values.get("MemAvailable")


def _percent(used: int | float, total: int | float) -> str:
    return f"{(used / total * 100):.0f}%" if total else "-"


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if amount < 1024 or unit == "TB":
            return f"{amount:.0f}{unit}" if unit == "B" else f"{amount:.1f}{unit}"
        amount /= 1024
    return f"{value}B"


def _config_table(summary: RunSummary) -> Table:
    table = Table.grid(padding=(0, 3))
    table.add_column(style=FG_MUTED, no_wrap=True, width=10)
    table.add_column()
    table.add_row("backend", Text(_backend(summary), style=f"bold {ACCENT}"))
    table.add_row("base", Text(_base(summary), style="bold"))
    table.add_row("datasets", Text(_dataset_summary(summary.datasets), style="bold"))
    table.add_row("executor", Text(_executor_label(summary), style=DONE if summary.executor_info.kind == "remote" else FG_MUTED))
    table.add_row("linear / α", Text(_rank_alpha(summary), style="bold"))
    conv = _conv_rank_alpha(summary)
    if conv != "-":
        table.add_row("conv / α", Text(conv, style="bold"))
    table.add_row("batch", Text(_batch(summary), style="bold"))
    table.add_row("lr / sched", Text(_lr_sched(summary), style="bold"))
    save_precision = summary.key_config.get("save_precision")
    if save_precision not in (None, ""):
        table.add_row("save", Text(str(save_precision), style="bold"))
    table.add_row("steps", Text(_steps_seed(summary), style="bold"))
    return table


def _events_tail(summary: RunSummary, *, lines: int) -> str:
    if not summary.run_dir:
        return "-"
    path = summary.run_dir / "logs" / "events.jsonl"
    try:
        raw = path.read_text(encoding="utf-8").splitlines()[-lines:]
    except (OSError, UnicodeDecodeError):
        return "-"
    rendered: list[str] = []
    import json

    for line in raw:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            rendered.append(line)
            continue
        event = item.get("event", "event")
        ts = item.get("timestamp") or item.get("observed_at") or ""
        state = f" state={item['state']}" if item.get("state") else ""
        exit_code = f" exit={item['exit_code']}" if item.get("exit_code") is not None else ""
        rendered.append(f"{str(ts).replace('T', ' ')[:16]} {event}{state}{exit_code}")
    return "\n".join(rendered) if rendered else "-"


def _events_table(summary: RunSummary, *, lines: int) -> Table | Text:
    if not summary.run_dir:
        return Text("-", style=MUTED)
    path = summary.run_dir / "logs" / "events.jsonl"
    try:
        raw = path.read_text(encoding="utf-8").splitlines()[-lines:]
    except (OSError, UnicodeDecodeError):
        return Text("-", style=MUTED)
    table = Table.grid(padding=(0, 2))
    table.add_column(style=MUTED, width=6)
    table.add_column(style=ACCENT, width=15)
    table.add_column()
    import json

    for line in raw:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            table.add_row("", "event", line)
            continue
        ts = str(item.get("timestamp") or item.get("observed_at") or "")
        event = str(item.get("event") or "event")
        table.add_row(ts.replace("T", " ")[11:16] or "--:--", Text(event, style=RUN if event == "run_started" else ACCENT), _event_detail(item))
    return table


def _keybar(targets: list[PathTarget], index: int, *, watch: bool = False) -> Text:
    current = targets[index].label if targets else "no path"
    open_key = "esc" if watch else "enter"
    open_label = " 3-column   " if watch else " watch   "
    return Text.assemble(("↑↓", f"bold {FG} on #222637"), (" select   ", FG_MUTED), (open_key, f"bold {FG} on #222637"), (open_label, FG_MUTED), ("tab", f"bold {FG} on #222637"), (f" path:{current}   ", FG_MUTED), ("r", f"bold {FG} on #222637"), (" runs   ", FG_MUTED), ("d", f"bold {FG} on #222637"), (" datasets   ", FG_MUTED), ("o", f"bold {FG} on #222637"), (" open   ", FG_MUTED), ("y", f"bold {FG} on #222637"), (" copy   ", FG_MUTED), ("q", f"bold {FG} on #222637"), (" quit", FG_MUTED))


def _run_fingerprint(run_dir: Path) -> tuple[tuple[str, int, int], ...]:
    paths = [
        run_dir / "run.yaml",
        run_dir / "resolved" / "manifest.lock.yaml",
        run_dir / "status.json",
        run_dir / "metrics" / "metrics.jsonl",
        run_dir / "logs" / "stdout.log",
        run_dir / "logs" / "events.jsonl",
    ]
    paths.extend(sorted((run_dir / "realizations").glob("*.json")))
    values: list[tuple[str, int, int]] = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        values.append((str(path.relative_to(run_dir)), stat.st_mtime_ns, stat.st_size))
    return tuple(values)


def _open_path(path: Path) -> bool:
    if not path.exists():
        return False
    if _is_wsl():
        converted = subprocess.run(["wslpath", "-w", str(path)], text=True, capture_output=True, check=False)
        converted_path = converted.stdout.strip()
        if converted.returncode or not converted_path:
            return False
        explorer = _windows_command("explorer.exe")
        cmd = _windows_command("cmd.exe")
        if explorer:
            command = [explorer, converted_path]
        elif cmd:
            command = [cmd, "/c", "start", "", converted_path]
        else:
            return False
    elif platform.system() == "Darwin":
        command = ["open", str(path)]
    else:
        if not shutil.which("xdg-open"):
            return False
        command = ["xdg-open", str(path)]
    try:
        subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        return False
    return True


def _open_url(url: str) -> bool:
    if _is_wsl():
        cmd = _windows_command("cmd.exe")
        explorer = _windows_command("explorer.exe")
        if cmd:
            command = [cmd, "/c", "start", "", url]
        elif explorer:
            command = [explorer, url]
        else:
            return False
    elif platform.system() == "Darwin":
        command = ["open", url]
    else:
        if not shutil.which("xdg-open"):
            return False
        command = ["xdg-open", url]
    try:
        subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        return False
    return True


def _windows_command(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    candidates = {
        "explorer.exe": ("/mnt/c/Windows/explorer.exe",),
        "cmd.exe": ("/mnt/c/Windows/System32/cmd.exe",),
    }.get(name, ())
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return None


def _runpod_pod_url(pod_id: str | None) -> str | None:
    return f"https://console.runpod.io/pods?id={pod_id}" if pod_id else None


def _copy_text(value: str, screen: Screen[Any]) -> None:
    if hasattr(screen.app, "copy_to_clipboard"):
        try:
            screen.app.copy_to_clipboard(value)  # type: ignore[attr-defined]
            return
        except Exception:
            pass
    try:
        import pyperclip

        pyperclip.copy(value)
        return
    except Exception:
        pass
    encoded = base64.b64encode(value.encode("utf-8")).decode("ascii")
    os.write(1, f"\033]52;c;{encoded}\a".encode("ascii"))


def _is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False


def _state_dot(summary: RunSummary) -> str:
    return "●"


def _state_style(summary: RunSummary) -> str:
    if summary.is_stale:
        return STALE
    return {"running": RUN, "completed": DONE, "failed": FAIL, "launch_failed": FAIL, "interrupted": FAIL, "queued": QUEUE, "staged": QUEUE, "compiled": QUEUE, "draft": QUEUE}.get(summary.state or "", QUEUE)


def _badge(summary: RunSummary) -> str:
    state = summary.state or "unknown"
    return state.upper()


def _badge_style(summary: RunSummary) -> str:
    return f"bold {_state_style(summary)}"


def _run_headline(summary: RunSummary) -> Text:
    text = Text()
    text.append(_badge(summary), style=_badge_style(summary))
    text.append("  ")
    text.append(_display_name(summary), style=f"bold {FG}")
    if summary.experiment and summary.experiment != _display_name(summary):
        text.append(" · ")
        text.append(_fit_plain(summary.experiment, 36), style=FG_MUTED)
    return text


def _progress_text(summary: RunSummary) -> Text:
    step = summary.progress.step or 0
    total = summary.progress.total or 0
    width = 34
    filled = round(step / total * width) if total else 0
    text = Text()
    text.append("█" * filled, style=RUN if (summary.state or "") in ACTIVE_STATES else ACCENT)
    text.append("░" * (width - filled), style="#343951")
    text.append("\n")
    text.append(f"step {step}/{total or '-'}", style=FG_MUTED)
    if summary.progress.seconds_per_iter is not None:
        text.append(f" · {_seconds_per_iter(summary)}", style=FG_MUTED)
    if summary.activity and (summary.state or "").lower() in ACTIVE_STATES and (step == 0 or not total):
        text.append("\n")
        text.append("phase ", style=FG_MUTED)
        text.append(summary.activity, style=f"bold {ACCENT}")
    return text


def _seconds_per_iter(summary: RunSummary) -> str:
    return _format_seconds_per_iter(summary.progress)


def _executor_label(summary: RunSummary) -> str:
    info = summary.executor_info
    if info.kind == "remote":
        return "☁ " + " · ".join(part for part in (info.provider, info.gpu) if part)
    return "⌂ " + (summary.executor or "local")


def _dataset_summary(datasets: tuple[RunDataset, ...]) -> str:
    if not datasets:
        return "-"
    return " / ".join((dataset.role + ":" if dataset.role else "") + (dataset.id or "-") for dataset in datasets)


def _params(summary: RunSummary) -> str:
    config = summary.key_config
    parts = []
    for key in ("rank", "batch_size", "lr", "steps"):
        if config.get(key) not in (None, ""):
            label = "batch" if key == "batch_size" else key
            parts.append(f"{label}={config[key]}")
    conv = _conv_rank_alpha(summary)
    if conv != "-":
        parts.append(f"conv={conv}")
    save_precision = config.get("save_precision")
    if save_precision not in (None, ""):
        parts.append(f"save={save_precision}")
    grad_accum = config.get("gradient_accumulation_steps")
    if grad_accum not in (None, "", 1):
        parts.append(f"accum={grad_accum}")
    return " · ".join(parts) or "-"


def _rank_alpha(summary: RunSummary) -> str:
    rank = summary.key_config.get("rank")
    alpha = summary.key_config.get("alpha")
    if rank is None and alpha is None:
        return "-"
    return f"{rank or '-'} / {alpha or '-'}"


def _conv_rank_alpha(summary: RunSummary) -> str:
    rank = summary.key_config.get("conv_rank")
    alpha = summary.key_config.get("conv_alpha")
    if rank is None and alpha is None:
        return "-"
    return f"{rank or '-'} / {alpha or '-'}"


def _batch(summary: RunSummary) -> str:
    batch = summary.key_config.get("batch_size")
    accum = summary.key_config.get("gradient_accumulation_steps")
    effective = summary.key_config.get("effective_batch_size")
    if batch in (None, "") and accum in (None, ""):
        return "-"
    if accum in (None, "", 1):
        return str(batch or "-")
    return f"{batch or '-'} × accum {accum} = effective {effective or '-'}"


def _lr_sched(summary: RunSummary) -> str:
    lr = summary.key_config.get("lr")
    scheduler = summary.key_config.get("scheduler") or "constant"
    if lr is None:
        return "-"
    return f"{lr} · {scheduler}"


def _steps_seed(summary: RunSummary) -> str:
    steps = summary.key_config.get("steps") or summary.progress.total
    seed = summary.key_config.get("seed")
    if steps is None:
        return "-"
    return f"{steps}" + (f" · seed {seed}" if seed is not None else "")


def _backend(summary: RunSummary) -> str:
    return str(summary.key_config.get("backend") or ("unknown" if summary.type == "train" else "comfyui"))


def _base(summary: RunSummary) -> str:
    return str(summary.key_config.get("base") or "-")


def _display_name(summary: RunSummary) -> str:
    return summary.experiment or _short_run_label(summary)


def _short_run_label(summary: RunSummary) -> str:
    parts = summary.id.split("_")
    return parts[1] if len(parts) >= 2 else summary.id


def _compact_path(path: Path, *, max_len: int = 52) -> str:
    if max_len <= 0:
        return ""
    if max_len == 1:
        return "…"
    value = str(path)
    if len(value) <= max_len:
        return value
    parts = path.parts
    if len(parts) >= 3:
        value = "…/" + "/".join(parts[-3:])
    if len(value) <= max_len:
        return value
    return "…" + value[-max(max_len - 1, 1):]


def _fit_plain(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    return value[: max(width - 1, 0)] + "…"


def _event_detail(item: dict[str, Any]) -> str:
    details: list[str] = []
    if item.get("state"):
        details.append(f"state {item['state']}")
    if item.get("exit_code") is not None:
        details.append(f"exit {item['exit_code']}")
    if item.get("container_id"):
        details.append("container " + str(item["container_id"])[:8])
    if item.get("detail"):
        details.append(str(item["detail"]))
    if not details and item.get("executor"):
        details.append(str(item["executor"]))
    return " · ".join(details) or "-"


def _age(value: datetime | None) -> str:
    return _duration_since(value) + " ago" if value else "-"


def _elapsed(summary: RunSummary) -> str:
    started = _aware_datetime(summary.started)
    finished = _aware_datetime(summary.finished)
    if started and finished:
        return _short_duration((finished - started).total_seconds())
    if started:
        return _duration_since(started)
    return "-"


def _duration_since(value: datetime | None) -> str:
    value = _aware_datetime(value)
    if not value:
        return "-"
    return _short_duration((datetime.now().astimezone() - value).total_seconds())


def _aware_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        return value.astimezone()
    return value


def _short_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes}m"
    if minutes:
        return f"{minutes}m"
    return f"{sec}s"


def _fmt_dt(value: datetime | None) -> str:
    return value.strftime("%m-%d %H:%M") if value else "-"


def _money(value: float | None) -> str:
    if value is None:
        return "$-"
    return f"${value:.3f}" if value < 10 else f"${value:.2f}"


def _money_per_hour(value: float | None) -> str:
    return "$-/h" if value is None else f"{_money(value)}/h"


def _digest_short(value: str | None) -> str:
    if not value:
        return "-"
    return value[:12] + "…" if len(value) > 13 else value


def _activity_percent(value: str | None) -> str | None:
    if not value:
        return None
    matches = re.findall(r"\b(\d{1,3})%", value)
    if not matches:
        return None
    percent = max(0, min(100, int(matches[-1])))
    return f"{percent}%"


def _plot_loss(values: tuple[float, ...], *, width: int, height: int) -> str:
    if not values:
        return ""
    try:
        import plotext as plt

        plt.clear_figure()
        plt.theme("clear")
        plt.plotsize(width, height)
        plt.plot(list(range(1, len(values) + 1)), list(values))
        built = plt.build()
        if isinstance(built, str):
            return ANSI_RE.sub("", built)
    except Exception:
        pass
    return _ascii_loss(values, width=width, height=height)


def _mini_loss_chart(values: tuple[float, ...], *, width: int, height: int) -> str:
    if not values:
        return "loss unavailable"
    width = max(width, 1)
    height = max(height, 1)
    sampled = _sample_values(values, width)
    low, high = min(sampled), max(sampled)
    if low == high:
        return "\n".join(("─" * len(sampled)) if row == height // 2 else (" " * len(sampled)) for row in range(height))
    rows = [[" " for _ in sampled] for _ in range(height)]
    for index, value in enumerate(sampled):
        row = height - 1 - round((value - low) / (high - low) * (height - 1))
        rows[row][index] = "█"
        if row + 1 < height:
            rows[row + 1][index] = "▌"
    return "\n".join("".join(row).rstrip() for row in rows)


def _ascii_loss(values: tuple[float, ...], *, width: int, height: int) -> str:
    if not values:
        return ""
    sampled = _sample_values(values, width)
    low, high = min(sampled), max(sampled)
    if low == high:
        return "─" * len(sampled)
    rows = [[" " for _ in sampled] for _ in range(height)]
    for index, value in enumerate(sampled):
        row = height - 1 - round((value - low) / (high - low) * (height - 1))
        rows[row][index] = "•"
    return "\n".join("".join(row) for row in rows)


def _sample_values(values: tuple[float, ...], width: int) -> list[float]:
    width = max(width, 1)
    sampled = list(values)
    if width == 1 and sampled:
        return [sampled[-1]]
    if len(sampled) > width:
        sampled = [sampled[round(index * (len(sampled) - 1) / (width - 1))] for index in range(width)]
    return sampled

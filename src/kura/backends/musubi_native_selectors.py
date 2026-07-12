"""Translate opaque Musubi Wan selectors into native command mechanics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WanNativeSelector:
    i2v_cache: bool = False
    clip_required: bool = False
    dual_dit_allowed: bool = False
    one_frame_allowed: bool = False


WAN_NATIVE_SELECTORS: dict[str, WanNativeSelector] = {
    "t2v-1.3B": WanNativeSelector(),
    "t2v-14B": WanNativeSelector(),
    "i2v-14B": WanNativeSelector(i2v_cache=True, clip_required=True, one_frame_allowed=True),
    "t2i-14B": WanNativeSelector(),
    "t2v-1.3B-FC": WanNativeSelector(),
    "t2v-14B-FC": WanNativeSelector(),
    "i2v-14B-FC": WanNativeSelector(i2v_cache=True, clip_required=True),
    "t2v-A14B": WanNativeSelector(dual_dit_allowed=True),
    "i2v-A14B": WanNativeSelector(i2v_cache=True, dual_dit_allowed=True),
    "flf2v-14B": WanNativeSelector(i2v_cache=True, clip_required=True, one_frame_allowed=True),
}


def wan_native_selector(value: str) -> WanNativeSelector:
    try:
        return WAN_NATIVE_SELECTORS[value]
    except KeyError as exc:
        supported = ", ".join(WAN_NATIVE_SELECTORS)
        raise ValueError(f"unsupported Musubi Wan native selector {value!r}; supported: {supported}") from exc

"""事件解析与校验。"""

from __future__ import annotations

from typing import Any

from mahjong_ai.rules.tiles import tile_to_index


def normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    """标准化外部事件格式。"""
    if "type" not in event:
        raise ValueError("事件缺少 type 字段")
    e = dict(event)
    event_type = e["type"]
    if event_type not in {
        "draw",
        "discard",
        "chi",
        "peng",
        "ming_gang",
        "an_gang",
        "bu_gang",
        "hu",
        "pass_all",
    }:
        raise ValueError(f"不支持的事件类型: {event_type}")

    if "seat" in e:
        e["seat"] = int(e["seat"])

    if "tile" in e and e["tile"] is not None:
        e["tile"] = tile_to_index(e["tile"])
    if "chi_start" in e and e["chi_start"] is not None:
        e["chi_start"] = tile_to_index(e["chi_start"])
    return e


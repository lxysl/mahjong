"""动作空间定义与编码。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from mahjong_ai.rules.tiles import NUM_TILE_TYPES, is_suited, tile_rank


@dataclass(frozen=True)
class Action:
    kind: str
    tile: int | None = None
    chi_start: int | None = None


@dataclass(frozen=True)
class ActionSpec:
    action_id: int
    action: Action
    label: str


CHI_STARTS = tuple(
    idx
    for idx in range(NUM_TILE_TYPES)
    if is_suited(idx) and tile_rank(idx) is not None and tile_rank(idx) <= 7
)


@lru_cache(maxsize=1)
def build_action_space() -> tuple[ActionSpec, ...]:
    specs: list[ActionSpec] = []
    action_id = 0

    def add(action: Action, label: str) -> None:
        nonlocal action_id
        specs.append(ActionSpec(action_id=action_id, action=action, label=label))
        action_id += 1

    add(Action("pass"), "pass")
    add(Action("hu"), "hu")

    for tile in range(NUM_TILE_TYPES):
        add(Action("discard", tile=tile), f"discard_{tile}")
    for tile in range(NUM_TILE_TYPES):
        add(Action("peng", tile=tile), f"peng_{tile}")
    for tile in range(NUM_TILE_TYPES):
        add(Action("ming_gang", tile=tile), f"ming_gang_{tile}")
    for tile in range(NUM_TILE_TYPES):
        add(Action("an_gang", tile=tile), f"an_gang_{tile}")
    for tile in range(NUM_TILE_TYPES):
        add(Action("bu_gang", tile=tile), f"bu_gang_{tile}")
    for chi_start in CHI_STARTS:
        add(Action("chi", chi_start=chi_start), f"chi_{chi_start}")

    return tuple(specs)


@lru_cache(maxsize=1)
def _action_to_id() -> dict[Action, int]:
    return {spec.action: spec.action_id for spec in build_action_space()}


@lru_cache(maxsize=1)
def _id_to_action() -> dict[int, Action]:
    return {spec.action_id: spec.action for spec in build_action_space()}


def action_to_id(action: Action) -> int:
    mapping = _action_to_id()
    if action not in mapping:
        raise ValueError(f"动作不在动作空间中: {action}")
    return mapping[action]


def id_to_action(action_id: int) -> Action:
    mapping = _id_to_action()
    if action_id not in mapping:
        raise ValueError(f"未知 action_id: {action_id}")
    return mapping[action_id]


def action_space_size() -> int:
    return len(build_action_space())


def mask_from_legal_ids(legal_ids: set[int]) -> list[int]:
    size = action_space_size()
    mask = [0] * size
    for action_id in legal_ids:
        if 0 <= action_id < size:
            mask[action_id] = 1
    return mask


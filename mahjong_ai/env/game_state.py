"""环境状态结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mahjong_ai.env.actions import Action
from mahjong_ai.rules.tiles import NUM_TILE_TYPES


@dataclass(frozen=True)
class Meld:
    kind: str
    tiles: tuple[int, ...]
    from_seat: int | None = None


@dataclass(frozen=True)
class PendingDiscard:
    discarder: int
    tile: int


@dataclass
class GameState:
    num_players: int = 4
    dealer: int = 0
    current_player: int = 0

    # 墙采用双指针：[wall_head, wall_tail] 闭区间
    wall: list[int] = field(default_factory=list)
    wall_head: int = 0
    wall_tail: int = -1

    laizi_indicator: int | None = None
    laizi_tile: int | None = None
    last_drawn_tile: int | None = None
    after_gang_draw: bool = False

    hands: list[list[int]] = field(default_factory=lambda: [[0] * NUM_TILE_TYPES for _ in range(4)])
    melds: list[list[Meld]] = field(default_factory=lambda: [[] for _ in range(4)])
    discards: list[list[int]] = field(default_factory=lambda: [[] for _ in range(4)])

    phase: str = "draw"  # draw / action / response / terminal
    pending_discard: PendingDiscard | None = None
    response_stage: str | None = None  # hu / peng_gang / chi
    response_order: list[int] = field(default_factory=list)
    response_index: int = 0
    response_claims: dict[int, Action] = field(default_factory=dict)

    winner: int | None = None
    win_mode: str | None = None  # dian_hu / zi_mo
    win_type: str | None = None
    gang_shang_hua: bool = False

    # 用于增量稠密奖励：记录“上一次落地手牌”的听牌进度与首次听牌奖励发放状态
    progress_snapshot: dict[int, tuple[bool, int] | None] = field(
        default_factory=lambda: {0: None, 1: None, 2: None, 3: None}
    )
    tenpai_enter_reward_used: dict[int, bool] = field(default_factory=lambda: {0: False, 1: False, 2: False, 3: False})

    rewards: dict[int, float] = field(default_factory=lambda: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0})
    history: list[dict[str, Any]] = field(default_factory=list)

    def wall_remaining(self) -> int:
        if self.wall_tail < self.wall_head:
            return 0
        return self.wall_tail - self.wall_head + 1

    def is_terminal(self) -> bool:
        return self.phase == "terminal"

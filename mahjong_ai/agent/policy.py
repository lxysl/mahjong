"""策略接口与基线策略。"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol

from mahjong_ai.env.actions import Action
from mahjong_ai.env.game_state import GameState
from mahjong_ai.serving.recommend import recommend_actions


class Policy(Protocol):
    def select_action(self, state: GameState, seat: int, legal_actions: set[Action]) -> Action:
        ...


@dataclass
class RandomPolicy:
    seed: int | None = None
    _rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def select_action(self, state: GameState, seat: int, legal_actions: set[Action]) -> Action:
        if not legal_actions:
            return Action("pass")
        return self._rng.choice(list(legal_actions))


@dataclass
class HeuristicPolicy:
    top_k: int = 1

    def select_action(self, state: GameState, seat: int, legal_actions: set[Action]) -> Action:
        if not legal_actions:
            return Action("pass")
        ranked = recommend_actions(state, seat, top_k=max(self.top_k, len(legal_actions)))
        for item in ranked:
            action = item["action"]
            if action in legal_actions:
                return action
        return next(iter(legal_actions))


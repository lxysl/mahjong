"""实战手动录入会话。"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from mahjong_ai.env.actions import Action
from mahjong_ai.env.game_state import GameState
from mahjong_ai.env.transition import apply_resolved_reaction, apply_turn_action
from mahjong_ai.rules.laizi import indicator_to_laizi
from mahjong_ai.rules.scoring import RewardConfig, evaluate_hand_progress
from mahjong_ai.rules.tiles import NUM_TILE_TYPES, make_counts
from mahjong_ai.serving.event_parser import normalize_event
from mahjong_ai.serving.recommend import recommend_actions


@dataclass
class GameSession:
    my_seat: int
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    state: GameState | None = None
    _snapshots: list[GameState] = field(default_factory=list)

    def start_round(
        self,
        dealer: int,
        laizi_indicator: int | str,
        initial_hands: dict[int, list[int | str]],
        current_player: int | None = None,
    ) -> None:
        """初始化人工录入会话。"""
        state = GameState()
        state.dealer = dealer
        state.current_player = dealer if current_player is None else current_player
        state.phase = "action"
        state.laizi_indicator = int(laizi_indicator) if isinstance(laizi_indicator, int) else None
        if not isinstance(laizi_indicator, int):
            from mahjong_ai.rules.tiles import tile_to_index

            state.laizi_indicator = tile_to_index(laizi_indicator)
        assert state.laizi_indicator is not None
        state.laizi_tile = indicator_to_laizi(state.laizi_indicator)
        state.hands = [[0] * NUM_TILE_TYPES for _ in range(state.num_players)]
        state.melds = [[] for _ in range(state.num_players)]
        state.discards = [[] for _ in range(state.num_players)]
        state.rewards = {seat: 0.0 for seat in range(state.num_players)}
        for seat, tiles in initial_hands.items():
            state.hands[seat] = make_counts(tiles)
        state.progress_snapshot = {seat: None for seat in range(state.num_players)}
        state.tenpai_enter_reward_used = {seat: False for seat in range(state.num_players)}
        for seat in range(state.num_players):
            if sum(state.hands[seat]) % 3 != 1:
                continue
            progress = evaluate_hand_progress(list(state.hands[seat]), laizi_idx=state.laizi_tile, enable_special=True)
            state.progress_snapshot[seat] = (progress.tenpai, progress.outs)
            state.tenpai_enter_reward_used[seat] = progress.tenpai
        self.state = state
        self._snapshots = []

    def apply_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """应用单条事件并返回当前推荐。"""
        if self.state is None:
            raise ValueError("会话未初始化，请先调用 start_round")
        e = normalize_event(event)
        self._snapshots.append(copy.deepcopy(self.state))

        event_type = e["type"]
        state = self.state

        # 如果仍有待响应弃牌，且输入的是下一步摸牌/打牌，默认视为无人响应
        if state.phase == "response" and event_type in {"draw", "discard"}:
            apply_resolved_reaction(state, None, None, config=self.reward_config)

        if event_type == "draw":
            seat = e["seat"]
            tile = e["tile"]
            state.hands[seat][tile] += 1
            state.current_player = seat
            state.last_drawn_tile = tile
            state.phase = "action"
            state.history.append({"type": "draw", "seat": seat, "tile": tile, "source": "manual"})
        elif event_type == "discard":
            seat = e["seat"]
            tile = e["tile"]
            apply_turn_action(state, seat, Action("discard", tile=tile), config=self.reward_config)
        elif event_type == "an_gang":
            seat = e["seat"]
            tile = e["tile"]
            apply_turn_action(state, seat, Action("an_gang", tile=tile), config=self.reward_config)
        elif event_type == "bu_gang":
            seat = e["seat"]
            tile = e["tile"]
            apply_turn_action(state, seat, Action("bu_gang", tile=tile), config=self.reward_config)
        elif event_type == "hu":
            seat = e["seat"]
            if state.phase == "action":
                apply_turn_action(state, seat, Action("hu"), config=self.reward_config)
            elif state.phase == "response":
                apply_resolved_reaction(state, seat, Action("hu"), config=self.reward_config)
            else:
                raise ValueError("当前阶段不允许胡牌事件")
        elif event_type == "peng":
            seat = e["seat"]
            apply_resolved_reaction(state, seat, Action("peng"), config=self.reward_config)
        elif event_type == "ming_gang":
            seat = e["seat"]
            apply_resolved_reaction(state, seat, Action("ming_gang"), config=self.reward_config)
        elif event_type == "chi":
            seat = e["seat"]
            chi_start = e["chi_start"]
            apply_resolved_reaction(state, seat, Action("chi", chi_start=chi_start), config=self.reward_config)
        elif event_type == "pass_all":
            apply_resolved_reaction(state, None, None, config=self.reward_config)
        else:
            raise ValueError(f"未知事件类型: {event_type}")

        return self.recommend_action()

    def recommend_action(self, top_k: int = 3, seat: int | None = None) -> dict[str, Any]:
        if self.state is None:
            raise ValueError("会话未初始化")
        state = self.state
        target_seat = state.current_player if seat is None else seat
        recs = recommend_actions(state, target_seat, top_k=top_k)
        return {
            "seat": target_seat,
            "phase": state.phase,
            "top_k": recs,
            "winner": state.winner,
            "win_mode": state.win_mode,
            "win_type": state.win_type,
        }

    def undo_last_event(self) -> None:
        if not self._snapshots:
            raise ValueError("没有可撤销事件")
        self.state = self._snapshots.pop()

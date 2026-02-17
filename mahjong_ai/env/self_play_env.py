"""自博弈环境（Gym 风格简化接口）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mahjong_ai.env.actions import Action, action_space_size, id_to_action, mask_from_legal_ids
from mahjong_ai.env.game_state import GameState
from mahjong_ai.env.legal_actions import (
    legal_action_ids,
    legal_actions_for_turn,
    legal_reactions_to_discard,
)
from mahjong_ai.env.transition import (
    apply_resolved_reaction,
    apply_turn_action,
    draw_for_current_player,
    initialize_round,
)
from mahjong_ai.rules.scoring import RewardConfig


@dataclass
class StepResult:
    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]


class SelfPlayEnv:
    """四人共享策略的回合同步环境。"""

    def __init__(self, seed: int | None = None, dealer: int = 0, reward_config: RewardConfig | None = None) -> None:
        self.seed = seed
        self.dealer = dealer
        self.reward_config = reward_config or RewardConfig()
        self.state: GameState = initialize_round(seed=seed, dealer=dealer)
        self._last_rewards = dict(self.state.rewards)

    def reset(self, seed: int | None = None, dealer: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = seed
        if dealer is not None:
            self.dealer = dealer
        self.state = initialize_round(seed=self.seed, dealer=self.dealer)
        self._last_rewards = dict(self.state.rewards)
        return self.observe()

    def observe(self, seat: int | None = None) -> dict[str, Any]:
        state = self.state
        if seat is None:
            seat = state.current_player
        return {
            "seat": seat,
            "current_player": state.current_player,
            "dealer": state.dealer,
            "phase": state.phase,
            "hand_counts": list(state.hands[seat]),
            "laizi_indicator": state.laizi_indicator,
            "laizi_tile": state.laizi_tile,
            "wall_remaining": state.wall_remaining(),
            "discards": [list(d) for d in state.discards],
            "melds": [[m.__dict__ for m in seat_melds] for seat_melds in state.melds],
            "pending_discard": (
                None
                if state.pending_discard is None
                else {"discarder": state.pending_discard.discarder, "tile": state.pending_discard.tile}
            ),
            "legal_action_mask": self.legal_action_mask(),
        }

    def legal_actions(self) -> set[Action]:
        state = self.state
        if state.phase == "terminal":
            return set()
        if state.phase == "draw":
            draw_for_current_player(state)
        if state.phase == "action":
            return legal_actions_for_turn(state, state.current_player)
        if state.phase == "response":
            pending = state.pending_discard
            if pending is None:
                return {Action("pass")}
            actions = legal_reactions_to_discard(
                state,
                seat=state.current_player,
                discarder=pending.discarder,
                tile=pending.tile,
            )
            return self._filter_response_actions_by_stage(actions)
        return set()

    def legal_action_mask(self) -> list[int]:
        actions = self.legal_actions()
        ids = legal_action_ids(actions)
        return mask_from_legal_ids(ids)

    def step(self, action: Action | int) -> StepResult:
        if self.state.phase == "terminal":
            return StepResult(self.observe(), reward=0.0, done=True, info={"reason": "terminal"})

        if self.state.phase == "draw":
            draw_for_current_player(self.state)

        actor = self.state.current_player
        if isinstance(action, int):
            action = id_to_action(action)

        legal = self.legal_actions()
        if action not in legal:
            raise ValueError(f"非法动作: {action}")

        if self.state.phase == "action":
            apply_turn_action(self.state, actor, action, config=self.reward_config)
        elif self.state.phase == "response":
            self._apply_response_action(actor, action)
        else:
            raise ValueError(f"未知阶段: {self.state.phase}")

        reward_delta = self._consume_reward_delta()
        done = self.state.phase == "terminal"
        info = {
            "winner": self.state.winner,
            "win_mode": self.state.win_mode,
            "win_type": self.state.win_type,
            "rewards": dict(self.state.rewards),
        }
        return StepResult(self.observe(), reward=reward_delta.get(actor, 0.0), done=done, info=info)

    def _apply_response_action(self, seat: int, action: Action) -> None:
        state = self.state
        if state.pending_discard is None or state.response_stage is None:
            raise ValueError("响应阶段状态不完整")

        if action.kind != "pass":
            state.response_claims[seat] = action
            if self._is_immediate_resolvable_claim(action):
                apply_resolved_reaction(state, seat, action, config=self.reward_config)
                return

        state.response_index += 1
        if state.response_index < len(state.response_order):
            state.current_player = state.response_order[state.response_index]
            return

        self._resolve_current_response_stage()

    def _resolve_current_response_stage(self) -> None:
        state = self.state
        assert state.pending_discard is not None
        assert state.response_stage is not None

        if state.response_stage == "hu":
            winner = self._first_claim_by_order({"hu"})
            if winner is not None:
                seat, action = winner
                apply_resolved_reaction(state, seat, action, config=self.reward_config)
                return
            state.response_stage = "peng_gang"
            state.response_order = [
                (state.pending_discard.discarder + i) % state.num_players for i in range(1, state.num_players)
            ]
            state.response_index = 0
            state.current_player = state.response_order[0]
            state.response_claims = {}
            return

        if state.response_stage == "peng_gang":
            winner = self._first_claim_by_order({"peng", "ming_gang"})
            if winner is not None:
                seat, action = winner
                apply_resolved_reaction(state, seat, action, config=self.reward_config)
                return
            next_seat = (state.pending_discard.discarder + 1) % state.num_players
            state.response_stage = "chi"
            state.response_order = [next_seat]
            state.response_index = 0
            state.current_player = next_seat
            state.response_claims = {}
            return

        if state.response_stage == "chi":
            winner = self._first_claim_by_order({"chi"})
            if winner is not None:
                seat, action = winner
                apply_resolved_reaction(state, seat, action, config=self.reward_config)
            else:
                apply_resolved_reaction(state, None, None, config=self.reward_config)
            return

        raise ValueError(f"未知响应阶段: {state.response_stage}")

    def _first_claim_by_order(self, allowed_kinds: set[str]) -> tuple[int, Action] | None:
        state = self.state
        for seat in state.response_order:
            action = state.response_claims.get(seat)
            if action and action.kind in allowed_kinds:
                return seat, action
        return None

    def _filter_response_actions_by_stage(self, actions: set[Action]) -> set[Action]:
        state = self.state
        stage = state.response_stage
        if stage == "hu":
            return {a for a in actions if a.kind in {"pass", "hu"}}
        if stage == "peng_gang":
            return {a for a in actions if a.kind in {"pass", "peng", "ming_gang"}}
        if stage == "chi":
            return {a for a in actions if a.kind in {"pass", "chi"}}
        return actions

    def _is_immediate_resolvable_claim(self, action: Action) -> bool:
        """当前座位声明后是否可立即结算（按顺序响应，后续座位优先级更低）。"""
        stage = self.state.response_stage
        if stage == "hu":
            return action.kind == "hu"
        if stage == "peng_gang":
            return action.kind in {"peng", "ming_gang"}
        if stage == "chi":
            return action.kind == "chi"
        return False

    def _consume_reward_delta(self) -> dict[int, float]:
        delta: dict[int, float] = {}
        for seat, value in self.state.rewards.items():
            prev = self._last_rewards.get(seat, 0.0)
            delta[seat] = value - prev
        self._last_rewards = dict(self.state.rewards)
        return delta

    @property
    def action_space_size(self) -> int:
        return action_space_size()

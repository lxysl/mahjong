"""推荐动作生成（默认启发式）。"""

from __future__ import annotations

from typing import Any

from mahjong_ai.env.actions import Action
from mahjong_ai.env.legal_actions import legal_actions_for_turn, legal_reactions_to_discard
from mahjong_ai.env.game_state import GameState
from mahjong_ai.rules.tiles import index_to_tile, is_suited, tile_rank


def recommend_actions(state: GameState, seat: int, top_k: int = 3) -> list[dict[str, Any]]:
    """返回 top-k 推荐动作。"""
    actions = _collect_legal_actions(state, seat)
    ranked = sorted(actions, key=lambda a: _action_score(state, seat, a), reverse=True)
    output = []
    for action in ranked[:top_k]:
        output.append(
            {
                "action": action,
                "score": _action_score(state, seat, action),
                "label": _action_label(action),
                "reason": _action_reason(action),
            }
        )
    return output


def _collect_legal_actions(state: GameState, seat: int) -> set[Action]:
    if state.phase == "action":
        return legal_actions_for_turn(state, seat)
    if state.phase == "response" and state.pending_discard is not None:
        return legal_reactions_to_discard(
            state,
            seat=seat,
            discarder=state.pending_discard.discarder,
            tile=state.pending_discard.tile,
        )
    return set()


def _action_score(state: GameState, seat: int, action: Action) -> float:
    # 先确保高优先级动作靠前
    if action.kind == "hu":
        return 100.0
    if action.kind == "an_gang":
        return 80.0
    if action.kind in {"ming_gang", "bu_gang"}:
        return 75.0
    if action.kind == "peng":
        return 65.0
    if action.kind == "chi":
        return 60.0
    if action.kind == "pass":
        return 0.0
    if action.kind == "discard":
        assert action.tile is not None
        return _discard_score(state, seat, action.tile)
    return -1.0


def _discard_score(state: GameState, seat: int, tile: int) -> float:
    """弃牌启发式分数，分数越高越推荐打出。"""
    hand = state.hands[seat]
    count = hand[tile]
    # 重复张通常更有价值，低分（不建议打）
    score = 10.0 - (count * 2.0)
    if not is_suited(tile):
        # 字牌默认略偏向先打
        score += 2.0
        return score

    rank = tile_rank(tile)
    assert rank is not None
    # 中张通常更有连接价值，降低打出倾向
    if rank in (4, 5, 6):
        score -= 2.0
    if rank in (1, 9):
        score += 1.0
    # 简单邻接评估
    neighbors = 0
    for delta in (-2, -1, 1, 2):
        adj_rank = rank + delta
        if not 1 <= adj_rank <= 9:
            continue
        adj_tile = tile + delta
        neighbors += hand[adj_tile]
    score -= min(neighbors, 4) * 0.8
    return score


def _action_label(action: Action) -> str:
    if action.kind in {"discard", "peng", "ming_gang", "an_gang", "bu_gang", "hu"} and action.tile is not None:
        return f"{action.kind}({index_to_tile(action.tile)})"
    if action.kind == "chi" and action.chi_start is not None:
        return f"chi({index_to_tile(action.chi_start)}-{index_to_tile(action.chi_start + 2)})"
    return action.kind


def _action_reason(action: Action) -> str:
    if action.kind == "hu":
        return "已满足和牌条件，优先胡牌。"
    if action.kind in {"an_gang", "ming_gang", "bu_gang"}:
        return "杠牌可获得额外收益并改善手牌结构。"
    if action.kind == "discard":
        return "该牌连接价值较低，优先作为弃牌。"
    if action.kind == "pass":
        return "当前不执行响应动作。"
    return "基于当前规则与手牌结构的启发式推荐。"


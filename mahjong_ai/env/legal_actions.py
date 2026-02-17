"""合法动作生成与响应裁决。"""

from __future__ import annotations

from typing import Iterable

from mahjong_ai.env.actions import Action, CHI_STARTS, action_to_id
from mahjong_ai.env.game_state import GameState, Meld
from mahjong_ai.rules.hand_checker import check_win
from mahjong_ai.rules.tiles import NUM_TILE_TYPES, is_suited, tile_rank


def legal_actions_for_turn(state: GameState, seat: int) -> set[Action]:
    """玩家自回合合法动作（不含 pass）。"""
    hand = state.hands[seat]
    laizi_idx = state.laizi_tile
    laizi_count = hand[laizi_idx] if laizi_idx is not None else 0
    actions: set[Action] = set()

    # 自摸胡
    if check_win(hand, laizi_idx=laizi_idx, enable_special=True).is_win:
        actions.add(Action("hu"))

    # 暗杠
    for tile in range(NUM_TILE_TYPES):
        if tile == laizi_idx:
            if hand[tile] >= 4:
                actions.add(Action("an_gang", tile=tile))
            continue
        if hand[tile] >= 4:
            actions.add(Action("an_gang", tile=tile))
            continue
        if laizi_idx is not None and hand[tile] + laizi_count >= 4 and hand[tile] > 0:
            actions.add(Action("an_gang", tile=tile))

    # 补杠：仅允许碰后自摸同张
    if state.last_drawn_tile is not None:
        for meld in state.melds[seat]:
            if meld.kind != "peng":
                continue
            tile = meld.tiles[0]
            if tile == state.last_drawn_tile and hand[tile] >= 1:
                actions.add(Action("bu_gang", tile=tile))

    # 打牌
    for tile in range(NUM_TILE_TYPES):
        if hand[tile] > 0:
            actions.add(Action("discard", tile=tile))

    return actions


def legal_reactions_to_discard(state: GameState, seat: int, discarder: int, tile: int) -> set[Action]:
    """弃牌响应窗口中某玩家可选动作（含 pass）。"""
    hand = state.hands[seat]
    laizi_idx = state.laizi_tile
    actions: set[Action] = {Action("pass")}

    # 胡（点胡）
    win_counts = list(hand)
    win_counts[tile] += 1
    if check_win(win_counts, laizi_idx=laizi_idx, enable_special=True).is_win:
        actions.add(Action("hu"))

    # 碰
    if _can_satisfy_requirements(hand, {tile: 2}, laizi_idx):
        actions.add(Action("peng", tile=tile))

    # 明杠
    if _can_satisfy_requirements(hand, {tile: 3}, laizi_idx):
        actions.add(Action("ming_gang", tile=tile))

    # 吃（仅上家）
    if seat == (discarder + 1) % state.num_players and is_suited(tile):
        rank = tile_rank(tile)
        assert rank is not None
        for start in _chi_starts_for_tile(tile):
            requirements = {}
            for t in (start, start + 1, start + 2):
                if t == tile:
                    continue
                requirements[t] = requirements.get(t, 0) + 1
            if _can_satisfy_requirements(hand, requirements, laizi_idx):
                actions.add(Action("chi", chi_start=start))

    return actions


def legal_action_ids(actions: Iterable[Action]) -> set[int]:
    """动作集合转 action_id 集合。"""
    return {action_to_id(action) for action in actions}


def resolve_reactions(discarder: int, claims: dict[int, Action], num_players: int = 4) -> tuple[int, Action] | None:
    """按规则裁决多家响应。"""
    ordered_seats = [(discarder + offset) % num_players for offset in range(1, num_players)]

    # 1) 胡优先
    for seat in ordered_seats:
        action = claims.get(seat)
        if action and action.kind == "hu":
            return seat, action

    # 2) 碰/杠优先
    for seat in ordered_seats:
        action = claims.get(seat)
        if action and action.kind in {"peng", "ming_gang"}:
            return seat, action

    # 3) 吃仅下家
    next_seat = (discarder + 1) % num_players
    next_action = claims.get(next_seat)
    if next_action and next_action.kind == "chi":
        return next_seat, next_action

    return None


def _chi_starts_for_tile(tile: int) -> list[int]:
    if not is_suited(tile):
        return []
    rank = tile_rank(tile)
    assert rank is not None
    starts: list[int] = []
    for delta in (-2, -1, 0):
        start_rank = rank + delta
        if 1 <= start_rank <= 7:
            start = tile - (rank - start_rank)
            if start in CHI_STARTS:
                starts.append(start)
    return starts


def _can_satisfy_requirements(hand: list[int], requirements: dict[int, int], laizi_idx: int | None) -> bool:
    """判断手牌是否可满足若干牌需求（考虑赖子替代且避免重复计数）。"""
    laizi_pool = hand[laizi_idx] if laizi_idx is not None else 0
    need_laizi = 0
    for tile, need in requirements.items():
        natural = hand[tile]
        if laizi_idx is not None and tile == laizi_idx:
            natural = 0
        deficit = max(0, need - natural)
        need_laizi += deficit
    return need_laizi <= laizi_pool

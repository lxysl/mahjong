"""状态转移与动作执行。"""

from __future__ import annotations

import random

from mahjong_ai.env.actions import Action
from mahjong_ai.env.game_state import GameState, Meld, PendingDiscard
from mahjong_ai.rules.hand_checker import check_win
from mahjong_ai.rules.laizi import indicator_to_laizi
from mahjong_ai.rules.scoring import (
    HandProgress,
    RewardConfig,
    dense_progress_reward,
    evaluate_hand_progress,
    immediate_gang_reward,
    terminal_rewards,
)
from mahjong_ai.rules.tiles import NUM_TILE_TYPES, is_suited, tile_rank


def build_wall(rng: random.Random) -> list[int]:
    wall = []
    for tile in range(NUM_TILE_TYPES):
        wall.extend([tile] * 4)
    rng.shuffle(wall)
    return wall


def initialize_round(seed: int | None = None, dealer: int = 0) -> GameState:
    """初始化一手牌，完成发牌与翻赖子指示牌。"""
    rng = random.Random(seed)
    state = GameState()
    state.dealer = dealer
    state.current_player = dealer
    state.wall = build_wall(rng)
    state.wall_head = 0
    state.wall_tail = len(state.wall) - 1

    # 发 13 张
    for _ in range(13):
        for seat in range(state.num_players):
            tile = draw_tile(state, seat, source="wall_head")
            state.last_drawn_tile = tile

    # 庄家补 1 张成为 14 张
    state.last_drawn_tile = draw_tile(state, dealer, source="wall_head")

    # 翻赖子指示牌（弃置，不参与后续）
    indicator = _pop_from_head(state)
    state.laizi_indicator = indicator
    state.laizi_tile = indicator_to_laizi(indicator)

    state.phase = "action"
    state.after_gang_draw = False
    state.pending_discard = None
    _initialize_progress_snapshots(state)
    return state


def draw_tile(state: GameState, seat: int, source: str = "wall_head") -> int:
    """摸牌（普通摸牌或杠尾补牌）。"""
    if state.wall_remaining() <= 0:
        raise ValueError("牌墙已空，无法继续摸牌")
    if source == "wall_head":
        tile = _pop_from_head(state)
    elif source == "wall_tail":
        tile = _pop_from_tail(state)
    else:
        raise ValueError(f"未知摸牌来源: {source}")
    state.hands[seat][tile] += 1
    state.last_drawn_tile = tile
    state.history.append({"type": "draw", "seat": seat, "tile": tile, "source": source})
    return tile


def draw_for_current_player(state: GameState) -> None:
    """执行当前玩家摸牌并进入行动阶段。"""
    if state.phase != "draw":
        return
    draw_tile(state, state.current_player, source="wall_head")
    state.phase = "action"
    state.after_gang_draw = False


def apply_turn_action(
    state: GameState, seat: int, action: Action, config: RewardConfig | None = None
) -> None:
    """执行玩家自回合动作。"""
    cfg = config or RewardConfig()
    if state.phase != "action":
        raise ValueError("当前不是 action 阶段")
    if seat != state.current_player:
        raise ValueError("非当前行动玩家")

    hand = state.hands[seat]
    laizi_idx = state.laizi_tile

    if action.kind == "discard":
        if action.tile is None or hand[action.tile] <= 0:
            raise ValueError("打牌非法：手牌不足")
        progress_before = _get_progress_snapshot(state, seat)
        hand[action.tile] -= 1
        progress_after = evaluate_hand_progress(list(hand), laizi_idx=laizi_idx, enable_special=True)
        allow_tenpai_enter = not state.tenpai_enter_reward_used.get(seat, False)
        if progress_before is not None:
            state.rewards[seat] += dense_progress_reward(
                progress_before,
                progress_after,
                cfg,
                allow_tenpai_enter=allow_tenpai_enter,
            )
            if allow_tenpai_enter and (not progress_before.tenpai) and progress_after.tenpai:
                state.tenpai_enter_reward_used[seat] = True
        elif progress_after.tenpai:
            # 无历史快照时不追溯发奖励，但标记首次听牌已占用，避免后续重复。
            state.tenpai_enter_reward_used[seat] = True
        _set_progress_snapshot(state, seat, progress_after)
        state.discards[seat].append(action.tile)
        state.pending_discard = PendingDiscard(discarder=seat, tile=action.tile)
        state.phase = "response"
        state.response_stage = "hu"
        state.response_order = [(seat + i) % state.num_players for i in range(1, state.num_players)]
        state.response_index = 0
        state.response_claims = {}
        state.current_player = state.response_order[0]
        state.after_gang_draw = False
        state.history.append({"type": "discard", "seat": seat, "tile": action.tile})
        return

    if action.kind == "hu":
        win_result = check_win(hand, laizi_idx=laizi_idx, enable_special=True)
        if not win_result.is_win:
            raise ValueError("自摸胡非法：当前手牌不可胡")
        state.winner = seat
        state.win_mode = "zi_mo"
        state.win_type = win_result.win_type
        state.gang_shang_hua = state.after_gang_draw
        settle = terminal_rewards(
            num_players=state.num_players,
            winner=seat,
            win_mode=state.win_mode,
            dealer=state.dealer,
            win_type=state.win_type,
            gang_shang_hua=state.gang_shang_hua,
            config=cfg,
        )
        _accumulate_rewards(state, settle)
        state.phase = "terminal"
        state.history.append({"type": "hu", "seat": seat, "mode": "zi_mo"})
        return

    if action.kind == "an_gang":
        if action.tile is None:
            raise ValueError("暗杠缺少牌参数")
        _remove_tiles_with_laizi(hand, action.tile, need=4, laizi_idx=laizi_idx)
        state.melds[seat].append(Meld(kind="an_gang", tiles=(action.tile,) * 4, from_seat=None))
        state.rewards[seat] += immediate_gang_reward("an_gang", cfg)
        if not _draw_after_gang_or_finish(state, seat):
            return
        state.after_gang_draw = True
        state.phase = "action"
        state.history.append({"type": "an_gang", "seat": seat, "tile": action.tile})
        return

    if action.kind == "bu_gang":
        if action.tile is None:
            raise ValueError("补杠缺少牌参数")
        if state.last_drawn_tile != action.tile:
            raise ValueError("补杠非法：仅允许碰后自摸同张")
        peng_idx = _find_peng_meld(state.melds[seat], action.tile)
        if peng_idx < 0 or hand[action.tile] <= 0:
            raise ValueError("补杠非法：不存在可升级碰，或手牌无对应牌")
        hand[action.tile] -= 1
        state.melds[seat][peng_idx] = Meld(kind="bu_gang", tiles=(action.tile,) * 4, from_seat=None)
        state.rewards[seat] += immediate_gang_reward("bu_gang", cfg)
        if not _draw_after_gang_or_finish(state, seat):
            return
        state.after_gang_draw = True
        state.phase = "action"
        state.history.append({"type": "bu_gang", "seat": seat, "tile": action.tile})
        return

    raise ValueError(f"动作 {action.kind} 不是自回合动作")


def apply_resolved_reaction(
    state: GameState,
    seat: int | None,
    action: Action | None,
    config: RewardConfig | None = None,
) -> None:
    """应用弃牌响应裁决结果。"""
    cfg = config or RewardConfig()
    if state.phase != "response":
        raise ValueError("当前不是 response 阶段")
    if state.pending_discard is None:
        raise ValueError("缺少待响应弃牌")

    pending = state.pending_discard
    discarder = pending.discarder
    tile = pending.tile
    laizi_idx = state.laizi_tile

    if seat is None or action is None or action.kind == "pass":
        state.pending_discard = None
        state.response_stage = None
        state.response_order = []
        state.response_index = 0
        state.response_claims = {}
        state.current_player = (discarder + 1) % state.num_players
        if state.wall_remaining() <= 0:
            state.phase = "terminal"
            state.history.append({"type": "draw_game"})
        else:
            state.phase = "draw"
        return

    hand = state.hands[seat]

    if action.kind == "hu":
        win_counts = list(hand)
        win_counts[tile] += 1
        win_result = check_win(win_counts, laizi_idx=laizi_idx, enable_special=True)
        if not win_result.is_win:
            raise ValueError("点胡非法：无法和牌")
        state.winner = seat
        state.win_mode = "dian_hu"
        state.win_type = win_result.win_type
        settle = terminal_rewards(
            num_players=state.num_players,
            winner=seat,
            win_mode=state.win_mode,
            dealer=state.dealer,
            win_type=state.win_type,
            gang_shang_hua=False,
            config=cfg,
        )
        _accumulate_rewards(state, settle)
        state.phase = "terminal"
        state.pending_discard = None
        state.response_stage = None
        state.response_order = []
        state.response_index = 0
        state.response_claims = {}
        state.history.append({"type": "hu", "seat": seat, "mode": "dian_hu", "tile": tile})
        return

    if action.kind == "peng":
        _remove_tiles_with_laizi(hand, tile, need=2, laizi_idx=laizi_idx)
        state.melds[seat].append(Meld(kind="peng", tiles=(tile, tile, tile), from_seat=discarder))
        state.pending_discard = None
        state.response_stage = None
        state.response_order = []
        state.response_index = 0
        state.response_claims = {}
        state.current_player = seat
        state.phase = "action"
        state.history.append({"type": "peng", "seat": seat, "tile": tile, "from": discarder})
        return

    if action.kind == "ming_gang":
        _remove_tiles_with_laizi(hand, tile, need=3, laizi_idx=laizi_idx)
        state.melds[seat].append(Meld(kind="ming_gang", tiles=(tile,) * 4, from_seat=discarder))
        state.rewards[seat] += immediate_gang_reward("ming_gang", cfg)
        state.pending_discard = None
        state.response_stage = None
        state.response_order = []
        state.response_index = 0
        state.response_claims = {}
        state.current_player = seat
        if not _draw_after_gang_or_finish(state, seat):
            return
        state.phase = "action"
        state.after_gang_draw = True
        state.history.append({"type": "ming_gang", "seat": seat, "tile": tile, "from": discarder})
        return

    if action.kind == "chi":
        if action.chi_start is None:
            raise ValueError("吃牌缺少 chi_start")
        if seat != (discarder + 1) % state.num_players:
            raise ValueError("吃牌非法：仅下家可吃")
        _apply_chi(hand, action.chi_start, tile, laizi_idx)
        state.melds[seat].append(
            Meld(kind="chi", tiles=(action.chi_start, action.chi_start + 1, action.chi_start + 2), from_seat=discarder)
        )
        state.pending_discard = None
        state.response_stage = None
        state.response_order = []
        state.response_index = 0
        state.response_claims = {}
        state.current_player = seat
        state.phase = "action"
        state.history.append(
            {"type": "chi", "seat": seat, "chi_start": action.chi_start, "tile": tile, "from": discarder}
        )
        return

    raise ValueError(f"未知响应动作: {action.kind}")


def _apply_chi(hand: list[int], chi_start: int, discarded_tile: int, laizi_idx: int | None) -> None:
    tiles = [chi_start, chi_start + 1, chi_start + 2]
    if discarded_tile not in tiles:
        raise ValueError("吃牌非法：弃牌不在顺子内")
    if not is_suited(chi_start) or not is_suited(chi_start + 2):
        raise ValueError("吃牌非法：必须是序数牌")
    if tile_rank(chi_start) is None or tile_rank(chi_start) > 7:
        raise ValueError("吃牌非法：起点越界")

    for t in tiles:
        if t == discarded_tile:
            continue
        _remove_tiles_with_laizi(hand, t, need=1, laizi_idx=laizi_idx)


def _initialize_progress_snapshots(state: GameState) -> None:
    """初始化每个座位的手牌进度快照，用于增量奖励差分。"""
    laizi_idx = state.laizi_tile
    state.progress_snapshot = {seat: None for seat in range(state.num_players)}
    state.tenpai_enter_reward_used = {seat: False for seat in range(state.num_players)}

    for seat in range(state.num_players):
        hand = state.hands[seat]

        # 开局庄家 14 张，快照应对应“上一次落地的 13 张”状态。
        if seat == state.dealer and state.last_drawn_tile is not None and hand[state.last_drawn_tile] > 0:
            hand[state.last_drawn_tile] -= 1
            progress = evaluate_hand_progress(list(hand), laizi_idx=laizi_idx, enable_special=True)
            hand[state.last_drawn_tile] += 1
        else:
            progress = evaluate_hand_progress(list(hand), laizi_idx=laizi_idx, enable_special=True)

        _set_progress_snapshot(state, seat, progress)
        if progress.tenpai:
            state.tenpai_enter_reward_used[seat] = True


def _get_progress_snapshot(state: GameState, seat: int) -> HandProgress | None:
    snapshot = state.progress_snapshot.get(seat)
    if snapshot is None:
        return None
    tenpai, outs = snapshot
    return HandProgress(tenpai=tenpai, outs=outs)


def _set_progress_snapshot(state: GameState, seat: int, progress: HandProgress) -> None:
    state.progress_snapshot[seat] = (progress.tenpai, progress.outs)


def _remove_tiles_with_laizi(hand: list[int], tile: int, need: int, laizi_idx: int | None) -> None:
    natural = hand[tile]
    if laizi_idx is not None and tile == laizi_idx:
        # 赖子牌统一从赖子池消耗，避免既当本牌又当万能牌的重复计数
        natural = 0
    real_use = min(natural, need)
    hand[tile] -= real_use
    remain = need - real_use
    if remain == 0:
        return
    if laizi_idx is None or hand[laizi_idx] < remain:
        raise ValueError("手牌不足，且赖子不足以补齐")
    hand[laizi_idx] -= remain


def _find_peng_meld(melds: list[Meld], tile: int) -> int:
    for idx, meld in enumerate(melds):
        if meld.kind == "peng" and meld.tiles[0] == tile:
            return idx
    return -1


def _accumulate_rewards(state: GameState, delta: dict[int, float]) -> None:
    for seat, value in delta.items():
        state.rewards[seat] = state.rewards.get(seat, 0) + value


def _draw_after_gang_or_finish(state: GameState, seat: int) -> bool:
    """开杠后补牌；若牌墙已空则直接流局终局。"""
    if state.wall_remaining() <= 0:
        state.phase = "terminal"
        state.pending_discard = None
        state.response_stage = None
        state.response_order = []
        state.response_index = 0
        state.response_claims = {}
        state.history.append({"type": "draw_game"})
        return False
    draw_tile(state, seat, source="wall_tail")
    return True


def _pop_from_head(state: GameState) -> int:
    if state.wall_head > state.wall_tail:
        raise ValueError("牌墙已空")
    tile = state.wall[state.wall_head]
    state.wall_head += 1
    return tile


def _pop_from_tail(state: GameState) -> int:
    if state.wall_head > state.wall_tail:
        raise ValueError("牌墙已空")
    tile = state.wall[state.wall_tail]
    state.wall_tail -= 1
    return tile

"""奖励结算模块。"""

from __future__ import annotations

from dataclasses import dataclass

from mahjong_ai.rules.hand_checker import check_win
from mahjong_ai.rules.tiles import NUM_TILE_TYPES


@dataclass(frozen=True)
class RewardConfig:
    """奖励配置。

    说明：庄家胡牌与七对不额外加分，规则固定。
    """

    dian_hu_reward: int = 1
    zi_mo_reward: int = 2
    ming_gang_reward: float = 0.2
    an_gang_reward: float = 0.4
    loser_penalty: int = -1

    # 稠密奖励（增量口径）
    tenpai_enter_reward: float = 0.05
    outs_delta_reward: float = 0.005
    outs_delta_clip: float = 0.05


def immediate_gang_reward(action_type: str, config: RewardConfig | None = None) -> float:
    """动作发生时即时奖励（仅杠牌）。"""
    cfg = config or RewardConfig()
    if action_type == "ming_gang":
        return cfg.ming_gang_reward
    if action_type == "an_gang":
        return cfg.an_gang_reward
    if action_type == "bu_gang":
        # 补杠默认按明杠处理（可按本地规则调整）
        return cfg.ming_gang_reward
    return 0


def terminal_rewards(
    num_players: int,
    winner: int | None,
    win_mode: str | None,
    dealer: int,
    win_type: str | None = None,
    gang_shang_hua: bool = False,
    config: RewardConfig | None = None,
) -> dict[int, float]:
    """终局奖励结算。

    Args:
        num_players: 玩家数量，当前固定 4。
        winner: 胡牌玩家；流局时为 None。
        win_mode: "dian_hu" / "zi_mo" / None。
        dealer: 庄家座位。
        win_type: "seven_pairs" 等胡型名称。
        gang_shang_hua: 是否杠上开花。
    """
    cfg = config or RewardConfig()
    rewards: dict[int, float] = {seat: 0.0 for seat in range(num_players)}
    if winner is None:
        return rewards

    if win_mode == "dian_hu":
        rewards[winner] += cfg.dian_hu_reward
    elif win_mode == "zi_mo":
        rewards[winner] += cfg.zi_mo_reward
    else:
        raise ValueError(f"未知胡牌类型: {win_mode}")

    if gang_shang_hua:
        rewards[winner] += cfg.an_gang_reward

    for seat in range(num_players):
        if seat != winner:
            rewards[seat] += cfg.loser_penalty
    return rewards


@dataclass(frozen=True)
class HandProgress:
    tenpai: bool
    outs: int


def evaluate_hand_progress(
    hand_counts: list[int],
    laizi_idx: int | None,
    enable_special: bool = True,
) -> HandProgress:
    """评估手牌进度：是否听牌 + 可胡张数（牌种数）。"""
    outs = winning_tiles(hand_counts, laizi_idx=laizi_idx, enable_special=enable_special)
    return HandProgress(tenpai=len(outs) > 0, outs=len(outs))


def dense_progress_reward(
    before: HandProgress,
    after: HandProgress,
    config: RewardConfig | None = None,
    allow_tenpai_enter: bool = True,
) -> float:
    """基于手牌进度增量计算稠密奖励。"""
    cfg = config or RewardConfig()
    reward = 0.0
    # 仅在“未听牌 -> 听牌”且允许发放时奖励
    if allow_tenpai_enter and (not before.tenpai) and after.tenpai:
        reward += cfg.tenpai_enter_reward
    delta_outs = after.outs - before.outs
    delta_reward = delta_outs * cfg.outs_delta_reward
    if delta_reward > cfg.outs_delta_clip:
        delta_reward = cfg.outs_delta_clip
    elif delta_reward < -cfg.outs_delta_clip:
        delta_reward = -cfg.outs_delta_clip
    reward += delta_reward
    return reward


def winning_tiles(
    hand_counts: list[int],
    laizi_idx: int | None = None,
    enable_special: bool = True,
    max_copies: int = 4,
) -> set[int]:
    """返回能让当前手牌和牌的摸牌集合（按牌种）。"""
    if len(hand_counts) != NUM_TILE_TYPES:
        raise ValueError(f"手牌计数长度必须为 {NUM_TILE_TYPES}")
    work = list(hand_counts)
    outs: set[int] = set()
    for tile in range(NUM_TILE_TYPES):
        if work[tile] >= max_copies:
            continue
        work[tile] += 1
        if check_win(work, laizi_idx=laizi_idx, enable_special=enable_special).is_win:
            outs.add(tile)
        work[tile] -= 1
    return outs

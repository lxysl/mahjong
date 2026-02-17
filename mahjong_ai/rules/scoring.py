"""奖励结算模块。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """奖励配置。

    说明：庄家胡牌与七对不额外加分，规则固定。
    """

    dian_hu_reward: int = 1
    zi_mo_reward: int = 2
    ming_gang_reward: int = 1
    an_gang_reward: int = 2
    loser_penalty: int = -1


def immediate_gang_reward(action_type: str, config: RewardConfig | None = None) -> int:
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
) -> dict[int, int]:
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
    rewards = {seat: 0 for seat in range(num_players)}
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

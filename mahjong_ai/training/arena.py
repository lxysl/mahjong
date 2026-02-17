"""策略对战评估。"""

from __future__ import annotations

from dataclasses import dataclass

from mahjong_ai.agent.policy import Policy
from mahjong_ai.env.self_play_env import SelfPlayEnv
from mahjong_ai.training.rollout import run_episode


@dataclass
class ArenaReport:
    episodes: int
    win_counts: dict[int, int]
    avg_reward: dict[int, float]


def evaluate(
    policies: dict[int, Policy],
    episodes: int = 100,
    seed: int = 42,
) -> ArenaReport:
    win_counts = {seat: 0 for seat in range(4)}
    sum_reward = {seat: 0.0 for seat in range(4)}

    for i in range(episodes):
        env = SelfPlayEnv(seed=seed + i, dealer=i % 4)
        result = run_episode(env, policies)
        if result.winner is not None:
            win_counts[result.winner] += 1
        for seat in range(4):
            sum_reward[seat] += result.rewards.get(seat, 0)

    avg_reward = {seat: sum_reward[seat] / episodes for seat in range(4)}
    return ArenaReport(episodes=episodes, win_counts=win_counts, avg_reward=avg_reward)


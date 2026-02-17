"""自博弈采样。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mahjong_ai.agent.policy import Policy
from mahjong_ai.env.self_play_env import SelfPlayEnv


@dataclass
class EpisodeResult:
    steps: int
    winner: int | None
    rewards: dict[int, float]
    transitions: list[dict[str, Any]]


def run_episode(env: SelfPlayEnv, policies: dict[int, Policy], max_steps: int = 2000) -> EpisodeResult:
    transitions: list[dict[str, Any]] = []
    obs = env.observe()
    for step_idx in range(max_steps):
        if env.state.phase == "terminal":
            return EpisodeResult(
                steps=step_idx,
                winner=env.state.winner,
                rewards=dict(env.state.rewards),
                transitions=transitions,
            )
        seat = env.state.current_player
        legal = env.legal_actions()
        policy = policies[seat]
        action = policy.select_action(env.state, seat, legal)
        result = env.step(action)
        transitions.append(
            {
                "seat": seat,
                "observation": obs,
                "action": action,
                "reward": result.reward,
                "done": result.done,
                "info": result.info,
            }
        )
        obs = result.observation

    return EpisodeResult(
        steps=max_steps,
        winner=env.state.winner,
        rewards=dict(env.state.rewards),
        transitions=transitions,
    )

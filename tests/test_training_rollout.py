from mahjong_ai.agent.policy import HeuristicPolicy, RandomPolicy
from mahjong_ai.env.self_play_env import SelfPlayEnv
from mahjong_ai.training.arena import evaluate
from mahjong_ai.training.rollout import run_episode


def test_run_episode_returns_result():
    env = SelfPlayEnv(seed=1, dealer=0)
    policies = {seat: RandomPolicy(seed=seat) for seat in range(4)}
    result = run_episode(env, policies, max_steps=200)
    assert result.steps <= 200
    assert isinstance(result.rewards, dict)


def test_arena_evaluate_runs():
    policies = {
        0: HeuristicPolicy(),
        1: RandomPolicy(seed=11),
        2: RandomPolicy(seed=22),
        3: RandomPolicy(seed=33),
    }
    report = evaluate(policies, episodes=3, seed=100)
    assert report.episodes == 3
    assert len(report.avg_reward) == 4


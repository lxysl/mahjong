from mahjong_ai.agent.policy import RandomPolicy
from mahjong_ai.env.self_play_env import SelfPlayEnv


def test_env_reset_basic_shape():
    env = SelfPlayEnv(seed=123, dealer=0)
    obs = env.reset(seed=123, dealer=0)
    assert obs["phase"] == "action"
    assert len(obs["hand_counts"]) == 34
    assert len(obs["legal_action_mask"]) == env.action_space_size
    hand_sizes = [sum(env.state.hands[seat]) for seat in range(4)]
    assert hand_sizes[0] == 14
    assert hand_sizes[1] == 13
    assert hand_sizes[2] == 13
    assert hand_sizes[3] == 13
    assert env.state.wall_remaining() == 82


def test_env_can_run_random_steps():
    env = SelfPlayEnv(seed=999, dealer=1)
    policy = RandomPolicy(seed=7)
    max_steps = 300
    for _ in range(max_steps):
        if env.state.phase == "terminal":
            break
        legal = env.legal_actions()
        action = policy.select_action(env.state, env.state.current_player, legal)
        result = env.step(action)
        assert len(result.observation["legal_action_mask"]) == env.action_space_size
    assert env.state.phase in {"terminal", "action", "response", "draw"}


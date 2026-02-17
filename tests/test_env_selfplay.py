from mahjong_ai.env.actions import Action
from mahjong_ai.env.game_state import GameState, PendingDiscard
from mahjong_ai.agent.policy import RandomPolicy
from mahjong_ai.env.self_play_env import SelfPlayEnv
from mahjong_ai.rules.tiles import make_counts, tile_to_index


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


def test_response_hu_should_settle_immediately_for_claiming_player():
    env = SelfPlayEnv(seed=1, dealer=0)
    state = GameState()
    state.phase = "response"
    state.current_player = 1
    state.dealer = 0
    state.pending_discard = PendingDiscard(discarder=0, tile=tile_to_index("2s"))
    state.response_stage = "hu"
    state.response_order = [1, 2, 3]
    state.response_index = 0
    state.response_claims = {}
    state.hands[1] = make_counts(["1m", "1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "2p", "3p", "4p", "2s"])

    env.state = state
    env._last_rewards = dict(state.rewards)

    assert Action("hu") in env.legal_actions()
    result = env.step(Action("hu"))
    assert result.done is True
    assert result.reward == 1
    assert result.info["winner"] == 1
    assert result.info["win_mode"] == "dian_hu"

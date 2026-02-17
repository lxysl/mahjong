from mahjong_ai.agent.observation_encoder import ObservationEncoder
from mahjong_ai.env.self_play_env import SelfPlayEnv


def test_encode_single_shapes():
    env = SelfPlayEnv(seed=123, dealer=0)
    obs = env.observe()
    encoder = ObservationEncoder()
    encoded = encoder.encode_single(obs)
    assert encoded["token_type_ids"].shape[0] == 128
    assert encoded["tile_ids"].shape[0] == 128
    assert encoded["value_ids"].shape[0] == 128
    assert encoded["attention_mask"].shape[0] == 128
    assert encoded["legal_action_mask"].shape[0] == env.action_space_size


def test_encode_batch_shapes():
    env = SelfPlayEnv(seed=456, dealer=1)
    encoder = ObservationEncoder()
    batch = encoder.encode_batch([env.observe(), env.observe()])
    assert batch["token_type_ids"].shape == (2, 128)
    assert batch["tile_ids"].shape == (2, 128)
    assert batch["value_ids"].shape == (2, 128)
    assert batch["attention_mask"].shape == (2, 128)
    assert batch["legal_action_mask"].shape[0] == 2


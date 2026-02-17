import torch

from mahjong_ai.agent.network import TransformerConfig, TransformerPolicyValueNet
from mahjong_ai.agent.observation_encoder import ObservationEncoder
from mahjong_ai.env.self_play_env import SelfPlayEnv


def test_transformer_forward_shape():
    env = SelfPlayEnv(seed=7, dealer=0)
    encoder = ObservationEncoder()
    net = TransformerPolicyValueNet(
        encoder=encoder,
        config=TransformerConfig(d_model=64, num_heads=4, num_layers=2, ffn_dim=128, dropout=0.0),
    )
    batch = encoder.encode_batch([env.observe(), env.observe()])
    out = net(batch)
    assert out.logits.shape == (2, env.action_space_size)
    assert out.value.shape == (2,)


def test_action_masking_blocks_illegal():
    env = SelfPlayEnv(seed=8, dealer=0)
    encoder = ObservationEncoder()
    net = TransformerPolicyValueNet(
        encoder=encoder,
        config=TransformerConfig(d_model=64, num_heads=4, num_layers=2, ffn_dim=128, dropout=0.0),
    )
    batch = encoder.encode_batch([env.observe()])
    out = net(batch)
    mask = batch["legal_action_mask"].clone()
    # 强制屏蔽所有动作，仅开放 action 0
    mask.zero_()
    mask[0, 0] = True
    masked = net.apply_action_mask(out.logits, mask)
    probs = torch.softmax(masked, dim=-1)
    assert torch.argmax(probs, dim=-1).item() == 0


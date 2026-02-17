import torch

from mahjong_ai.training.ppo_trainer import PPOConfig, PPOTrainer


def test_ppo_trainer_smoke():
    cfg = PPOConfig(
        total_steps=16,
        rollout_steps=8,
        mini_batch_size=4,
        epochs=1,
        learning_rate=1e-3,
        checkpoint_interval=0,
        model_d_model=64,
        model_num_heads=4,
        model_num_layers=2,
        model_ffn_dim=128,
        model_dropout=0.0,
    )
    trainer = PPOTrainer(config=cfg)
    logs = trainer.train()
    assert len(logs) == 2
    assert "loss" in logs[-1]
    assert trainer.global_step >= 16


def test_ppo_trainer_multi_env_rollout_smoke():
    cfg = PPOConfig(
        total_steps=32,
        rollout_steps=8,
        num_envs_per_worker=2,
        mini_batch_size=8,
        epochs=1,
        learning_rate=1e-3,
        checkpoint_interval=0,
        model_d_model=64,
        model_num_heads=4,
        model_num_layers=2,
        model_ffn_dim=128,
        model_dropout=0.0,
    )
    trainer = PPOTrainer(config=cfg)
    logs = trainer.train()
    assert len(logs) == 2
    assert logs[-1]["rollout/num_envs_per_worker"] == 2.0
    assert trainer.global_step >= 32


def test_pending_reward_flush_assigns_offturn_delta_to_latest_turn():
    rewards_t = [torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0])]
    pending_rewards = [{0: -1.0, 1: 0.0, 2: 0.0, 3: 0.0}]
    last_transition_step_by_env = [{0: 1, 1: 2, 2: None, 3: None}]

    PPOTrainer._flush_pending_rewards_to_last_transitions(
        env_idx=0,
        pending_rewards=pending_rewards,
        last_transition_step_by_env=last_transition_step_by_env,
        rewards_t=rewards_t,
    )

    assert rewards_t[1][0].item() == -1.0
    assert pending_rewards[0][0] == 0.0

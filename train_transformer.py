"""Transformer + PPO 训练入口。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from mahjong_ai.training.ppo_trainer import PPOConfig, PPOTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练麻将 Transformer 策略模型")
    parser.add_argument("--total-steps", type=int, default=20000)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--mini-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dealer", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="mahjong-ai")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--ddp-backend", type=str, default="", choices=["", "nccl", "gloo"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tags = tuple(tag.strip() for tag in args.wandb_tags.split(",") if tag.strip())
    rank = int(os.environ.get("RANK", "0"))
    is_main = rank == 0
    cfg = PPOConfig(
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        mini_batch_size=args.mini_batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        dealer=args.dealer,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
        wandb_tags=tags,
        ddp_backend=args.ddp_backend,
        model_d_model=args.d_model,
        model_num_heads=args.num_heads,
        model_num_layers=args.num_layers,
        model_ffn_dim=args.ffn_dim,
        model_dropout=args.dropout,
    )
    trainer = PPOTrainer(config=cfg)
    logs = trainer.train()
    if logs and is_main:
        last = logs[-1]
        print("training finished", flush=True)
        print(
            f"update={int(last['update'])} step={int(last['global_step'])} "
            f"loss={last['loss']:.4f} policy={last['policy_loss']:.4f} value={last['value_loss']:.4f}",
            flush=True,
        )
    latest_ckpt = Path(cfg.checkpoint_dir) / "ppo_latest.pt"
    if is_main:
        print(f"latest checkpoint: {latest_ckpt}", flush=True)


if __name__ == "__main__":
    main()

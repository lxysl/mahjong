"""PPO 训练器（PyTorch + Transformer）。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from mahjong_ai.agent.network import TransformerConfig, TransformerPolicyValueNet
from mahjong_ai.agent.observation_encoder import ObservationEncoder
from mahjong_ai.env.self_play_env import SelfPlayEnv


@dataclass
class PPOConfig:
    total_steps: int = 200_000
    rollout_steps: int = 2_048
    mini_batch_size: int = 256
    epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    seed: int = 42
    dealer: int = 0
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 20
    log_interval: int = 1

    use_wandb: bool = False
    wandb_project: str = "mahjong-ai"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str = "online"
    wandb_tags: tuple[str, ...] = ()

    model_d_model: int = 256
    model_num_heads: int = 8
    model_num_layers: int = 16
    model_ffn_dim: int = 1024
    model_dropout: float = 0.1


class PPOTrainer:
    """自博弈 PPO 训练器。"""

    def __init__(
        self,
        config: PPOConfig | None = None,
        encoder: ObservationEncoder | None = None,
        model: TransformerPolicyValueNet | None = None,
    ) -> None:
        self.config = config or PPOConfig()
        self.device = self._resolve_device(self.config.device)
        self.encoder = encoder or ObservationEncoder()
        self.model = model or TransformerPolicyValueNet(
            encoder=self.encoder,
            config=TransformerConfig(
                d_model=self.config.model_d_model,
                num_heads=self.config.model_num_heads,
                num_layers=self.config.model_num_layers,
                ffn_dim=self.config.model_ffn_dim,
                dropout=self.config.model_dropout,
            ),
        )
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        self.global_step = 0
        self.update_count = 0
        self._seed_cursor = self.config.seed + 1
        self._wandb_module = None
        self._wandb_run = None

    def train(self) -> list[dict[str, float]]:
        """启动训练循环，返回每次 update 的指标。"""
        self._init_wandb_if_needed()
        env = SelfPlayEnv(seed=self.config.seed, dealer=self.config.dealer)
        obs = env.observe()

        updates = max(1, self.config.total_steps // self.config.rollout_steps)
        logs: list[dict[str, float]] = []
        try:
            for update_idx in range(1, updates + 1):
                rollout_start = time.time()
                batch, obs, rollout_metrics = self.collect_rollout(env, obs)
                rollout_sec = time.time() - rollout_start

                update_start = time.time()
                metrics = self.update(batch)
                update_sec = time.time() - update_start

                metrics.update(rollout_metrics)
                metrics["timing/rollout_sec"] = float(rollout_sec)
                metrics["timing/update_sec"] = float(update_sec)
                metrics["timing/steps_per_sec"] = float(self.config.rollout_steps / max(rollout_sec, 1e-6))
                metrics["train/lr"] = float(self.optimizer.param_groups[0]["lr"])
                metrics["update"] = float(update_idx)
                metrics["global_step"] = float(self.global_step)

                logs.append(metrics)
                self.update_count += 1
                self._log_metrics(metrics)

                if self.config.log_interval > 0 and update_idx % self.config.log_interval == 0:
                    self._print_metrics(metrics)

                if self.config.checkpoint_interval > 0 and update_idx % self.config.checkpoint_interval == 0:
                    self.save_checkpoint(Path(self.config.checkpoint_dir) / f"ppo_update_{update_idx}.pt")

            self.save_checkpoint(Path(self.config.checkpoint_dir) / "ppo_latest.pt")
        finally:
            self._finish_wandb()
        return logs

    def collect_rollout(
        self, env: SelfPlayEnv, start_observation: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, float]]:
        obs = start_observation
        token_type_ids = []
        tile_ids = []
        value_ids = []
        attention_masks = []
        legal_action_masks = []
        actions = []
        log_probs = []
        values = []
        rewards = []
        dones = []

        for _ in range(self.config.rollout_steps):
            encoded = self.encoder.encode_single(obs)
            model_input = self._model_input_from_encoded(encoded)
            legal_mask = encoded["legal_action_mask"].to(self.device).unsqueeze(0)

            with torch.no_grad():
                output = self.model(model_input)
                masked_logits = self.model.apply_action_mask(output.logits, legal_mask)
                dist = Categorical(logits=masked_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                value = output.value

            step_result = env.step(int(action.item()))

            token_type_ids.append(encoded["token_type_ids"])
            tile_ids.append(encoded["tile_ids"])
            value_ids.append(encoded["value_ids"])
            attention_masks.append(encoded["attention_mask"])
            legal_action_masks.append(encoded["legal_action_mask"])
            actions.append(int(action.item()))
            log_probs.append(float(log_prob.item()))
            values.append(float(value.item()))
            rewards.append(float(step_result.reward))
            done = 1.0 if step_result.done else 0.0
            dones.append(done)
            self.global_step += 1

            obs = step_result.observation
            if step_result.done:
                obs = env.reset(seed=self._next_seed(), dealer=(env.dealer + 1) % 4)

        with torch.no_grad():
            encoded_last = self.encoder.encode_single(obs)
            last_input = self._model_input_from_encoded(encoded_last)
            last_value = float(self.model(last_input).value.item())

        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        values_t = torch.tensor(values, dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32)
        advantages, returns = self.compute_gae(rewards_t, values_t, dones_t, last_value)

        batch = {
            "token_type_ids": torch.stack(token_type_ids),
            "tile_ids": torch.stack(tile_ids),
            "value_ids": torch.stack(value_ids),
            "attention_mask": torch.stack(attention_masks),
            "legal_action_mask": torch.stack(legal_action_masks),
            "actions": torch.tensor(actions, dtype=torch.long),
            "old_log_probs": torch.tensor(log_probs, dtype=torch.float32),
            "advantages": advantages,
            "returns": returns,
        }
        rollout_metrics = {
            "rollout/avg_reward": float(rewards_t.mean().item()),
            "rollout/reward_std": float(rewards_t.std(unbiased=False).item()),
            "rollout/done_count": float(dones_t.sum().item()),
            "rollout/done_rate": float(dones_t.mean().item()),
            "rollout/avg_value_pred": float(values_t.mean().item()),
            "rollout/avg_advantage": float(advantages.mean().item()),
            "rollout/avg_return": float(returns.mean().item()),
            "rollout/illegal_action_count": 0.0,
        }
        return batch, obs, rollout_metrics

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        n = batch["actions"].shape[0]
        advantages = batch["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        batch["advantages"] = advantages

        metrics = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "explained_var": 0.0,
        }
        updates = 0

        for _ in range(self.config.epochs):
            indices = torch.randperm(n)
            for start in range(0, n, self.config.mini_batch_size):
                end = min(start + self.config.mini_batch_size, n)
                idx = indices[start:end]

                mb = self._batch_to_device(batch, idx)
                output = self.model(
                    {
                        "token_type_ids": mb["token_type_ids"],
                        "tile_ids": mb["tile_ids"],
                        "value_ids": mb["value_ids"],
                        "attention_mask": mb["attention_mask"],
                    }
                )
                logits = self.model.apply_action_mask(output.logits, mb["legal_action_mask"])
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb["actions"])
                entropy = dist.entropy().mean()

                ratio = (new_log_probs - mb["old_log_probs"]).exp()
                surr1 = ratio * mb["advantages"]
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * mb[
                    "advantages"
                ]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(output.value, mb["returns"])
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (mb["old_log_probs"] - new_log_probs).mean().item()
                    clip_fraction = ((ratio - 1.0).abs() > self.config.clip_ratio).float().mean().item()
                    var_y = torch.var(mb["returns"])
                    explained_var = 1.0 - (
                        torch.var(mb["returns"] - output.value) / (var_y + 1e-8)
                    ).item()
                metrics["loss"] += float(loss.item())
                metrics["policy_loss"] += float(policy_loss.item())
                metrics["value_loss"] += float(value_loss.item())
                metrics["entropy"] += float(entropy.item())
                metrics["approx_kl"] += float(approx_kl)
                metrics["clip_fraction"] += float(clip_fraction)
                metrics["explained_var"] += float(explained_var)
                updates += 1

        if updates > 0:
            for key in metrics:
                metrics[key] /= updates
        return metrics

    def compute_gae(
        self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, last_value: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(rewards.shape[0])):
            if t == rewards.shape[0] - 1:
                next_value = last_value
                next_non_terminal = 1.0 - dones[t].item()
            else:
                next_value = values[t + 1].item()
                next_non_terminal = 1.0 - dones[t].item()
            delta = rewards[t].item() + self.config.gamma * next_value * next_non_terminal - values[t].item()
            gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def save_checkpoint(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "global_step": self.global_step,
            "update_count": self.update_count,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(payload, p)

    def load_checkpoint(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location=self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.global_step = int(payload.get("global_step", 0))
        self.update_count = int(payload.get("update_count", 0))

    def _model_input_from_encoded(self, encoded: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            "token_type_ids": encoded["token_type_ids"].to(self.device).unsqueeze(0),
            "tile_ids": encoded["tile_ids"].to(self.device).unsqueeze(0),
            "value_ids": encoded["value_ids"].to(self.device).unsqueeze(0),
            "attention_mask": encoded["attention_mask"].to(self.device).unsqueeze(0),
        }

    def _batch_to_device(self, batch: dict[str, torch.Tensor], idx: torch.Tensor) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            out[key] = value[idx].to(self.device)
        return out

    def _next_seed(self) -> int:
        seed = self._seed_cursor
        self._seed_cursor += 1
        return seed

    def _init_wandb_if_needed(self) -> None:
        if not self.config.use_wandb:
            return
        import wandb

        self._wandb_module = wandb
        self._wandb_run = wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=self.config.wandb_run_name,
            mode=self.config.wandb_mode,
            tags=list(self.config.wandb_tags),
            config=asdict(self.config),
        )

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        if self._wandb_module is None:
            return
        self._wandb_module.log(metrics, step=int(metrics["global_step"]))

    def _finish_wandb(self) -> None:
        if self._wandb_run is None:
            return
        self._wandb_run.finish()
        self._wandb_run = None
        self._wandb_module = None

    def _print_metrics(self, metrics: dict[str, float]) -> None:
        print(
            "update={update} step={step} loss={loss:.4f} policy={policy:.4f} value={value:.4f} "
            "entropy={entropy:.4f} kl={kl:.4f} reward={reward:.4f} sps={sps:.1f}".format(
                update=int(metrics["update"]),
                step=int(metrics["global_step"]),
                loss=metrics["loss"],
                policy=metrics["policy_loss"],
                value=metrics["value_loss"],
                entropy=metrics["entropy"],
                kl=metrics["approx_kl"],
                reward=metrics["rollout/avg_reward"],
                sps=metrics["timing/steps_per_sec"],
            )
        )

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

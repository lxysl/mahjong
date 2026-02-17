"""PPO 训练器（PyTorch + Transformer）。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import time
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from torch.nn.parallel import DistributedDataParallel as DDP

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

    # 每个 worker 的并行环境数（每卡多环境并行 rollout）
    num_envs_per_worker: int = 1

    use_wandb: bool = False
    wandb_project: str = "mahjong-ai"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str = "online"
    wandb_tags: tuple[str, ...] = ()

    ddp_backend: str = ""

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
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.is_distributed = self.world_size > 1
        self.device = self._resolve_device(self.config.device, self.local_rank, self.is_distributed)
        self._init_distributed_if_needed()

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
        if self.is_distributed:
            if self.device.type == "cuda":
                self.model = DDP(
                    self.model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=False,
                )
            else:
                self.model = DDP(self.model, find_unused_parameters=False)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        self.global_step = 0
        self.update_count = 0
        self._seed_cursor = self.config.seed + 1 + self.rank * 1_000_000
        self._wandb_module = None
        self._wandb_run = None

    def train(self) -> list[dict[str, float]]:
        """启动训练循环，返回每次 update 的指标。"""
        self._init_wandb_if_needed()

        num_envs = max(1, int(self.config.num_envs_per_worker))
        envs: list[SelfPlayEnv] = []
        obs_list: list[dict[str, Any]] = []
        for env_idx in range(num_envs):
            seed = self.config.seed + self.rank * 1_000_000 + env_idx
            dealer = (self.config.dealer + self.rank + env_idx) % 4
            env = SelfPlayEnv(seed=seed, dealer=dealer)
            envs.append(env)
            obs_list.append(env.observe())

        steps_per_update_local = self.config.rollout_steps * num_envs
        updates = max(1, self.config.total_steps // steps_per_update_local)

        logs: list[dict[str, float]] = []
        if self.is_main_process:
            print(
                f"training start: updates={updates} rollout_steps={self.config.rollout_steps} "
                f"world_size={self.world_size} device={self.device}",
                flush=True,
            )
        try:
            for update_idx in range(1, updates + 1):
                rollout_start = time.time()
                batch, obs_list, rollout_metrics = self.collect_rollout(envs, obs_list)
                rollout_sec = time.time() - rollout_start

                update_start = time.time()
                metrics = self.update(batch)
                update_sec = time.time() - update_start

                metrics.update(rollout_metrics)
                metrics = self._sync_metrics(metrics)
                local_sps = float(steps_per_update_local / max(rollout_sec, 1e-6))
                metrics["timing/rollout_sec"] = float(rollout_sec)
                metrics["timing/update_sec"] = float(update_sec)
                metrics["timing/steps_per_sec_local"] = local_sps
                metrics["timing/steps_per_sec"] = float(local_sps * self.world_size)
                metrics["train/lr"] = float(self.optimizer.param_groups[0]["lr"])
                metrics["update"] = float(update_idx)
                metrics["global_step"] = float(self.global_step * self.world_size)
                metrics["ddp/world_size"] = float(self.world_size)
                metrics["rollout/num_envs_per_worker"] = float(num_envs)

                logs.append(metrics)
                self.update_count += 1
                self._log_metrics(metrics)

                if self.is_main_process and self.config.log_interval > 0 and update_idx % self.config.log_interval == 0:
                    self._print_metrics(metrics)

                if (
                    self.is_main_process
                    and self.config.checkpoint_interval > 0
                    and update_idx % self.config.checkpoint_interval == 0
                ):
                    self.save_checkpoint(Path(self.config.checkpoint_dir) / f"ppo_update_{update_idx}.pt")

            if self.is_main_process:
                self.save_checkpoint(Path(self.config.checkpoint_dir) / "ppo_latest.pt")
        finally:
            self._finish_wandb()
            self._cleanup_distributed()
        return logs

    def collect_rollout(
        self,
        envs: list[SelfPlayEnv],
        start_observations: list[dict[str, Any]],
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], dict[str, float]]:
        obs_list = list(start_observations)
        num_envs = len(envs)
        rollout_steps = self.config.rollout_steps

        token_type_ids_t = []
        tile_ids_t = []
        value_ids_t = []
        attention_masks_t = []
        legal_action_masks_t = []
        actions_t = []
        log_probs_t = []
        values_t = []
        rewards_t = []
        dones_t = []

        for _ in range(rollout_steps):
            encoded_batch = self.encoder.encode_batch(obs_list)
            model_input = self._model_input_from_batch(encoded_batch)
            legal_mask = encoded_batch["legal_action_mask"].to(self.device)

            with torch.no_grad():
                output = self.model(model_input)
                masked_logits = self._policy_model().apply_action_mask(output.logits, legal_mask)
                dist_policy = Categorical(logits=masked_logits)
                actions = dist_policy.sample()
                log_probs = dist_policy.log_prob(actions)
                values = output.value

            step_rewards = []
            step_dones = []
            next_obs_list = []
            for env_idx, env in enumerate(envs):
                step_result = env.step(int(actions[env_idx].item()))
                done = 1.0 if step_result.done else 0.0
                step_rewards.append(float(step_result.reward))
                step_dones.append(done)
                next_obs = step_result.observation
                if step_result.done:
                    next_obs = env.reset(seed=self._next_seed(), dealer=(env.dealer + 1) % 4)
                next_obs_list.append(next_obs)

            token_type_ids_t.append(encoded_batch["token_type_ids"])
            tile_ids_t.append(encoded_batch["tile_ids"])
            value_ids_t.append(encoded_batch["value_ids"])
            attention_masks_t.append(encoded_batch["attention_mask"])
            legal_action_masks_t.append(encoded_batch["legal_action_mask"])
            actions_t.append(actions.detach().cpu())
            log_probs_t.append(log_probs.detach().cpu())
            values_t.append(values.detach().cpu())
            rewards_t.append(torch.tensor(step_rewards, dtype=torch.float32))
            dones_t.append(torch.tensor(step_dones, dtype=torch.float32))
            self.global_step += num_envs

            obs_list = next_obs_list

        with torch.no_grad():
            encoded_last = self.encoder.encode_batch(obs_list)
            last_input = self._model_input_from_batch(encoded_last)
            last_values = self.model(last_input).value.detach().cpu()

        rewards = torch.stack(rewards_t, dim=0)  # [T, N]
        values = torch.stack(values_t, dim=0)  # [T, N]
        dones = torch.stack(dones_t, dim=0)  # [T, N]
        advantages, returns = self.compute_gae(rewards, values, dones, last_values)

        token_type_ids = torch.stack(token_type_ids_t, dim=0).reshape(rollout_steps * num_envs, -1)
        tile_ids = torch.stack(tile_ids_t, dim=0).reshape(rollout_steps * num_envs, -1)
        value_ids = torch.stack(value_ids_t, dim=0).reshape(rollout_steps * num_envs, -1)
        attention_mask = torch.stack(attention_masks_t, dim=0).reshape(rollout_steps * num_envs, -1)
        legal_action_mask = torch.stack(legal_action_masks_t, dim=0).reshape(rollout_steps * num_envs, -1)
        actions = torch.stack(actions_t, dim=0).reshape(-1)
        old_log_probs = torch.stack(log_probs_t, dim=0).reshape(-1)
        advantages_flat = advantages.reshape(-1)
        returns_flat = returns.reshape(-1)

        batch = {
            "token_type_ids": token_type_ids,
            "tile_ids": tile_ids,
            "value_ids": value_ids,
            "attention_mask": attention_mask,
            "legal_action_mask": legal_action_mask,
            "actions": actions.long(),
            "old_log_probs": old_log_probs.float(),
            "advantages": advantages_flat.float(),
            "returns": returns_flat.float(),
        }

        rollout_metrics = {
            "rollout/avg_reward": float(rewards.mean().item()),
            "rollout/reward_std": float(rewards.std(unbiased=False).item()),
            "rollout/done_count": float(dones.sum().item()),
            "rollout/done_rate": float(dones.mean().item()),
            "rollout/avg_value_pred": float(values.mean().item()),
            "rollout/avg_advantage": float(advantages.mean().item()),
            "rollout/avg_return": float(returns.mean().item()),
            "rollout/illegal_action_count": 0.0,
        }
        return batch, obs_list, rollout_metrics

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
                logits = self._policy_model().apply_action_mask(output.logits, mb["legal_action_mask"])
                dist_policy = Categorical(logits=logits)
                new_log_probs = dist_policy.log_prob(mb["actions"])
                entropy = dist_policy.entropy().mean()

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
                    explained_var = 1.0 - (torch.var(mb["returns"] - output.value) / (var_y + 1e-8)).item()
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
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """向量化 GAE，输入形状为 [T, N]。"""
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros_like(last_values)
        for t in reversed(range(rewards.shape[0])):
            if t == rewards.shape[0] - 1:
                next_values = last_values
            else:
                next_values = values[t + 1]
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.config.gamma * next_values * next_non_terminal - values[t]
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
            "model_state_dict": self._policy_model().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(payload, p)

    def load_checkpoint(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location=self.device)
        self._policy_model().load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.global_step = int(payload.get("global_step", 0))
        self.update_count = int(payload.get("update_count", 0))

    def _model_input_from_batch(self, encoded: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            "token_type_ids": encoded["token_type_ids"].to(self.device),
            "tile_ids": encoded["tile_ids"].to(self.device),
            "value_ids": encoded["value_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
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
        if not self.config.use_wandb or not self.is_main_process:
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
        if self._wandb_module is None or not self.is_main_process:
            return
        self._wandb_module.log(metrics, step=int(metrics["global_step"]))

    def _finish_wandb(self) -> None:
        if self._wandb_run is None:
            return
        self._wandb_run.finish()
        self._wandb_run = None
        self._wandb_module = None

    def _sync_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        if not self.is_distributed:
            return metrics
        synced: dict[str, float] = {}
        for key, value in metrics.items():
            tensor = torch.tensor(float(value), dtype=torch.float64, device=self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            if key.endswith("done_count"):
                synced[key] = float(tensor.item())
            else:
                synced[key] = float(tensor.item() / self.world_size)
        return synced

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
            ),
            flush=True,
        )

    @staticmethod
    def _resolve_device(device: str, local_rank: int, is_distributed: bool) -> torch.device:
        if device == "cuda" and torch.cuda.is_available():
            if is_distributed:
                torch.cuda.set_device(local_rank)
            return torch.device("cuda")
        return torch.device("cpu")

    def _init_distributed_if_needed(self) -> None:
        if not self.is_distributed:
            return
        if dist.is_initialized():
            return
        backend = self.config.ddp_backend
        if not backend:
            backend = "nccl" if self.device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    def _cleanup_distributed(self) -> None:
        if not self.is_distributed:
            return
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def _policy_model(self) -> TransformerPolicyValueNet:
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model


"""PyTorch Transformer 策略价值网络。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mahjong_ai.agent.observation_encoder import ObservationEncoder


@dataclass(frozen=True)
class TransformerConfig:
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 16
    ffn_dim: int = 1024
    dropout: float = 0.1


@dataclass(frozen=True)
class PolicyValueOutput:
    logits: torch.Tensor
    value: torch.Tensor


class TransformerPolicyValueNet(nn.Module):
    """16 层 Transformer 策略价值网络（默认配置）。"""

    def __init__(
        self,
        encoder: ObservationEncoder,
        config: TransformerConfig | None = None,
        action_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.config = config or TransformerConfig()
        self.action_dim = action_dim if action_dim is not None else encoder.action_dim
        self.seq_len = encoder.config.seq_len

        self.token_type_embedding = nn.Embedding(encoder.token_type_vocab_size, self.config.d_model)
        self.tile_embedding = nn.Embedding(encoder.tile_vocab_size, self.config.d_model)
        self.value_embedding = nn.Embedding(encoder.config.value_vocab_size, self.config.d_model)
        self.position_embedding = nn.Embedding(self.seq_len, self.config.d_model)
        self.input_norm = nn.LayerNorm(self.config.d_model)
        self.input_dropout = nn.Dropout(self.config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.ffn_dim,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.config.num_layers)
        self.output_norm = nn.LayerNorm(self.config.d_model)

        self.policy_head = nn.Sequential(
            nn.LayerNorm(self.config.d_model),
            nn.Linear(self.config.d_model, self.action_dim),
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(self.config.d_model),
            nn.Linear(self.config.d_model, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> PolicyValueOutput:
        token_type_ids = batch["token_type_ids"]
        tile_ids = batch["tile_ids"]
        value_ids = batch["value_ids"]
        attention_mask = batch["attention_mask"]

        if token_type_ids.dim() == 1:
            token_type_ids = token_type_ids.unsqueeze(0)
            tile_ids = tile_ids.unsqueeze(0)
            value_ids = value_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        bsz, seq_len = token_type_ids.shape
        if seq_len != self.seq_len:
            raise ValueError(f"输入序列长度错误: {seq_len} != {self.seq_len}")

        position_ids = torch.arange(self.seq_len, device=token_type_ids.device).unsqueeze(0).expand(bsz, -1)
        x = (
            self.token_type_embedding(token_type_ids)
            + self.tile_embedding(tile_ids)
            + self.value_embedding(value_ids)
            + self.position_embedding(position_ids)
        )
        x = self.input_norm(x)
        x = self.input_dropout(x)

        key_padding_mask = ~attention_mask.bool()
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        x = self.output_norm(x)

        pooled = masked_mean_pooling(x, attention_mask)
        logits = self.policy_head(pooled)
        value = self.value_head(pooled).squeeze(-1)
        return PolicyValueOutput(logits=logits, value=value)

    @staticmethod
    def apply_action_mask(logits: torch.Tensor, legal_action_mask: torch.Tensor) -> torch.Tensor:
        if legal_action_mask.dtype != torch.bool:
            legal_action_mask = legal_action_mask.bool()
        return logits.masked_fill(~legal_action_mask, -1e9)


def masked_mean_pooling(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.float().unsqueeze(-1)
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


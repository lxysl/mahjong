"""推理辅助函数。"""

from __future__ import annotations

import torch
from torch.distributions import Categorical

from mahjong_ai.env.actions import id_to_action


def masked_argmax(logits: list[float], legal_mask: list[int]) -> int:
    if len(logits) != len(legal_mask):
        raise ValueError("logits 与 mask 长度不一致")
    best_idx = -1
    best_val = float("-inf")
    for idx, (v, m) in enumerate(zip(logits, legal_mask)):
        if m == 0:
            continue
        if v > best_val:
            best_val = v
            best_idx = idx
    if best_idx < 0:
        raise ValueError("无合法动作")
    return best_idx


def action_from_logits(logits: list[float], legal_mask: list[int]):
    action_id = masked_argmax(logits, legal_mask)
    return id_to_action(action_id)


def masked_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    if legal_mask.dtype != torch.bool:
        legal_mask = legal_mask.bool()
    return logits.masked_fill(~legal_mask, -1e9)


def sample_action_id(logits: torch.Tensor, legal_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dist = Categorical(logits=masked_logits(logits, legal_mask))
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action, log_prob

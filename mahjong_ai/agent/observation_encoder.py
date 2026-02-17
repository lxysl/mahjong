"""观测编码器：将环境 observation 转为 Transformer 输入。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from mahjong_ai.env.actions import action_space_size
from mahjong_ai.rules.tiles import NUM_TILE_TYPES

PHASE_TO_ID = {"draw": 0, "action": 1, "response": 2, "terminal": 3}

TOKEN_TYPE_HAND = 0
TOKEN_TYPE_DISCARD = 1
TOKEN_TYPE_MELD = 2
TOKEN_TYPE_META = 3
TOKEN_TYPE_PAD = 4

TILE_META = NUM_TILE_TYPES
TILE_PAD = NUM_TILE_TYPES + 1


@dataclass(frozen=True)
class EncoderConfig:
    seq_len: int = 128
    value_vocab_size: int = 128


class ObservationEncoder:
    """把 observation 编码为固定长度 token 序列。"""

    def __init__(self, config: EncoderConfig | None = None) -> None:
        self.config = config or EncoderConfig()
        self._action_dim = action_space_size()

    @property
    def token_type_vocab_size(self) -> int:
        return 5

    @property
    def tile_vocab_size(self) -> int:
        return NUM_TILE_TYPES + 2

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def encode_single(self, observation: dict[str, Any]) -> dict[str, torch.Tensor]:
        token_types: list[int] = []
        tile_ids: list[int] = []
        value_ids: list[int] = []

        hand_counts = observation["hand_counts"]
        discard_counts = self._count_discards(observation["discards"])
        meld_counts = self._count_meld_tiles(observation["melds"])

        self._append_tile_group(token_types, tile_ids, value_ids, TOKEN_TYPE_HAND, hand_counts)
        self._append_tile_group(token_types, tile_ids, value_ids, TOKEN_TYPE_DISCARD, discard_counts)
        self._append_tile_group(token_types, tile_ids, value_ids, TOKEN_TYPE_MELD, meld_counts)
        self._append_meta_tokens(token_types, tile_ids, value_ids, observation)

        valid_len = len(token_types)
        if valid_len > self.config.seq_len:
            raise ValueError(f"序列长度超限: {valid_len} > {self.config.seq_len}")

        pad_len = self.config.seq_len - valid_len
        token_types.extend([TOKEN_TYPE_PAD] * pad_len)
        tile_ids.extend([TILE_PAD] * pad_len)
        value_ids.extend([0] * pad_len)

        attention_mask = [1] * valid_len + [0] * pad_len
        legal_action_mask = observation["legal_action_mask"]
        if len(legal_action_mask) != self._action_dim:
            raise ValueError("legal_action_mask 长度与动作空间不一致")

        return {
            "token_type_ids": torch.tensor(token_types, dtype=torch.long),
            "tile_ids": torch.tensor(tile_ids, dtype=torch.long),
            "value_ids": torch.tensor(value_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "legal_action_mask": torch.tensor(legal_action_mask, dtype=torch.bool),
        }

    def encode_batch(self, observations: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = [self.encode_single(obs) for obs in observations]
        keys = encoded[0].keys()
        batch: dict[str, torch.Tensor] = {}
        for key in keys:
            batch[key] = torch.stack([item[key] for item in encoded], dim=0)
        return batch

    def _append_tile_group(
        self,
        token_types: list[int],
        tile_ids: list[int],
        value_ids: list[int],
        token_type: int,
        counts: list[int],
    ) -> None:
        for tile in range(NUM_TILE_TYPES):
            token_types.append(token_type)
            tile_ids.append(tile)
            value_ids.append(self._clip_value(counts[tile]))

    def _append_meta_tokens(
        self,
        token_types: list[int],
        tile_ids: list[int],
        value_ids: list[int],
        observation: dict[str, Any],
    ) -> None:
        pending = observation.get("pending_discard")
        pending_tile = TILE_META
        pending_discarder = 4
        if pending is not None:
            pending_tile = int(pending["tile"])
            pending_discarder = int(pending["discarder"])

        laizi_tile = observation.get("laizi_tile")
        if laizi_tile is None:
            laizi_tile = TILE_META

        meta_values = [
            int(observation["seat"]),
            int(observation["current_player"]),
            int(observation["dealer"]),
            PHASE_TO_ID.get(observation["phase"], 0),
            int(observation["wall_remaining"]),
            int(pending_discarder),
        ]
        meta_tiles = [
            TILE_META,
            TILE_META,
            TILE_META,
            TILE_META,
            TILE_META,
            TILE_META,
            int(laizi_tile),
            int(pending_tile),
        ]
        # laizi_tile 与 pending_tile 这两个 meta token 的 value 设为 1（存在）或 0（不存在）
        meta_values.extend([1 if observation.get("laizi_tile") is not None else 0, 1 if pending is not None else 0])

        for t, v in zip(meta_tiles, meta_values):
            token_types.append(TOKEN_TYPE_META)
            tile_ids.append(t)
            value_ids.append(self._clip_value(v))

    def _count_discards(self, discards: list[list[int]]) -> list[int]:
        counts = [0] * NUM_TILE_TYPES
        for seat_discards in discards:
            for tile in seat_discards:
                counts[tile] += 1
        return counts

    def _count_meld_tiles(self, melds: list[list[dict[str, Any]]]) -> list[int]:
        counts = [0] * NUM_TILE_TYPES
        for seat_melds in melds:
            for meld in seat_melds:
                tiles = meld.get("tiles", [])
                for tile in tiles:
                    counts[int(tile)] += 1
        return counts

    def _clip_value(self, value: int) -> int:
        return max(0, min(int(value), self.config.value_vocab_size - 1))


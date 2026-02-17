"""赖子规则相关函数。"""

from __future__ import annotations

from mahjong_ai.rules.tiles import (
    DRAGONS,
    HONORS,
    NUM_TILE_TYPES,
    WIND_CODES,
    WINDS,
    is_suited,
    tile_rank,
    tile_suit,
    tile_to_index,
)

# 东南西北中发白
_HONOR_CYCLE = (27, 28, 29, 30, 31, 32, 33)


def indicator_to_laizi(indicator: int | str) -> int:
    """根据翻牌（赖子指示牌）计算本局赖子牌索引。"""
    idx = tile_to_index(indicator)
    if is_suited(idx):
        suit = tile_suit(idx)
        rank = tile_rank(idx)
        assert suit is not None and rank is not None
        next_rank = 1 if rank == 9 else rank + 1
        base = {"m": 0, "p": 9, "s": 18}[suit]
        return base + (next_rank - 1)

    pos = _HONOR_CYCLE.index(idx)
    return _HONOR_CYCLE[(pos + 1) % len(_HONOR_CYCLE)]


def pop_laizi_from_counts(counts: list[int], laizi_idx: int) -> int:
    """从计数中移除赖子并返回数量。"""
    if len(counts) != NUM_TILE_TYPES:
        raise ValueError("计数向量长度错误")
    laizi_count = counts[laizi_idx]
    counts[laizi_idx] = 0
    return laizi_count


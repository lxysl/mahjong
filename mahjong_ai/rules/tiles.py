"""麻将牌编码与转换工具。

采用 34 种基础牌索引：
- 0..8   : 1m..9m（万）
- 9..17  : 1p..9p（筒/饼）
- 18..26 : 1s..9s（索）
- 27..30 : E,S,W,N（东南西北）
- 31..33 : C,F,P（中发白）
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

SUIT_CODES = ("m", "p", "s")
WIND_CODES = ("E", "S", "W", "N")
DRAGON_CODES = ("C", "F", "P")

SUIT_TILE_CODES = tuple(f"{rank}{suit}" for suit in SUIT_CODES for rank in range(1, 10))
HONOR_TILE_CODES = WIND_CODES + DRAGON_CODES
ALL_TILE_CODES = SUIT_TILE_CODES + HONOR_TILE_CODES

TILE_CODE_TO_INDEX = {code: idx for idx, code in enumerate(ALL_TILE_CODES)}

SUIT_START = {"m": 0, "p": 9, "s": 18}
NUM_TILE_TYPES = 34

WINDS = tuple(range(27, 31))
DRAGONS = tuple(range(31, 34))
HONORS = tuple(range(27, 34))
TERMINALS = tuple([0, 8, 9, 17, 18, 26])


def tile_to_index(tile: int | str) -> int:
    """将牌编码（int/str）转为基础索引。"""
    if isinstance(tile, int):
        if 0 <= tile < NUM_TILE_TYPES:
            return tile
        raise ValueError(f"牌索引越界: {tile}")
    if tile in TILE_CODE_TO_INDEX:
        return TILE_CODE_TO_INDEX[tile]
    raise ValueError(f"未知牌编码: {tile}")


def index_to_tile(index: int) -> str:
    """将基础索引转为字符串编码。"""
    if 0 <= index < NUM_TILE_TYPES:
        return ALL_TILE_CODES[index]
    raise ValueError(f"牌索引越界: {index}")


def is_suited(index: int) -> bool:
    """是否为序数牌（万/筒/索）。"""
    return 0 <= index <= 26


def tile_suit(index: int) -> str | None:
    """返回牌花色；字牌返回 None。"""
    index = tile_to_index(index)
    if 0 <= index <= 8:
        return "m"
    if 9 <= index <= 17:
        return "p"
    if 18 <= index <= 26:
        return "s"
    return None


def tile_rank(index: int) -> int | None:
    """返回序数牌点数（1..9）；字牌返回 None。"""
    index = tile_to_index(index)
    if not is_suited(index):
        return None
    return (index % 9) + 1


def is_258(index: int) -> bool:
    """是否为 2/5/8 的序数牌。"""
    rank = tile_rank(index)
    return rank in (2, 5, 8)


def make_counts(tiles: Iterable[int | str]) -> List[int]:
    """将牌列表转为长度 34 的计数向量。"""
    counts = [0] * NUM_TILE_TYPES
    for tile in tiles:
        idx = tile_to_index(tile)
        counts[idx] += 1
    return counts


def counts_to_tiles(counts: Sequence[int]) -> List[int]:
    """将计数向量展开为索引列表。"""
    if len(counts) != NUM_TILE_TYPES:
        raise ValueError(f"计数长度必须为 {NUM_TILE_TYPES}，实际为 {len(counts)}")
    tiles: List[int] = []
    for idx, c in enumerate(counts):
        if c < 0:
            raise ValueError("计数不能为负数")
        tiles.extend([idx] * c)
    return tiles


def validate_counts(counts: Sequence[int], max_copies: int = 4) -> None:
    """校验计数向量合法性。"""
    if len(counts) != NUM_TILE_TYPES:
        raise ValueError(f"计数长度必须为 {NUM_TILE_TYPES}")
    for c in counts:
        if c < 0:
            raise ValueError("存在负计数")
        if c > max_copies:
            raise ValueError(f"单牌数量超限: {c} > {max_copies}")


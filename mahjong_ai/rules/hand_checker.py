"""胡牌判定模块。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
from typing import Callable, Sequence

from mahjong_ai.rules.laizi import pop_laizi_from_counts
from mahjong_ai.rules.tiles import (
    HONORS,
    NUM_TILE_TYPES,
    TERMINALS,
    is_258,
    is_suited,
    make_counts,
    tile_rank,
)

PAIR_258_INDICES = tuple(idx for idx in range(27) if is_258(idx))
ALL_INDICES = tuple(range(NUM_TILE_TYPES))
ORPHAN_INDICES = tuple(sorted(set(TERMINALS + HONORS)))

# 全不靠允许的三组点数
RANK_GROUPS = ((1, 4, 7), (2, 5, 8), (3, 6, 9))


@dataclass(frozen=True)
class WinResult:
    is_win: bool
    win_type: str | None = None
    detail: str | None = None


def check_win(
    counts: Sequence[int] | Sequence[str] | Sequence[int | str],
    laizi_idx: int | None = None,
    enable_special: bool = True,
) -> WinResult:
    """统一胡牌判定入口。"""
    vec = _normalize_counts(counts)
    total_tiles = sum(vec)
    if total_tiles % 3 != 2:
        return WinResult(False)

    if enable_special:
        if is_seven_pairs(vec, laizi_idx):
            return WinResult(True, "seven_pairs")
        if is_thirteen_orphans(vec, laizi_idx):
            return WinResult(True, "thirteen_orphans")
        if is_quan_bu_kao(vec, laizi_idx):
            return WinResult(True, "quan_bu_kao")
        if is_zuhe_long(vec, laizi_idx):
            return WinResult(True, "zuhe_long")

    if is_standard_win(vec, laizi_idx, pair_restrict_258=True):
        return WinResult(True, "standard")
    return WinResult(False)


def is_standard_win(
    counts: Sequence[int] | Sequence[str] | Sequence[int | str],
    laizi_idx: int | None = None,
    pair_restrict_258: bool = True,
) -> bool:
    """标准 4 面子 + 1 对将判定。"""
    vec = _normalize_counts(counts)
    total_tiles = sum(vec)
    if total_tiles % 3 != 2:
        return False

    work = list(vec)
    laizi_count = 0
    if laizi_idx is not None:
        laizi_count = pop_laizi_from_counts(work, laizi_idx)

    pair_candidates = PAIR_258_INDICES if pair_restrict_258 else ALL_INDICES
    for pair_idx in pair_candidates:
        need_pair = max(0, 2 - work[pair_idx])
        if need_pair > laizi_count:
            continue
        after = list(work)
        use_real = min(2, after[pair_idx])
        after[pair_idx] -= use_real
        if _can_form_all_melds(after, laizi_count - need_pair):
            return True

    return False


def is_seven_pairs(
    counts: Sequence[int] | Sequence[str] | Sequence[int | str], laizi_idx: int | None = None
) -> bool:
    """七对判定（赖子可配对）。"""
    vec = _normalize_counts(counts)
    if sum(vec) != 14:
        return False

    work = list(vec)
    laizi_count = 0
    if laizi_idx is not None:
        laizi_count = pop_laizi_from_counts(work, laizi_idx)

    singles = sum(c % 2 for c in work)
    if singles > laizi_count:
        return False
    pairs = sum(c // 2 for c in work)
    laizi_left = laizi_count - singles
    return pairs + singles + (laizi_left // 2) >= 7


def is_thirteen_orphans(
    counts: Sequence[int] | Sequence[str] | Sequence[int | str], laizi_idx: int | None = None
) -> bool:
    """十三幺判定（含赖子）。"""
    vec = _normalize_counts(counts)
    if sum(vec) != 14:
        return False

    work = list(vec)
    laizi_count = 0
    if laizi_idx is not None:
        laizi_count = pop_laizi_from_counts(work, laizi_idx)

    orphan_set = set(ORPHAN_INDICES)
    for idx, c in enumerate(work):
        if c and idx not in orphan_set:
            return False

    missing = sum(1 for idx in ORPHAN_INDICES if work[idx] == 0)
    if missing > laizi_count:
        return False

    laizi_left = laizi_count - missing
    has_pair = any(work[idx] >= 2 for idx in ORPHAN_INDICES)
    if has_pair:
        return True

    has_single = any(work[idx] >= 1 for idx in ORPHAN_INDICES)
    if has_single and laizi_left >= 1:
        return True

    return laizi_left >= 2


def is_quan_bu_kao(
    counts: Sequence[int] | Sequence[str] | Sequence[int | str], laizi_idx: int | None = None
) -> bool:
    """全不靠判定（14 张互不成靠，允许赖子补足）。"""
    vec = _normalize_counts(counts)
    if sum(vec) != 14:
        return False

    work = list(vec)
    laizi_count = 0
    if laizi_idx is not None:
        laizi_count = pop_laizi_from_counts(work, laizi_idx)

    # 非赖子部分若有重复，不满足全不靠“全单张”要求
    if any(c > 1 for c in work):
        return False

    used = {idx for idx, c in enumerate(work) if c}
    non_laizi_tiles = len(used)
    if non_laizi_tiles + laizi_count < 14:
        return False

    suit_offsets = {"m": 0, "p": 9, "s": 18}
    honors = set(HONORS)

    for group_order in permutations(RANK_GROUPS, 3):
        allowed = set(honors)
        for suit_idx, group in enumerate(group_order):
            base = suit_idx * 9
            for rank in group:
                allowed.add(base + rank - 1)
        if not used.issubset(allowed):
            continue
        need = 14 - non_laizi_tiles
        if need < 0:
            continue
        if need > laizi_count:
            continue
        # 可用不同牌种数量必须够补齐到 14 张全单张
        if len(allowed - used) >= need:
            return True

    return False


def is_zuhe_long(
    counts: Sequence[int] | Sequence[str] | Sequence[int | str], laizi_idx: int | None = None
) -> bool:
    """组合龙判定。

    判定口径：手牌可拆出 147/258/369 跨三门各一组共 9 张，
    剩余 5 张可组成 1 面子 + 1 对将（将牌不限 2/5/8）。
    """
    vec = _normalize_counts(counts)
    if sum(vec) != 14:
        return False

    work = list(vec)
    laizi_count = 0
    if laizi_idx is not None:
        laizi_count = pop_laizi_from_counts(work, laizi_idx)

    suit_bases = [0, 9, 18]
    rank_groups = ((1, 4, 7), (2, 5, 8), (3, 6, 9))
    for suit_perm in permutations(suit_bases, 3):
        req_tiles = []
        for base, group in zip(suit_perm, rank_groups):
            req_tiles.extend(base + rank - 1 for rank in group)

        after = list(work)
        need = 0
        for idx in req_tiles:
            if after[idx] > 0:
                after[idx] -= 1
            else:
                need += 1
        if need > laizi_count:
            continue

        remain_laizi = laizi_count - need
        remain_count = sum(after) + remain_laizi
        if remain_count != 5:
            continue

        if _is_one_meld_one_pair(after, remain_laizi):
            return True
    return False


def _is_one_meld_one_pair(counts: Sequence[int], laizi_count: int) -> bool:
    """判断剩余 5 张是否可组成 1 面子 + 1 对将（将牌不限制）。"""
    work = list(counts)
    for pair_idx in range(NUM_TILE_TYPES):
        need_pair = max(0, 2 - work[pair_idx])
        if need_pair > laizi_count:
            continue
        after = list(work)
        use_real = min(2, after[pair_idx])
        after[pair_idx] -= use_real
        if _can_form_all_melds(after, laizi_count - need_pair):
            return True
    return False


def _can_form_all_melds(counts: Sequence[int], laizi_count: int) -> bool:
    """判断剩余牌是否可全部拆为面子。"""
    if (sum(counts) + laizi_count) % 3 != 0:
        return False

    @lru_cache(maxsize=None)
    def dfs(counts_key: tuple[int, ...], laizi_left: int) -> bool:
        total = sum(counts_key)
        if total == 0:
            return laizi_left % 3 == 0
        if (total + laizi_left) % 3 != 0:
            return False

        first = -1
        for i, c in enumerate(counts_key):
            if c:
                first = i
                break
        if first == -1:
            return laizi_left % 3 == 0

        # 方案 1：刻子
        c = counts_key[first]
        need_triplet = 0 if c >= 3 else 3 - c
        if need_triplet <= laizi_left:
            next_counts = list(counts_key)
            next_counts[first] -= min(3, c)
            if dfs(tuple(next_counts), laizi_left - need_triplet):
                return True

        # 方案 2：顺子（仅序数牌且点数 <= 7）
        if is_suited(first) and (first % 9) <= 6:
            second = first + 1
            third = first + 2
            next_counts = list(counts_key)
            next_counts[first] -= 1
            need = 0
            if next_counts[second] > 0:
                next_counts[second] -= 1
            else:
                need += 1
            if next_counts[third] > 0:
                next_counts[third] -= 1
            else:
                need += 1
            if need <= laizi_left and dfs(tuple(next_counts), laizi_left - need):
                return True

        return False

    return dfs(tuple(counts), laizi_count)


def _normalize_counts(
    counts: Sequence[int] | Sequence[str] | Sequence[int | str],
) -> list[int]:
    """统一输入为长度 34 的计数向量。"""
    if len(counts) == NUM_TILE_TYPES and all(isinstance(x, int) for x in counts):
        return list(counts)  # type: ignore[arg-type]
    return make_counts(counts)  # type: ignore[arg-type]


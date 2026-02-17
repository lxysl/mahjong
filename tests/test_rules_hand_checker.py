from mahjong_ai.rules.hand_checker import (
    check_win,
    is_quan_bu_kao,
    is_seven_pairs,
    is_standard_win,
    is_thirteen_orphans,
    is_zuhe_long,
)
from mahjong_ai.rules.tiles import make_counts, tile_to_index


def _counts(*tiles: str):
    return make_counts(tiles)


def test_standard_win_with_258_pair():
    counts = _counts(
        "1m",
        "1m",
        "1m",
        "2m",
        "3m",
        "4m",
        "3p",
        "4p",
        "5p",
        "7s",
        "8s",
        "9s",
        "5p",
        "5p",
    )
    assert is_standard_win(counts, pair_restrict_258=True)


def test_standard_win_reject_non_258_pair():
    counts = _counts(
        "1m",
        "1m",
        "1m",
        "2m",
        "3m",
        "4m",
        "3p",
        "4p",
        "5p",
        "7s",
        "8s",
        "9s",
        "1p",
        "1p",
    )
    assert not is_standard_win(counts, pair_restrict_258=True)


def test_standard_win_with_laizi_pair_completion():
    laizi_idx = tile_to_index("P")
    counts = _counts(
        "1m",
        "1m",
        "1m",
        "2m",
        "3m",
        "4m",
        "3p",
        "4p",
        "5p",
        "7s",
        "8s",
        "9s",
        "5p",
        "P",
    )
    assert is_standard_win(counts, laizi_idx=laizi_idx, pair_restrict_258=True)


def test_seven_pairs():
    counts = _counts("1m", "1m", "2m", "2m", "3p", "3p", "4p", "4p", "5s", "5s", "6s", "6s", "C", "C")
    assert is_seven_pairs(counts)
    result = check_win(counts)
    assert result.is_win
    assert result.win_type == "seven_pairs"


def test_thirteen_orphans():
    counts = _counts("1m", "1m", "9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "C", "F", "P")
    assert is_thirteen_orphans(counts)


def test_thirteen_orphans_with_laizi():
    laizi_idx = tile_to_index("5m")
    counts = _counts("1m", "1m", "9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "C", "F", "5m")
    assert is_thirteen_orphans(counts, laizi_idx=laizi_idx)


def test_quan_bu_kao():
    counts = _counts("1m", "4m", "7m", "2p", "5p", "8p", "3s", "6s", "9s", "E", "S", "W", "N", "C")
    assert is_quan_bu_kao(counts)
    result = check_win(counts)
    assert result.is_win
    assert result.win_type == "quan_bu_kao"


def test_zuhe_long():
    counts = _counts("1m", "4m", "7m", "2p", "5p", "8p", "3s", "6s", "9s", "2m", "2m", "3m", "4m", "5m")
    assert is_zuhe_long(counts)
    result = check_win(counts)
    assert result.is_win
    assert result.win_type == "zuhe_long"


def test_non_win():
    counts = _counts("1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "5s")
    result = check_win(counts)
    assert not result.is_win


import pytest

from mahjong_ai.rules.laizi import indicator_to_laizi
from mahjong_ai.rules.tiles import index_to_tile, tile_to_index


@pytest.mark.parametrize(
    ("indicator", "expected"),
    [
        ("1m", "2m"),
        ("5m", "6m"),
        ("9m", "1m"),
        ("9p", "1p"),
        ("8s", "9s"),
        ("9s", "1s"),
    ],
)
def test_indicator_to_laizi_suited(indicator, expected):
    assert index_to_tile(indicator_to_laizi(indicator)) == expected


@pytest.mark.parametrize(
    ("indicator", "expected"),
    [
        ("E", "S"),
        ("S", "W"),
        ("W", "N"),
        ("N", "C"),
        ("C", "F"),
        ("F", "P"),
        ("P", "E"),
    ],
)
def test_indicator_to_laizi_honors(indicator, expected):
    assert index_to_tile(indicator_to_laizi(indicator)) == expected


def test_indicator_accept_index():
    n_idx = tile_to_index("N")
    assert index_to_tile(indicator_to_laizi(n_idx)) == "C"


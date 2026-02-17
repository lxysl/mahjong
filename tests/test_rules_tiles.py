from mahjong_ai.rules.tiles import (
    ALL_TILE_CODES,
    NUM_TILE_TYPES,
    counts_to_tiles,
    index_to_tile,
    is_258,
    make_counts,
    tile_rank,
    tile_to_index,
    validate_counts,
)


def test_tile_round_trip_all_types():
    assert len(ALL_TILE_CODES) == NUM_TILE_TYPES
    for idx in range(NUM_TILE_TYPES):
        code = index_to_tile(idx)
        assert tile_to_index(code) == idx


def test_make_counts_and_expand():
    tiles = ["1m", "1m", "9s", "E", "C", "C"]
    counts = make_counts(tiles)
    assert counts[tile_to_index("1m")] == 2
    assert counts[tile_to_index("9s")] == 1
    assert counts[tile_to_index("E")] == 1
    assert counts[tile_to_index("C")] == 2
    expanded = counts_to_tiles(counts)
    assert len(expanded) == len(tiles)


def test_is_258_only_on_suited():
    positives = {"2m", "5m", "8m", "2p", "5p", "8p", "2s", "5s", "8s"}
    for code in ALL_TILE_CODES:
        idx = tile_to_index(code)
        if code in positives:
            assert is_258(idx)
        else:
            assert not is_258(idx)


def test_tile_rank_honors_none():
    assert tile_rank(tile_to_index("E")) is None
    assert tile_rank(tile_to_index("C")) is None
    assert tile_rank(tile_to_index("1m")) == 1
    assert tile_rank(tile_to_index("9p")) == 9


def test_validate_counts_reject_overflow():
    counts = [0] * NUM_TILE_TYPES
    counts[0] = 5
    try:
        validate_counts(counts)
    except ValueError:
        return
    raise AssertionError("预期应抛出 ValueError")


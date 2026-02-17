from mahjong_ai.env.actions import Action, action_space_size
from mahjong_ai.env.game_state import GameState
from mahjong_ai.env.legal_actions import legal_reactions_to_discard, resolve_reactions
from mahjong_ai.rules.tiles import make_counts, tile_to_index


def test_action_space_size():
    assert action_space_size() == 193


def test_resolve_reactions_priority_hu_first():
    claims = {
        1: Action("peng", tile=tile_to_index("3m")),
        2: Action("hu", tile=tile_to_index("3m")),
    }
    seat, action = resolve_reactions(discarder=0, claims=claims, num_players=4)
    assert seat == 2
    assert action.kind == "hu"


def test_resolve_reactions_seat_order_for_hu():
    claims = {
        1: Action("hu"),
        2: Action("hu"),
    }
    seat, action = resolve_reactions(discarder=0, claims=claims, num_players=4)
    assert seat == 1
    assert action.kind == "hu"


def test_legal_reaction_chi_only_next_player():
    state = GameState()
    state.laizi_tile = tile_to_index("P")
    state.hands[1] = make_counts(["1m", "2m", "3m", "4m", "5m", "P", "P", "E", "E", "C", "C", "F", "N"])
    state.hands[2] = make_counts(["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "E", "S", "W", "N"])

    tile = tile_to_index("3m")
    next_actions = legal_reactions_to_discard(state, seat=1, discarder=0, tile=tile)
    other_actions = legal_reactions_to_discard(state, seat=2, discarder=0, tile=tile)

    assert any(a.kind == "chi" for a in next_actions)
    assert not any(a.kind == "chi" for a in other_actions)


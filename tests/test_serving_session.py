from mahjong_ai.serving.session import GameSession


def test_session_start_and_recommend():
    session = GameSession(my_seat=0)
    session.start_round(
        dealer=0,
        laizi_indicator="5m",
        initial_hands={
            0: ["1m", "1m", "1m", "2m", "3m", "4m", "5p", "5p", "7s", "8s", "9s", "E", "C", "P"],
            1: ["1p"] * 13,
            2: ["2p"] * 13,
            3: ["3p"] * 13,
        },
        current_player=0,
    )
    rec = session.recommend_action(top_k=3)
    assert rec["seat"] == 0
    assert rec["phase"] == "action"
    assert len(rec["top_k"]) <= 3


def test_session_apply_draw_discard_and_undo():
    session = GameSession(my_seat=0)
    session.start_round(
        dealer=0,
        laizi_indicator="9p",
        initial_hands={
            0: ["1m"] * 14,
            1: ["2m"] * 13,
            2: ["3m"] * 13,
            3: ["4m"] * 13,
        },
        current_player=0,
    )
    session.apply_event({"type": "discard", "seat": 0, "tile": "1m"})
    assert session.state is not None
    assert session.state.phase == "response"
    session.apply_event({"type": "pass_all"})
    assert session.state.phase in {"draw", "terminal"}
    snapshot_phase = session.state.phase
    session.undo_last_event()
    assert session.state.phase != snapshot_phase


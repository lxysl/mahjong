import pytest

from mahjong_ai.env.actions import Action
from mahjong_ai.env.game_state import GameState
from mahjong_ai.env.transition import apply_turn_action
from mahjong_ai.rules.scoring import HandProgress, RewardConfig


def test_discard_dense_reward_uses_snapshot_and_one_time_enter_bonus(monkeypatch):
    state = GameState()
    state.phase = "action"
    state.current_player = 0
    state.laizi_tile = None
    state.hands[0][0] = 1
    state.hands[0][1] = 1
    state.progress_snapshot[0] = (False, 0)
    state.tenpai_enter_reward_used[0] = False

    progress_seq = iter([HandProgress(tenpai=True, outs=4), HandProgress(tenpai=True, outs=6)])

    def fake_evaluate_hand_progress(hand_counts, laizi_idx, enable_special=True):
        return next(progress_seq)

    monkeypatch.setattr("mahjong_ai.env.transition.evaluate_hand_progress", fake_evaluate_hand_progress)

    apply_turn_action(state, 0, Action("discard", tile=0), config=RewardConfig())
    assert state.rewards[0] == pytest.approx(0.07)
    assert state.tenpai_enter_reward_used[0] is True
    assert state.progress_snapshot[0] == (True, 4)

    # 模拟进入下一次同座位 action（绕过 response，聚焦奖励增量逻辑）
    state.phase = "action"
    state.current_player = 0
    state.pending_discard = None
    state.response_stage = None
    state.response_order = []
    state.response_index = 0
    state.response_claims = {}

    apply_turn_action(state, 0, Action("discard", tile=1), config=RewardConfig())
    # 第二次不再发首次听牌奖励，仅按 outs 增量 +2*0.005 = +0.01
    assert state.rewards[0] == pytest.approx(0.08)
    assert state.progress_snapshot[0] == (True, 6)


def test_discard_without_snapshot_does_not_backfill_bonus(monkeypatch):
    state = GameState()
    state.phase = "action"
    state.current_player = 0
    state.laizi_tile = None
    state.hands[0][0] = 1
    state.progress_snapshot[0] = None
    state.tenpai_enter_reward_used[0] = False

    monkeypatch.setattr(
        "mahjong_ai.env.transition.evaluate_hand_progress",
        lambda hand_counts, laizi_idx, enable_special=True: HandProgress(tenpai=True, outs=3),
    )

    apply_turn_action(state, 0, Action("discard", tile=0), config=RewardConfig())
    assert state.rewards[0] == pytest.approx(0.0)
    assert state.tenpai_enter_reward_used[0] is True
    assert state.progress_snapshot[0] == (True, 3)

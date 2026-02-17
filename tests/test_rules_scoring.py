import pytest

from mahjong_ai.rules.scoring import (
    HandProgress,
    RewardConfig,
    dense_progress_reward,
    evaluate_hand_progress,
    immediate_gang_reward,
    terminal_rewards,
)
from mahjong_ai.rules.tiles import make_counts, tile_to_index


def test_immediate_gang_reward():
    cfg = RewardConfig()
    assert immediate_gang_reward("ming_gang", cfg) == 0.2
    assert immediate_gang_reward("an_gang", cfg) == 0.4
    assert immediate_gang_reward("bu_gang", cfg) == 0.2
    assert immediate_gang_reward("discard", cfg) == 0


def test_terminal_dian_hu_default_no_dealer_bonus():
    rewards = terminal_rewards(
        num_players=4,
        winner=0,
        win_mode="dian_hu",
        dealer=0,
        win_type="standard",
    )
    assert rewards[0] == 1
    assert rewards[1] == -1
    assert rewards[2] == -1
    assert rewards[3] == -1


def test_terminal_zi_mo_with_gang_shang_hua():
    rewards = terminal_rewards(
        num_players=4,
        winner=2,
        win_mode="zi_mo",
        dealer=1,
        win_type="standard",
        gang_shang_hua=True,
    )
    assert rewards[2] == pytest.approx(2.4)  # 自摸 +2 + 杠上开花暗杠 +0.4
    assert rewards[0] == -1
    assert rewards[1] == -1
    assert rewards[3] == -1


def test_dense_progress_reward_enter_tenpai_once_and_delta_outs():
    cfg = RewardConfig(tenpai_enter_reward=0.05, outs_delta_reward=0.005, outs_delta_clip=0.05)
    before = HandProgress(tenpai=False, outs=0)
    after = HandProgress(tenpai=True, outs=4)
    # 进入听牌 +0.05，听牌张数增加 4 张 -> +0.02，总计 +0.07
    assert dense_progress_reward(before, after, cfg) == pytest.approx(0.07)


def test_dense_progress_reward_clip_and_negative_delta():
    cfg = RewardConfig(tenpai_enter_reward=0.05, outs_delta_reward=0.005, outs_delta_clip=0.05)
    before = HandProgress(tenpai=True, outs=20)
    after = HandProgress(tenpai=True, outs=0)
    # 变化 -20 张，原始 -0.1，裁剪为 -0.05
    assert dense_progress_reward(before, after, cfg) == pytest.approx(-0.05)


def test_dense_progress_reward_disable_enter_bonus():
    cfg = RewardConfig(tenpai_enter_reward=0.05, outs_delta_reward=0.005, outs_delta_clip=0.05)
    before = HandProgress(tenpai=False, outs=0)
    after = HandProgress(tenpai=True, outs=4)
    # 禁用首次听牌奖励时，仅保留听牌张数增量奖励 +0.02
    assert dense_progress_reward(before, after, cfg, allow_tenpai_enter=False) == pytest.approx(0.02)


def test_evaluate_hand_progress_counts_outs():
    # 标准 13 张听牌：缺一张 5p 即和（将牌 5p）
    counts = make_counts(["1m", "1m", "1m", "2m", "3m", "4m", "3p", "4p", "5p", "7s", "8s", "9s", "5p"])
    progress = evaluate_hand_progress(counts, laizi_idx=tile_to_index("P"), enable_special=True)
    assert progress.tenpai
    assert progress.outs >= 1

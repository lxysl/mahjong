from mahjong_ai.rules.scoring import RewardConfig, immediate_gang_reward, terminal_rewards


def test_immediate_gang_reward():
    cfg = RewardConfig()
    assert immediate_gang_reward("ming_gang", cfg) == 1
    assert immediate_gang_reward("an_gang", cfg) == 2
    assert immediate_gang_reward("bu_gang", cfg) == 1
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
    assert rewards[2] == 4  # 自摸 +2 + 杠上开花暗杠 +2
    assert rewards[0] == -1
    assert rewards[1] == -1
    assert rewards[3] == -1

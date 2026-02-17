"""规则模块导出。"""

from mahjong_ai.rules.hand_checker import WinResult, check_win
from mahjong_ai.rules.laizi import indicator_to_laizi
from mahjong_ai.rules.scoring import RewardConfig, immediate_gang_reward, terminal_rewards
from mahjong_ai.rules.tiles import index_to_tile, make_counts, tile_to_index

__all__ = [
    "WinResult",
    "check_win",
    "indicator_to_laizi",
    "RewardConfig",
    "immediate_gang_reward",
    "terminal_rewards",
    "index_to_tile",
    "make_counts",
    "tile_to_index",
]

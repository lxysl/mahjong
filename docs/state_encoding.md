# 状态编码说明

当前环境观测由 `SelfPlayEnv.observe()` 给出，包含：

- `seat`: 当前观察座位
- `current_player`: 当前行动玩家
- `dealer`: 庄家
- `phase`: `draw/action/response/terminal`
- `hand_counts`: 长度 34 的手牌计数
- `laizi_indicator` / `laizi_tile`
- `wall_remaining`
- `discards`: 四家弃牌
- `melds`: 四家副露
- `pending_discard`: 响应窗口中的待裁决弃牌
- `legal_action_mask`: 与固定动作空间同长度的掩码

说明：

- 训练输入建议只使用可观测信息，避免“训练可见、实战不可见”的信息泄漏。

# 动作空间说明（T001）

采用固定离散动作表，便于模型输出统一 `action_id`。

## 动作类型

- `pass`
- `hu`
- `discard(tile)`
- `peng(tile)`
- `ming_gang(tile)`
- `an_gang(tile)`
- `bu_gang(tile)`
- `chi(chi_start)`（`chi_start` 为顺子起点）

## 编码顺序

1. `pass`
2. `hu`
3. `discard(0..33)`
4. `peng(0..33)`
5. `ming_gang(0..33)`
6. `an_gang(0..33)`
7. `bu_gang(0..33)`
8. `chi(21 种起点)`

动作空间总大小由 `mahjong_ai/env/actions.py` 中 `action_space_size()` 给出。

## 使用建议

- 训练时始终结合 `legal_action_mask`。
- 自回合动作不包含 `pass`；响应窗口中允许 `pass`。

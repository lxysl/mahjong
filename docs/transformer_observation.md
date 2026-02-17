# Transformer 观测编码

`ObservationEncoder` 将环境 `observation` 转换为固定长度序列（默认 `seq_len=128`），用于 `TransformerPolicyValueNet`。

## Token 组成

1. 手牌计数 token：34 个（每种牌一个）
2. 弃牌计数 token：34 个
3. 副露计数 token：34 个
4. 元信息 token：8 个
  - seat
  - current_player
  - dealer
  - phase
  - wall_remaining
  - pending_discarder
  - laizi_tile
  - pending_discard_tile

总有效 token 为 110，其余位置补 PAD 到 128。

## 输入字段

- `token_type_ids`
- `tile_ids`
- `value_ids`
- `attention_mask`
- `legal_action_mask`（用于策略头动作掩码）

## 说明

- 模型输出为动作 logits（193 维）与状态价值标量。
- 训练和推理都必须应用 `legal_action_mask`，确保不采样非法动作。

# 训练指标建议

建议最少跟踪以下指标：

1. 平均回报（按座位、按模型版本）
2. 胜率（vs Random / vs Heuristic）
3. 非法动作率（目标长期为 0）
4. 平均对局长度
5. 策略熵（避免过早塌缩）
6. 胡牌方式占比（点胡/自摸）
7. 杠牌频率（明杠/暗杠/补杠）

当前 `PPOTrainer` 已默认输出并可上报到 wandb 的关键指标：

- `loss`
- `policy_loss`
- `value_loss`
- `entropy`
- `approx_kl`
- `clip_fraction`
- `explained_var`
- `rollout/avg_reward`
- `rollout/reward_std`
- `rollout/done_rate`
- `rollout/avg_value_pred`
- `rollout/avg_advantage`
- `rollout/avg_return`
- `timing/steps_per_sec`
- `train/lr`

配套建议：

- 每个 checkpoint 固定种子跑一轮 Arena，对比历史最佳。
- 指标异常时优先检查规则模块与合法动作掩码。

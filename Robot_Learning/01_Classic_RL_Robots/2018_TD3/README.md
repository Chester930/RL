# TD3 — Twin Delayed Deep Deterministic (2018)

論文：Fujimoto et al., arXiv:1802.09477 | 機構：McGill University

## DDPG 的三個改進
1. 雙 Critic：Q = min(Q₁, Q₂) → 防止過估計
2. 延遲更新：每 2 次 Critic 更新才更新 1 次 Actor
3. 目標噪音：a' = π'(s') + clip(ε, -c, c) → 平滑 Q 估計

## 本專案結果
環境：Pendulum-v1 | 訓練步數：200K | best eval：-117.6

## 相關資源
- 論文：https://arxiv.org/abs/1802.09477
- 本專案實作：../../04_Actor_Critic_Continuous/2018_TD3/


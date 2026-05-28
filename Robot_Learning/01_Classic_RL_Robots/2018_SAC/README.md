# SAC — Soft Actor-Critic (2018)

論文：Haarnoja et al., arXiv:1801.01290 | 機構：UC Berkeley

## 最大熵目標
max_π E[Σ γᵗ (r(sₜ,aₜ) + α·H(π(·|sₜ)))]

α 自動調整：min_α E[-α log π(aₜ|sₜ) - α·H̄]
目標熵 H̄ = -dim(A)（動作維度的負數）

## 機器人優勢
樣本效率最高、不需精細調參、隨機策略避免過擬合

## 本專案結果
Pendulum：-171.8（100K steps）| LunarLanderContinuous：262.4

## 相關資源
- 論文：https://arxiv.org/abs/1801.01290
- 本專案實作：../../04_Actor_Critic_Continuous/2018_SAC/


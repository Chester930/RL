# DDPG — Deep Deterministic Policy Gradient (2015)

論文：Lillicrap et al., arXiv:1509.02971 | 機構：Google DeepMind

## 核心公式

Actor 更新：∇_θ J ≈ E[∇_a Q(s,a)|_{a=π(s)} · ∇_θ π_θ(s)]
Critic 更新：L = E[(Q(s,a) - (r + γ Q'(s',π'(s'))))²]

## 機器人應用
- 關節角度控制（Reacher, Ant）
- 末端執行器位置控制

## 本專案結果
環境：Pendulum-v1 | 訓練步數：200K | best eval：-86.1

## 相關資源
- 論文：https://arxiv.org/abs/1509.02971
- 本專案實作：../../04_Actor_Critic_Continuous/2015_DDPG/


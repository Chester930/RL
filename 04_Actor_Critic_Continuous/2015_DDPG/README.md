# DDPG — 深度確定性策略梯度 (Deep Deterministic Policy Gradient)

## 論文

Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., & Wierstra, D. (2015).  
*Continuous Control with Deep Reinforcement Learning*. ICLR 2016. arXiv:1509.02971.

---

## 核心思想 (Key Idea)

DDPG 將 DQN 的概念擴充套件到了**連續動作空間 (Continuous action spaces)**，並使用確定性策略。演員 (Actor) 不再從分佈中取樣，而是直接輸出動作值：

```
pi(s) -> a    (無機率分佈，無取樣過程)
```

評論家 (Critic) 負責估計 $Q(s, \pi(s))$，而演員則透過**確定性策略梯度 (Deterministic Policy Gradient)** 來學習最大化該 Q 值：

```
grad_theta J(theta) = E_s [ grad_a Q(s,a)|_{a=pi(s)} * grad_theta pi(s) ]
```

---

## 架構 (Architecture)

```
演員 (Actor)：  s -> MLP -> tanh -> a  (連續動作限制在 [-1,1] 區間，再乘以動作縮放倍率)
評論家 (Critic)： (s, a) -> MLP -> Q(s,a)   (將狀態與動作串聯後作為輸入)

目標網路： theta_target <- tau * theta + (1 - tau) * theta_target  (軟更新)
```

---

## 探索 (Exploration)

由於策略是確定性的，因此探索需要加入**疊加雜訊 (Additive noise)**：
- **OU 雜訊** (Ornstein-Uhlenbeck noise，原始論文採用)：具有時間相關性，對物理系統來說探索過程較平滑。
- **高斯雜訊** (較簡單的替代方案)：在實務中通常也能達到類似甚至更好的效果。

```
a = pi(s) + 雜訊 (OU 或 高斯)
```

---

## 更新規則 (Update Rules)

**評論家 (貝爾曼更新)：**
```
y = r + gamma * Q_target(s', pi_target(s'))
L_critic = E[(Q(s,a) - y)^2]
```

**演員 (策略梯度)：**
```
L_actor = -E[Q(s, pi(s))]    (最大化期望 Q 值)
```

**軟目標更新 (兩套網路皆適用)：**
```
theta_target <- tau * theta + (1 - tau) * theta_target
```

---

## 侷限性 (Limitations)

- 對超引數非常敏感。
- OU 雜訊難以調校。
- 傾向於高估 Q 值（後續的 TD3 演演算法解決了此問題）。
- 樣本效率 (Sample efficiency) 不如 SAC。

---

## 關鍵公式 (Key Equations)

```
DPG 定理： grad J = E_s[grad_a Q(s,a)|_{a=mu(s)} * grad_theta mu_theta(s)]
軟更新：   theta' <- tau * theta + (1 - tau) * theta'
```

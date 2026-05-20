# MADDPG — 多代理人深度確定性策略梯度 (Multi-Agent Deep Deterministic Policy Gradient)

## 論文

Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., & Mordatch, I. (2017).  
*Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments*.  
NeurIPS 2017. arXiv:1706.02275.

---

## 核心思想 (Key Idea)

**集中式訓練，分散式執行 (Centralized Training, Decentralized Execution, CTDE):**

```
訓練階段：  評論家 (Critic_i) 可以看到「所有」代理人的觀測 + 動作 (全域性資訊)。
執行階段：  演員 (Actor_i) 僅能看到代理人 i 自己的「區域性」觀測 (分散式執行)。
```

這種架構解決了多代理人強化學習中的**「非平穩性 (Non-stationarity)」**問題：在傳統 MARL 中，當多個代理人同時學習時，對任何一個代理人而言，環境的轉移機率會隨其他代理人策略的改變而改變，導致學習難以收斂。MADDPG 透過讓 Critic 觀察全域性資訊，使環境重新變得平穩。

---

## 架構 (Architecture)

```
針對每個代理人 i:
    Actor_i(o_i) -> a_i                           # 區域性策略 (Local policy)
    Critic_i(o_1,...,o_N, a_1,...,a_N) -> Q_i     # 集中式 Q 函式 (Centralized Q-function)
```

---

## 更新規則 (Update Rules)

```
評論家損失 (Critic loss): MSE(Q_i(x, a_1,...,a_N), y_i)
    其中 y_i = r_i + gamma * Q'_i(x', mu'_1(o'_1), ..., mu'_N(o'_N))

演員損失 (Actor loss): -E[Q_i(x, a_1, ..., mu_i(o_i), ..., a_N)]
    (注意：梯度僅流經 Actor_i；計算時其他代理人的動作被視為固定值)
```

---

## 支援的環境 (Supported Environments)

- **合作型 (Cooperative)**：代理人共享同一個獎勵訊號，目標是一致的。
- **競爭型 (Competitive)**：零和賽局（例如：掠食者與獵物遊戲 Predator-Prey）。
- **混合型 (Mixed)**：多個小團隊內部合作，團隊之間則進行競爭。

建議使用 **PettingZoo** 函式庫 (`pip install pettingzoo`) 來進行多代理人環境的開發與測試。

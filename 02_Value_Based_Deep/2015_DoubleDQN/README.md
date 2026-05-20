# Double DQN | 雙重 DQN

## 論文

van Hasselt, H., Guez, A., & Silver, D. (2015). *Deep Reinforcement Learning with Double Q-Learning*.  
arXiv:1509.06461. AAAI 2016.

---

## 核心思想 (Key Idea)

DQN 會高估 Q 值，因為它使用同一個網路來「選擇」與「評估」最佳的下一動作。這會導致樂觀偏差 (Optimistic Bias) 的價值估計，並隨著時間累積。

Double DQN 將這兩個操作解耦：
- **線上網路 (Online network)** 負責選擇動作：`a* = argmax_{a'} Q_online(s', a')`
- **目標網路 (Target network)** 負責評估該動作：`y = r + gamma * Q_target(s', a*)`

---

## 對比 (Comparison)

| 特性 | DQN | Double DQN |
|---|-----|-----------|
| 動作選擇 | 目標網路 (target_net) | 線上網路 (online_net) |
| 動作評估 | 目標網路 (target_net) | 目標網路 (target_net) |
| 目標計算 | `r + gamma * max_{a'} Q_target(s', a')` | `r + gamma * Q_target(s', argmax Q_online)` |
| 偏差 | 高估 (Overestimation) | 偏差降低 (Reduced bias) |

---

## 目標計算 (Target Computation)

**DQN:**
```
a* = argmax_{a'} Q_target(s', a')       # 同一個網路進行選擇與評估
y  = r + gamma * Q_target(s', a*)
```

**Double DQN:**
```
a* = argmax_{a'} Q_online(s', a')       # 線上網路選擇
y  = r + gamma * Q_target(s', a*)       # 目標網路評估
```

這一行程式碼的改動顯著降低了 Atari 基準測試中的高估問題。

---

## 結果 (Results)

在 49 個 Atari 遊戲中，Double DQN 達成：
- 更精確的 Q 值估計（透過與蒙特卡羅回報對比驗證）。
- 在 41/49 個遊戲中表現優於原本的 DQN。
- 在 DQN 容易高估的遊戲中，效能提升尤其顯著。

---

## 實作注意 (Implementation Note)

相對於 DQN 代理人，程式碼變動極小 — 僅有目標計算方式不同：

```python
# DQN:
next_q = target_net(next_states).max(dim=1)[0]

# Double DQN:
best_actions = online_net(next_states).argmax(dim=1, keepdim=True)
next_q = target_net(next_states).gather(1, best_actions).squeeze(1)
```

# HER — 事後經驗重播 (Hindsight Experience Replay)

## 論文

Andrychowicz, M., Wolski, F., Ray, A., Schneider, J., Fong, R., Welinder, P., McGrew, B., Tobin, J., Abbeel, P., & Zaremba, W. (2017).  
*Hindsight Experience Replay*. NeurIPS 2017. arXiv:1707.01495.

---

## 核心思想 (Key Idea)

在稀疏獎勵 (Sparse rewards) 的環境中，大多數的回合 (Episodes) 都會失敗。HER 透過**「假設代理人實際上正試圖達到它最終到達的狀態」**，將失敗的回合重新賦予意義並轉化為成功的經驗。

```
原始目標：  想要達到 (5, 5) — 代理人最後停在 (3, 2)  [失敗，獎勵 = -1]
HER 重標註： 「目標」改設為 (3, 2) — 此時獎勵 = 0         [成功！]

這兩種轉換資料 (Transitions) 同時都會被存入重播緩衝區。
```

---

## 演演算法流程 (Algorithm)

```
針對每個回合：
    1. 使用原始目標 g 進行資料收集
    2. 將 {(s_t, a_t, r_t, s_{t+1}, g)} 存入重播緩衝區

HER 事後重標註 (以 future 策略為例)：
    針對時間點 t，從同一個回合中取樣未來的一個時間點 t' > t
    g_her = 該點實際達到的目標 achieved_goal[t']
    r_her = 計算獎勵 compute_reward(achieved[t], g_her)   # 通常為 0 或 -1
    將 {(s_t, a_t, r_her, s_{t+1}, g_her)} 存入重播緩衝區

在混合批次（例如 80% HER 資料 + 20% 原始資料）上訓練 DDPG
```

---

## 重標註策略 (Relabeling Strategies)

| 策略 | 描述 | 效能表現 |
|----------|-------------|-------------|
| **future** | 同一回合中隨機選擇一個未來的狀態作為目標 | **最佳 (Best)** |
| final | 該回合最後一個狀態作為目標 | 良好 (Good) |
| episode | 該回合中隨機選擇任一狀態作為目標 | 普通 (OK) |
| random | 從所有回合中隨機選擇任一狀態作為目標 | 最弱 (Weakest) |

---

## 目標條件介面 (Goal-Conditioned Interface)

HER 預期接收 Gymnasium `GoalEnv` 的觀測字典結構：
```python
obs = {
    "observation":    np.ndarray,  # 機器人狀態 (Robot state)
    "achieved_goal":  np.ndarray,  # 實際上達到的位置
    "desired_goal":   np.ndarray,  # 預期要達到的位置 (原始目標)
}
```

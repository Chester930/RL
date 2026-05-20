# Dyna-Q — 整合學習、規劃與反應的架構

## 論文

Sutton, R. S. (1990). Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming. *Machine Learning Proceedings*, 216–224.

Sutton & Barto, *RL: An Introduction* (2nd ed.), Chapter 8.

---

## 核心思想 (Key Idea)

Dyna-Q 將**無模型 (Model-free)** 與**基於模型 (Model-based)** 的強化學習整合進同一個框架中：

```
真實經驗 -> 直接強化學習 (Direct RL / Q-Learning) + 模型學習 (Model Learning)
                                                |
                                            規劃 (Planning，執行 K 次模擬更新)
```

在環境中的每一次真實互動，都會利用學到的模型額外產生 K 次「虛擬」更新，從而顯著提高**樣本效率 (Sample efficiency)**。

---

## 演演算法 (Algorithm)

```
初始化 Q(s,a) 以及模型 Model
針對每一步：
    a = epsilon-greedy(Q(s))
    r, s' = env.step(a)
    
    (1) 直接強化學習： Q(s,a) += alpha * [r + gamma * max Q(s') - Q(s,a)]
    
    (2) 模型更新： Model[s][a] = (r, s')
    
    (3) 規劃 (重複執行 K 次)：
        s_sim, a_sim = 隨機從先前見過的狀態-動作對中挑選
        r_sim, s_sim' = Model[s_sim][a_sim]
        Q(s_sim, a_sim) += alpha * [r_sim + gamma * max Q(s_sim') - Q(s_sim, a_sim)]
```

---

## 規劃步數 K 的影響 (Effect of K)

```
K = 0:    純粹的 Q-Learning (無模型，無規劃)
K = 5:    每一步額外執行 5 次虛擬更新
K = 50:   每一步額外執行 50 次虛擬更新 (收斂速度顯著加快)
K = inf:  趨近於動態規劃 (對整個模型進行全面掃描遍歷)
```

更多的規劃步數意味著達成同樣效能所需的真實環境互動次數更少（樣本效率更高）。

---

## 模型質量 (Model Quality)

在查表式 (Tabular) 環境中，模型是精確的（直接記錄每一個造訪過的轉換關係）。對於連續狀態空間，模型則通常需要使用神經網路來近似學習（可參考 DreamerV3 或 MuZero）。

模型中的任何誤差都會導致規劃出錯誤的策略，這被稱為**模型偏差 (Model bias)** 問題。

---

## Dyna 架構 (Dyna Architecture)

```
真實世界 (Real World) -> 經驗 (Experience) -> 直接強化學習更新 (Direct RL Update)
                                         -> 模型更新 (Model Update)

學得模型 (Model) -> 模擬經驗 (Simulated Experience) -> 規劃更新 (Planning Updates)
```

規劃的本質是更充分地利用有限的真實經驗，從而完美橋接了無模型與基於模型的強化學習方法。

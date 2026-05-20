# CQL — 保守 Q 學習 (Conservative Q-Learning) 與離線強化學習 (Offline RL)

## 論文

Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020).  
*Conservative Q-Learning for Offline Reinforcement Learning*.  
NeurIPS 2020. arXiv:2006.04779.

---

## 核心思想 (Key Idea)

在離線資料 (Offline data) 上執行標準 Q 學習時，代理人往往會**「高估 (Overestimate)」**那些在資料集中未曾出現過的「分佈外 (Out-of-Distribution, OOD)」動作之 Q 值。CQL 透過引入一個**「保守懲罰項 (Conservative penalty)」**來壓低這些 OOD 動作的 Q 值。

```
L_CQL = L_Bellman + alpha * (E_{a~pi}[Q(s,a)] - E_{a~D}[Q(s,a)])
                               ^^^^^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^
                               壓低 (分佈外動作)    推高 (資料集動作)
```

這種做法能確保學習到的 Q 函式成為真實價值的**「下界 (Lower bound)」**，從而避免代理人因為貪婪地選擇被高估的 OOD 動作而導致策略崩潰。

---

## CQL 懲罰項（實踐形式）(Practical Form)

```
CQL_penalty = logsumexp_{a~Unif + pi}[Q(s,a)] - Q(s, a_dataset)

透過重要性取樣 (Importance sampling) 進行估計：
    logsumexp ≈ log(1/N * sum_i exp(Q(s, a_i) - log q(a_i)))
    其中 a_i 取樣自目前策略 pi(a|s) 或均勻分佈 Uniform(-1, 1)
```

---

## 變體 (Variants)

| 變體 | 描述 |
|---------|-------------|
| **CQL(H)** | 使用拉格朗日乘子 (Lagrange multiplier) 控制保守程度（自動調整 alpha）|
| **CQL(rho)** | 基於 Softmax 的懲罰項（預設推薦，最實用） |

---

## 資料集 (Dataset)

CQL 主要是為 **D4RL** (Datasets for Deep Data-Driven Reinforcement Learning) 設計的：
```python
pip install d4rl
import d4rl
env = gym.make("halfcheetah-medium-v2")
dataset = env.get_dataset()
# 鍵值包含：observations, actions, rewards, next_observations, terminals
```

---

## 主要結果 (Key Result)

CQL 在 D4RL 的運動基準測試中表現優異，顯著超越了行為複製 (BC) 與部分的離線演演算法。它在「中等專家 (Medium-expert)」與「隨機 (Random)」資料集上的表現尤為突出，因為在這些場景中**「分佈偏移 (Distribution shift)」**的問題最為嚴重。

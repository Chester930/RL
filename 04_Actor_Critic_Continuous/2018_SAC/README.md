# SAC — 軟演員-評論家 (Soft Actor-Critic)

## 論文

Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*. ICML 2018. arXiv:1801.01290.

Haarnoja, T., et al. (2018). *Soft Actor-Critic Algorithms and Applications*. arXiv:1812.05905. (自動溫度調節版本)

---

## 核心思想 (Key Idea)

SAC 旨在最大化**最大熵目標 (Maximum entropy objective)** — 即同時最大化期望回報以及策略的熵 (Entropy)：

```
J(pi) = E_pi [ sum_t r_t + alpha * H(pi(.|s_t)) ]
```

熵項 $H(\pi)$ 能鼓勵模型進行探索、防止策略過早收斂到區域性最優解，並使策略更具魯棒性 (Robustness)，能更好地適應環境的隨機變化。

---

## 為什麼要最大化熵？ (Why Maximum Entropy?)

1. **更好的探索能力**：高熵意味著會探索多種可能的動作，而非僅侷限於當前認為最優的動作。
2. **魯棒性**：模型會維持多種近乎最優的行為模式，當環境發生微小變化時更具適應力。
3. **樣本效率**：結合了離策略 (Off-policy) 學習與隨機性策略，通常比 DDPG 訓練速度更快且更穩定。
4. **避免區域性最優**：機率性取樣確保了模型不會輕易陷入次優的確定性策略中。

---

## 演演算法 (Algorithm)

```
1. 收集資料 (s, a, r, s') 並存入回放緩衝區 (動作 a ~ pi_theta(.|s))

2. 評論家更新 (軟貝爾曼更新)：
   a' ~ pi(.|s')
   y = r + gamma * (min(Q1_target, Q2_target)(s', a') - alpha * log_pi(a'|s'))
   最小化 (Q1(s,a) - y)^2 + (Q2(s,a) - y)^2

3. 策略更新 (演員)：
   最大化 E_a~pi [min(Q1, Q2)(s,a) - alpha * log_pi(a|s)]
   (使用重引數化技巧：a = tanh(mu + sigma * epsilon))

4. 溫度引數 alpha 更新 (自動調校)：
   alpha* = argmin_alpha E_a~pi [-alpha * (log_pi(a|s) + H_target)]
```

---

## 重引數化技巧 (Reparameterization Trick)

SAC 使用**壓縮高斯策略 (Squashed Gaussian policy)** 來確保梯度能傳回演員網路：
```
z ~ N(mu(s), sigma(s))   (從高斯分佈取樣)
a = tanh(z) * scale       (壓縮至有限的動作區間)
```

對數機率計算（包含對 `tanh` 變換的 Jacobian 修正）：
```
log pi(a|s) = log N(z|mu,sigma) - sum log(1 - tanh^2(z))
```

---

## 自動溫度調節 (Automatic Temperature Tuning)

與其手動設定超引數 $\alpha$，不如自動調校它以達到目標熵 $H_{target} = -|\mathcal{A}|$（動作空間維度的負值）：

```
L(alpha) = -alpha * (log_pi(a|s) + H_target)
```

---

## SAC vs TD3 比較

| 特性 | SAC | TD3 |
|----------|-----|-----|
| 策略型別 | 隨機策略 (高斯) | 確定性策略 |
| 探索方式 | 熵最大化 (Entropy maximization) | 疊加高斯雜訊 |
| 溫度引數 alpha | 自動調校 | 不適用 |
| 目標演員網路 | 不需要 | 需要 |
| 樣本效率 | 較高 | 相似 |
| 目前地位 | 連續動作空間的首選 (SOTA) | 極具競爭力的替代方案 |

---

## 訓練結果 (Training Results)

### Pendulum-v1（100K steps）

| 步數 | Eval |
|------|------|
| 10K | -168.5 ± 93.4 |
| 100K | **-171.8 ± 78.8** |

Alpha 從 0.32 自動降至 0.02，解決標準（< -200）✅

### LunarLanderContinuous-v3（訓練至 ~171K，提前終止）

| 步數 | Eval |
|------|------|
| 50K | 160.5 ± 93.7 |
| **100K** | **262.4 ± 22.2** ← 收斂 |
| 150K | 215.7 ± 71.0 |

**50K steps 突破 >200 目標**，比 PPO on-policy（需 ~163K）快 3 倍。
100K 時標準差僅 ±22，策略穩定。

# ICM — 內在好奇心模組 (Intrinsic Curiosity Module)

## 論文

Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017).  
*Curiosity-driven Exploration by Self-Supervised Prediction*.  
ICML 2017. arXiv:1705.05363.

---

## 核心思想 (Key Idea)

好奇心 = 學習到的特徵空間中的**「預測誤差」**。

那些難以透過學習到的前向模型 (Forward model) 預測的狀態被視為「新奇」且值得探索。

```
總獎勵 r_total = 外在獎勵 r_extrinsic + eta * ||phi(s') - phi_hat(s')||^2
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                            內在好奇心獎勵 (Intrinsic curiosity reward)
```

---

## 三個核心網路 (Three Networks)

```
1. 特徵編碼器 (Feature Encoder): s -> phi(s)
   透過逆向模型訓練，僅捕捉環境中「可控」的部分。

2. 逆向模型 (Inverse Model): (phi(s), phi(s')) -> a_hat
   預測是哪一個動作導致了目前的狀態轉移。
   強迫 phi 忽略那些無法被動作影響的背景幹擾因素。

3. 前向模型 (Forward Model): (phi(s), a) -> phi(s')_hat
   預測下一個狀態的特徵。預測誤差越高，代表該區域越新奇，給予的好奇心獎勵就越高。
```

---

## 損失函式 (Loss Functions)

```
L_inverse = CE(a_hat, a)                      # 離散動作預測 (Cross-Entropy)
L_forward = 0.5 * ||phi(s')_hat - phi(s')||^2   # 特徵預測誤差 (MSE)

ICM 總損失 = beta * L_forward + (1-beta) * L_inverse

內在獎勵： r_i = eta * L_forward (計算時需 detach 斷開梯度連結)
```

---

## 為什麼不用原始畫素誤差？ (Why Not Raw Pixel Error?)

畫素級別的預測誤差會包含許多「不可控的雜訊」（例如背景電視的閃爍、風吹草動）。
ICM 透過**「特徵編碼器 + 逆向模型」**的設計來過濾掉這些無關雜訊，只關注那些會受到代理人動作影響的環境特徵，從而解決了「喧鬧電視機問題 (Noisy TV problem)」。

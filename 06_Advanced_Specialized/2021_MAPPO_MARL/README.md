# MAPPO — 多代理人 PPO (Multi-Agent PPO)

## 論文

Yu, C., Velu, A., Vinitsky, E., Wang, Y., Bayen, A., & Wu, Y. (2021).  
*The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*.  
NeurIPS 2022 (Datasets and Benchmarks). arXiv:2103.01955.

---

## 核心思想 (Key Idea)

MAPPO = 搭配「集中式評論家 (Centralized Critic)」的 PPO，採用 **CTDE**（集中式訓練，分散式執行）架構。

MAPPO 的設計哲學是「大道至簡」：雖然結構簡單，但在許多複雜的多代理人合作任務中，其表現往往優於 QMIX 與 MADDPG 等更為複雜的演演算法。

```
MAPPO:  演員 Actor_i(o_i) -> a_i          (分散式，僅需區域性觀測)
        評論家 Critic_i(s_global) -> V_i  (集中式，訓練時使用全域性狀態)

IPPO:   演員 Actor_i(o_i) -> a_i          (分散式，僅需區域性觀測)
        評論家 Critic_i(o_i) -> V_i       (分散式，僅限區域性觀測)
```

---

## 演演算法 (Algorithm)

```
針對每一次取樣 (Rollout):
    1. 收集所有代理人的 T 步取樣資料。
    2. 使用全域性 V(s) 與聯合觀測計算 GAE 優勢值 (Advantages)。
    3. 對所有演員網路執行 PPO 截斷 (Clip) 更新。
    4. 對所有評論家網路執行 MSE 更新。

演員損失 (Actor loss) = -E[min(r*A, clip(r, 1-eps, 1+eps)*A)] - ent_coef * 熵
評論家損失 (Critic loss) = MSE(V(s), 回報 returns)
```

---

## 全域性狀態 vs 區域性觀測 (Global State vs Local Obs)

MAPPO 相對於獨立 PPO (IPPO) 的核心優勢在於：
- **更精準的價值估計**：評論家可以看到全域性狀態（Joint state），能更準確地評估當前局勢的價值。
- **穩定的優勢值**：更準確的 V(s) 帶來更穩定的優勢值，進而引導演員網路進行更高質量的更新。
- **分散式執行**：儘管訓練時使用了全域性狀態，但測試和部署時只需要演員網路處理區域性觀測即可，符合實際場景。

---

## 適用場景 (When to Use)

| 情境 | 建議演演算法 |
|----------|---------------|
| **完全合作** | MAPPO 或 IPPO |
| **混合合作與競爭** | MADDPG |
| **極大量的代理人群體** | **共享引數的 MAPPO** (Parameter Sharing) |
| **棋盤遊戲 / 零和賽局** | MADDPG 或 AlphaZero 風格演演算法 |

基準測試環境：StarCraft II (SMACv2)、Multi-agent MuJoCo、Hanabi (花札)。

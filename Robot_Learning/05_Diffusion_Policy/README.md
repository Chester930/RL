# 05 — Diffusion Policy（2023–2024）

> **時代定位**：用生成模型解決機器人動作的多模態問題  
> **核心突破**：Diffusion/Flow Matching 生成動作分佈，而非預測單一動作

---

## 為什麼需要 Diffusion Policy

```
傳統策略的問題：輸出單一動作
  情境：桌上有一個瓶子，你要抓它
  多模態解：從左邊抓、從右邊抓、從正上方抓...都對

BC/RL 的處理方式：
  → 對所有等效動作取「平均」
  → 輸出一個「平均」動作（夾住空氣）
  → 實際上哪個方向都沒抓到

Diffusion Policy：
  → 學習動作的完整分佈 p(a|s)
  → 推理時從分佈中採樣一個具體動作
  → 自然捕捉多種等效解法
```

---

## Diffusion Policy 原理

### 核心思想

```
訓練（加噪）：
  clean action a₀ → 加噪 → a₁ → a₂ → ... → aT（純雜訊）
  學習：εθ(aₜ, t, s) ≈ 加入的噪音

推理（去噪）：
  隨機噪音 aT → 去噪 → ... → a₁ → a₀（乾淨動作）
  條件：當前觀測 s

類比：雕刻家從大理石（純噪音）一刀一刀雕出雕像（動作）
```

### 兩種主要架構

```
UNet Diffusion Policy（原始版）：
  用 UNet 作為去噪網路
  時序卷積處理動作序列
  條件輸入：視覺特徵（ResNet 提取）

Transformer Diffusion Policy：
  Transformer 作為去噪網路
  Cross-attention 注入觀測條件
  更好的長序列動作建模
```

### 關鍵特性

```
Action Chunking（動作分塊）：
  不輸出單步動作，而是輸出未來 H 步動作序列
  H = 8~16（論文），π₀ 中 H = 50
  好處：減少高頻抖動，動作更平滑連貫
```

---

## Flow Matching（PI0 使用的改進版）

```
Diffusion 的問題：
  訓練路徑是隨機擴散（彎曲路徑）
  推理需要 50-100 步去噪

Flow Matching 的改進：
  訓練：學習從噪音到動作的直線路徑（vector field）
  推理：沿直線流場走，只需 10 步
  
直覺對比：
  Diffusion：從 A 到 B，沿隨機彎曲路徑走
  Flow：從 A 到 B，走直線（歐拉積分）
  
PI0 使用 Flow Matching 的原因：
  推理速度快（10 步 vs 50 步）
  低延遲控制（機器人需要 50Hz+ 的控制頻率）
```

---

## 本章節演算法

### 2023_DiffusionPolicy
```
論文：Chi et al. (2023), arXiv:2303.04137
機構：Columbia University / MIT

實驗環境：
  Push-T（推塊到目標）
  Block Pushing（多物體）
  真實機器手臂操作

vs BC 對比：
  Push-T 成功率：BC 60% → Diffusion Policy 90%+
  靈巧手任務：BC 幾乎失敗 → Diffusion 可行

代碼：github.com/real-stanford/diffusion_policy
```

### 2024_FlowMatching
```
背景論文：
  Flow Matching (Lipman 2022) [arXiv:2210.02747]
  Rectified Flow (Liu 2022) [arXiv:2209.03003]

機器人應用：
  π₀ 的 Action Expert 使用 Flow Matching
  比標準 Diffusion 快 5-10×，且效果相當

實作細節（以 π₀ 為例）：
  - 動作噪音初始化：a_noise ~ N(0, I)
  - 去噪步驟：10 步 Euler 積分
  - 條件：RL Token 或 VLM 特徵
```

---

## 對比：BC / RL / Diffusion Policy

| 維度 | BC | SAC/TD3 | Diffusion Policy |
|---|---|---|---|
| 多模態動作 | ❌ 平均化 | ❌ 確定性 | ✅ 自然捕捉 |
| 長動作序列 | ❌ 逐步誤差 | ❌ 逐步 | ✅ Action Chunk |
| 訓練穩定性 | ✅ 高 | ⚠️ 中 | ✅ 高 |
| 推理速度 | 快 | 快 | 較慢（需多步去噪）|
| 示範資料 | 多 | 不需要 | 中等 |

---

## 參考論文
- Diffusion Policy: [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)
- Flow Matching: [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
- ACT (Action Chunking Transformer): [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)

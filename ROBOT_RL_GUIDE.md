# 機器人強化學習方法全景指南

> 涵蓋從傳統 RL 到 VLA 基礎模型的完整技術演進路線。  
> 最後更新：2026-05-28

---

## 目錄

1. [技術演進全景](#1-技術演進全景)
2. [Layer 1：傳統 RL 方法](#2-layer-1傳統-rl-方法)
3. [Layer 2：模仿學習](#3-layer-2模仿學習)
4. [Layer 3：基於目標的學習](#4-layer-3基於目標的學習)
5. [Layer 4：Diffusion Policy](#5-layer-4diffusion-policy)
6. [Layer 5：VLA 基礎模型](#6-layer-5vla-基礎模型)
7. [PI0（π₀）詳解](#7-pi0π₀詳解)
8. [RLT（RL Token）詳解](#8-rltrl-token詳解)
9. [方法比較表](#9-方法比較表)
10. [學習路線建議](#10-學習路線建議)

---

## 1. 技術演進全景

```
時間軸 →

2013─────2016──────2018──────2020──────2022──────2024──2026
  │         │          │         │         │         │     │
  DQN     DDPG       TD3       HER      RT-2     PI0   RLT
  │        SAC       GAIL      MBPO     RLHF    π0.7   │
  │        HER       BC+RL    Dreamer   GRPO     │      │
  │         │          │         │      DPO    VLA大  RL Token
  │      Model-     Imitation  Model-  家族      模型   精修
  │      Free        + RL     Based
  │
「感知→動作」  ────→  「理解→推理→動作」  ────→  「基礎模型+精修」
```

**三個時代：**
- **Era 1（2013–2019）**：純強化學習，從像素/感測器學動作
- **Era 2（2019–2022）**：預訓練 + 微調，Transformer 進入機器人
- **Era 3（2022–now）**：大型視覺語言動作模型（VLA）+ 少量 RL 精修

---

## 2. Layer 1：傳統 RL 方法

### 適用場景
低維狀態空間、連續控制、無需語言理解的操作任務。

### 主要方法

#### DDPG（2015）
```
架構：Deterministic Actor + Critic + Replay Buffer + Target Network
特點：連續動作空間，離線策略
機器人應用：關節角度控制、末端執行器軌跡
問題：訓練不穩定，對超參敏感
```

#### TD3（2018）
```
改進 DDPG：雙 Critic 防過估計 + 延遲 Actor 更新 + 目標平滑
機器人應用：Pendulum / Hopper / HalfCheetah 類連續控制
優點：比 DDPG 穩定 2-3×
```

#### SAC（2018）
```
架構：Maximum Entropy RL，自動調整溫度參數 α
核心：最大化獎勵的同時最大化策略熵
機器人應用：桌面操作（reaching, grasping）
優點：樣本效率高，不需精細調參
實際效果：Pendulum -171（本專案），LunarLander 262
```

#### PPO（2017）
```
架構：Clip 目標函數，防止策略更新過大
機器人應用：人形機器人步態學習（配合物理模擬器）
優點：穩定、易並行化、支援連續/離散動作
```

---

## 3. Layer 2：模仿學習

### 為什麼需要模仿學習？
機器人任務的獎勵函數難以設計（稀疏、延遲），人工示範能快速提供「什麼是好行為」的先驗。

### 主要方法

#### BC（Behavioral Cloning）
```
原理：直接監督學習，複製專家動作
訓練：argmin_θ Σ ||π_θ(s) - a_expert||²
問題：分佈偏移（Covariate Shift）——測試時遇到訓練未見的狀態，誤差累積爆炸
適用：任務簡單、示範資料充足時的快速基線
```

#### IRL（Inverse Reinforcement Learning）
```
原理：從示範中反推獎勵函數，再用 RL 最佳化
優點：學到的獎勵函數可泛化到新場景
問題：計算昂貴，需大量示範資料，部署困難
代表：MaxEnt IRL, GAIL
```

#### GAIL（Generative Adversarial Imitation Learning）
```
架構：生成器（策略）+ 判別器（區分示範/策略動作）
原理：GAN 框架，策略被訓練成讓判別器無法區分
優點：解決 BC 的分佈偏移問題
問題：GAN 訓練不穩定，收斂難

對比示意：
  BC：  示範 ─→ 監督學習 ─→ 策略
  GAIL：示範 ─┐
              ├→ 判別器 ─→ 獎勵 ─→ RL ─→ 策略
  策略動作 ─┘
```

---

## 4. Layer 3：基於目標的學習

#### HER（Hindsight Experience Replay，2017）
```
核心洞察：失敗的嘗試也包含有用資訊
          「雖然沒達到目標 g，但達到了 g'」→ 把這次當成以 g' 為目標的成功

適用：稀疏獎勵 + 目標條件任務（機器人夾取、FetchReach）
實際效果：FetchReach-v4，400–500 ep 全部 100%（本專案）

虛擬示意：
  原始軌跡：(s₀,g) → (s₁,g) → ... → (sT,g) 失敗
  事後標記：(s₀,sT) → (s₁,sT) → ... → (sT,sT) ✓ 成功！
           把最終狀態 sT 當成目標，得到一組成功經驗
```

---

## 5. Layer 4：Diffusion Policy

### 背景
傳統策略輸出單一動作，但機器人任務往往有**多模態**解法（同一場景，多種抓取方式都對）。

#### Diffusion Policy（2023）
```
原理：把動作生成視為去噪過程（DDPM）
      從隨機噪音 aT → aT-1 → ... → a0（乾淨動作）
優點：
  - 捕捉多模態動作分佈
  - 比 BC 更穩定的長期規劃
  - 自然支援高維動作空間（如靈巧手 25 DoF）
架構：U-Net 或 Transformer 作為去噪網路
```

#### Flow Matching（PI0 使用）
```
改進 Diffusion：用直線流場取代隨機擴散路徑
優點：推理步驟更少（10 步 vs Diffusion 的 50-100 步）
       訓練更穩定（不需要 noise schedule 設計）
```

---

## 6. Layer 5：VLA 基礎模型

### 什麼是 VLA？

VLA = Vision + Language + Action，將視覺語言模型（VLM）延伸到機器人控制。

```
VLM（視覺語言）：
  輸入：圖像 + 文字
  輸出：文字（描述、問答）

VLA（視覺語言動作）：
  輸入：圖像 + 文字指令 + 機器人狀態
  輸出：動作序列（關節角度、末端執行器位置）
```

### VLA 演進史

| 年份 | 模型 | 機構 | 特點 |
|---|---|---|---|
| 2022 | **RT-1** | Google | Transformer + 大規模機器人資料（130K 示範）|
| 2023 | **RT-2** | Google | 用 VLM（PaLI）初始化，語言指令理解能力大幅提升 |
| 2023 | **SayCan** | Google | LLM 規劃 + 可行性函數（RL affordance）|
| 2024 | **OpenVLA** | 開源社群 | 7B Llama 基底，開源 VLA 基準 |
| 2024 | **π₀（PI0）** | Physical Intelligence | Flow Matching 動作專家 + PaliGemma，最強通用機器人 |
| 2025 | **Octo** | Berkeley | 開源，支援多種機器人 |
| 2026 | **RLT** | Physical Intelligence | 在 VLA 上加 RL Token 做精修 |

### 三大核心元件

```
┌─────────────────────────────────────────────────┐
│                  VLA 模型                        │
│                                                  │
│  [觀測編碼器]     [推理主幹]      [動作解碼器]    │
│  RGB 圖像 ─→   Transformer ─→  動作 Token /     │
│  深度圖          或 VLM           擴散 / Flow    │
│  語言指令 ─→   (reasoning)   ─→ 關節角度序列    │
│  機器人狀態                                      │
└─────────────────────────────────────────────────┘
```

---

## 7. PI0（π₀）詳解

**全名**：π₀: A Vision-Language-Action Flow Model for General Robot Control  
**機構**：Physical Intelligence（pi.website）  
**發布**：2024 年 10 月  
**論文**：[arXiv](https://arxiv.org/abs/2410.24164)

### 架構

```
輸入層：
  ├─ 3 × RGB 相機圖像（腕部 + 外部視角）
  ├─ 語言指令（"pick up the apple"）
  └─ 機器人本體感知（關節角度、速度）

主幹：
  ├─ PaliGemma 3B（Google VLM，處理視覺 + 語言）
  └─ Action Expert 300M（處理動作生成）
      ↕ 跨注意力交互（cross-attention）

輸出層：
  └─ 未來 H=50 步的動作序列（Flow Matching 去噪）
```

### 訓練策略

```
Stage 1：預訓練
  資料：10,000+ 小時機器人操作資料
        7 種機器人類型 + 68 種任務
        + Open X-Embodiment 公開資料集
  目標：學習通用物理知識和操作先驗

Stage 2：後訓練（任務特化微調）
  方式：少量任務示範資料 fine-tune
  保留：預訓練知識（不從頭訓練）
```

### Block-wise Causal Masking

```
VLM Block ←→ 自注意力（雙向）
Proprioception Block → 看 VLM + 自身（單向）
Action Block → 看所有 Block（全局）
```

### PI0 版本演進

| 版本 | 發布 | 特點 |
|---|---|---|
| π₀ | 2024-10 | 基礎通用版 |
| π₀-FAST | 2024-12 | 加速推理（語言動作 Token 化）|
| π₀.7 | 2025 | 可引導版本，具備湧現能力 |

---

## 8. RLT（RL Token）詳解

**全名**：RL Token: Bootstrapping Online RL with Vision-Language-Action Models  
**機構**：Physical Intelligence  
**論文**：[arXiv:2604.23073](https://arxiv.org/abs/2604.23073)  
**發布**：2026 年 3 月

### 解決的問題

PI0 等 VLA 模型在通用任務上表現優異，但**高精度操作任務**（螺絲鎖入、精確插拔）仍有瓶頸：
- 重訓整個 VLA 代價極高
- 傳統 RL 從頭學習需要大量資料
- 人工遙控示範耗費大量工程資源

### 架構

```
┌──────────────────────────────────────────────┐
│           預訓練 VLA（π₀，凍結）              │
│                                              │
│  圖像 + 語言 + 本體感知 → ... → [RL TOKEN]  │
│                                    ↓         │
│                           緊湊向量表示        │
│                      （壓縮任務關鍵資訊）     │
└───────────────────────┬──────────────────────┘
                        │（凍結，不更新）
            ┌───────────▼───────────┐
            │   小型 Actor-Critic   │
            │   ← 參數量極少        │
            │   ← 離線策略學習      │
            │   ← Replay Buffer     │
            │   ← 每秒數百次更新    │
            └───────────┬───────────┘
                        │
                  精修後動作輸出
```

### 訓練流程

```
Step 1：SFT 適應
  少量示範資料 → 微調 VLA 輸出 RL Token
  目的：讓 VLA 學會輸出 RL 可用的表示

Step 2：Online RL（VLA 完全凍結）
  環境互動 → Replay Buffer
  Actor-Critic 讀 RL Token → 預測動作修正
  獎勵：任務成功信號（二元或密集）
  更新速度：數百次/秒（輕量網路）
```

### 關鍵效能

| 任務 | 說明 | 提升 |
|---|---|---|
| M3 螺絲鎖入 | 3mm 精度旋轉插入 | 最快 15 分鐘學會 |
| 束線帶扣緊 | 柔性物體操作 | 速度 3× 提升 |
| 充電器插入 | USB-C 精確對準 | 超越人工遙控速度 |
| 網路線插入 | RJ45 精確對準 | 同上 |

### RLT 與傳統方法的根本差異

```
傳統線上 RL（from scratch）：
  環境 → 隨機探索 → 稀疏獎勵 → 慢速收斂（數百萬步）

RLT：
  VLA 先驗（已知「大概怎麼抓」）
    ↓ RL Token 萃取關鍵資訊
    ↓ 小型 Actor-Critic 精修「最後一毫米」
  結果：數百步收斂（分鐘級）
```

---

## 9. 方法比較表

| 方法 | 類型 | 資料需求 | 泛化能力 | 精度 | 計算成本 | 適用場景 |
|---|---|---|---|---|---|---|
| **BC** | 模仿 | 中（示範）| 低（分佈偏移）| 中 | 低 | 快速基線 |
| **SAC/TD3** | Model-Free RL | 高（環境互動）| 中 | 高（收斂後）| 中 | 低維連續控制 |
| **HER** | Goal-Cond RL | 中 | 中 | 高 | 中 | 稀疏獎勵操作 |
| **Diffusion Policy** | 模仿 | 中 | 中高 | 高（多模態）| 中高 | 靈巧操作 |
| **RT-2** | VLA | 極高 | 高（語言泛化）| 中 | 極高 | 通用語言指令任務 |
| **π₀（PI0）** | VLA+Flow | 極高 | 極高 | 高 | 極高 | 通用機器人基礎 |
| **RLT** | VLA+線上RL | 低（精修用）| 極高（繼承 VLA）| 極高 | 低（精修階段）| 高精度操作 |

---

## 10. 學習路線建議

### 基礎路線（本專案現有）
```
1. Model-Free 連續控制
   DDPG → TD3 → SAC（Pendulum，理解 Actor-Critic）

2. 稀疏獎勵操作
   HER（FetchReach，理解目標條件學習）

3. 模仿學習
   BC（行為複製的極限）

4. 安全約束
   PPO-Lag / CPO（SafePendulum，工業機器人安全需求）
```

### 進階路線（機器人前沿）
```
5. Diffusion Policy
   學習多模態動作分佈
   論文：Chi et al. (2023), arXiv:2303.04137

6. VLA 基礎
   RT-2 → OpenVLA（理解大模型如何進入機器人）

7. PI0（π₀）
   Flow Matching + Action Expert + 預訓練策略
   代碼：github.com/allenzren/open-pi-zero

8. RLT（RL Token）
   VLA + 線上 RL 精修
   論文：arXiv:2604.23073
```

### 關鍵洞察

```
機器人學習的核心矛盾：
  泛化 vs 精度
  └→ VLA 解決泛化，RL Token 解決精度

樣本效率的核心矛盾：
  從零學習（慢）vs 利用先驗（快）
  └→ Pretrain-then-finetune（π₀ 的哲學）
     └→ 極端案例：RLT 用凍結 VLA 作為感知器，只學「最後一步」

2026 年趨勢：
  大模型（VLA）提供語義理解和粗略動作
  小型 RL（RLT 類）負責精修高精度步驟
  → 分工明確，各司其職
```

---

## 參考資料

### 論文
- [π₀ (PI0): A Vision-Language-Action Flow Model](https://arxiv.org/abs/2410.24164) — Physical Intelligence, 2024
- [RLT: Bootstrapping Online RL with VLAs](https://arxiv.org/abs/2604.23073) — Physical Intelligence, 2026
- [Diffusion Policy](https://arxiv.org/abs/2303.04137) — Chi et al., 2023
- [RT-2: Vision-Language-Action Models](https://arxiv.org/abs/2307.15818) — Google DeepMind, 2023
- [HER: Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) — OpenAI, 2017
- [SAC: Soft Actor-Critic](https://arxiv.org/abs/1801.01290) — Haarnoja et al., 2018
- [Survey: RL for VLA Robotic Manipulation](https://www.techrxiv.org/users/934012/articles/1366553) — TechRxiv, 2025

### 官方資源
- [Physical Intelligence 研究頁](https://www.pi.website/research/rlt)
- [Open-pi-zero 實作](https://github.com/allenzren/open-pi-zero)
- [SakanaAI/RLT GitHub](https://github.com/SakanaAI/RLT)（注意：這是不同的 RLT）
- [Awesome-VLA-Papers](https://github.com/Psi-Robot/Awesome-VLA-Papers)

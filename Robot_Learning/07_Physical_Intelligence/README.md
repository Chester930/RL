# 07 — Physical Intelligence 系列（2024–2025）

> **時代定位**：目前最強的通用機器人基礎模型系列  
> **核心突破**：Flow Matching 動作專家 + 大規模多機器人預訓練

---

## Physical Intelligence（π）公司背景

```
成立：2023 年，前 Google/DeepMind/OpenAI 核心研究員
使命：建立通用機器人智能的基礎（類似 OpenAI 之於 LLM）
代表作：π₀（pi-zero）系列

核心理念：
  機器人 AI 需要與 LLM 相同的「預訓練 → 微調」正規範式
  大規模多樣化資料預訓練 → 任意任務快速適配
```

---

## 本章節演算法

### 2024_PI0 — π₀: 通用機器人基礎模型

```
論文：Black et al. (2024), arXiv:2410.24164
官網：pi.website/blog/pi0
代碼：github.com/allenzren/open-pi-zero（社群復現）

訓練資料規模：
  10,000+ 小時機器人操作資料
  7 種不同機器人硬體
  68 種任務
  + Open X-Embodiment 公開資料集

架構概覽：
┌────────────────────────────────────────────────┐
│              π₀ 雙流架構                        │
│                                                 │
│  [PaliGemma 3B Stream]   [Action Expert Stream] │
│  RGB×3 相機               機器人本體感知        │
│  語言指令                  動作噪音              │
│       ↕ Cross-Attention ↕                       │
│                   ↓                             │
│        Flow Matching 去噪輸出                   │
│        未來 H=50 步動作序列                     │
└────────────────────────────────────────────────┘

兩個核心元件：
  PaliGemma (3B)：Google 的視覺語言模型
    - 理解圖像內容和語言指令
    - 提供豐富的語義特徵

  Action Expert (300M)：π₀ 的核心貢獻
    - Flow Matching 訓練（非自迴歸，非分類）
    - 輸入：本體感知 + 動作噪音
    - 輸出：去噪後的動作向量場
    - 預測 50 步動作 Chunk

訓練配方：
  Stage 1（預訓練）：
    學習率 5e-5，batch 1024
    多機器人、多任務聯合訓練
    目標：學習跨任務的物理操作先驗

  Stage 2（後訓練/微調）：
    少量任務示範資料
    保留預訓練知識，特化任務性能
```

### 2025_PI0_FAST — π₀-FAST
```
論文：Black et al. (2024), arXiv:2412.10677

問題：π₀ 的自迴歸 Action Chunk 推理慢（每個 Token 都要 forward）
解法：用頻率編碼把整個動作序列壓縮為單一 Token 預測

速度對比：
  π₀：需要 H=50 次 Transformer forward
  π₀-FAST：1 次 forward → 立即得到完整序列

代價：稍低的動作精度（壓縮帶來信息損失）
適用：需要高頻控制（100Hz+）的任務
```

---

## Block-wise Causal Masking 詳解

```
π₀ 的 Transformer 注意力設計：

┌──────────────┬──────────────┬──────────────┐
│  VLM Block   │  Proprio     │  Action      │
│  (圖像+語言) │  Block       │  Block       │
├──────────────┼──────────────┼──────────────┤
│  ← 自身雙向  │  ← VLM + 自 │  ← 所有 Block│
│  注意力 →   │  身注意力    │  全局注意力  │
└──────────────┴──────────────┴──────────────┘

設計邏輯：
  VLM 完全雙向：圖像 Token 相互 attend（標準 VLM 做法）
  Proprio 看 VLM：知道當前指令，調整本體感知的解讀
  Action 看全部：動作生成需要完整的上下文
```

---

## π₀ vs 其他方法

| 維度 | OpenVLA | π₀ | RLT（在 π₀ 上精修）|
|---|---|---|---|
| 動作解碼 | 離散 Token | Flow Matching | RL Token + Actor-Critic |
| 推理精度 | 中 | 高 | 極高（精修後）|
| 泛化能力 | 高 | 極高 | 繼承 π₀ |
| 精細操作 | 弱 | 中 | 極強 |
| 學習速度 | 需微調（小時）| 需微調（小時）| 15 分鐘線上學習 |

---

## 參考資料
- π₀ 論文: [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)
- π₀-FAST 論文: [arXiv:2412.10677](https://arxiv.org/abs/2412.10677)
- 官方部落格: [pi.website/blog/pi0](https://www.pi.website/blog/pi0)
- 社群復現: [allenzren/open-pi-zero](https://github.com/allenzren/open-pi-zero)
- HuggingFace: [lerobot/pi0_base](https://huggingface.co/lerobot/pi0_base)

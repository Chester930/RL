# 04 — 世界模型用於機器人（2018–2020）

> **時代定位**：讓機器人在「想像」中規劃，減少昂貴的真實環境互動  
> **核心突破**：學習環境動態模型，在潛空間中做夢式規劃

---

## 核心概念

```
純 Model-Free RL（傳統）：
  真實環境 → 觀察 → RL 更新
  問題：每次學習都需要真實互動，機器人磨損昂貴

World Model 方法：
  真實環境（少量）→ 訓練世界模型
  世界模型（大量）→ 在模型中模擬 → RL 學習
  → 大幅減少真實環境需求
```

---

## 本章節演算法

### 2018_WorldModels — Ha & Schmidhuber
```
論文：Ha & Schmidhuber (2018), arXiv:1803.10122

架構三元組：
  V（視覺模型）：VAE 壓縮圖像 → 潛變量 z
  M（記憶模型）：MDN-RNN 預測下一個 z（學環境動態）
  C（控制器）：CMA-ES 在 z 空間找最優動作

機器人意義：
  視覺 → 壓縮表示 → 在壓縮空間預測未來 → 無需存儲原始圖像

本專案結果：CarRacing-v2，42.5（CPU 限制，需 GPU 才能達 900+）
```

### 2020_Dreamer — Dream to Control
```
論文：Hafner et al. (2020), arXiv:1912.01603（Dreamer v1）
後續：Dreamer v2 (2020), Dreamer v3 (2023)

架構：RSSM（Recurrent State Space Model）
  確定性狀態 h（GRU 隱藏狀態）
  隨機潛狀態 z（VAE 採樣）
  組合：s = (h, z)

「做夢」訓練：
  Step 1：用真實資料訓練 RSSM（學環境動態）
  Step 2：在 RSSM 內部展開想像軌跡（不用真實環境）
  Step 3：Actor-Critic 在想像軌跡上學習

機器人應用：
  DeepMind Control Suite（關節控制模擬）
  本專案：Pendulum -868（目標 -200，CPU 環境限制）
  論文結果（GPU）：Humanoid Stand 達到 800+ 分

Dreamer v3（2023）的改進：
  - 工作在多種環境（Atari/DMC/Minecraft）
  - 可預測獎勵稀疏的長序列任務
  - 無需任何超參調整
```

---

## 世界模型 vs 無模型的實際選擇

```
選無模型（SAC/TD3）：
  ✓ 任務簡單，環境互動便宜
  ✓ 不需要準確的環境動態模型
  ✓ 訓練更穩定可靠

選世界模型（Dreamer）：
  ✓ 真實機器人互動昂貴（每次都是硬體磨損）
  ✓ 需要大量規劃的複雜長序列任務
  ✓ 環境本身有規律可以被模型化
```

---

## 世界模型在 VLA 時代的角色

```
PI0 / RLT 不使用顯式世界模型，而是用大型 VLM 的「隱式」世界知識：
  VLM 在大規模語言+圖像資料上預訓練
  → 隱含了物理世界的常識（重力、物體形狀、接觸動態）
  → 比 Dreamer 的小型 RNN 世界模型更豐富

未來：大型世界模型（如 Genie 2、VideoPreTraining）
  → 視頻生成模型作為機器人的世界模型
```

---

## 參考論文
- WorldModels: [arXiv:1803.10122](https://arxiv.org/abs/1803.10122)
- Dreamer v1: [arXiv:1912.01603](https://arxiv.org/abs/1912.01603)
- Dreamer v2: [arXiv:2010.02193](https://arxiv.org/abs/2010.02193)
- Dreamer v3: [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)

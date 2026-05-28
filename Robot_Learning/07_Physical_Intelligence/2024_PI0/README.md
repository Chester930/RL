# π₀（PI0）— 通用機器人基礎模型（2024）

論文：Black et al., arXiv:2410.24164 | 機構：Physical Intelligence

## 架構速覽

```
輸入：RGB×3 + 語言 + 本體感知
  ↓
PaliGemma 3B（視覺語言理解）
  ↕ Cross-Attention
Action Expert 300M（Flow Matching 動作生成）
  ↓
未來 50 步動作序列
```

## 訓練規模
- 10,000+ 小時操作資料
- 7 種機器人硬體 × 68 種任務
- Stage 1 預訓練 → Stage 2 任務微調

## 關鍵創新
Flow Matching 取代 Diffusion：推理 10 步（vs Diffusion 50-100 步）
Action Chunking：輸出 H=50 步序列，動作更平滑

## 相關資源
- 論文：https://arxiv.org/abs/2410.24164
- 官方：https://www.pi.website/blog/pi0
- 復現：https://github.com/allenzren/open-pi-zero
- HuggingFace：https://huggingface.co/lerobot/pi0_base

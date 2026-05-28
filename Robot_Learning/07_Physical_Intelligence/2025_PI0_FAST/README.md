# π₀-FAST — 快速版本（2024）

論文：Black et al., arXiv:2412.10677 | 機構：Physical Intelligence

## 問題與解法

π₀ 的瓶頸：每步都需要 H=50 次 Transformer forward
π₀-FAST：用頻率編碼把整個動作序列壓縮為單一預測

```
π₀：  forward × 50 → 動作序列（慢）
FAST：forward × 1  → 動作序列（快 50×）
```

## 速度對比
- 推理頻率：π₀ ~10Hz → FAST ~500Hz（估計）
- 適用：需要高頻控制的精密任務

## 相關資源
- 論文：https://arxiv.org/abs/2412.10677

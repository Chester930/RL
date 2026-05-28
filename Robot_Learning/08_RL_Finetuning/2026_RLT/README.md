# RLT — RL Token（2026）

論文：Cetin, Zhao, Tang, arXiv:2604.23073 | 機構：Physical Intelligence

## 一句話說明
凍結 π₀ 大模型，只訓練讀取 RL Token 的小型 Actor-Critic，
15 分鐘學會鎖 M3 螺絲（3mm 精度），速度超越人工遙控。

## 架構

```
π₀（凍結）→ RL Token（緊湊向量）
                  ↓
          小型 Actor-Critic
          （每秒數百次更新）
                  ↓
            精修後動作
```

## 測試任務

| 任務 | 學習時間 | 效果 |
|---|---|---|
| M3 螺絲鎖入 | 15 分鐘 | 90%+ 成功率 |
| USB-C 插入 | 1-2 小時 | 超越人工遙控 |
| RJ45 插入 | 1-2 小時 | 3× 速度提升 |

## 相關資源
- 論文：https://arxiv.org/abs/2604.23073
- 官方：https://www.pi.website/research/rlt

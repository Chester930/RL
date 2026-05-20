# Dreamer 訓練日誌

## Pendulum-v1（2026-05-17）

| 引數 | 數值 |
|------|------|
| n_episodes | 100 |
| obs | 500×500 RGB → resize 64×64（bilinear） |
| deter_dim | 200 |
| stoch_dim | 30 |
| embed_dim | 1024 |
| imagine_horizon | 15 |
| gamma / lambda | 0.99 / 0.95 |
| seed_steps | 5,000（前 25 集隨機探索） |
| update_every | 50 steps（每 50 步做 1 次更新） |
| SEQ / B | 16 / 4（CPU 友好小批次） |

### 訓練過程

| 集數 | 回報 |
|------|------|
| 10 | -1070 |
| 20 | -900 |
| 40 | -970 |
| 80 | -838 |
| 100 | -836（最終） |

隨機策略基線：~-1500 ~ -1600。100 集後回報提升至 -836，世界模型有在學習但策略尚未收斂。

### Bug 修正記錄

**`network.py` decoder Linear bug：**
```python
# 修正前（錯誤）
self.linear = nn.Linear(latent_dim, 32 * 32)  # 輸出 1024，view(-1,32,1,1) 錯誤放大 batch 32×
# 修正後
self.linear = nn.Linear(latent_dim, 32)       # 輸出 32，view(-1,32,1,1) 正確 → (N,32,1,1)
```

**`agent.py` TODO 補完（主要變更）：**
1. `store()` — 加入 resize (H,W,C)→(C,64,64) uint8（Pendulum render 是 500×500）
2. `select_action()` — 加入 resize (C,H,W)→64×64 再送 encoder
3. `update()` — 完整實作世界模型訓練（encoder + RSSM + decoder + reward head）與行為學習（imagination + lambda-returns + actor/critic）

### 結論

- 100 集（20K steps）的 image-based Dreamer 在 CPU 上屬於**概念驗證**訓練
- 原始 Dreamer 需要 2M+ steps 在視覺環境（DMControl / Atari）才能完整收斂
- 此版本用向量狀態環境（Pendulum）搭配渲染影像，本質上繞過了 Dreamer 的最大優勢（從畫素中學習）
- 世界模型結構正確（RSSM posterior/prior + KL + 重建 + 獎勵頭），梯度流動正常，可作為學習 Dreamer 架構的參考實作
- 如需完整效果，需 GPU + dm_control 視覺環境 + 500K+ steps

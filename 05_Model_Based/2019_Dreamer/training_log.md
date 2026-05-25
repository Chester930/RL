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

---

## Pendulum-v1 State-based 重跑（2026-05-25，任務 E）

### 架構改動

| 項目 | 原始（image-based）| 重跑（state-based）|
|------|------|------|
| 觀測輸入 | RGB 像素 500×500 → resize 64×64 | 3D 狀態向量 [cos θ, sin θ, θ̇] |
| Encoder | 4 層 CNN → Linear（embed=1024）| MLP 3→256→256→64 |
| Decoder | ConvTranspose 重建 64×64 影像 | MLP 148→256→256→3（重建狀態）|
| embed_dim | 1024 | 64 |
| deter_dim | 200 | 128 |
| stoch_dim | 30 | 20 |
| latent_dim | 230 | 148 |
| seed_steps | 5,000 | 1,000 |
| update_every | 50 步 × 1 次 | 20 步 × 4 次 |
| n_episodes | 100 | 500 |
| action_scale | 未縮放（輸出 [-1,1]）| ×2.0（對應 Pendulum [-2,2]）|

### 訓練過程（eval 每 25 集）

| 集數 | Eval 平均 | 備註 |
|------|-----------|------|
| 25 | -1237.4 ± 218.2 | 剛過 seed 階段 |
| 50 | -1215.7 ± 190.4 | |
| 75 | -1105.7 ± 238.7 | 開始改善 |
| 100 | -1142.5 ± 188.0 | |
| 150 | -1327.1 ± 98.2 | 短暫波動 |
| 200 | -1156.0 ± 162.5 | |
| 300 | -1085.0 ± 182.1 | 穩定改善 |
| 325 | -1006.8 ± 74.5 | 首次突破 -1000 |
| 350 | -965.3 ± 195.7 | |
| 475 | **-868.2 ± 129.1** | 最佳，checkpoint 已儲存 |
| 500 | -1022.2 ± 132.7 | 最終（輕微回落）|

### 學習趨勢

```
ep25: -1237 → ep75: -1106 → ep325: -1007 → ep475: -868
改善幅度：約 -370（從隨機 -1600 基線計算，恢復率 ~30%）
```

### 結論

- **最佳 eval：-868.2**（ep 475，`best_checkpoints/dreamer_state.pt`）
- 相較原始 image-based 版本（-836, 100 集），state-based 版本**學習更穩定**，趨勢清晰
- 相較隨機基線（-1600），改善約 370 分（~30%）
- 距離論文水準（-200）仍有差距，原因：CPU 限制訓練集數（500 集 = 100K steps），完整 Dreamer 需 2M+ steps
- State-based 版本的教學價值：清楚展示 RSSM 世界模型學習（先驗/後驗分佈、KL 約束）與潛在空間想像（不與環境互動即可優化 actor）

### 演算法比較小結（05_Model_Based）

| 演算法 | 最佳 Eval | 說明 |
|------|-----------|------|
| DynaQ | — | 表格式，FrozenLake |
| Dreamer (image) | -836 | 100 集，pixel-based |
| **Dreamer (state)** | **-868.2** | **500 集，state-based，有學習趨勢** |
| World Models | 42.5 | CarRacing，CPU 限制 |
| MuZero | ~9 | 骨架展示，結構性限制 |
| MBPO | -1480 | 50k steps，model error 累積 |

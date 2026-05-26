> ⚠️ **本章節尚未實作**，作為延伸閱讀參考。程式碼目錄存在但無訓練記錄。

# 08 Meta-RL | 元強化學習

學習如何快速適應新任務——「學會學習」的能力。

傳統 RL 每次面對新環境都要從頭訓練；Meta-RL 的目標是訓練出一個 Agent，只需要少量樣本（few-shot）就能適應新任務。

---

## 核心概念

```
傳統 RL：任務 A 訓練 → 任務 A 的策略（換任務就重來）
Meta-RL：大量任務訓練 → 一個可以快速適應任何新任務的策略
```

Meta-RL 的訓練分兩層：
- **外層（meta-update）**：跨任務更新，讓模型學會「怎麼學」
- **內層（inner-update）**：在單一新任務上快速微調

---

## 演演算法列表

| 目錄 | 演演算法 | 中文名 | 論文 |
|------|--------|--------|------|
| `2017_MAML` | Model-Agnostic Meta-Learning | 模型無關元學習 | Finn et al. 2017 |
| `2016_RL2` | RL² (Learning to Reinforce) | 用 RNN 記憶元學習 | Wang et al. 2016 |
| `2019_PEARL` | Probabilistic Embeddings for Actor-Critic RL | 機率嵌入元強化學習 | Rakelly et al. 2019 |

---

## 學習路徑

**前置知識**：Policy Gradient（03）、Actor-Critic（04）

**建議順序**：MAML → RL² → PEARL

---

## 與其他方法的關係

| 問題 | 解決方法 |
|:---|:---|
| 新任務需要大量樣本重新訓練 | **Meta-RL**（本章） |
| 過去資料無法再與環境互動 | Offline RL（06） |
| 訓練時不能踩危險區域 | Safe RL（10） |

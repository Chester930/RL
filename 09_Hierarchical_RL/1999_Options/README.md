# Options Framework | 選項框架

> Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning  
> Sutton, Precup, Singh — Artificial Intelligence 1999

---

## 核心思想

定義「選項（Option）」作為時間上的抽象動作：

```
一個 Option o = (I, π, β)
  I ：起始條件（哪些狀態可以啟動這個選項）
  π ：選項內部的低層策略（每步怎麼走）
  β ：終止條件（什麼時候結束這個選項）
```

高層策略選擇「用哪個 Option」，低層策略負責執行到 β 觸發為止。

**直觀例子**：
- Option A：「往右走走廊」（執行若干步，直到到達右側房間）
- Option B：「拿起鑰匙」（執行若干步，直到拿到鑰匙）

---

## 環境 / 實驗設定

- 環境：Rooms（四房間格子世界）、FourRooms

---

## 檔案結構

```
1999_Options/
├── agent.py        # Option-Critic 或手動定義 Option
├── train.py        # 訓練指令碼
└── training_log.md # 訓練日誌
```

---

## 論文

- Paper: https://www.sciencedirect.com/science/article/pii/S0004370299000521

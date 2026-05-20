# MAML | 模型無關元學習

> Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks  
> Finn, Abbeel, Levine — ICML 2017

---

## 核心思想

訓練一個初始化引數 θ，使得從 θ 出發，**只需少量梯度步驟**就能適應任何新任務。

```
Meta-train：從大量任務中學到 θ*
Inner-loop ：θ' = θ - α ∇L_task(θ)     ← 任務專屬微調（1~5步）
Outer-loop ：θ  = θ - β ∇L_meta(θ')    ← 跨任務更新初始點
```

**模型無關**：可套用在任何用梯度訓練的模型（RL、監督學習都可以）。

---

## 環境 / 實驗設定

- 環境：Half-Cheetah / Ant（MuJoCo） 或 CartPole 變體
- 任務分佈：不同目標速度 / 不同重力設定

---

## 檔案結構

```
2017_MAML/
├── agent.py        # MAML Agent（inner/outer loop）
├── train.py        # 訓練指令碼
└── training_log.md # 訓練日誌
```

---

## 論文

- Paper: https://arxiv.org/abs/1703.03400

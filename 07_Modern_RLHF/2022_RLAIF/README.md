# RLAIF | AI 反饋強化學習

> Constitutional AI: Harmlessness from AI Feedback  
> Bai et al. (Anthropic) — 2022  
> RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback  
> Lee et al. (Google) — 2023

---

## 核心思想

把 RLHF 中「人類打分」的部分，改由**另一個 AI（強模型）**來打分：

```
RLHF 流程：
  LLM 產生回答 A、B → 人類選「哪個比較好」→ 訓練 Reward Model → PPO

RLAIF 流程：
  LLM 產生回答 A、B → AI（Claude/GPT-4）打分 → 訓練 Reward Model → PPO
```

**Constitutional AI（CAI）**：
  Anthropic 的變體，先給 AI 一套「憲法原則」（安全準則），  
  AI 根據原則自我批評並修改輸出 → 生成偏好資料 → 訓練 RM。

---

## RLHF vs RLAIF 對比

| 面向 | RLHF | RLAIF |
|:---|:---:|:---:|
| 標記成本 | 高（人工） | 低（API） |
| 擴充套件性 | 受限於人力 | 可大量平行 |
| 一致性 | 人與人標準不同 | AI 較一致 |
| 偏見來源 | 人類標記員偏見 | 裁判 AI 的偏見 |
| Reward Hacking 風險 | 中 | 較高（討好裁判 AI） |

---

## 環境 / 實驗設定

- 模型：SFT 後的語言模型（如 LLaMA、Mistral）
- 裁判 AI：GPT-4 / Claude 或同類更強模型
- 訓練：與 RLHF InstructGPT 流程相同，替換標記來源

---

## 檔案結構

```
2022_RLAIF/
├── generate_preferences.py  # 用 AI 生成偏好資料
├── train_reward_model.py    # 訓練 Reward Model
├── train_ppo.py             # PPO 微調主模型
└── README.md
```

---

## 論文

- Constitutional AI: https://arxiv.org/abs/2212.08073
- RLAIF (Google): https://arxiv.org/abs/2309.00267

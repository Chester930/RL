# 學習路線規劃

---

## 路線 A：快速入門（2 週）

適合：已有深度學習基礎，想快速掌握機器人 RL 核心概念

```
Week 1：傳統 RL 基礎
  Day 1-2：SAC 理論 + Pendulum/HalfCheetah 實驗
            → 理解連續動作空間 Actor-Critic
  Day 3-4：HER 理論 + FetchReach 實驗
            → 理解稀疏獎勵和目標條件學習
  Day 5-7：BC 和 GAIL 對比
            → 理解模仿學習 vs RL 的權衡

Week 2：現代方法
  Day 8-9：Diffusion Policy 概念
            → 理解多模態動作生成
  Day 10-11：VLA 大模型概覽（RT-2, OpenVLA）
              → 理解預訓練遷移到機器人
  Day 12-14：PI0 + RLT 架構
              → 掌握 2024-2026 前沿
```

---

## 路線 B：深度研究（2 個月）

適合：研究生或計劃在機器人 RL 領域發論文

```
Month 1：基礎紮實
  Week 1：經典連續控制 RL（DDPG→TD3→SAC）
           實作：Pendulum/HalfCheetah/Ant
  Week 2：稀疏獎勵操作（HER）
           實作：FetchReach/FetchPush/FetchPickAndPlace
  Week 3：模仿學習全家桶（BC→GAIL→IRL）
           實作：CartPole 示範複製對比實驗
  Week 4：世界模型機器人應用（Dreamer）
           閱讀：Dreamer v1/v2/v3 論文

Month 2：前沿技術
  Week 5：Diffusion Policy 原理 + 實作
           代碼：github.com/real-stanford/diffusion_policy
  Week 6：VLA 架構（RT-1→RT-2→OpenVLA）
           實作：OpenVLA 推理 + LoRA 微調
  Week 7：PI0 深度解析
           代碼：github.com/allenzren/open-pi-zero
  Week 8：RLT 論文精讀 + 複現設計
           論文：arXiv:2604.23073
```

---

## 路線 C：工程實作（針對想部署真實機器人）

```
Stage 1：模擬器熟悉
  MuJoCo / Isaac Sim / PyBullet / Genesis
  → 先在模擬器驗證演算法

Stage 2：Sim-to-Real 基礎
  Domain Randomization（隨機化物理參數）
  Domain Adaptation（調整感知模型）

Stage 3：真實機器人 RL
  SAC on real robot（從真實資料學）
  HER + 真實操作臂（FetchReach on real arm）

Stage 4：現代部署框架
  使用 OpenVLA 或 open-pi-zero 作為基礎
  用 RLT 對特定任務精修
```

---

## 推薦閱讀順序（論文）

| 順序 | 論文 | 重要性 | 預計時間 |
|---|---|---|---|
| 1 | SAC (Haarnoja 2018) | ⭐⭐⭐⭐⭐ | 3h |
| 2 | HER (Andrychowicz 2017) | ⭐⭐⭐⭐⭐ | 2h |
| 3 | BC/GAIL (Ho & Ermon 2016) | ⭐⭐⭐⭐ | 2h |
| 4 | Dreamer v2 (Hafner 2020) | ⭐⭐⭐⭐ | 4h |
| 5 | Diffusion Policy (Chi 2023) | ⭐⭐⭐⭐⭐ | 3h |
| 6 | RT-2 (Brohan 2023) | ⭐⭐⭐⭐ | 3h |
| 7 | π₀ (Black 2024) | ⭐⭐⭐⭐⭐ | 4h |
| 8 | RLT (Cetin 2026) | ⭐⭐⭐⭐⭐ | 3h |

# Robot Learning — 機器人學習方法全覽

> 本資料夾系統化整理機器人強化學習的完整演進路線，  
> 從 2015 年的 DDPG 到 2026 年的 RLT（RL Token）。

---

## 結構概覽

```
Robot_Learning/
├── 00_Overview/          # 技術全景、時間軸、核心概念
├── 01_Classic_RL_Robots/ # 傳統 RL 用於機器人（DDPG/TD3/SAC/HER）
├── 02_Imitation_Learning/# 模仿學習（BC/GAIL/IRL）
├── 03_Goal_Conditioned/  # 目標條件學習（HER+/GCSL）
├── 04_World_Models_Robots/# 世界模型（Dreamer 機器人應用）
├── 05_Diffusion_Policy/  # 擴散策略（Diffusion + Flow Matching）
├── 06_VLA_Foundation/    # VLA 基礎模型（RT-1/RT-2/OpenVLA）
├── 07_Physical_Intelligence/ # PI0 系列（π₀/π₀-FAST）
└── 08_RL_Finetuning/     # RL 精修（RLT/RL Token）
```

---

## 技術演進時間軸

```
2015 ──────────────────────────────────────────────── 2026
  │                                                      │
DDPG      GAIL    HER     RT-1    Diffusion  π₀     RLT
TD3        BC    Goal-RL  RT-2    Policy    π₀.7    RL Token
SAC       IRL            OpenVLA  Flow Match         精修

Era 1          Era 2              Era 3               Era 4
傳統 RL    模仿+目標學習      擴散+VLA 大模型     大模型+RL 精修
```

---

## 各章節速覽

| 章節 | 時代 | 核心技術 | 代表論文 | 狀態 |
|---|---|---|---|---|
| 01 Classic RL | 2015–2018 | Actor-Critic，連續控制 | DDPG, TD3, SAC | 📄 說明文件 |
| 02 Imitation | 2004–2019 | 示範學習，去除 reward 設計 | BC, GAIL, MaxEnt IRL | 📄 說明文件 |
| 03 Goal-Cond | 2017–2021 | 稀疏獎勵，目標重標記 | HER, GCSL | 📄 說明文件 |
| 04 World Models | 2018–2020 | 內部世界模型，想像規劃 | WorldModels, Dreamer | 📄 說明文件 |
| 05 Diffusion | 2023–2024 | 多模態動作分佈 | Diffusion Policy, Flow Match | 📄 說明文件 |
| 06 VLA | 2022–2024 | 視覺語言動作大模型 | RT-1, RT-2, OpenVLA | 📄 說明文件 |
| 07 PI | 2024–2025 | 通用機器人基礎模型 | π₀, π₀-FAST | 📄 說明文件 |
| 08 RL Finetune | 2026 | VLA + 線上 RL 精修 | RLT (RL Token) | 📄 說明文件 |

---

## 快速導覽

- **想了解全局演進** → [`00_Overview/evolution.md`](00_Overview/evolution.md)
- **學習路線規劃** → [`00_Overview/learning_path.md`](00_Overview/learning_path.md)
- **最新技術 PI0** → [`07_Physical_Intelligence/2024_PI0/README.md`](07_Physical_Intelligence/2024_PI0/README.md)
- **最新技術 RLT** → [`08_RL_Finetuning/2026_RLT/README.md`](08_RL_Finetuning/2026_RLT/README.md)

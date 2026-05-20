# Reinforcement Learning Study Project | 強化學習學習專案

A structured collection of RL algorithm implementations from tabular methods to Meta-RL, Hierarchical RL, and Safe RL, organized chronologically within each category.

---

## Directory Structure | 目錄結構

### 01 Tabular Basics | 表格型基礎方法

| Directory | Algorithm | Chinese Name | Paper |
|-----------|-----------|--------------|-------|
| `1950s_DP` | Dynamic Programming | 動態規劃 | Bellman 1957 |
| `1980s_MC` | Monte Carlo Methods | 蒙特卡羅方法 | Sutton & Barto |
| `1988_TD_Lambda` | TD(λ) | 時序差分λ / 資格跡 | Sutton 1988 |
| `1989_QLearning` | Q-Learning | Q學習 | Watkins 1989 |
| `1994_SARSA` | SARSA | SARSA同策略控制 | Rummery & Niranjan 1994 |
| `1995_NStep_TD` | N-Step TD | N步時序差分 | Sutton 1995 |

### 02 Value-Based Deep RL | 基於價值的深度強化學習

| Directory | Algorithm | Chinese Name | Paper |
|-----------|-----------|--------------|-------|
| `2013_DQN` | Deep Q-Network | 深度Q網路 | Mnih et al. 2013/2015 |
| `2015_DoubleDQN` | Double DQN | 雙重DQN | van Hasselt et al. 2015 |
| `2015_PER` | Prioritized Experience Replay | 優先經驗回放 | Schaul et al. 2015 |
| `2016_A3C` | Asynchronous Advantage Actor-Critic | 非同步優勢演員-評論家 | Mnih et al. 2016 |
| `2016_DuelingDQN` | Dueling DQN | 競爭DQN | Wang et al. 2016 |
| `2017_RainbowDQN` | Rainbow DQN | 彩虹DQN | Hessel et al. 2017 |

### 03 Policy Gradient | 策略梯度方法

| Directory | Algorithm | Chinese Name | Paper |
|-----------|-----------|--------------|-------|
| `1992_REINFORCE` | REINFORCE | 強化演演算法 | Williams 1992 |
| `2015_TRPO` | Trust Region Policy Optimization | 信任域策略最佳化 | Schulman et al. 2015 |
| `2016_A2C` | Advantage Actor-Critic | 優勢演員-評論家 | Mnih et al. 2016 |
| `2017_PPO` | Proximal Policy Optimization | 近端策略最佳化 | Schulman et al. 2017 |

### 04 Actor-Critic Continuous | 演員-評論家連續動作

| Directory | Algorithm | Chinese Name | Paper |
|-----------|-----------|--------------|-------|
| `2015_DDPG` | Deep Deterministic Policy Gradient | 深度確定性策略梯度 | Lillicrap et al. 2015 |
| `2018_TD3` | Twin Delayed DDPG | 雙延遲深度確定性策略梯度 | Fujimoto et al. 2018 |
| `2018_SAC` | Soft Actor-Critic | 軟演員-評論家 | Haarnoja et al. 2018 |

### 05 Model-Based RL | 基於模型的強化學習

| Directory | Algorithm | Chinese Name | Paper |
|-----------|-----------|--------------|-------|
| `1990_DynaQ` | Dyna-Q | Dyna-Q規劃架構 | Sutton 1990 |
| `2018_WorldModels` | World Models | 世界模型 | Ha & Schmidhuber 2018 |
| `2019_Dreamer` | Dreamer | 夢想家 | Hafner et al. 2019 |
| `2019_MuZero` | MuZero | MuZero | Schrittwieser et al. 2019 |
| `2020_MBPO` | Model-Based Policy Optimization | 基於模型的策略最佳化 | Janner et al. 2019 |

### 06 Advanced & Specialized | 進階與專題方法

| Directory | Algorithm | Chinese Name | Paper |
|-----------|-----------|--------------|-------|
| `2017_C51_DistRL` | C51 Distributional RL | C51分散式強化學習 | Bellemare et al. 2017 |
| `2017_HER` | Hindsight Experience Replay | 事後經驗回放 | Andrychowicz et al. 2017 |
| `2017_ICM` | Intrinsic Curiosity Module | 內在好奇心模組 | Pathak et al. 2017 |
| `2017_MADDPG_MARL` | Multi-Agent DDPG | 多智慧體DDPG | Lowe et al. 2017 |
| `2020_CQL_Offline` | Conservative Q-Learning | 保守Q學習（離線RL） | Kumar et al. 2020 |
| `2021_IQL_Offline` | Implicit Q-Learning | 隱式Q學習（離線RL） | Kostrikov et al. 2021 |
| `2021_MAPPO_MARL` | Multi-Agent PPO | 多智慧體PPO | Yu et al. 2021 |

### 07 Modern RLHF | 現代人類回饋強化學習

| Directory | Algorithm | Chinese Name | Paper |
|-----------|-----------|--------------|-------|
| `2022_RLHF_InstructGPT` | RLHF / InstructGPT | 人類回饋強化學習 | Ouyang et al. 2022 |
| `2022_RLAIF` | RLAIF / Constitutional AI | AI回饋強化學習 | Anthropic 2022 / Google 2023 |
| `2023_DPO` | Direct Preference Optimization | 直接偏好最佳化 | Rafailov et al. 2023 |
| `2024_GRPO` | Group Relative Policy Optimization | 群組相對策略最佳化 | DeepSeek 2024 |

### 08 Meta-RL | 元強化學習

| Directory | Algorithm | Chinese Name | Paper |
|-----------|-----------|--------------|-------|
| `2016_RL2` | RL² (Learning to Reinforce) | 迴圈記憶元學習 | Wang et al. 2016 |
| `2017_MAML` | Model-Agnostic Meta-Learning | 模型無關元學習 | Finn et al. 2017 |
| `2019_PEARL` | Probabilistic Embeddings for Actor-Critic RL | 機率嵌入元強化學習 | Rakelly et al. 2019 |

### 09 Hierarchical RL | 層次強化學習

| Directory | Algorithm | Chinese Name | Paper |
|-----------|-----------|--------------|-------|
| `1999_Options` | Options Framework | 選項框架 / 時間抽象 | Sutton et al. 1999 |
| `2017_FeUdal` | FeUdal Networks | 封建網路層次架構 | Vezhnevets et al. 2017 |
| `2018_HIRO` | Hierarchical RL with Off-Policy Correction | 離策略修正層次RL | Nachum et al. 2018 |

### 10 Safe RL | 安全強化學習

| Directory | Algorithm | Chinese Name | Paper |
|-----------|-----------|--------------|-------|
| `2017_CPO` | Constrained Policy Optimization | 約束策略最佳化 | Achiam et al. 2017 |
| `2019_PPO_Lagrangian` | Lagrangian PPO | 拉格朗日安全PPO | Ray et al. 2019 |

---

## Common Infrastructure | 共用基礎設施

```
common/
├── base_agent.py          # Abstract base class for all agents
├── buffers/
│   ├── replay_buffer.py   # Standard experience replay
│   └── priority_buffer.py # Prioritized experience replay (SumTree)
├── networks/
│   ├── mlp.py             # Multi-layer perceptron
│   └── cnn.py             # Nature CNN for Atari
├── envs/
│   └── wrappers.py        # Gymnasium environment wrappers
└── utils/
    ├── logger.py           # TensorBoard logger
    ├── scheduler.py        # Epsilon schedulers
    └── evaluator.py        # Policy evaluation utility
```

---

## Quick Start | 快速開始

```bash
# Install dependencies
pip install -r requirements.txt

# Run Q-Learning on FrozenLake
cd 01_Tabular_Basics/1989_QLearning
python train.py

# Run DQN on CartPole
cd 02_Value_Based_Deep/2013_DQN
python train.py

# Run PPO on LunarLander
cd 03_Policy_Gradient/2017_PPO
python train.py
```

---

## Learning Path | 學習路徑

**Beginner | 初級**: DP → MC → Q-Learning → SARSA  
**Intermediate | 中級**: DQN → Double DQN → Dueling DQN → REINFORCE → PPO  
**Advanced | 高階**: SAC → TD3 → Rainbow → Dreamer → MuZero  
**Specialized | 專題**: HER → ICM → C51 → Offline RL → MARL → RLHF → RLAIF  
**Frontier | 前沿**: Meta-RL (MAML → RL² → PEARL) → Hierarchical RL (Options → FeUdal → HIRO) → Safe RL (CPO → PPO-Lagrangian)  

---

## References | 參考資料

- Sutton & Barto: *Reinforcement Learning: An Introduction* (2nd ed.)
- Spinning Up in Deep RL: https://spinningup.openai.com
- Stable Baselines3: https://stable-baselines3.readthedocs.io

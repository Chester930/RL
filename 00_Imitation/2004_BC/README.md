# Behavioral Cloning（行為複製）

## 論文

Pomerleau, D. A. (1991).  
*Efficient training of artificial neural networks for autonomous navigation.*  
Neural Computation, 3(1), 88–97.

---

## 核心思想

Behavioral Cloning 是最直覺的模仿學習方法：
**收集人類（或現有 policy）的示範，直接用監督學習複製它。**

```python
# 訓練：讓 policy 的輸出盡量接近專家動作
loss = MSE(policy(state), expert_action)

# 推論：直接拿 policy 輸出當動作
action = policy(state)
```

RL 中最難的部分是「獎勵設計」和「探索」——BC 完全繞開這兩個問題。
這讓它成為許多機器人任務的起點。

---

## 架構

```
Pendulum-v1 狀態（3 維）：[cos(θ), sin(θ), θ̇]
                ↓
        MLP  256 × 256
                ↓
            tanh × 2.0
                ↓
    扭矩動作（1 維，範圍 [-2, 2]）
```

與 SAC 的 PolicyNetwork 比較：

| | BC | SAC |
|---|---|---|
| 策略型別 | 確定性（直接輸出動作）| 隨機性（輸出均值 + std）|
| 訓練訊號 | MSE（監督學習）| Bellman + Entropy |
| 需要獎勵？ | ❌ | ✅ |
| 需要探索？ | ❌（從 demo 取樣）| ✅（ε / entropy）|

---

## 核心限制：Distribution Shift（分佈偏移）

BC 只學過「專家訪問過的狀態」。在這些狀態裡，BC 能做出接近專家的動作。

**問題來自部署時的誤差累積：**

```
時刻 t：   正常狀態 s_t（訓練分佈內）→ BC 動作接近正確
時刻 t+1： 因為 BC 動作有小誤差 → s_{t+1} 略微偏離訓練分佈
時刻 t+2： 更偏離 → BC 動作誤差更大
    ...
時刻 t+k： 完全出軌，BC 從未見過這種狀態 → 隨機亂動
```

比喻：學生只讀過考試範本，遇到一道沒見過的題目就緊張，
緊張導致下一題也做錯，誤差雪球般越滾越大。

**Pendulum 的量化展示：**

SAC expert 在 Pendulum 上快速將擺錘直立（θ→0），
因此 demo 資料 80%+ 都是 θ ≈ 0 附近的狀態。

| 初始角度 | SAC Expert | BC Policy | 說明 |
|---|---|---|---|
| 0°（直立）| -xxx | -xxx | BC 在訓練分佈內，表現接近專家 |
| 90°（水平）| -xxx | -xxx | BC 開始失準 |
| 180°（朝下）| -xxx | -xxx | BC 幾乎從未見過此狀態 → 崩潰 |

*（實際數字在跑完 train.py 後填入）*

---

## 使用方式

```bash
cd 00_Imitation/2004_BC

# 完整流程：收集 demo → 訓練 → 分佈偏移測試
python train.py

# 已有 demos.npz 時跳過收集
python train.py --skip-collect

# 單獨收集 demo（100 集）
python collect_demos.py --episodes 100
```

**前置條件**：SAC checkpoint 必須存在於
`04_Actor_Critic_Continuous/2018_SAC/checkpoints_pendulum/sac.pt`

---

## 延伸方向

| 問題 | 解法 |
|---|---|
| Distribution Shift | **DAgger**（2011）：讓專家在 BC 走錯的地方補充示範 |
| 需要精確動作標註 | **GAIL**（2016）：用 GAN 判別「像不像專家」代替 MSE |
| 無法從失敗中學習 | **RLHF**（2022）：BC 熱身 + PPO 用人類偏好微調 |

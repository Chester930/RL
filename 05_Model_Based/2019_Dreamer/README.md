# Dreamer — 透過潛在想像學習行為

## 論文

Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2019).  
*Dream to Control: Learning Behaviors by Latent Imagination*. ICLR 2020. arXiv:1912.01603.

後續研究：DreamerV2 (2020), DreamerV3 (2023, arXiv:2301.04104)

---

## 核心思想 (Key Idea)

Dreamer 從畫素觀測中學習一個**精簡的世界模型 (Compact world model)**，接著完全在「潛在想像 (Latent imagination)」中訓練行為 — 在行為學習過程中完全不需要與真實環境互動。

```
真實環境 -> 觀測影像 -> 世界模型 -> 精簡潛在空間
                                      |
                                  想像 (想像預測 H 個步數)
                                      |
                                  演員-評論家 (Actor-Critic) 訓練
```

---

## RSSM — 迴圈狀態空間模型 (Recurrent State Space Model)

世界模型為每個潛在狀態使用兩個核心元件：

```
確定性狀態：h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})  (記憶，不含隨機雜訊)
隨機性狀態：z_t ~ q(z_t | h_t, x_t)                (後驗分佈，使用觀測影像)
      或者：z_t ~ p(z_t | h_t)                      (先驗分佈，用於預測未來)
```

結合狀態：`s_t = (h_t, z_t)` — 這就是供演員與評論家使用的完整狀態表示。

---

## 世界模型訓練 (World Model Training)

透過最佳化三個損失函式來同時學習：

```
1. 影像重建： E[||x_t - decoder(h_t, z_t)||^2]
2. 獎勵預測： E[||r_t - reward_head(h_t, z_t)||^2]
3. KL 散度：  E[KL(q(z|h,x) || p(z|h))]  (強迫先驗分佈在無觀測時也能精準預測後驗分佈)
```

---

## 行為學習：想像 (Behavior Learning: Imagination)

從真實的潛在狀態出發，使用**先驗分佈 (PRIOR)**（完全不依賴新的觀測影像）想像 $H=15$ 個時間步：

```
針對步數 t = 1..H:
    a_t = actor(h_t, z_t)              (演員根據想像選擇動作)
    h_{t+1}, z_{t+1} = RSSM.img_step(h_t, z_t, a_t)  (先驗狀態轉移)
    r_t = reward_head(h_t, z_t)        (想像的獎勵)
    v_t = critic(h_t, z_t)             (想像的價值)

Lambda 回報： V_t = r_t + gamma * [(1-lambda) * v_{t+1} + lambda * V_{t+1}]
```

演員 (Actor) 目標是極大化 $V_t$，評論家 (Critic) 則負責預測 $V_t$。

---

## 架構總結 (Architecture Summary)

```
編碼器 (Encoder)： 影像 (C, H, W) -> 嵌入向量 (1024)
RSSM：            (h, z, a, 嵌入向量) -> (h', z')  [後驗，用於編碼真實經驗]
                  (h, z, a)           -> (h', z')  [先驗，用於想像未來]
解碼器 (Decoder)： (h, z) -> 原始影像
獎勵 (Reward)：    (h, z) -> 純量 r
演員 (Actor)：     (h, z) -> 連續動作
價值 (Value)：     (h, z) -> 純量 V
```

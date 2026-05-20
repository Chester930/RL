# MBPO — 透過模型信賴機制最佳化策略 (Model-Based Policy Optimization)

## 論文

Janner, M., Fu, J., Zhang, M., & Levine, S. (2019).  
*When to Trust Your Model: Model-Based Policy Optimization*.  
NeurIPS 2019. arXiv:1906.08253.

---

## 核心思想 (Key Idea)

短步數的模型生成取樣（分支長度 $k$）能顯著提升樣本效率，同時避免了長步數想像軌跡所產生的累積誤差 (Compounding error)。

```
理論基礎：k 步模型誤差的增長量級為 O(k * epsilon_m / (1-gamma))
=> 短步數取樣能將模型偏差 (Model bias) 控制在有限範圍內
=> 混合真實資料 (5%) 與想像資料 (95%) 來進行 SAC 更新
```

---

## 演演算法流程 (Algorithm)

```
初始化：整合模型 (Ensemble model)、SAC 策略、真實緩衝區 (Real_buffer)、模型緩衝區 (Model_buffer)

針對每個真實環境步數：
    1. 收集真實轉換資料 (Real transition) -> 存入 real_buffer
    2. 每隔 N 步：在 real_buffer 上訓練整合模型
    3. 從真實狀態出發，透過模型進行 k 步分支取樣 -> 存入 model_buffer
    4. 在混合批次（5% 真實 + 95% 模型資料）上執行 G 次 SAC 梯度更新
```

---

## 整合動態模型 (Ensemble Dynamics Model)

- 包含 7 個機率網路，根據驗證集上的負對數似然 (NLL) 選擇 5 個「精銳 (Elite)」網路。
- 每個網路皆預測：`mu, log_var = f(s, a)` 用於估計狀態變化 `delta_s` 與獎勵 `r`。
- 使用具備可學習變異數邊界的 Gaussian NLL 損失函式。
- 透過整合模型間的不一致性 (Disagreement) 來衡量認知不確定性 (Epistemic uncertainty)。

---

## 展開長度排程 (Rollout Length Schedule)

原始論文在訓練過程中將取樣長度 $k$ 從 1 逐步增加至 15：
```python
k = min(1 + step // 40000, 15)  # 範例排程方式
```

訓練初期使用短 $k$（較安全），後期使用長 $k$（待模型完善後，增加取樣長度可進一步提升資料效率）。

---

## 演演算法對比 (Comparison)

| 方法 | 樣本效率 (Sample Efficiency) | 漸近效能 (最終表現) |
|--------|------------------|-----------------|
| SAC (無模型) | 低 | 高 |
| MBPO k=1 | 極高 | 高 |
| MBPO k=15 | 高 | 高 |
| Dyna-Q | 中 | 中 |

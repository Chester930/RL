# 06 — VLA 基礎模型（2022–2024）

> **時代定位**：將大型視覺語言模型能力遷移到機器人控制  
> **核心突破**：一個模型理解語言指令 + 看圖 + 輸出動作

---

## VLA 的誕生背景

```
語言模型（LLM）發展到 2022 年：
  GPT-4 / PaLM / LLaMA → 強大的推理和語言理解
  
機器人的困境：
  每個任務都要從頭訓練
  「把紅球放進藍盒子」vs「把藍球放進紅盒子」= 兩個不同模型

VLA 的願景：
  一個模型 + 語言描述任意任務
  「pick up the red cup」→ 動作序列
  「open the drawer」→ 動作序列
  （不需要重新訓練！）
```

---

## 模型演進

### 2022_RT1 — Robotics Transformer 1
```
論文：Brohan et al. (2022), arXiv:2212.06817
機構：Google

規模：
  130,000 次示範軌跡
  700 種任務
  13 種物體類別

架構：
  EfficientNet（視覺）+ Token Learner（壓縮）
  + Transformer（序列建模）
  → 直接輸出離散動作 Token

結果：
  泛化到新任務：97% 成功率（訓練任務）
  遷移到新物體：25% 成功率（未見物體）
  
局限：視覺理解和語言推理能力有限
```

### 2023_RT2 — Robotics Transformer 2
```
論文：Brohan et al. (2023), arXiv:2307.15818
機構：Google DeepMind

核心創新：用大型 VLM（PaLI-X 55B）初始化
  VLM 預訓練：理解圖像 + 語言 → 知道「可口可樂是紅色的」
  機器人微調：把這個知識用到動作預測

結果 vs RT-1：
  泛化新任務：62% → 92%（顯著提升！）
  緊急推理（新能力）：
    「把可以清潔電腦的東西給我」→ 去拿洗手液（零樣本推理）

架構：VLM 的文字 Token 直接改為動作 Token
```

### 2024_OpenVLA — 開源版本
```
論文：Kim et al. (2024), arXiv:2406.09246
機構：Stanford, UC Berkeley, Toyota...

動機：RT-2 不開源，研究社群無法複現和改進
做法：
  基底：Llama 2 7B + DINOv2 視覺編碼器
  訓練資料：Open X-Embodiment（970K 示範）
  開源：模型權重 + 訓練代碼全公開

效能：
  比 RT-2 小 10× 但效果相近
  LoRA 微調可適配特定機器人（<1 小時）

代碼：github.com/openvla/openvla
```

---

## VLA 的通用架構

```
輸入處理：
  RGB 圖像 ──→ 視覺編碼器（CLIP/DINOv2/SigLIP）
                    ↓ 視覺 Token
  語言指令 ──→ 語言 Token
  機器人狀態 → 狀態 Token（可選）
                    ↓
                    全部合併

推理主幹：
  Transformer / VLM
  （處理視覺 + 語言 + 狀態的聯合理解）

動作解碼：
  方式 1：自迴歸輸出離散動作 Token（RT-1/RT-2/OpenVLA）
  方式 2：連續動作專家（Diffusion/Flow，如 π₀）
```

---

## 三大 VLA 對比

| 維度 | RT-1 | RT-2 | OpenVLA |
|---|---|---|---|
| 開源 | ❌ | ❌ | ✅ |
| 基底模型 | 從頭訓練 | PaLI-X 55B | Llama 2 7B |
| 訓練資料 | 130K | 130K+ | 970K (OXE) |
| 語言理解 | 基本 | 強（VLM 繼承）| 強 |
| 參數量 | 35M | 55B | 7.5B |
| 推理速度 | 快 | 慢 | 中 |

---

## VLA 的局限與下一步

```
局限：
  1. 動作精度不夠（大模型輸出粗略動作 Token）
  2. 推理延遲高（55B 模型需要 GPU 加速）
  3. 微調成本高（全量微調需要大量 GPU）

下一步（→ 07 PI0, 08 RLT）：
  PI0：換 Diffusion/Flow 動作解碼器 → 精度大幅提升
  RLT：凍結 VLA，用 RL Token 精修 → 推理快 + 高精度
```

---

## 參考論文
- RT-1: [arXiv:2212.06817](https://arxiv.org/abs/2212.06817)
- RT-2: [arXiv:2307.15818](https://arxiv.org/abs/2307.15818)
- OpenVLA: [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)
- Open X-Embodiment: [arXiv:2310.08864](https://arxiv.org/abs/2310.08864)

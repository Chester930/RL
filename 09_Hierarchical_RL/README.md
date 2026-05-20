# 09 Hierarchical RL | 層次強化學習

把長時間跨度的複雜任務，拆解成「高層目標」+「低層執行」的兩級結構。

---

## 核心概念

```
普通 RL：每一步都要自己決定（動幅細、序列長）
Hierarchical RL：
  高層 Manager → 決定「子目標」（去哪個區域）
  低層 Worker  → 決定「細部動作」（怎麼走到那裡）
```

**解決的問題**：
- 獎勵非常稀疏（要走幾百步才能拿到一次分）
- 任務需要長期規劃（煮飯 = 買菜 → 備料 → 烹飪）

---

## 演演算法列表

| 目錄 | 演演算法 | 中文名 | 論文 |
|------|--------|--------|------|
| `1999_Options` | Options Framework | 選項框架 / 時間抽象 | Sutton et al. 1999 |
| `2017_FeUdal` | FeUdal Networks | 封建網路層次架構 | Vezhnevets et al. 2017 |
| `2018_HIRO` | Hierarchical RL with Off-Policy Correction | 離策略修正層次RL | Nachum et al. 2018 |

---

## 學習路徑

**前置知識**：Policy Gradient（03）、Actor-Critic（04）

**建議順序**：Options → FeUdal → HIRO

---

## 直覺比喻

> 你要去日本旅遊（長序列任務）。  
> **高層**決定：先去東京、再去京都、最後去大阪。  
> **低層**決定：現在要往左走、搭哪班車、在哪裡買票。  
> 兩層各自學習，互不幹擾。

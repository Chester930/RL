# HER — Hindsight Experience Replay (2017)

論文：Andrychowicz et al., arXiv:1707.01495 | 機構：OpenAI

## 核心洞察
失敗的嘗試中藏有成功的資訊：
「雖然沒到目標 g，但到了 g'」→ 把這次當成目標 g' 的成功！

## 重標記策略
- future（推薦）：從軌跡後半段隨機取 k 個狀態作為新目標
- episode：從整集任意位置取
- random：從所有歷史集取

## 本專案結果
環境：FetchReach-v4 | Epoch 160 首達 100% | 400-500 ep 穩定 100%

## 相關資源
- 論文：https://arxiv.org/abs/1707.01495
- 本專案實作：../../06_Advanced_Specialized/2017_HER/


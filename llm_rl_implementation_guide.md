# 強化學習 (RL) 實作指南：如何與 LLM 結對程式設計

這份指南旨在幫助你利用大型語言模型 (LLM) 作為「蘇格拉底式的助教」，來協助完成任何強化學習 (Reinforcement Learning) 演算法的程式碼實作。

雖然本指南以 **Q-Learning** 為具體範例，但此工作流程完全適用於其他的 RL 演算法實作（例如：Sarsa, DQN, REINFORCE, PPO 等）。

---

## 1. 設定 LLM 助教角色 (System Prompt)

在開始撰寫程式碼之前，首先要「限制」LLM 的行為，避免它直接把完整答案丟給你而失去練習的意義。請在新的對話開頭貼上這段提示詞：

> [!IMPORTANT]
> **通用提示詞範本：**
> 「你現在是我的強化學習助教。我正在實作 `[填入演算法名稱，如 Q-Learning]` 演算法的 `[填入函數名稱，如 update]` 函數。請 **不要** 直接給我完整的程式碼解答。請用引導、提問的方式，幫助我理解數學公式如何轉換為 Python 程式碼。當我給出錯誤的程式碼時，請點出我的邏輯盲點。」

---

## 2. 提供上下文 (Context)

LLM 需要知道你目前的變數命名與系統架構，才能給出精準的建議。請將相關的程式碼骨架 (Skeleton) 或介面定義貼給它。

> **互動範例：**
> 「助教，這是我的函數介面與 docstring。請幫我釐清接下來的實作步驟。」
> ```python
> def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> dict:
>     # TODO: 實作單步更新
>     pass
> ```

---

## 3. 引導式實作步驟 (Prompting Workflow)

將複雜的 RL 演算法拆解為三個階段與 LLM 進行討論。

### Step 3.1: 拆解數學公式對應
強化學習的核心是數學公式（如 Bellman Equation 或 Policy Gradient）。第一步是確認數學符號如何對應到你的程式碼變數。

> **🗣 你的提問 (以 Q-Learning 為例):**
> 「Q-Learning 的 TD Target 公式是 $r + \gamma \max_{a'} Q(s', a')$。請幫我對應這個公式裡的變數，與我函數中的 `reward`, `self.gamma`, `self.Q` 之間的關係？在 NumPy 裡面 $\max_{a'}$ 要怎麼寫？」

### Step 3.2: 處理環境的邊界條件 (Terminal States)
RL 環境中最重要的邊界條件就是 Episode 的結束 (`done=True`)。這通常會改變預測目標 (Target) 的計算方式。

> **🗣 你的提問:**
> 「程式碼裡面有一個 `done` 變數。當環境回傳 `done=True`（代表 Episode 結束）時，我的 Target 計算方式應該有什麼不同？我該如何寫 if/else 來處理這段邏輯？」

### Step 3.3: 實作核心邏輯與更新
當公式拆解完畢後，嘗試自己寫出程式碼，並請 LLM 檢查語法或邏輯錯誤（例如常見的 in-place update 寫錯）。

> **🗣 你的提問:**
> 「我已經算好 `target` 了。接下來要計算 TD Error 以及更新 Q Table。請檢查我以下的程式碼實作是否有觀念上的錯誤：
> ```python
> td_error = target - self.Q[state, action]
> self.Q[state, action] = self.alpha * td_error # 我這樣寫對嗎？
> ```」
> *(註：LLM 應該要能抓出這裡漏寫了 `+` 號，正確應為 `+=`)*

---

## 4. 驗證與 Code Review

當你完成了所有的 `# TODO` 區塊，請將完整的函數貼給 LLM，進行最後的 Code Review。

> **🗣 你的提問:**
> 「這是我完成的 `[演算法名稱]` 實作。請幫我做 Code Review：
> 1. 檢查有沒有潛在的邊界條件錯誤（例如 array index out of bounds 或 tensor shape 不對）。
> 2. 演算法邏輯是否完全符合原始論文的定義？
> 3. 在 Python/NumPy (或 PyTorch) 的執行效能上，有沒有可以向量化 (Vectorize) 的優化空間？」

---

## 5. 進階挑戰與延伸探討 (Extensions)

實作完成後，可以利用 LLM 探討如何擴充演算法，加深對超參數的理解。

> [!TIP]
> **探索方向提問範例：**
> *   **超參數調整:** 「如果我想把目前的 epsilon-greedy 策略改成隨著訓練步數慢慢衰減 (Epsilon Decay)，我應該在哪裡修改架構？」
> *   **演算法變體:** 「如果要將這份程式碼從 Q-Learning 改成 Sarsa，我需要修改介面中的哪些輸入變數？更新規則需要做什麼樣的微調？」
> *   **分析與視覺化:** 「我應該記錄哪些指標 (Metrics) 才能畫出漂亮的 Learning Curve 來證明我的 Agent 有在學習？」

"""
TD(lambda) 代理人，使用資格跡 (Eligibility Traces)。

透過資格跡銜接蒙特卡羅 (lambda=1) 與 TD(0) (lambda=0)。
實作同策略控制的後向觀點 (Backward-view，累加式跡)。

參考文獻：
    Sutton, R. S. (1988). Learning to predict by the methods of temporal
    differences. Machine Learning, 3(1), 9–44.
    Sutton & Barto, RL: An Introduction, Ch. 12
"""

import numpy as np


class TDLambdaAgent:
    """
    表格型 TD(lambda) SARSA，使用累加式資格跡。

    資格跡提供跨時間的歸因 (Credit Assignment)：每個被訪問過的「狀態-動作」對都會獲得一個跡，該跡每步會以 (gamma*lambda) 的幾何比例衰減。TD 誤差會分配給所有最近訪問過的對。

    引數：
        n_states:  離散狀態的數量。
        n_actions: 離散動作的數量。
        alpha:     學習率（步長）。
        gamma:     折扣因子。
        lam:       跡衰減引數 lambda，範圍 [0, 1]。
                   0 = TD(0) (僅限當前步驟)，1 = 蒙特卡羅。
        epsilon:   用於 epsilon-greedy 探索的 epsilon。
        trace_type: "accumulate" (累加) 或 "replace" (替換，替換式跡可避免數值失控)。
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        lam: float = 0.9,
        epsilon: float = 0.1,
        trace_type: str = "accumulate",
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.trace_type = trace_type

        # Q-table
        self.Q: np.ndarray = np.zeros((n_states, n_actions), dtype=np.float64)

        # 資格跡 E(s, a) — 每個集數開始時重置
        self.E: np.ndarray = np.zeros((n_states, n_actions), dtype=np.float64)

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def select_action(self, state: int, evaluate: bool = False) -> int:
        """當前策略下的 Epsilon-greedy 動作選擇。"""
        if evaluate or np.random.random() > self.epsilon:
            return int(np.argmax(self.Q[state]))
        return int(np.random.randint(self.n_actions))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def reset_traces(self) -> None:
        """在每個集數開始時清除資格跡。"""
        self.E[:] = 0.0

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: int,
        done: bool,
    ) -> dict:
        """
        使用資格跡進行 TD(lambda) 後向更新。

        步驟：
        1. 計算 TD 誤差 delta (與 SARSA 相同)
        2. 增加當前 (s, a) 的跡
        3. 按比例更新所有 (s, a) 對，權重為其跡 E(s, a)
        4. 依 gamma * lambda 衰減所有跡

        引數：
            state:       當前狀態 s。
            action:      採取的動作 a。
            reward:      獎勵 r。
            next_state:  下一個狀態 s'。
            next_action: 下一個動作 a' (來自當前策略，同策略)。
            done:        集數是否結束。

        回傳：
            指標字典，包含 "td_error" 與 "max_trace"。
        """
        # --- 步驟 1: TD 誤差 (與 SARSA(0) 相同) ---
        # TODO: delta = r + gamma * Q(s', a') - Q(s, a)   (同策略目標)
        if done:
            td_error = reward - self.Q[state, action]
        else:
            td_error = (
                reward
                + self.gamma * self.Q[next_state, next_action]
                - self.Q[state, action]
            )

        # --- 步驟 2: 更新訪問過的 (s, a) 的資格跡 ---
        # TODO: 累加式跡：E(s, a) += 1
        #        替換式跡：E(s, :) = 0; E(s, a) = 1
        if self.trace_type == "replace":
            self.E[state, :] = 0.0
            self.E[state, action] = 1.0
        else:
            self.E[state, action] += 1.0

        # --- 步驟 3: 按跡比例更新所有 Q 值 ---
        # TODO: Q(s, a) <- Q(s, a) + alpha * delta * E(s, a) 對所有 s, a 進行更新
        self.Q += self.alpha * td_error * self.E

        # --- 步驟 4: 衰減跡 ---
        # TODO: E(s, a) <- gamma * lambda * E(s, a) 對所有 s, a 進行更新
        if done:
            self.E[:] = 0.0   # 集數結束，重置跡
        else:
            self.E *= self.gamma * self.lam

        return {
            "td_error": float(td_error),
            "max_trace": float(self.E.max()),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        import os
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "Q.npy"), self.Q)

    def load(self, path: str) -> None:
        import os
        self.Q = np.load(os.path.join(path, "Q.npy"))

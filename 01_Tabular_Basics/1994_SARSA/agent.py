"""
SARSA 代理人 — 同策略時序差分控制 (On-policy TD Control)。

參考文獻：
    Rummery, G. A., & Niranjan, M. (1994). On-Line Q-Learning Using
    Connectionist Systems. Technical Report, Cambridge University.
"""

import numpy as np


class SARSAAgent:
    """
    同策略 SARSA 代理人，使用 epsilon-greedy 探索。

    與 Q-Learning 的核心差異：SARSA 使用在下一狀態「實際採取」的動作 (a') 進行更新，而非貪婪動作。這使其成為同策略 — 學習到的價值反映了探索行為。

    更新規則：
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)]

    其中 a' 是從當前 epsilon-greedy 策略取樣，而非取最大值 (max)。

    引數：
        n_states:  離散狀態的數量。
        n_actions: 離散動作的數量。
        alpha:     學習率。
        gamma:     折扣因子。
        epsilon:   用於 epsilon-greedy 策略的初始 epsilon。
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.Q: np.ndarray = np.zeros((n_states, n_actions), dtype=np.float64)

        # 儲存下一個動作，以便訓練迴圈可以將其傳回作為 'action'
        self._next_action: int = None

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def select_action(self, state: int, evaluate: bool = False) -> int:
        """
        Epsilon-greedy 動作選擇（當前策略 pi）。

        注意：在 SARSA 中，同一個策略同時用於行為和學習。
        """
        if evaluate or np.random.random() > self.epsilon:
            return int(np.argmax(self.Q[state]))
        return int(np.random.randint(self.n_actions))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

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
        SARSA 單步更新 (狀態-動作-獎勵-狀態-動作)。

        五元組 (s, a, r, s', a') 是 SARSA 名稱的由來。

        TD 目標 (同策略):
            y = r + gamma * Q(s', a')   [其中 a' ~ pi(s')，而非取最大值 max]

        引數：
            state:       s
            action:      a
            reward:      r
            next_state:  s'
            next_action: a'  (從當前策略取樣，而非貪婪動作)
            done:        集數結束標記

        回傳：
            指標字典，包含 "td_error"。
        """
        # TODO: 計算同策略 TD 目標
        #   若結束：目標 = r
        #   否則：  目標 = r + gamma * Q(s', a')   <-- 與 Q-Learning 的關鍵差異
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.Q[next_state, next_action]

        td_error = target - self.Q[state, action]
        # TODO: 應用更新
        self.Q[state, action] += self.alpha * td_error

        return {"td_error": float(td_error)}

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

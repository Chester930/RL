"""
Q-Learning 代理人 (Watkins, 1989)。

異策略時序差分控制 (Off-policy TD control) — 無論使用何種行為策略收集資料，都能直接學習最佳 Q*。

參考文獻：
    Watkins, C. J. C. H. (1989). Learning from Delayed Rewards.
    PhD thesis, Cambridge University.
"""

import numpy as np


class QLearningAgent:
    """
    表格型 Q-Learning，使用 epsilon-greedy 探索。

    更新規則：
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]

    對下一動作取最大值使其成為異策略：它始終以 s' 的貪婪（最佳）動作為目標，無論實際採取了什麼動作。

    引數：
        n_states:  離散狀態的數量。
        n_actions: 離散動作的數量。
        alpha:     學習率。
        gamma:     折扣因子。
        epsilon:   用於 epsilon-greedy 探索的初始 epsilon。
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

        # Q 表初始化為零（樂觀：零對被限制的獎勵有效）
        self.Q: np.ndarray = np.zeros((n_states, n_actions), dtype=np.float64)

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def select_action(self, state: int, evaluate: bool = False) -> int:
        """
        Epsilon-greedy 動作選擇。

        引數：
            state:    當前離散狀態索引。
            evaluate: 若為 True，則始終選擇貪婪動作。

        回傳：
            動作 (int)
        """
        if evaluate or np.random.random() > self.epsilon:
            # 貪婪：隨機打破平手
            q_vals = self.Q[state]
            best_q = q_vals.max()
            best_actions = np.where(q_vals == best_q)[0]
            return int(np.random.choice(best_actions))
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
        done: bool,
    ) -> dict:
        """
        單步 Q-Learning 更新。

        TD 目標 (異策略):
            y = r + gamma * max_{a'} Q(s', a')  (若結束則僅為 r)

        TD 誤差:
            delta = y - Q(s, a)

        更新:
            Q(s, a) <- Q(s, a) + alpha * delta

        引數：
            state:      當前狀態 s。
            action:     採取的動作 a。
            reward:     觀察到的獎勵 r。
            next_state: 結果狀態 s'。
            done:       集數是否結束。

        回傳：
            指標字典，包含 "td_error"。
        """
        # TODO: 計算 TD 目標
        #   若結束：目標 = r
        #   否則：  目標 = r + gamma * max_a' Q[next_state, a']
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.Q[next_state].max()

        td_error = target - self.Q[state, action]
        # TODO: 應用更新規則
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

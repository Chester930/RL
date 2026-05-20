"""
蒙特卡羅代理人 (Monte Carlo Agent) — 同策略首次存取 MC 控制 (On-policy first-visit MC control)。

不需要環境模型；從完整的集數 (Episode) 中學習。

參考文獻：
    Sutton & Barto, RL: An Introduction, Ch. 5
"""

import numpy as np
from collections import defaultdict
from typing import List, Tuple


class MCAgent:
    """
    同策略首次存取蒙特卡羅控制，使用 epsilon-soft 策略。

    維護一個 Q 表 Q[s][a]，估計為每個集數中首次存取 (s, a) 後觀察到的所有回報之平均值。

    引數：
        n_states:  離散狀態的數量。
        n_actions: 離散動作的數量。
        gamma:     折扣因子。
        epsilon:   用於 epsilon-greedy/soft 策略的 epsilon。
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon

        # Q[s][a] = 估計的動作價值
        self.Q: np.ndarray = np.zeros((n_states, n_actions), dtype=np.float64)

        # returns_sum[s][a] 與 returns_count[s][a] 用於增量平均
        self.returns_sum: np.ndarray = np.zeros((n_states, n_actions), dtype=np.float64)
        self.returns_count: np.ndarray = np.zeros((n_states, n_actions), dtype=np.int64)

        # 當前集數緩衝：(state, action, reward) 的列表
        self._episode: List[Tuple[int, int, float]] = []

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def select_action(self, state: int, evaluate: bool = False) -> int:
        """
        Epsilon-greedy (epsilon-soft) 動作選擇。

        評估期間，使用完全貪婪策略。
        """
        if evaluate or np.random.random() > self.epsilon:
            return int(np.argmax(self.Q[state]))
        return int(np.random.randint(self.n_actions))

    def store_transition(self, state: int, action: int, reward: float) -> None:
        """將 (s, a, r) 附加到當前集數緩衝中。"""
        self._episode.append((state, action, reward))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self) -> dict:
        """
        使用首次存取 MC 處理完成的集數。

        每個集數結束時呼叫一次。

        回傳：
            metrics 指標字典，包含 "n_updates" (更新的 (s,a) 對數量)。
        """
        if not self._episode:
            return {"n_updates": 0}

        # --- 由後往前計算折扣回報 G_t ---
        G = 0.0
        returns: List[Tuple[int, int, float]] = []
        for s, a, r in reversed(self._episode):
            # TODO: G_t = r_t + gamma * G_{t+1}
            G = r + self.gamma * G
            returns.append((s, a, G))
        returns.reverse()  # 依時間順序排序

        # --- 首次存取：僅更新每個 (s,a) 對的第一次出現 ---
        visited = set()
        n_updates = 0
        for s, a, G in returns:
            if (s, a) not in visited:
                visited.add((s, a))
                # TODO: 增量平均更新：Q(s,a) <- Q(s,a) + (1/N) * (G - Q(s,a))
                self.returns_sum[s, a] += G
                self.returns_count[s, a] += 1
                self.Q[s, a] = self.returns_sum[s, a] / self.returns_count[s, a]
                n_updates += 1

        self._episode.clear()
        return {"n_updates": n_updates}

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

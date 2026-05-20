"""
N-步時序差分代理人 (N-Step TD Agent)。

透過在引導 (Bootstrapping) 前向前看 n 步，歸納了 TD(0) (n=1) 與蒙特卡羅 (n=無限)。

參考文獻：
    Sutton, R. S. (1988). Learning to predict by the methods of temporal differences.
    Sutton & Barto, RL: An Introduction, Ch. 7
"""

import numpy as np
from collections import deque
from typing import Deque, Tuple


class NStepTDAgent:
    """
    表格型 n-步 SARSA 代理人，使用 n-步回報緩衝。

    N-步回報：
        G_{t:t+n} = r_{t+1} + gamma*r_{t+2} + ... + gamma^{n-1}*r_{t+n}
                    + gamma^n * Q(s_{t+n}, a_{t+n})

    在觀察到 n 次轉移後，對 Q(s_t, a_t) 應用此更新。
    緩衝區儲存最後 n 個 (s, a, r) 元組。

    引數：
        n_states:  離散狀態的數量。
        n_actions: 離散動作的數量。
        n:         前瞻步數 (n=1 為 TD，n=無限 為 MC)。
        alpha:     學習率。
        gamma:     折扣因子。
        epsilon:   Epsilon-greedy 探索引數。
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        n: int = 4,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.Q: np.ndarray = np.zeros((n_states, n_actions), dtype=np.float64)

        # 用於 n-步回報的 (state, action, reward) 元組滾動緩衝區
        self._buffer: Deque[Tuple[int, int, float]] = deque()

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def select_action(self, state: int, evaluate: bool = False) -> int:
        """Epsilon-greedy 動作選擇。"""
        if evaluate or np.random.random() > self.epsilon:
            return int(np.argmax(self.Q[state]))
        return int(np.random.randint(self.n_actions))

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def store(self, state: int, action: int, reward: float) -> None:
        """將轉移附加到 n-步緩衝區中。"""
        self._buffer.append((state, action, reward))

    def is_ready(self) -> bool:
        """當緩衝區包含至少 n 個轉移時為 True。"""
        return len(self._buffer) >= self.n

    def clear_buffer(self) -> None:
        """清空緩衝區 (在集數結束時呼叫以清空剩餘轉移)。"""
        pass  # 由 update(flush=True) 處理

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(
        self,
        bootstrap_state: int,
        bootstrap_action: int,
        done: bool,
        flush: bool = False,
    ) -> dict:
        """
        計算 n-步回報並更新 Q(s_t, a_t)。

        當緩衝區有 n 個專案 (正常) 或集數結束 (清空) 時呼叫。

        引數：
            bootstrap_state:  用於引導 (Bootstrapping) 的狀態 s_{t+n}。
            bootstrap_action: 用於引導的動作 a_{t+n}。
            done:             集數是否結束。
            flush:            若為 True，則排乾剩餘的緩衝區專案。

        回傳：
            指標字典，包含 "td_error"，若轉移不足則為空。
        """
        if not flush and not self.is_ready():
            return {}

        n_updates = max(1, len(self._buffer)) if flush else 1
        total_td_error = 0.0

        for _ in range(n_updates):
            if not self._buffer:
                break

            # 更新目標狀態是緩衝區中最舊的專案
            s_t, a_t, _ = self._buffer[0]

            # --- 計算 n-步回報 G_{t:t+n} ---
            # TODO: G = sum_{k=0}^{n-1} gamma^k * r_{t+k+1}
            #           + gamma^n * Q(s_{t+n}, a_{t+n})  (if not done)
            G = 0.0
            for k, (_, _, r) in enumerate(self._buffer):
                G += (self.gamma ** k) * r

            # 除非集數結束，否則使用 Q(s_{t+n}, a_{t+n}) 進行引導 (Bootstrap)
            if not done or len(self._buffer) < self.n:
                G += (self.gamma ** len(self._buffer)) * self.Q[bootstrap_state, bootstrap_action]

            # --- 更新 Q(s_t, a_t) ---
            td_error = G - self.Q[s_t, a_t]
            self.Q[s_t, a_t] += self.alpha * td_error
            total_td_error += td_error

            self._buffer.popleft()

        return {"td_error": float(total_td_error / max(1, n_updates))}

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

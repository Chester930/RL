"""
Dyna-Q 代理人 — 整合基於模型與無模型之強化學習。

參考文獻：
    Sutton, R. S. (1990). Integrated Architectures for Learning, Planning, and
    Reacting Based on Approximating Dynamic Programming.
    Machine Learning Proceedings, 216-224.
    Sutton & Barto, RL: An Introduction, Ch. 8.
"""

import numpy as np
import random
from typing import Dict, Tuple


class DynaQAgent:
    """
    Dyna-Q：藉由簡單的查表式世界模型 (Tabular world model) 增強 Q-Learning。

    Dyna 架構結合了：
    1. 直接強化學習：從真實經驗中學習（Q-Learning 更新）
    2. 模型學習：記住觀察到的狀態轉換 (s, a) -> (r, s')
    3. 規劃：從模型中模擬 K 次額外更新

    規劃能有效地在每一次真實步數中提供 K 次額外的「虛擬」經驗，
    從而顯著提高查表式問題的樣本效率 (Sample efficiency)。

    引數：
        n_states:    離散狀態的數量。
        n_actions:   離散動作的數量。
        alpha:       Q-learning 學習率。
        gamma:       折扣因子。
        epsilon:     Epsilon-greedy 探索引數。
        n_planning:  每一步真實環境互動後進行的規劃步數（論文中的 K）。
                     K=0 為純 Q-Learning，K=inf 則趨近於動態規劃。
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        n_planning: int = 10,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_planning = n_planning

        # Q 表 (Q-table)
        self.Q: np.ndarray = np.zeros((n_states, n_actions), dtype=np.float64)

        # 查表式世界模型：model[s][a] = (獎勵, 下一個狀態)
        # 僅儲存確定性轉換（簡化實作）
        self.model: Dict[int, Dict[int, Tuple[float, int]]] = {}

        # 已造訪過的 (s, a) 對集合（用於規劃時的取樣）
        self._visited: list = []

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def select_action(self, state: int, evaluate: bool = False) -> int:
        """Epsilon-greedy 動作選擇。"""
        if evaluate or np.random.random() > self.epsilon:
            return int(np.argmax(self.Q[state]))
        return int(np.random.randint(self.n_actions))

    # ------------------------------------------------------------------
    # 學習 (直接強化學習 + 模型更新 + 規劃)
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
        針對單次真實環境步數的 Dyna-Q 更新。

        步驟：
        1. 對 (s, a, r, s') 執行直接 Q-Learning 更新
        2. 模型更新：儲存 (s, a) -> (r, s')
        3. 規劃：從模型中執行 K 次模擬 Q-Learning 更新

        回傳：
            包含 "td_error"、"model_size"、"n_planning_steps" 的指標字典。
        """
        # --- 步驟 1: 直接強化學習 (Q-Learning) ---
        # TODO: 標準 Q-Learning 更新
        target = reward if done else reward + self.gamma * self.Q[next_state].max()
        td_error = target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

        # --- 步驟 2: 模型學習 ---
        # TODO: 在查表式模型中儲存觀察到的轉換
        if state not in self.model:
            self.model[state] = {}
        self.model[state][action] = (reward, next_state)

        if (state, action) not in [pair for pair in self._visited]:
            self._visited.append((state, action))

        # --- 步驟 3: 規劃 ---
        # TODO: 取樣 K 個先前見過的 (s,a) 對並進行模擬 Q 更新
        for _ in range(self.n_planning):
            if not self._visited:
                break
            s_sim, a_sim = random.choice(self._visited)
            r_sim, ns_sim = self.model[s_sim][a_sim]
            # 針對模擬轉換執行 Q-Learning 更新
            sim_target = r_sim + self.gamma * self.Q[ns_sim].max()
            self.Q[s_sim, a_sim] += self.alpha * (sim_target - self.Q[s_sim, a_sim])

        return {
            "td_error": float(td_error),
            "model_size": sum(len(v) for v in self.model.values()),
            "n_planning_steps": self.n_planning,
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

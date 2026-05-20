"""
動態規劃代理人 (Dynamic Programming Agent) — 價值疊代 (Value Iteration) 與 策略疊代 (Policy Iteration)。

需要完全已知的 MDP 模型（轉移機率與獎勵）。
執行於離散且有限的狀態與動作空間。

參考文獻：
    Bellman, R. (1957). Dynamic Programming. Princeton University Press.
    Sutton & Barto, RL: An Introduction, Ch. 4
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class DPAgent:
    """
    表格型 DP 代理人，使用價值疊代或策略疊代在已知模型的有限 MDP 中尋找最佳策略。

    環境必須提供：
        env.n_states   (int) — 狀態總數
        env.n_actions  (int) — 動作總數
        env.P          (dict) — Gymnasium 風格的轉移字典
                       P[s][a] = List[(prob, next_s, reward, done)]

    引數：
        n_states:  MDP 中的狀態數量。
        n_actions: MDP 中的動作數量。
        gamma:     折扣因子，範圍為 [0, 1)。
        theta:     價值函式更新的收斂閾值。
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = 0.99,
        theta: float = 1e-6,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.theta = theta

        # 狀態價值函式 V(s) — 初始化為零
        self.V: np.ndarray = np.zeros(n_states, dtype=np.float64)

        # 確定性策略 pi(s) -> a — 隨機初始化
        self.policy: np.ndarray = np.zeros(n_states, dtype=np.int64)

    # ------------------------------------------------------------------
    # Value Iteration
    # ------------------------------------------------------------------

    def value_iteration(self, env) -> int:
        """
        透過價值疊代尋找 V* 及其對應的貪婪策略。

        遍歷所有狀態，應用貝爾曼最佳化運算元：
            V(s) <- max_a sum_{s'} p(s'|s,a) [r + gamma * V(s')]

        直到每次遍歷的最大變化量低於 theta。

        引數：
            env: 包含 P[s][a] 轉移表的環境。

        回傳：
            n_sweeps: 直到收斂的遍歷次數。
        """
        n_sweeps = 0
        while True:
            delta = 0.0
            for s in range(self.n_states):
                v_old = self.V[s]
                # TODO: 貝爾曼最佳化更新
                #   q_values = [每個動作 a 的 (機率 * (獎勵 + gamma * V[s'])) 之和]
                q_values = self._compute_q_values(s, env.P)
                self.V[s] = np.max(q_values)
                delta = max(delta, abs(v_old - self.V[s]))
            n_sweeps += 1
            if delta < self.theta:
                break

        # 從 V* 提取貪婪策略
        self._extract_policy(env.P)
        return n_sweeps

    # ------------------------------------------------------------------
    # Policy Iteration
    # ------------------------------------------------------------------

    def policy_iteration(self, env) -> int:
        """
        透過交替進行策略評估與策略改進來尋找最佳策略。

        步驟：
            1. 策略評估：計算 V^pi 直到收斂。
            2. 策略改進：使策略對當前 V 呈現貪婪 (Greedy)。
            重複上述步驟直到策略穩定。

        引數：
            env: 包含 P[s][a] 轉移表的環境。

        回傳：
            n_iterations: 策略改進的迭代次數。
        """
        n_iterations = 0
        while True:
            # --- 步驟 1: 策略評估 ---
            self._policy_evaluation(env.P)

            # --- 步驟 2: 策略改進 ---
            policy_stable = True
            for s in range(self.n_states):
                old_action = self.policy[s]
                # TODO: 貪婪改進
                #   self.policy[s] = argmax_a Q(s, a)
                q_values = self._compute_q_values(s, env.P)
                self.policy[s] = np.argmax(q_values)
                if old_action != self.policy[s]:
                    policy_stable = False

            n_iterations += 1
            if policy_stable:
                break

        return n_iterations

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _policy_evaluation(self, P: dict) -> None:
        """迭代評估當前策略，直到 V 收斂。"""
        while True:
            delta = 0.0
            for s in range(self.n_states):
                v_old = self.V[s]
                a = self.policy[s]
                # TODO: 固定策略的貝爾曼期望更新
                #   V(s) = sum_{s',r} p(s',r|s,pi(s)) [r + gamma * V(s')]
                self.V[s] = sum(
                    prob * (r + self.gamma * self.V[s_next] * (1 - done))
                    for prob, s_next, r, done in P[s][a]
                )
                delta = max(delta, abs(v_old - self.V[s]))
            if delta < self.theta:
                break

    def _compute_q_values(self, s: int, P: dict) -> np.ndarray:
        """
        根據當前價值函式計算所有動作 a 的 Q(s, a)。

        Q(s, a) = sum_{s'} p(s'|s,a) [r + gamma * V(s')]
        """
        q_values = np.zeros(self.n_actions, dtype=np.float64)
        for a in range(self.n_actions):
            for prob, s_next, r, done in P[s][a]:
                q_values[a] += prob * (r + self.gamma * self.V[s_next] * (1 - done))
        return q_values

    def _extract_policy(self, P: dict) -> None:
        """從收斂的價值函式中推匯出貪婪策略。"""
        for s in range(self.n_states):
            q_values = self._compute_q_values(s, P)
            self.policy[s] = np.argmax(q_values)

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def select_action(self, state: int) -> int:
        """回傳當前策略指定的動作。"""
        return int(self.policy[state])

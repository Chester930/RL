"""
Options 代理人 — 時間抽象的階層式強化學習。

核心思想：
  普通 RL：每步選一個原始動作 a ∈ A
  Options：每步選一個 option o ∈ O
            每個 option 自帶：
              初始集（何時可啟動）
              終止條件（何時終止）
              內部策略（如何執行）

  高層策略（High-level）：選擇 option → SMDP Q-learning
  低層策略（Low-level） ：每個 option 各有一個 Q-table，
                          目標是抵達自己的子目標（走廊位置）

整個架構：
  step:
    若無執行中 option → 高層選 option
    執行 option 的低層動作
    若 option 終止 → 更新高層 Q(s, o)
    更新低層 Q_o(s, a) with option-specific reward

參考：
    Sutton, R.S., Precup, D., & Singh, S. (1999).
    Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning.
    Artificial Intelligence, 112(1-2), 181-211.
"""

import numpy as np
from typing import Optional, Tuple, List
from env import DOORWAYS, GOAL, N_ACTIONS, H, W


class OptionsAgent:
    """
    Options 框架代理人。

    Options 設計（4 個走廊 option + 1 個目標 option）：
      option 0–3：前往對應 DOORWAYS[i]
      （終止條件：到達走廊位置 OR 達到最大步數）

    高層：SMDP Q-learning，狀態 = flat(r,c)，動作 = option index
    低層：每個 option 有獨立 Q-table，狀態 = flat(r,c)，動作 = 4 個移動

    引數：
        n_states:     格子世界的狀態數（H×W）
        n_options:    option 數量（= 走廊數）
        alpha_hi:     高層 Q-learning 學習率
        alpha_lo:     低層 Q-learning 學習率
        gamma:        折扣因子
        eps_hi:       高層探索率（ε-greedy）
        eps_lo:       低層探索率
        option_steps: 每個 option 最多執行幾步
    """

    def __init__(
        self,
        n_states: int,
        n_options: int = 4,
        alpha_hi: float = 0.1,
        alpha_lo: float = 0.1,
        gamma: float = 0.99,
        eps_hi: float = 0.3,
        eps_lo: float = 0.1,
        option_steps: int = 50,
    ):
        self.n_states = n_states
        self.n_options = n_options
        self.alpha_hi = alpha_hi
        self.alpha_lo = alpha_lo
        self.gamma = gamma
        self.eps_hi = eps_hi
        self.eps_lo = eps_lo
        self.option_steps = option_steps

        # 高層 Q-table：Q_hi[state, option]
        self.Q_hi = np.zeros((n_states, n_options))

        # 低層 Q-table：Q_lo[option][state, action]
        self.Q_lo = [np.zeros((n_states, N_ACTIONS)) for _ in range(n_options)]

        # 執行時狀態
        self.current_option: Optional[int] = None
        self.option_start_state: Optional[int] = None
        self.option_cumulative_reward: float = 0.0
        self.option_gamma_acc: float = 1.0
        self.option_step_count: int = 0
        self.total_steps: int = 0

    # ------------------------------------------------------------------
    # 走廊 / 目標 判斷
    # ------------------------------------------------------------------

    def _doorway_of_option(self, option: int) -> Tuple[int, int]:
        return DOORWAYS[option]

    def _option_terminates(self, state_rc: Tuple[int, int], option: int) -> bool:
        """判斷 option 是否應在此狀態終止。"""
        target = self._doorway_of_option(option)
        return state_rc == target or state_rc == GOAL

    def _option_reward(self, state_rc: Tuple[int, int], option: int) -> float:
        """低層 option-specific 獎勵：抵達子目標 +1，其他 -0.001。"""
        target = self._doorway_of_option(option)
        if state_rc == target or state_rc == GOAL:
            return 1.0
        return -0.001

    # ------------------------------------------------------------------
    # 動作選擇
    # ------------------------------------------------------------------

    def select_option(self, state: int) -> int:
        """高層 ε-greedy：選 option。"""
        if np.random.random() < self.eps_hi:
            return np.random.randint(self.n_options)
        return int(np.argmax(self.Q_hi[state]))

    def select_primitive_action(self, state: int, option: int) -> int:
        """低層 ε-greedy：在 option 內選原始動作。"""
        if np.random.random() < self.eps_lo:
            return np.random.randint(N_ACTIONS)
        return int(np.argmax(self.Q_lo[option][state]))

    # ------------------------------------------------------------------
    # 更新
    # ------------------------------------------------------------------

    def update_low(
        self,
        option: int,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        """低層 Q-learning 更新。"""
        target = reward + (0.0 if done else self.gamma * np.max(self.Q_lo[option][next_state]))
        self.Q_lo[option][state, action] += self.alpha_lo * (target - self.Q_lo[option][state, action])

    def update_high(
        self,
        option: int,
        start_state: int,
        cumulative_reward: float,
        gamma_acc: float,
        next_state: int,
        done: bool,
    ) -> None:
        """
        高層 SMDP Q-learning 更新。

        SMDP 的 Q-learning target（k 步的 option 折扣）：
          Q(s, o) ← Q(s, o) + α × [G_option + γ^k × max_o' Q(s', o') - Q(s, o)]
        """
        if done:
            target = cumulative_reward
        else:
            target = cumulative_reward + gamma_acc * np.max(self.Q_hi[next_state])
        self.Q_hi[start_state, option] += self.alpha_hi * (target - self.Q_hi[start_state, option])

    # ------------------------------------------------------------------
    # 完整 step 介面（給訓練迴圈使用）
    # ------------------------------------------------------------------

    def act(self, state: int, state_rc: Tuple[int, int]) -> int:
        """
        在當前狀態選擇原始動作（高層選 option → 低層選動作）。
        """
        if self.current_option is None or self._option_terminates(state_rc, self.current_option):
            # 先更新高層（若有執行中的 option 即將終止）
            if self.current_option is not None:
                self.update_high(
                    self.current_option,
                    self.option_start_state,
                    self.option_cumulative_reward,
                    self.option_gamma_acc,
                    state,
                    done=False,
                )
            # 選新 option
            self.current_option = self.select_option(state)
            self.option_start_state = state
            self.option_cumulative_reward = 0.0
            self.option_gamma_acc = 1.0
            self.option_step_count = 0

        return self.select_primitive_action(state, self.current_option)

    def observe(
        self,
        state: int,
        action: int,
        next_state: int,
        state_rc: Tuple[int, int],
        next_state_rc: Tuple[int, int],
        env_reward: float,
        done: bool,
    ) -> None:
        """
        觀察環境回傳，更新低層 Q-table，並在 option 終止時更新高層。
        """
        if self.current_option is None:
            return

        option = self.current_option
        opt_reward = self._option_reward(next_state_rc, option)

        # 低層更新
        terminates = self._option_terminates(next_state_rc, option)
        self.update_low(option, state, action, opt_reward, next_state, terminates or done)

        # 累積 option 獎勵（以環境獎勵為基礎，供高層使用）
        self.option_cumulative_reward += self.option_gamma_acc * env_reward
        self.option_gamma_acc *= self.gamma
        self.option_step_count += 1
        self.total_steps += 1

        # option 終止條件：到達子目標、環境終止、或超過最大步數
        if terminates or done or self.option_step_count >= self.option_steps:
            self.update_high(
                option,
                self.option_start_state,
                self.option_cumulative_reward,
                self.option_gamma_acc,
                next_state,
                done,
            )
            self.current_option = None

    def reset_option(self) -> None:
        """集數結束時重置 option 狀態。"""
        self.current_option = None
        self.option_start_state = None
        self.option_cumulative_reward = 0.0
        self.option_gamma_acc = 1.0
        self.option_step_count = 0

    def save(self, path: str) -> None:
        import os
        import pickle
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "options_agent.pkl"), "wb") as f:
            pickle.dump({"Q_hi": self.Q_hi, "Q_lo": self.Q_lo}, f)

    def load(self, path: str) -> None:
        import os
        import pickle
        with open(os.path.join(path, "options_agent.pkl"), "rb") as f:
            ckpt = pickle.load(f)
        self.Q_hi = ckpt["Q_hi"]
        self.Q_lo = ckpt["Q_lo"]

    def save_resume(self, path: str) -> None:
        self.save(path)

    def load_resume(self, path: str) -> None:
        self.load(path)

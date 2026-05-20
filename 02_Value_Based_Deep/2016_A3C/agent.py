"""
A3C 代理人 — 非同步優勢演員-評論家 (Asynchronous Advantage Actor-Critic)。

注意：這是一個單執行緒的實作，旨在展現 A3C 的核心數學思想。
完整的非同步版本會使用 Python 的 multiprocessing 模組來執行多個工作者，
並同時更新共享的全域網路（請參見 train.py 中關於如何擴充套件的說明）。

參考文獻：
    Mnih, V., et al. (2016). Asynchronous Methods for Deep Reinforcement
    Learning. ICML 2016. arXiv:1602.01783.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from common.base_agent import BaseAgent
from network import ActorCriticNetwork


class A3CAgent(BaseAgent):
    """
    Single-threaded A3C (equivalent to A2C when synchronous).

    A3C uses n-step returns to estimate the advantage:
        A(s_t, a_t) = G_{t:t+n} - V(s_t)

    Loss functions:
        Actor loss:  -log(pi(a|s)) * A(s, a)
        Critic loss: (V(s) - G_target)^2
        Entropy bonus: -entropy(pi(s))  [prevents premature convergence]

        Total loss = actor_loss + c_v * critic_loss - c_e * entropy

    Args:
        state_dim:   Input state dimension.
        action_dim:  Number of discrete actions.
        lr:          Learning rate for the shared network.
        gamma:       Discount factor.
        n_steps:     Number of steps before computing returns (rollout length).
        c_v:         Critic loss coefficient.
        c_e:         Entropy bonus coefficient.
        device:      "cpu" or "cuda".
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 7e-4,
        gamma: float = 0.99,
        n_steps: int = 5,
        c_v: float = 0.5,
        c_e: float = 0.01,
        device: str = "cpu",
    ):
        super().__init__(state_dim, action_dim, device)
        self.gamma = gamma
        self.n_steps = n_steps
        self.c_v = c_v
        self.c_e = c_e

        # 全域/共享網路 (在非同步版本中，工作者會從此處複製權重)
        self.global_net = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.global_net.parameters(), lr=lr)

        # 資料收集緩衝區 (Rollout buffer，每 n_steps 重置一次)
        self._rollout_states = []
        self._rollout_actions = []
        self._rollout_rewards = []
        self._rollout_dones = []

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        從當前策略中取樣動作。

        在評估期間，使用貪婪 (argmax) 動作。
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.global_net(state_t)
            if evaluate:
                return int(logits.argmax(dim=1).item())
            dist = torch.distributions.Categorical(logits=logits)
            return int(dist.sample().item())

    def store(self, state, action, reward, done):
        """將一次轉移存入 n-步收集緩衝區 (Rollout buffer)。"""
        self._rollout_states.append(state)
        self._rollout_actions.append(action)
        self._rollout_rewards.append(reward)
        self._rollout_dones.append(done)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self, next_state: np.ndarray = None, last_done: bool = False) -> dict:
        """
        計算 n-步回報並更新演員-評論家網路。

        步驟：
        1. 計算引導值 V(s_{t+n}) (若已結束則為 0)
        2. 反向累積折扣回報：G_t = r_t + gamma * G_{t+1}
        3. 計算優勢：A_t = G_t - V(s_t)
        4. 演員損失 = -mean(log_pi(a_t|s_t) * A_t)
        5. 評論家損失 = mean((G_t - V(s_t))^2)
        6. 熵獎勵 = -mean(entropy(pi(s_t)))
        7. 總損失 = 演員損失 + c_v * 評論家損失 - c_e * 熵獎勵

        引數：
            next_state: 用於引導 (Bootstrapping) 的狀態 s_{t+n} (若集數結束則為 None)。
            last_done:  Rollout 是否結束於終止狀態。

        回傳：
            包含各項損失的指標字典。
        """
        if not self._rollout_states:
            return {}

        T = len(self._rollout_states)

        # --- 引導值 (Bootstrap value) ---
        if last_done or next_state is None:
            R = 0.0
        else:
            s_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, bootstrap_v = self.global_net(s_t)
            R = bootstrap_v.item()

        # --- 反向計算 n-步回報 ---
        # TODO: G_t = r_t + gamma * G_{t+1}
        returns = []
        for r, done in zip(reversed(self._rollout_rewards), reversed(self._rollout_dones)):
            R = r + self.gamma * R * (1.0 - float(done))
            returns.insert(0, R)

        returns_t = torch.FloatTensor(returns).to(self.device)
        states_t = torch.FloatTensor(np.array(self._rollout_states)).to(self.device)
        actions_t = torch.LongTensor(self._rollout_actions).to(self.device)

        # --- 前向傳遞 (Forward pass) ---
        log_probs, entropy, values = self.global_net.evaluate_actions(states_t, actions_t)
        values = values.squeeze(1)

        # --- 優勢 (Advantage) = G_t - V(s_t) ---
        advantages = (returns_t - values).detach()

        # TODO: Actor loss = -E[log pi(a|s) * A(s,a)]
        actor_loss = -(log_probs * advantages).mean()

        # TODO: Critic loss = E[(V(s) - G_t)^2]
        critic_loss = nn.functional.mse_loss(values, returns_t)

        # TODO: Entropy bonus = E[H(pi(s))]
        entropy_loss = -entropy.mean()

        total_loss = actor_loss + self.c_v * critic_loss + self.c_e * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.global_net.parameters(), max_norm=40.0)    # 梯度裁剪
        self.optimizer.step()

        self.total_steps += T
        self._rollout_states.clear()
        self._rollout_actions.clear()
        self._rollout_rewards.clear()
        self._rollout_dones.clear()

        return {
            "total_loss": float(total_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(-entropy_loss.item()),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        self._ensure_dir(path)
        torch.save({
            "global_net":  self.global_net.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, os.path.join(path, "a3c.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "a3c.pt"), map_location=self.device)
        self.global_net.load_state_dict(ckpt["global_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt.get("total_steps", 0)

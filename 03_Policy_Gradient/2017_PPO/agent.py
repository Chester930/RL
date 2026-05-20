"""
PPO 代理人 — 近端策略最佳化 (Proximal Policy Optimization)。

參考文獻：
    Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
    Proximal Policy Optimization Algorithms. arXiv:1707.06347.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List

from common.base_agent import BaseAgent
from network import ActorCriticNetwork


class PPOAgent(BaseAgent):
    """
    具備剪裁代理目標函式與 GAE 的 PPO。

    PPO 透過使用「剪裁 (Clipping)」啟發式方法取代強 KL 約束，簡化了 TRPO 的信任域。
    剪裁後的目標函式可防止策略在單次更新中變動過大：

        L_CLIP(theta) = E_t [ min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t) ]

    在收集 T 個步數後，PPO 會在相同的資料上執行 K 個梯度的更新週期 (Epochs)。
    這是與 A2C 等在策略 (On-policy) 方法的主要區別：資料重複利用。

    引數：
        n_steps:     每次更新之間的環境步數。
        n_epochs:    每次取樣後的梯度更新週期數。
        n_minibatch: 每個週期的小批次 (Mini-batch) 數量。
        clip_eps:    PPO 剪裁引數 epsilon (通常為 0.1 或 0.2)。
        gae_lambda:  優勢估計所需的 GAE lambda 引數。
        ent_coef:    熵係數。
        vf_coef:     價值函式損失係數。
        max_grad_norm: 梯度的最大範數限制。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        n_steps: int = 2048,
        n_epochs: int = 10,
        n_minibatch: int = 32,
        clip_eps: float = 0.2,
        gae_lambda: float = 0.95,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        super().__init__(state_dim, action_dim, device)
        self.gamma = gamma
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.n_minibatch = n_minibatch
        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.net = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)

        # 資料取樣儲存區 (預先配置)
        self._obs = np.zeros((n_steps, state_dim), dtype=np.float32)
        self._actions = np.zeros(n_steps, dtype=np.int64)
        self._rewards = np.zeros(n_steps, dtype=np.float32)
        self._dones = np.zeros(n_steps, dtype=np.float32)
        self._log_probs = np.zeros(n_steps, dtype=np.float32)
        self._values = np.zeros(n_steps, dtype=np.float32)
        self._step = 0  # 取樣指標

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, _, value = self.net.get_action_and_value(state_t)

        if evaluate:
            logits, _ = self.net(state_t)
            return int(logits.argmax(dim=1).item())

        if self._step < self.n_steps:
            self._obs[self._step] = state
            self._actions[self._step] = int(action.item())
            self._log_probs[self._step] = float(log_prob.item())
            self._values[self._step] = float(value.item())

        return int(action.item())

    def store_reward_done(self, reward: float, done: bool) -> None:
        if self._step < self.n_steps:
            self._rewards[self._step] = reward
            self._dones[self._step] = float(done)
            self._step += 1

    def is_ready(self) -> bool:
        return self._step >= self.n_steps

    # ------------------------------------------------------------------
    # 學習 (剪裁代理目標函式)
    # ------------------------------------------------------------------

    def update(self, next_state: np.ndarray, last_done: bool) -> dict:
        """
        對收集到的取樣資料進行 PPO 更新。

        步驟：
        1. 計算 GAE 優勢函式與回報
        2. 重複進行 n_epochs 個週期：
           a. 打亂資料並分割為多個小批次 (Mini-batches)
           b. 針對每個小批次：
              - 計算新的對數機率與價值
              - 計算機率比率 r = pi_new / pi_old
              - 剪裁代理目標：min(r*A, clip(r, 1-eps, 1+eps)*A)
              - 計算總損失並更新網路
        """
        T = self._step
        if T == 0:
            return {}

        # 引導值 (Bootstrap value)
        if last_done:
            next_value = 0.0
        else:
            ns_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, next_value_t = self.net(ns_t)
            next_value = float(next_value_t.item())

        # --- GAE 優勢函式 (GAE advantages) ---
        # TODO: A_t = sum_k (gamma*lambda)^k * delta_{t+k}
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        values_ext = np.append(self._values[:T], next_value)
        for t in reversed(range(T)):
            d = self._dones[t]
            delta = self._rewards[t] + self.gamma * values_ext[t+1] * (1 - d) - values_ext[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - d) * gae
            advantages[t] = gae

        returns = advantages + self._values[:T]

        # Convert to tensors
        b_obs = torch.FloatTensor(self._obs[:T]).to(self.device)
        b_actions = torch.LongTensor(self._actions[:T]).to(self.device)
        b_log_probs = torch.FloatTensor(self._log_probs[:T]).to(self.device)
        b_advantages = torch.FloatTensor(advantages).to(self.device)
        b_returns = torch.FloatTensor(returns).to(self.device)

        # 優勢函式歸一化 (Normalize advantages)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        metrics_list = []
        minibatch_size = max(1, T // self.n_minibatch)

        for _ in range(self.n_epochs):
            indices = torch.randperm(T, device=self.device)

            for start in range(0, T, minibatch_size):
                end = min(start + minibatch_size, T)
                mb_idx = indices[start:end]

                mb_obs = b_obs[mb_idx]
                mb_actions = b_actions[mb_idx]
                mb_old_log_probs = b_log_probs[mb_idx]
                mb_advantages = b_advantages[mb_idx]
                mb_returns = b_returns[mb_idx]

                _, new_log_probs, entropy, new_values = self.net.get_action_and_value(
                    mb_obs, mb_actions
                )
                new_values = new_values.squeeze(1)

                # 機率比率 r = pi_new / pi_old
                # TODO: ratio = exp(new_log_probs - old_log_probs)
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = log_ratio.exp()

                # TODO: 剪裁代理目標函式 (Clipped surrogate objective)
                # L_CLIP = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # 價值函式損失 (採用 PPO 論文中的剪裁版本)
                critic_loss = nn.functional.mse_loss(new_values, mb_returns)

                # 熵獎勵 (Entropy bonus)
                entropy_loss = -entropy.mean()

                loss = actor_loss + self.vf_coef * critic_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()

                metrics_list.append({
                    "loss": loss.item(),
                    "actor_loss": actor_loss.item(),
                    "critic_loss": critic_loss.item(),
                    "entropy": -entropy_loss.item(),
                    "approx_kl": approx_kl,
                    "clip_frac": ((ratio - 1).abs() > self.clip_eps).float().mean().item(),
                })

        self.total_steps += T
        self._step = 0  # 重置取樣指標

        return {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0]}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        self._ensure_dir(path)
        torch.save({
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, os.path.join(path, "ppo.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "ppo.pt"), map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

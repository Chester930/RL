"""
A2C 代理人 — 優勢演員-評論家 (Advantage Actor-Critic，A3C 的同步版本)。

參考文獻：
    Mnih, V., et al. (2016). Asynchronous Methods for Deep Reinforcement Learning.
    A2C 是由 OpenAI 推廣的同步版本。
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


class A2CAgent(BaseAgent):
    """
    同步優勢演員-評論家 (A2C)。

    A2C 同時從 N 個環境中各收集 n 步資料，等待所有環境完成後，
    再使用合併後的批次資料統一進行更新。這比非同步的 A3C 更適合 GPU，
    因為它支援批次化處理。

    為了簡化起見，本實作使用單一環境。

    核心特性：
    - 搭配 GAE (廣義優勢估計) 的 n-步回報
    - 用於探索的熵獎勵 (Entropy bonus)
    - 共享的演員-評論家網路
    - 不使用回放緩衝區 (在策略，On-policy)

    引數：
        gae_lambda: GAE 平滑引數 (0 = TD 優勢, 1 = MC 優勢)。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 7e-4,
        gamma: float = 0.99,
        n_steps: int = 5,
        gae_lambda: float = 0.95,
        c_v: float = 0.5,
        c_e: float = 0.01,
        device: str = "cpu",
    ):
        super().__init__(state_dim, action_dim, device)
        self.gamma = gamma
        self.n_steps = n_steps
        self.gae_lambda = gae_lambda
        self.c_v = c_v
        self.c_e = c_e

        self.net = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.RMSprop(self.net.parameters(), lr=lr, eps=1e-5)

        # 資料取樣儲存區 (Rollout storage)
        self._states: List[np.ndarray] = []
        self._actions: List[int] = []
        self._rewards: List[float] = []
        self._values: List[float] = []
        self._dones: List[bool] = []

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.net(state_t)
        if evaluate:
            return int(logits.argmax(dim=1).item())
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        # 儲存價值以便後續用於 GAE 計算
        self._values.append(float(value.item()))
        self._states.append(state)
        self._actions.append(int(action.item()))
        return int(action.item())

    def store_reward_done(self, reward: float, done: bool) -> None:
        """儲存當前步的獎勵與結束狀態。"""
        self._rewards.append(reward)
        self._dones.append(done)

    # ------------------------------------------------------------------
    # Learning (GAE + Actor-Critic update)
    # ------------------------------------------------------------------

    def update(self, next_state: np.ndarray = None, last_done: bool = False) -> dict:
        """
        計算 GAE 優勢函式並更新演員-評論家網路。

        GAE 優勢估計：
            delta_t = r_t + gamma * V(s_{t+1}) * (1-done) - V(s_t)
            A_t = sum_{k=0}^{T-t-1} (gamma * lambda)^k * delta_{t+k}

        回傳：
            指標字典。
        """
        T = len(self._states)
        if T == 0:
            return {}

        # 引導值 (Bootstrap value)
        if last_done or next_state is None:
            next_value = 0.0
        else:
            ns_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, v = self.net(ns_t)
            next_value = float(v.item())

        # --- GAE 優勢函式計算 (GAE advantage computation) ---
        # TODO: A_t = sum_k (gamma*lambda)^k * delta_{t+k}
        advantages = torch.zeros(T, device=self.device)
        gae = 0.0
        values = self._values + [next_value]
        for t in reversed(range(T)):
            d = float(self._dones[t])
            delta = self._rewards[t] + self.gamma * values[t+1] * (1 - d) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - d) * gae
            advantages[t] = gae

        returns = advantages + torch.FloatTensor(self._values).to(self.device)

        # 優勢函式歸一化 (Normalize advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std(correction=0) + 1e-8)

        states_t = torch.FloatTensor(np.array(self._states)).to(self.device)
        actions_t = torch.LongTensor(self._actions).to(self.device)

        log_probs, entropy, values_t = self.net.evaluate_actions(states_t, actions_t)
        values_t = values_t.squeeze(1)

        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = nn.functional.mse_loss(values_t, returns.detach())
        entropy_loss = -entropy.mean()

        loss = actor_loss + self.c_v * critic_loss + self.c_e * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.optimizer.step()

        self.total_steps += T
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._values.clear()
        self._dones.clear()

        return {
            "total_loss": float(loss.item()),
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(-entropy_loss.item()),
        }

    def save(self, path: str) -> None:
        self._ensure_dir(path)
        torch.save({"net": self.net.state_dict(), "optimizer": self.optimizer.state_dict()},
                   os.path.join(path, "a2c.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "a2c.pt"), map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

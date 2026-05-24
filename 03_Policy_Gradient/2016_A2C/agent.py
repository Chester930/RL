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
        """單狀態動作選擇（用於評估）。不儲存內部轉換資料。"""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.net(state_t)
        if evaluate:
            return int(logits.argmax(dim=1).item())
        dist = torch.distributions.Categorical(logits=logits)
        return int(dist.sample().item())

    def select_actions(self, states: np.ndarray) -> np.ndarray:
        """VecEnv 批次動作選擇：states (N, state_dim) → actions (N,)。儲存轉換資料。"""
        states_t = torch.FloatTensor(states).to(self.device)
        with torch.no_grad():
            logits, values = self.net(states_t)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        self._states.append(states.copy())
        self._actions.append(actions.cpu().numpy())
        self._values.append(values.squeeze(1).cpu().numpy())
        return actions.cpu().numpy()

    def store_reward_done(self, rewards, dones) -> None:
        """儲存獎勵與結束狀態（接受純量或陣列）。"""
        self._rewards.append(np.asarray(rewards, dtype=float))
        self._dones.append(np.asarray(dones, dtype=float))

    # ------------------------------------------------------------------
    # Learning (GAE + Actor-Critic update)
    # ------------------------------------------------------------------

    def update(self, next_states: np.ndarray, last_dones: np.ndarray) -> dict:
        """
        GAE 優勢函式計算並更新演員-評論家網路（支援 VecEnv）。

        next_states: (N, state_dim) — 最後一步的下一觀測（VecEnv 已自動重置）
        last_dones:  (N,)          — 最後一步的結束旗標（用於 bootstrap 截斷）
        """
        T = len(self._states)
        if T == 0:
            return {}

        states_arr  = np.stack(self._states)   # (T, N, state_dim)
        actions_arr = np.stack(self._actions)  # (T, N)
        rewards_arr = np.stack(self._rewards)  # (T, N)
        dones_arr   = np.stack(self._dones)    # (T, N)
        values_arr  = np.stack(self._values)   # (T, N)
        N = states_arr.shape[1]

        # 引導值：done 的環境 bootstrap = 0
        ns_t = torch.FloatTensor(next_states).to(self.device)
        with torch.no_grad():
            _, next_vals = self.net(ns_t)
        next_values = next_vals.squeeze(1).cpu().numpy()                      # (N,)
        next_values = next_values * (1.0 - np.asarray(last_dones, dtype=float))

        # GAE (T, N)
        advantages = np.zeros_like(rewards_arr)
        gae = np.zeros(N)
        all_vals = np.concatenate([values_arr, next_values[np.newaxis]], axis=0)  # (T+1, N)

        for t in reversed(range(T)):
            d = dones_arr[t]
            delta = rewards_arr[t] + self.gamma * all_vals[t+1] * (1-d) - all_vals[t]
            gae = delta + self.gamma * self.gae_lambda * (1-d) * gae
            advantages[t] = gae

        returns = advantages + values_arr  # (T, N)

        # 展平 (T*N, ...)
        state_dim = states_arr.shape[2]
        states_flat     = states_arr.reshape(-1, state_dim)
        actions_flat    = actions_arr.reshape(-1)
        advantages_flat = advantages.reshape(-1)
        returns_flat    = returns.reshape(-1)

        adv_t = torch.FloatTensor(advantages_flat).to(self.device)
        ret_t = torch.FloatTensor(returns_flat).to(self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        states_t  = torch.FloatTensor(states_flat).to(self.device)
        actions_t = torch.LongTensor(actions_flat).to(self.device)

        log_probs, entropy, values_t = self.net.evaluate_actions(states_t, actions_t)
        values_t = values_t.squeeze(1)

        actor_loss   = -(log_probs * adv_t.detach()).mean()
        critic_loss  = nn.functional.mse_loss(values_t, ret_t.detach())
        entropy_loss = -entropy.mean()
        loss = actor_loss + self.c_v * critic_loss + self.c_e * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.optimizer.step()

        self.total_steps += T * N
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._values.clear()
        self._dones.clear()

        return {
            "total_loss":  float(loss.item()),
            "actor_loss":  float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy":     float(-entropy_loss.item()),
        }

    def save(self, path: str) -> None:
        self._ensure_dir(path)
        torch.save({"net": self.net.state_dict(), "optimizer": self.optimizer.state_dict()},
                   os.path.join(path, "a2c.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "a2c.pt"), map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

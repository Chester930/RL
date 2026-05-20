"""
HER 代理人 — 事後經驗重播 (Hindsight Experience Replay)。

透過將失敗的回合重標註為替代目標（即實際上達到的狀態），實現從稀疏獎勵 (Sparse rewards) 中學習。

可與任何離線策略 (Off-policy) 演演算法結合使用（此處使用 DDPG）。
主要的改變在於重播緩衝區：在一個回合結束後，增加額外的轉換資料，並將原始目標替換為稍後實際達到的狀態。

參考文獻：
    Andrychowicz, M., et al. (2017). Hindsight Experience Replay.
    NeurIPS 2017. arXiv:1707.01495.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from common.base_agent import BaseAgent


class ActorNetwork(nn.Module):
    """確定性演員 (Deterministic actor)：(obs, goal) -> 動作。"""

    def __init__(self, obs_dim: int, goal_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + goal_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, goal], dim=-1))


class CriticNetwork(nn.Module):
    """Q 網路：(obs, goal, action) -> Q 值。"""

    def __init__(self, obs_dim: int, goal_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + goal_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, goal: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, goal, action], dim=-1)).squeeze(-1)


class HERReplayBuffer:
    """
    具備事後經驗重播 (Hindsight Experience Replay) 功能的重播緩衝區。

    儲存完整的回合資料，並在取樣時應用 HER 重標註策略。
    支援「未來 (future)」策略（效果最佳）。

    策略選項：
        'future':  將目標替換為同一回合中隨機的一個未來狀態。
        'final':   將目標替換為該回合的最終狀態。
        'episode': 將目標替換為該回合中的任一隨機狀態。
    """

    def __init__(self, capacity: int, her_ratio: float = 0.8, strategy: str = "future"):
        self.capacity = capacity
        self.her_ratio = her_ratio  # 批次中使用 HER 目標的比例
        self.strategy = strategy
        self.episodes = deque(maxlen=capacity)
        self._current_episode = []

    def store_transition(self, obs, action, reward, next_obs, done, goal, achieved_goal):
        """儲存一個步數；當 done=True 時呼叫 finish_episode()。"""
        self._current_episode.append({
            "obs": obs.copy(),
            "action": np.array(action).copy(),
            "reward": float(reward),
            "next_obs": next_obs.copy(),
            "done": float(done),
            "goal": goal.copy(),
            "achieved": achieved_goal.copy(),
        })
        if done:
            self.finish_episode()

    def finish_episode(self):
        if self._current_episode:
            self.episodes.append(list(self._current_episode))
            self._current_episode = []

    def _compute_reward(self, achieved: np.ndarray, goal: np.ndarray,
                        threshold: float = 0.05) -> float:
        """二元稀疏獎勵：未達成則為 -1，達成則為 0。"""
        dist = np.linalg.norm(achieved - goal)
        return 0.0 if dist < threshold else -1.0

    def sample(self, batch_size: int) -> dict:
        """
        進行具備 HER 重標註的小批次取樣。

        針對每個取樣的轉換資料：
            - 以 her_ratio 的機率：替換為 HER 事後目標。
            - 以 (1-her_ratio) 的機率：保留原始目標。
        """
        obs_list, action_list, reward_list = [], [], []
        next_obs_list, done_list, goal_list = [], [], []

        n_her = int(batch_size * self.her_ratio)
        n_real = batch_size - n_her

        for _ in range(batch_size):
            # Sample random episode
            ep = random.choice(self.episodes)
            t = random.randint(0, len(ep) - 1)
            transition = ep[t]

            if len(obs_list) < n_her and len(ep) > 1:
                # HER 重標註：選擇一個未來狀態作為替代目標
                if self.strategy == "future":
                    future_t = random.randint(t, len(ep) - 1)
                    her_goal = ep[future_t]["achieved"]
                elif self.strategy == "final":
                    her_goal = ep[-1]["achieved"]
                else:  # episode
                    her_goal = random.choice(ep)["achieved"]

                goal = her_goal
                reward = self._compute_reward(transition["achieved"], goal)
            else:
                goal = transition["goal"]
                reward = transition["reward"]

            obs_list.append(transition["obs"])
            action_list.append(transition["action"])
            reward_list.append(reward)
            next_obs_list.append(transition["next_obs"])
            done_list.append(transition["done"])
            goal_list.append(goal)

        return {
            "obs": np.array(obs_list, dtype=np.float32),
            "actions": np.array(action_list, dtype=np.float32),
            "rewards": np.array(reward_list, dtype=np.float32),
            "next_obs": np.array(next_obs_list, dtype=np.float32),
            "dones": np.array(done_list, dtype=np.float32),
            "goals": np.array(goal_list, dtype=np.float32),
        }

    def is_ready(self, batch_size: int) -> bool:
        return len(self.episodes) >= 1 and sum(len(e) for e in self.episodes) >= batch_size


class HERAgent(BaseAgent):
    """
    DDPG + HER 適用於具備目標條件 (Goal-conditioned) 的稀疏獎勵環境。

    預期接收的觀測字典包含： "observation", "achieved_goal", "desired_goal"
    （標準 Gymnasium GoalEnv 介面）。

    引數：
        obs_dim:       觀測維度（不包含目標）。
        goal_dim:      目標維度。
        action_dim:    動作維度。
        action_scale:  動作縮放因子（最大動作值）。
        lr:            Adam 學習率。
        gamma:         折扣因子。
        tau:           目標網路軟更新係數。
        buffer_size:   欲儲存的回合數量。
        batch_size:    小批次維度。
        her_ratio:     每批次中 HER 重標註資料的比例。
        noise_std:     高斯探索雜訊標準差。
        device:        "cpu" 或 "cuda"。
    """

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        action_dim: int,
        action_scale: float = 1.0,
        lr: float = 1e-3,
        gamma: float = 0.98,
        tau: float = 0.05,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        her_ratio: float = 0.8,
        noise_std: float = 0.2,
        device: str = "cpu",
    ):
        super().__init__(obs_dim + goal_dim, action_dim, device)

        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.action_scale = action_scale
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_std = noise_std

        # Networks
        self.actor = ActorNetwork(obs_dim, goal_dim, action_dim).to(device)
        self.actor_target = ActorNetwork(obs_dim, goal_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = CriticNetwork(obs_dim, goal_dim, action_dim).to(device)
        self.critic_target = CriticNetwork(obs_dim, goal_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # HER buffer
        self.buffer = HERReplayBuffer(
            capacity=buffer_size, her_ratio=her_ratio
        )

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, goal: np.ndarray,
                      evaluate: bool = False) -> np.ndarray:
        """
        給定觀測與預期目標，選擇動作。

        引數：
            obs:  (obs_dim,) 目前觀測。
            goal: (goal_dim,) 預期目標。
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        goal_t = torch.FloatTensor(goal).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(obs_t, goal_t).squeeze(0).cpu().numpy()

        if not evaluate:
            noise = np.random.randn(*action.shape) * self.noise_std
            action = np.clip(action + noise, -1.0, 1.0)

        return action * self.action_scale

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self) -> dict:
        """在 HER 重標註的小批次上執行一步 DDPG 梯度更新。"""
        if not self.buffer.is_ready(self.batch_size):
            return {}

        batch = self.buffer.sample(self.batch_size)
        obs = torch.FloatTensor(batch["obs"]).to(self.device)
        actions = torch.FloatTensor(batch["actions"]).to(self.device)
        rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_obs = torch.FloatTensor(batch["next_obs"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).to(self.device)
        goals = torch.FloatTensor(batch["goals"]).to(self.device)

        # 針對網路將動作限制在 [-1, 1]（代理人內部儲存未縮放的動作）
        actions_norm = (actions / self.action_scale).clamp(-1.0, 1.0)

        # --- 評論家 (Critic) 更新 ---
        with torch.no_grad():
            next_actions = self.actor_target(next_obs, goals)
            q_next = self.critic_target(next_obs, goals, next_actions)
            # HER 使用稀疏獎勵 (-1 或 0)；將其限制在有效範圍內
            q_target = (rewards + self.gamma * (1 - dones) * q_next).clamp(
                -1.0 / (1.0 - self.gamma), 0.0
            )

        q_pred = self.critic(obs, goals, actions_norm)
        critic_loss = nn.functional.mse_loss(q_pred, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- 演員 (Actor) 更新 (最大化 Q 值) ---
        actor_actions = self.actor(obs, goals)
        actor_loss = -self.critic(obs, goals, actor_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- 目標網路軟更新 (Soft target updates) ---
        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

        self.total_steps += 1

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        self._ensure_dir(path)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "total_steps": self.total_steps,
        }, os.path.join(path, "her_checkpoint.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "her_checkpoint.pt"), map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.total_steps = ckpt["total_steps"]

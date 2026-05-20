"""
SAC 代理人 — 軟演員-評論家 (Soft Actor-Critic)。

參考文獻：
    Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft Actor-Critic:
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.
    ICML 2018. arXiv:1801.01290.

    Haarnoja, T., et al. (2018). Soft Actor-Critic Algorithms and Applications.
    arXiv:1812.05905. (自動溫度調節版本)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

from common.base_agent import BaseAgent
from common.buffers.replay_buffer import ReplayBuffer
from network import PolicyNetwork, TwinQNetwork


class SACAgent(BaseAgent):
    """
    具備自動溫度調節功能的軟演員-評論家 (SAC)。

    SAC 旨在最大化「最大熵目標」：
        J(pi) = E_pi [ sum_t r_t + alpha * H(pi(.|s_t)) ]

    熵項 H(pi) 能鼓勵探索並防止過早收斂。
    Alpha (溫度引數) 控制獎勵與熵之間的權衡。

    核心特性：
    - 離策略 (Off-policy，使用回放緩衝區) — 樣本效率優於 PPO
    - 隨機性策略 (高斯分佈，配合重引數化技巧)
    - 雙 Q 網路 (Twin Q-networks，類似 TD3) 以減少高估問題
    - 自動溫度調節 (alpha = exp(log_alpha))
    - 無須目標演員網路 (使用 min(Q1, Q2) - alpha*log_pi 進行軟貝爾曼更新)

    引數：
        auto_alpha:       若為 True，則自動調校溫度引數。
        init_alpha:       初始溫度值。
        target_entropy:   自動調校的目標熵。預設為 -動作維度。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_scale: float = 1.0,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        auto_alpha: bool = True,
        init_alpha: float = 0.2,
        target_entropy: float = None,
        device: str = "cpu",
    ):
        super().__init__(state_dim, action_dim, device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.action_scale = action_scale

        # Networks
        self.policy = PolicyNetwork(state_dim, action_dim, action_scale=action_scale).to(self.device)
        self.critic = TwinQNetwork(state_dim, action_dim).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)
        self.target_critic.eval()

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # 自動溫度調節 (Automatic temperature tuning)
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = target_entropy or -float(action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = init_alpha

        self.buffer = ReplayBuffer(capacity=buffer_size)

    def _soft_update(self):
        for sp, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if evaluate:
                action = self.policy.get_deterministic_action(state_t)
            else:
                action, _ = self.policy.get_action(state_t)
        return action.cpu().numpy()[0]

    def update(self) -> dict:
        """
        SAC 更新流程：
        1. 評論家 (Critic) 更新：
           y = r + gamma * (min(Q1_target, Q2_target)(s', a') - alpha * log_pi(a'|s'))
           其中 a' ~ pi(.|s') (無須目標演員網路！)
        2. 策略更新 (演員)：
           最大化： E_a~pi [min(Q1, Q2)(s, a) - alpha * log_pi(a|s)]
        3. 溫度引數更新 (若開啟自動調校)：
           最小化： -log_alpha * (log_pi + target_entropy)
        """
        if not self.buffer.is_ready(self.batch_size):
            return {}

        batch = self.buffer.sample(self.batch_size)
        states = torch.FloatTensor(batch["states"]).to(self.device)
        actions = torch.FloatTensor(batch["actions"]).to(self.device)
        rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).to(self.device)

        # --- 評論家更新 (軟貝爾曼更新，Soft Bellman backup) ---
        with torch.no_grad():
            # TODO: 從「當前」策略中取樣下一步動作 (而非目標演員 — SAC 沒有目標策略)
            next_actions, next_log_probs = self.policy.get_action(next_states)
            q1_next, q2_next = self.target_critic(next_states, next_actions)
            # TODO: 軟目標值：y = r + gamma * (min(Q1,Q2) - alpha * log_pi)
            min_q_next = torch.min(q1_next, q2_next).squeeze(1)
            targets = rewards + self.gamma * (min_q_next - self.alpha * next_log_probs) * (1 - dones)

        q1, q2 = self.critic(states, actions)
        q1, q2 = q1.squeeze(1), q2.squeeze(1)
        critic_loss = nn.functional.mse_loss(q1, targets) + nn.functional.mse_loss(q2, targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- 策略更新 (Policy update) ---
        new_actions, log_probs = self.policy.get_action(states)
        q1_new, q2_new = self.critic(states, new_actions)
        min_q_new = torch.min(q1_new, q2_new).squeeze(1)

        # TODO: 最大化 E[min(Q) - alpha * log_pi] = 最小化 -(min(Q) - alpha * log_pi)
        policy_loss = (self.alpha * log_probs - min_q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        metrics = {
            "critic_loss": float(critic_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "alpha": float(self.alpha),
            "entropy": float(-log_probs.mean().item()),
        }

        # --- Automatic temperature tuning ---
        if self.auto_alpha:
            # TODO: minimize -log_alpha * (log_pi + target_entropy)
            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
            metrics["alpha_loss"] = float(alpha_loss.item())

        # --- 執行目標評論家軟更新 (Soft update target critic) ---
        self._soft_update()
        self.total_steps += 1

        return metrics

    def save(self, path: str) -> None:
        self._ensure_dir(path)
        torch.save({
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "log_alpha": self.log_alpha if self.auto_alpha else None,
        }, os.path.join(path, "sac.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "sac.pt"), map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.critic.load_state_dict(ckpt["critic"])
        self.target_critic = copy.deepcopy(self.critic)
        if self.auto_alpha and ckpt["log_alpha"] is not None:
            self.log_alpha.data.copy_(ckpt["log_alpha"].data)

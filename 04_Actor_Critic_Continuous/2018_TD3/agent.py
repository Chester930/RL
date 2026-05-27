"""
TD3 代理人 — 雙延遲深度確定性策略梯度 (Twin Delayed Deep Deterministic Policy Gradient)。

參考文獻：
    Fujimoto, S., van Hoof, H., & Meger, D. (2018). Addressing Function
    Approximation Error in Actor-Critic Methods. ICML 2018. arXiv:1802.09477.
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
from network import ActorNetwork, TwinCriticNetwork


class TD3Agent(BaseAgent):
    """
    TD3 在 DDPG 的基礎上增加了三項改進，以解決高估偏差 (Overestimation bias)：

    1. **雙評論家 (Twin Critics / Clipped Double Q)**：使用兩個獨立的評論家網路，
       在計算目標值時取其最小值，以防止 Q 值被高估。

    2. **延遲策略更新 (Delayed Policy Updates)**：演員網路（及其目標網路）的
       更新頻率低於評論家（例如每 d 步更新一次）。這能讓評論家在演員網路跟進前先趨於穩定。

    3. **目標策略平滑 (Target Policy Smoothing)**：在目標動作中加入經過剪裁的雜訊，
       以使 Q 值變化更平滑，並防止模型利用 Q 函式中的誤差：
           a' = clip(pi_target(s') + clip(noise, -c, c), -action_max, action_max)

    引數：
        policy_noise:  加入至目標策略的高斯雜訊標準差 (Sigma)。
        noise_clip:    目標策略雜訊的最大幅值。
        policy_delay:  每進行 `policy_delay` 次評論家更新後才更新一次演員。
        expl_noise:    探索雜訊的標準差（採用高斯雜訊而非 OU 雜訊）。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_scale: float = 1.0,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        expl_noise: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__(state_dim, action_dim, device)
        self.action_scale = action_scale
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.policy_noise = policy_noise * action_scale
        self.noise_clip = noise_clip * action_scale
        self.policy_delay = policy_delay
        self.expl_noise = expl_noise * action_scale
        self._critic_updates = 0

        self.actor = ActorNetwork(state_dim, action_dim, action_scale=action_scale).to(self.device)
        self.target_actor = copy.deepcopy(self.actor)

        self.critic = TwinCriticNetwork(state_dim, action_dim).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)

        for net in [self.target_actor, self.target_critic]:
            net.eval()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = ReplayBuffer(capacity=buffer_size)

    def _soft_update(self, source, target):
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]
        if not evaluate:
            noise = np.random.normal(0, self.expl_noise, size=action.shape)
            action = np.clip(action + noise, -self.action_scale, self.action_scale)
        return action

    def update(self) -> dict:
        """
        TD3 更新流程：
        1. 評論家 (Critic) 更新（每一步執行）：
           - 計算包含策略平滑雜訊的目標動作
           - 取雙 Q 網路的最小值作為目標值
        2. 演員 (Actor) 更新（每隔 policy_delay 步執行）：
           - 最大化 Q1(s, pi(s))
        3. 執行兩套目標網路的軟更新（每隔 policy_delay 步執行）
        """
        if not self.buffer.is_ready(self.batch_size):
            return {}

        batch = self.buffer.sample(self.batch_size)
        states = torch.FloatTensor(batch["states"]).to(self.device)
        actions = torch.FloatTensor(batch["actions"]).to(self.device)
        rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).to(self.device)

        self._critic_updates += 1

        # --- 評論家更新 (Critic update) ---
        with torch.no_grad():
            # TODO: 目標策略平滑 (Target policy smoothing) — 在目標動作中加入剪裁雜訊
            noise = torch.randn_like(actions) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.target_actor(next_states) + noise).clamp(
                -self.action_scale, self.action_scale
            )

            # TODO: 剪裁雙 Q 學習 (Clipped Double Q) — 取雙評論家的最小值
            q1_target, q2_target = self.target_critic(next_states, next_actions)
            min_q_target = torch.min(q1_target, q2_target).squeeze(1)
            targets = rewards + self.gamma * min_q_target * (1 - dones)

        q1, q2 = self.critic(states, actions)
        q1, q2 = q1.squeeze(1), q2.squeeze(1)
        critic_loss = nn.functional.mse_loss(q1, targets) + nn.functional.mse_loss(q2, targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        metrics = {"critic_loss": float(critic_loss.item())}

        # --- 延遲演員更新 (Delayed actor update) ---
        if self._critic_updates % self.policy_delay == 0:
            # TODO: 最大化 Q1(s, pi(s))
            actor_loss = -self.critic.q1_only(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update(self.actor, self.target_actor)
            self._soft_update(self.critic, self.target_critic)

            metrics["actor_loss"] = float(actor_loss.item())

        self.total_steps += 1
        return metrics

    def save(self, path: str) -> None:
        self._ensure_dir(path)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, os.path.join(path, "td3.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "td3.pt"), map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

    def save_resume(self, path: str) -> None:
        self._ensure_dir(path)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_opt": self.actor_optimizer.state_dict(),
            "critic_opt": self.critic_optimizer.state_dict(),
        }, os.path.join(path, "td3_resume.pt"))

    def load_resume(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "td3_resume.pt"), map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.target_actor.load_state_dict(ckpt["target_actor"])
        self.target_critic.load_state_dict(ckpt["target_critic"])
        self.actor_optimizer.load_state_dict(ckpt["actor_opt"])
        self.critic_optimizer.load_state_dict(ckpt["critic_opt"])

"""
MBPO 代理人 — 透過模型信賴機制最佳化策略 (Model-Based Policy Optimization)。

結合了整合動態模型 (Ensemble dynamics model) 與 SAC 策略。
核心思想：從真實狀態出發的短步數模型生成取樣能擴大有效資料集，進而實現具備樣本效率的學習。

參考文獻：
    Janner, M., Fu, J., Zhang, M., & Levine, S. (2019).
    When to Trust Your Model: Model-Based Policy Optimization.
    NeurIPS 2019. arXiv:1906.08253.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from common.base_agent import BaseAgent
from common.buffers.replay_buffer import ReplayBuffer
from network import EnsembleDynamicsModel, PolicyNetwork, TwinQNetwork


class MBPOAgent(BaseAgent):
    """
    MBPO：透過模型信賴機制最佳化策略。

    架構：
        - 整合動態模型（7 個成員，5 個精銳成員）：用於生成想像取樣。
        - SAC 策略（演員 + 雙 Q 網路 + 自動 Alpha）：在真實與模型混合資料上訓練。
        - 兩個重播緩衝區：real_buffer（環境轉換資料）+ model_buffer（想像轉換資料）。

    每個真實步數的訓練迴圈：
        1. 收集 1 個真實轉換資料 (Real transition) -> 存入 real_buffer。
        2. 每隔 `model_train_freq` 步：在 real_buffer 上重新訓練動態模型。
        3. 從真實狀態出發進行 k 步模型取樣 -> 存入 model_buffer。
        4. 在來自 model_buffer（以及部分真實資料）的批次上執行 G 次 SAC 梯度更新。

    引數：
        state_dim:         狀態維度。
        action_dim:        動作維度。
        hidden_dim:        策略與 Q 網路的隱藏層維度。
        ensemble_members:  動態模型整合成員的數量。
        n_elite:           用於取樣的精銳成員數量。
        rollout_length:    k — 模型取樣步數 (Horizon)。
        real_ratio:        來自 real_buffer 的 SAC 更新比例。
        gamma:             折扣因子。
        tau:               軟更新係數。
        alpha:             初始 SAC 溫度引數（若 auto_alpha=True 則會自動調整）。
        auto_alpha:        是否自動調整溫度引數。
        lr:                策略與評論家網路的學習率。
        model_lr:          動態模型的學習率。
        real_buffer_size:  真實經驗緩衝區的容量。
        model_buffer_size: 想像轉換資料緩衝區的容量。
        batch_size:        SAC 小批次 (Minibatch) 維度。
        device:            "cpu" 或 "cuda"。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        ensemble_members: int = 7,
        n_elite: int = 5,
        rollout_length: int = 1,
        real_ratio: float = 0.05,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        lr: float = 3e-4,
        model_lr: float = 1e-3,
        real_buffer_size: int = 100_000,
        model_buffer_size: int = 400_000,
        batch_size: int = 256,
        device: str = "cpu",
    ):
        super().__init__(state_dim, action_dim, device)

        self.gamma = gamma
        self.tau = tau
        self.real_ratio = real_ratio
        self.rollout_length = rollout_length
        self.batch_size = batch_size
        self.n_elite = n_elite
        self.elite_indices = list(range(n_elite))  # updated after model training

        # --- 動態模型 (Dynamics model) ---
        self.model = EnsembleDynamicsModel(
            state_dim, action_dim, n_members=ensemble_members
        ).to(device)
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=model_lr)

        # --- SAC 元件 ---
        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic = TwinQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = TwinQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # --- 自動調整 Alpha (Auto-alpha) ---
        self.auto_alpha = auto_alpha
        self.target_entropy = -float(action_dim)
        if auto_alpha:
            self.log_alpha = nn.Parameter(torch.zeros(1, device=device))
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha

        # --- 重播緩衝區 (Replay buffers) ---
        self.real_buffer = ReplayBuffer(capacity=real_buffer_size)
        self.model_buffer = ReplayBuffer(capacity=model_buffer_size)

    # --- 執行動作 (Acting) ---

    @torch.no_grad()
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if evaluate:
            mu, _ = self.actor(s)
            return torch.tanh(mu).squeeze(0).cpu().numpy()
        action, _ = self.actor.sample(s)
        return action.squeeze(0).cpu().numpy()

    # --- 動態模型訓練 (Dynamics model training) ---

    def update_model(self, n_epochs: int = 5) -> dict:
        """在 real_buffer 資料上重新訓練整合模型；並選擇精銳 (Elite) 成員。"""
        if not self.real_buffer.is_ready(self.batch_size):
            return {}

        total_loss = 0.0
        for _ in range(n_epochs):
            batch = self.real_buffer.sample(min(len(self.real_buffer), 1024))

            states = torch.FloatTensor(batch["states"]).to(self.device)
            actions = torch.FloatTensor(batch["actions"]).to(self.device)
            next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
            rewards = torch.FloatTensor(batch["rewards"]).to(self.device)

            delta_state = next_states - states
            loss = self.model.nll_loss(states, actions, delta_state, rewards)

            self.model_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.model_optimizer.step()
            total_loss += loss.item()

        # TODO: 透過留存的真實資料在驗證集上的 NLL 選擇精銳成員
        # self.elite_indices = sorted_by_val_loss[:self.n_elite]

        return {"model_loss": total_loss / n_epochs}

    # --- 模型取樣 (Rollouts) -> model_buffer ---

    @torch.no_grad()
    def model_rollout(self, n_rollouts: int) -> int:
        """
        使用學習到的動態模型，從真實狀態出發進行短步數取樣。

        回傳新增至 model_buffer 的轉換資料數量。
        """
        if not self.real_buffer.is_ready(n_rollouts):
            return 0

        batch = self.real_buffer.sample(n_rollouts)
        states = torch.FloatTensor(batch["states"]).to(self.device)

        added = 0
        for _ in range(self.rollout_length):
            actions, _ = self.actor.sample(states)
            next_states, rewards = self.model.sample_next_state(
                states, actions, self.elite_indices
            )

            s_np = states.cpu().numpy()
            a_np = actions.cpu().numpy()
            r_np = rewards.cpu().numpy()
            ns_np = next_states.cpu().numpy()

            for i in range(len(s_np)):
                self.model_buffer.push(s_np[i], a_np[i], r_np[i], ns_np[i], False)
                added += 1

            states = next_states

        return added

    # --- SAC 更新 ---

    def update(self) -> dict:
        """
        在來自真實與模型緩衝區的混合批次上執行一次 SAC 梯度更新。

        其中 real_ratio 比例來自 real_buffer，其餘來自 model_buffer。
        """
        total_needed = self.batch_size
        n_real = max(1, int(total_needed * self.real_ratio))
        n_model = total_needed - n_real

        if not self.real_buffer.is_ready(n_real):
            return {}
        if not self.model_buffer.is_ready(n_model):
            return {}

        # 建立混合批次 (Build mixed batch)
        def _to_tensor(arr):
            return torch.FloatTensor(arr).to(self.device)

        real_batch = self.real_buffer.sample(n_real)
        model_batch = self.model_buffer.sample(n_model)

        states = _to_tensor(np.concatenate([real_batch["states"], model_batch["states"]]))
        actions = _to_tensor(np.concatenate([real_batch["actions"], model_batch["actions"]]))
        rewards = _to_tensor(np.concatenate([real_batch["rewards"], model_batch["rewards"]]))
        next_states = _to_tensor(np.concatenate([real_batch["next_states"], model_batch["next_states"]]))
        dones = _to_tensor(np.concatenate([real_batch["dones"], model_batch["dones"]]))

        # --- 評論家 (Critic) 更新 ---
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q_next = self.critic_target.q_min(next_states, next_actions)
            q_target = rewards + self.gamma * (1 - dones) * (q_next - self.alpha * next_log_probs)

        q1, q2 = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(q1, q_target) + nn.functional.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # --- 演員 (Actor) 更新 ---
        new_actions, log_probs = self.actor.sample(states)
        q_actor = self.critic.q_min(states, new_actions)
        actor_loss = (self.alpha * log_probs - q_actor).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # --- Alpha 更新 ---
        alpha_loss = torch.tensor(0.0)
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # --- 目標網路軟更新 (Soft target update) ---
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

        self.total_steps += 1

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": self.alpha,
        }

    # --- 持久化 (Persistence) ---

    def save(self, path: str) -> None:
        self._ensure_dir(path)
        torch.save({
            "model": self.model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "total_steps": self.total_steps,
        }, os.path.join(path, "mbpo_checkpoint.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "mbpo_checkpoint.pt"), map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.total_steps = ckpt["total_steps"]

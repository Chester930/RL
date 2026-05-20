"""
ICM 代理人 — 透過內在好奇心模組 (Intrinsic Curiosity Module) 驅動探索。

在基礎強化學習演演算法（此處為 PPO）之上封裝一個 ICM 模組，
根據前向模型 (Forward model) 的預測誤差生成內在獎勵。

總獎勵 r_total = 外在獎勵 r_extrinsic + eta * 內在獎勵 r_intrinsic

參考文獻：
    Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017).
    Curiosity-driven Exploration by Self-Supervised Prediction. ICML 2017.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from common.base_agent import BaseAgent
from network import ICMModule


class PolicyNetwork(nn.Module):
    """簡單的離散策略 + 價值函式（用於 PPO 基礎）。"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def get_action(self, x: torch.Tensor):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value


class ICMAgent(BaseAgent):
    """
    結合了內在好奇心模組 (ICM) 的 PPO 代理人。

    ICM 模組提供了內在獎勵，鼓勵代理人造訪那些難以預測的狀態（即新奇狀態）。

    適用場景：
        - 外在獎勵稀疏或完全沒有獎勵的環境。
        - 探索巨大的狀態空間。
        - 在獎勵密集的環境中避免陷入區域性最優解。

    引數：
        state_dim:       狀態維度。
        action_dim:      離散動作數量。
        feature_dim:     ICM 特徵空間維度。
        hidden_dim:      策略/價值網路的隱藏層維度。
        eta:             內在獎勵縮放因子。
        beta:            ICM 前向/逆向損失的平衡權重。
        lr:              策略與 ICM 的學習率。
        gamma:           折扣因子。
        gae_lambda:      GAE 平滑引數。
        clip_eps:        PPO 截斷 (Clip) 引數。
        n_epochs:        PPO 小週期 (Mini-epoch) 數量。
        rollout_steps:   每次 PPO 取樣的步數。
        batch_size:      PPO 更新的小批次維度。
        device:          "cpu" 或 "cuda"。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        feature_dim: int = 256,
        hidden_dim: int = 256,
        eta: float = 0.01,
        beta: float = 0.2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        n_epochs: int = 4,
        rollout_steps: int = 2048,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        super().__init__(state_dim, action_dim, device)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.rollout_steps = rollout_steps
        self.batch_size = batch_size
        self.eta = eta

        # PPO 策略網路 (PPO policy)
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # ICM 模組 (ICM module)
        self.icm = ICMModule(
            state_dim=state_dim,
            action_dim=action_dim,
            feature_dim=feature_dim,
            discrete=True,
            eta=eta,
            beta=beta,
        ).to(device)
        self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=lr)

        # 取樣資料儲存 (Rollout storage)
        self._reset_rollout()

    def _reset_rollout(self):
        self._states = []
        self._next_states = []
        self._actions = []
        self._log_probs = []
        self._ext_rewards = []
        self._values = []
        self._dones = []

    # --- 執行動作 (Acting) ---

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if evaluate:
            with torch.no_grad():
                logits, _ = self.policy(s)
                return int(logits.argmax(dim=-1).item())
        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(s)
        # 儲存於取樣緩衝區（呼叫者必須在 env.step 後呼叫 store_transition）
        self._last_state = state
        self._last_action = action.item()
        self._last_log_prob = log_prob.item()
        self._last_value = value.item()
        return self._last_action

    def store_transition(self, next_state: np.ndarray, ext_reward: float, done: bool):
        """在 env.step() 後呼叫此函式以填充取樣緩衝區 (Rollout buffer)。"""
        self._states.append(self._last_state)
        self._next_states.append(next_state)
        self._actions.append(self._last_action)
        self._log_probs.append(self._last_log_prob)
        self._ext_rewards.append(ext_reward)
        self._values.append(self._last_value)
        self._dones.append(float(done))

    # --- 學習更新 (Learning) ---

    def update(self) -> dict:
        """
        計算 ICM 內在獎勵，接著在完整的取樣資料上執行 PPO 更新。

        當取樣緩衝區已滿（達到 rollout_steps）時由外部呼叫。
        """
        if len(self._states) < self.rollout_steps:
            return {}

        states_np = np.array(self._states, dtype=np.float32)
        next_states_np = np.array(self._next_states, dtype=np.float32)
        actions_np = np.array(self._actions)
        log_probs_np = np.array(self._log_probs, dtype=np.float32)
        ext_rewards_np = np.array(self._ext_rewards, dtype=np.float32)
        values_np = np.array(self._values, dtype=np.float32)
        dones_np = np.array(self._dones, dtype=np.float32)

        states_t = torch.FloatTensor(states_np).to(self.device)
        next_states_t = torch.FloatTensor(next_states_np).to(self.device)
        actions_t = torch.LongTensor(actions_np).to(self.device)

        # --- 計算內在獎勵 (Compute intrinsic rewards) ---
        with torch.no_grad():
            intr_rewards, _, icm_info = self.icm(states_t, next_states_t, actions_t)
        intr_rewards_np = intr_rewards.cpu().numpy()

        # --- 組合獎勵 (Combined reward) ---
        total_rewards = ext_rewards_np + intr_rewards_np

        # --- GAE 優勢估計計算 (GAE advantage computation) ---
        advantages = np.zeros_like(total_rewards)
        returns = np.zeros_like(total_rewards)
        gae = 0.0

        # 對最終價值進行引導 (Bootstrap final value)
        last_s = torch.FloatTensor(next_states_np[-1]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, last_val = self.policy(last_s)
        next_val = last_val.item() * (1.0 - dones_np[-1])

        for t in reversed(range(self.rollout_steps)):
            nv = next_val if t == self.rollout_steps - 1 else values_np[t + 1]
            delta = total_rewards[t] + self.gamma * nv * (1 - dones_np[t]) - values_np[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones_np[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values_np[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        adv_t = torch.FloatTensor(advantages).to(self.device)
        ret_t = torch.FloatTensor(returns).to(self.device)
        old_lp_t = torch.FloatTensor(log_probs_np).to(self.device)

        # --- PPO 小週期更新 (PPO mini-epoch updates) ---
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_icm_loss = 0.0

        indices = np.arange(self.rollout_steps)
        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, self.rollout_steps, self.batch_size):
                mb = indices[start: start + self.batch_size]
                mb_states = states_t[mb]
                mb_next = next_states_t[mb]
                mb_actions = actions_t[mb]
                mb_adv = adv_t[mb]
                mb_ret = ret_t[mb]
                mb_old_lp = old_lp_t[mb]

                # PPO 損失 (PPO loss)
                logits, values = self.policy(mb_states)
                dist = torch.distributions.Categorical(logits=logits)
                new_lp = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = (new_lp - mb_old_lp).exp()
                surr1 = ratio * mb_adv
                surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, mb_ret)
                ppo_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                self.policy_optimizer.zero_grad()
                ppo_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.policy_optimizer.step()

                # ICM 損失 (ICM loss)
                _, icm_loss, _ = self.icm(mb_states, mb_next, mb_actions)
                self.icm_optimizer.zero_grad()
                icm_loss.backward()
                self.icm_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_icm_loss += icm_loss.item()

        self.total_steps += self.rollout_steps
        self._reset_rollout()

        n_updates = self.n_epochs * (self.rollout_steps // self.batch_size)
        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "icm_loss": total_icm_loss / n_updates,
            "mean_intr_reward": float(intr_rewards_np.mean()),
        }

    # --- 持久化 (Persistence) ---

    def save(self, path: str) -> None:
        self._ensure_dir(path)
        torch.save({
            "policy": self.policy.state_dict(),
            "icm": self.icm.state_dict(),
            "total_steps": self.total_steps,
        }, os.path.join(path, "icm_checkpoint.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "icm_checkpoint.pt"), map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.icm.load_state_dict(ckpt["icm"])
        self.total_steps = ckpt["total_steps"]

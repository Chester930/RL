"""
PPO-Lagrangian 代理人 — 帶拉格朗日乘數自動調整的安全 PPO。

核心思想：
  原始 CMDP：max E[Σr_t]  s.t.  E[Σc_t] ≤ d
  拉格朗日鬆弛（去掉常數 λd）：
      L(π, λ) = E[Σr_t] - λ × E[Σc_t]

  λ 雙層最佳化：
      固定 λ → 更新 π（PPO 步驟）
      固定 π → 更新 λ：λ ← max(0, λ + lr_λ × (J_C - d))
        違規（J_C > d）→ λ ↑ → 代價懲罰加重 → 策略更保守
        安全（J_C ≤ d）→ λ ↓ → 策略更積極

  Actor 損失：-[L_CLIP(A^r) - λ × L_CLIP(A^c)]
  Critic 損失：MSE(V^r) + MSE(V^c)

參考：
    Ray, J., Achiam, J., & Amodei, D. (2019).
    Benchmarking Safe Exploration in Deep Reinforcement Learning.
    arXiv:1910.12156
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from common.base_agent import BaseAgent
from network import SafeActorCriticNetwork


class PPOLagrangianAgent(BaseAgent):
    """
    連續動作空間 PPO + 拉格朗日安全約束。

    相較標準 PPO 的新增部分：
      - cost_critic：估計代價值函式 V^c(s)
      - lambda_（λ）：拉格朗日乘數，自動在每次 update 後調整
      - store_transition：額外接收 cost 信號
      - update：同時計算獎勵 / 代價 GAE，並更新 λ

    引數：
        action_scale:  動作上界（Pendulum 為 2.0）
        cost_limit:    每集允許的最大代價總和（安全預算 d）
        lambda_init:   λ 初始值（0 = 完全不考慮安全）
        lr_lambda:     λ 的學習率
        其餘引數與標準 PPO 相同
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_scale: float = 1.0,
        lr: float = 3e-4,
        gamma: float = 0.99,
        n_steps: int = 2048,
        n_epochs: int = 10,
        n_minibatch: int = 32,
        clip_eps: float = 0.2,
        gae_lambda: float = 0.95,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        cost_limit: float = 25.0,
        lambda_init: float = 0.0,
        lr_lambda: float = 0.05,
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
        self.action_scale = action_scale
        self.cost_limit = cost_limit
        self.lr_lambda = lr_lambda

        # λ 不參與梯度，透過顯式規則更新
        self.lambda_ = float(lambda_init)

        self.net = SafeActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)

        # 預先配置資料緩衝區
        self._obs = np.zeros((n_steps, state_dim), dtype=np.float32)
        self._actions = np.zeros((n_steps, action_dim), dtype=np.float32)
        self._rewards = np.zeros(n_steps, dtype=np.float32)
        self._costs = np.zeros(n_steps, dtype=np.float32)
        self._dones = np.zeros(n_steps, dtype=np.float32)
        self._log_probs = np.zeros(n_steps, dtype=np.float32)
        self._reward_values = np.zeros(n_steps, dtype=np.float32)
        self._cost_values = np.zeros(n_steps, dtype=np.float32)
        self._step = 0

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, evaluate: bool = False):
        """
        選擇動作。

        evaluate=True  → 回傳確定性均值動作（不採樣，不存緩衝區）
        evaluate=False → 從高斯策略採樣，並將資料存入緩衝區
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if evaluate:
                action = self.net.get_mean_action(state_t)
                return action.cpu().numpy()[0]

            action, log_prob, _, rv, cv = self.net.get_action_and_value(state_t)
            if self._step < self.n_steps:
                self._obs[self._step] = state
                self._actions[self._step] = action.cpu().numpy()[0]
                self._log_probs[self._step] = log_prob.item()
                self._reward_values[self._step] = rv.item()
                self._cost_values[self._step] = cv.item()

        return action.cpu().numpy()[0]

    def store_transition(self, reward: float, cost: float, done: bool) -> None:
        """儲存環境回傳的 reward、cost 和 done（接在 select_action 後呼叫）。"""
        if self._step < self.n_steps:
            self._rewards[self._step] = reward
            self._costs[self._step] = cost
            self._dones[self._step] = float(done)
            self._step += 1

    def is_ready(self) -> bool:
        return self._step >= self.n_steps

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def _gae(
        self,
        signals: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float,
        T: int,
    ):
        """計算 GAE 優勢函式與折扣回報（reward 或 cost 共用）。"""
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        values_ext = np.append(values[:T], next_value)
        for t in reversed(range(T)):
            d = dones[t]
            delta = signals[t] + self.gamma * values_ext[t + 1] * (1 - d) - values_ext[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - d) * gae
            advantages[t] = gae
        returns = advantages + values[:T]
        return advantages, returns

    def update(self, next_state: np.ndarray, last_done: bool) -> dict:
        """
        PPO-Lagrangian 更新：

        1. 計算獎勵 GAE 與代價 GAE
        2. 估計本次 rollout 的集數平均代價 J_C
        3. 執行 n_epochs 的小批次 PPO 更新（actor + 雙 critic）
        4. λ 更新：λ ← max(0, λ + lr_λ × (J_C - d))
        """
        T = self._step
        if T == 0:
            return {}

        # Bootstrap value
        ns_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if last_done:
                next_rv, next_cv = 0.0, 0.0
            else:
                next_rv, next_cv = self.net.get_values(ns_t)
                next_rv, next_cv = next_rv.item(), next_cv.item()

        r_adv, r_ret = self._gae(self._rewards, self._reward_values, self._dones, next_rv, T)
        c_adv, c_ret = self._gae(self._costs, self._cost_values, self._dones, next_cv, T)

        # 計算集數平均代價（用於 λ 更新）
        episode_costs = []
        ep_cost = 0.0
        for t in range(T):
            ep_cost += self._costs[t]
            if self._dones[t]:
                episode_costs.append(ep_cost)
                ep_cost = 0.0
        # 若 rollout 跨集但最後集未結束，仍納入估計
        if not episode_costs:
            episode_costs.append(ep_cost)
        mean_ep_cost = float(np.mean(episode_costs))

        # 轉 Tensor
        b_obs = torch.FloatTensor(self._obs[:T]).to(self.device)
        b_actions = torch.FloatTensor(self._actions[:T]).to(self.device)
        b_old_lp = torch.FloatTensor(self._log_probs[:T]).to(self.device)
        b_r_adv = torch.FloatTensor(r_adv).to(self.device)
        b_c_adv = torch.FloatTensor(c_adv).to(self.device)
        b_r_ret = torch.FloatTensor(r_ret).to(self.device)
        b_c_ret = torch.FloatTensor(c_ret).to(self.device)

        # 歸一化優勢（獎勵和代價分別歸一化）
        b_r_adv = (b_r_adv - b_r_adv.mean()) / (b_r_adv.std() + 1e-8)
        b_c_adv = (b_c_adv - b_c_adv.mean()) / (b_c_adv.std() + 1e-8)

        metrics_list = []
        minibatch_size = max(1, T // self.n_minibatch)

        for _ in range(self.n_epochs):
            indices = torch.randperm(T, device=self.device)

            for start in range(0, T, minibatch_size):
                end = min(start + minibatch_size, T)
                mb_idx = indices[start:end]

                mb_obs = b_obs[mb_idx]
                mb_actions = b_actions[mb_idx]
                mb_old_lp = b_old_lp[mb_idx]
                mb_r_adv = b_r_adv[mb_idx]
                mb_c_adv = b_c_adv[mb_idx]
                mb_r_ret = b_r_ret[mb_idx]
                mb_c_ret = b_c_ret[mb_idx]

                _, new_lp, entropy, new_rv, new_cv = self.net.get_action_and_value(
                    mb_obs, mb_actions
                )

                log_ratio = new_lp - mb_old_lp
                ratio = log_ratio.exp()

                # 獎勵代理目標（PPO 剪裁）
                r_surr1 = ratio * mb_r_adv
                r_surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_r_adv
                reward_pg = torch.min(r_surr1, r_surr2).mean()

                # 代價代理目標（PPO 剪裁，同樣取悲觀下界）
                c_surr1 = ratio * mb_c_adv
                c_surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_c_adv
                cost_pg = torch.min(c_surr1, c_surr2).mean()

                # Actor 損失：最大化（獎勵 - λ × 代價）
                actor_loss = -(reward_pg - self.lambda_ * cost_pg)

                # Critic 損失：獎勵值函式 + 代價值函式
                critic_loss = (
                    nn.functional.mse_loss(new_rv, mb_r_ret)
                    + nn.functional.mse_loss(new_cv, mb_c_ret)
                )

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
                    "entropy": entropy.mean().item(),
                    "approx_kl": approx_kl,
                    "lambda": self.lambda_,
                    "mean_ep_cost": mean_ep_cost,
                })

        # λ 更新（在所有 epoch 完成後）
        self.lambda_ = max(0.0, self.lambda_ + self.lr_lambda * (mean_ep_cost - self.cost_limit))

        self.total_steps += T
        self._step = 0

        if not metrics_list:
            return {}
        return {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0]}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        self._ensure_dir(path)
        torch.save({
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lambda_": self.lambda_,
        }, os.path.join(path, "ppo_lagrangian.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(
            os.path.join(path, "ppo_lagrangian.pt"), map_location=self.device
        )
        self.net.load_state_dict(ckpt["net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.lambda_ = ckpt.get("lambda_", 0.0)

    def save_resume(self, path: str) -> None:
        """儲存可供接續訓練的完整檢查點（與 save 相同，λ 也一併儲存）。"""
        self.save(path)

    def load_resume(self, path: str) -> None:
        """載入接續訓練的檢查點。"""
        self.load(path)

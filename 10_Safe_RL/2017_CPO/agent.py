"""
CPO 代理人 — 約束策略最佳化 (Constrained Policy Optimization)。

核心思想：
  TRPO：max E[Σr_t]  s.t.  KL(π_old||π_new) ≤ δ
  CPO ：max E[Σr_t]  s.t.  KL(π_old||π_new) ≤ δ  AND  E[Σc_t] ≤ d

CPO vs PPO-Lagrangian：
  PPO-Lag：用拉格朗日乘數λ把代價軟性加入目標，每步都更新λ
  CPO    ：直接在「信任域內」求解帶代價約束的最佳化問題（硬約束）

更新流程：
  1. 計算獎勵梯度 g 和代價梯度 b
  2. 用共軛梯度法計算自然梯度 x = H⁻¹g, y = H⁻¹b
  3. 計算純量 p = xᵀHx, q = yᵀHy, r = xᵀHy, c = J_C - d
  4. 若 c ≤ 0（安全）：TRPO 步驟
     若 c > 0（違規）：CPO 雙重規劃求解，混合 x 和 y
  5. 回溯線搜尋：同時確認 KL ≤ δ 且代價改善
  6. 獨立更新獎勵 / 代價評論家

參考：
    Achiam, J., Held, D., Tamar, A., & Abbeel, P. (2017).
    Constrained Policy Optimization. ICML 2017. arXiv:1705.10528
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math

from common.base_agent import BaseAgent
from network import SafeActorCriticNetwork


class CPOAgent(BaseAgent):
    """
    連續動作空間的 Constrained Policy Optimization。

    引數：
        max_kl:       KL 信任域半徑 δ
        cost_limit:   每集代價預算 d
        damping:      Fisher 向量積阻尼（數值穩定）
        cg_iters:     共軛梯度法迭代次數
        ls_iters:     回溯線搜尋最大步數
        n_critic_epochs: 評論家更新輪數
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_scale: float = 1.0,
        gamma: float = 0.99,
        gae_lambda: float = 0.97,
        max_kl: float = 0.01,
        cost_limit: float = 25.0,
        damping: float = 0.1,
        cg_iters: int = 10,
        ls_iters: int = 10,
        lr_critic: float = 3e-4,
        n_critic_epochs: int = 5,
        device: str = "cpu",
    ):
        super().__init__(state_dim, action_dim, device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_kl = max_kl
        self.cost_limit = cost_limit
        self.damping = damping
        self.cg_iters = cg_iters
        self.ls_iters = ls_iters
        self.n_critic_epochs = n_critic_epochs
        self.action_scale = action_scale

        self.net = SafeActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_optimizer = optim.Adam(
            list(self.net.reward_critic.parameters()) +
            list(self.net.cost_critic.parameters()),
            lr=lr_critic,
        )

        self._buffer: list = []

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, evaluate: bool = False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if evaluate:
                return self.net.get_mean_action(state_t).cpu().numpy()[0]
            dist = self.net.get_dist(state_t)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            rv, cv = self.net.get_values(state_t)
        return action.cpu().numpy()[0]

    def store(self, state, action, reward, cost, done) -> None:
        """儲存一步轉換資料（reward 和 cost 分開儲存）。"""
        self._buffer.append({
            "s": state, "a": action,
            "r": reward, "c": cost, "done": done,
        })

    def is_ready(self) -> bool:
        return len(self._buffer) > 0

    # ------------------------------------------------------------------
    # 自然梯度工具函式
    # ------------------------------------------------------------------

    def _actor_params(self):
        return [p for p in self.net.actor_trunk.parameters()] + \
               [p for p in self.net.actor_mean.parameters()] + \
               [self.net.actor_log_std]

    def _flat_params(self) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for p in self._actor_params()])

    def _set_flat_params(self, flat: torch.Tensor) -> None:
        idx = 0
        for p in self._actor_params():
            n = p.numel()
            p.data.copy_(flat[idx: idx + n].view(p.shape))
            idx += n

    def _fvp(self, v: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Fisher-向量積 Hv（KL Hessian 乘以 v），加阻尼。"""
        dist = self.net.get_dist(states)
        with torch.no_grad():
            old_dist = self.net.get_dist(states)

        kl = torch.distributions.kl_divergence(old_dist, dist).sum(-1).mean()
        grads = torch.autograd.grad(kl, self._actor_params(), create_graph=True)
        flat_grad = torch.cat([g.reshape(-1) for g in grads])

        gvp = (flat_grad * v.detach()).sum()
        hvp = torch.autograd.grad(gvp, self._actor_params())
        flat_hvp = torch.cat([g.contiguous().reshape(-1) for g in hvp])
        return flat_hvp + self.damping * v

    def _cg(self, b: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """共軛梯度法求解 Hx = b，回傳 x = H⁻¹b。"""
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdot = torch.dot(r, r)

        for _ in range(self.cg_iters):
            Hp = self._fvp(p, states)
            alpha = rdot / (torch.dot(p, Hp) + 1e-8)
            x = x + alpha * p
            r = r - alpha * Hp
            rdot_new = torch.dot(r, r)
            beta = rdot_new / (rdot + 1e-8)
            p = r + beta * p
            rdot = rdot_new
            if rdot < 1e-10:
                break
        return x

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------

    def _gae(self, signals, values, dones, next_value, T):
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        vext = np.append(values[:T], next_value)
        for t in reversed(range(T)):
            d = dones[t]
            delta = signals[t] + self.gamma * vext[t+1] * (1 - d) - vext[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - d) * gae
            adv[t] = gae
        return adv, adv + values[:T]

    # ------------------------------------------------------------------
    # 主更新
    # ------------------------------------------------------------------

    def update(self, next_state: np.ndarray = None, last_done: bool = True) -> dict:
        """
        CPO 更新：

        1. 計算 reward / cost GAE 優勢
        2. 計算 g（獎勵策略梯度）和 b（代價策略梯度）
        3. 共軛梯度求解自然梯度 x = H⁻¹g, y = H⁻¹b
        4. CPO 雙重規劃：若代價違規，混合 x 和 y 方向
        5. 回溯線搜尋：同時檢查 KL 和代價約束
        6. 評論家更新（reward + cost，多輪 Adam）
        """
        if not self._buffer:
            return {}

        T = len(self._buffer)
        states_np = np.array([x["s"] for x in self._buffer], dtype=np.float32)
        actions_np = np.array([x["a"] for x in self._buffer], dtype=np.float32)
        rewards_np = np.array([x["r"] for x in self._buffer], dtype=np.float32)
        costs_np = np.array([x["c"] for x in self._buffer], dtype=np.float32)
        dones_np = np.array([x["done"] for x in self._buffer], dtype=np.float32)

        states = torch.FloatTensor(states_np).to(self.device)
        actions = torch.FloatTensor(actions_np).to(self.device)

        # 計算集數平均代價（用於判斷約束是否違規）
        ep_costs = []
        ep_c = 0.0
        for t in range(T):
            ep_c += costs_np[t]
            if dones_np[t]:
                ep_costs.append(ep_c)
                ep_c = 0.0
        if not ep_costs:
            ep_costs.append(ep_c)
        mean_ep_cost = float(np.mean(ep_costs))
        c_violation = mean_ep_cost - self.cost_limit  # > 0 = 違規

        # 價值函式推論（GAE 用）
        with torch.no_grad():
            rv, cv = self.net.get_values(states)
            rv_np = rv.cpu().numpy()
            cv_np = cv.cpu().numpy()
            if last_done or next_state is None:
                next_rv, next_cv = 0.0, 0.0
            else:
                ns_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                nrv, ncv = self.net.get_values(ns_t)
                next_rv, next_cv = nrv.item(), ncv.item()

        r_adv, r_ret = self._gae(rewards_np, rv_np, dones_np, next_rv, T)
        c_adv, c_ret = self._gae(costs_np, cv_np, dones_np, next_cv, T)

        r_adv_t = torch.FloatTensor(r_adv).to(self.device)
        c_adv_t = torch.FloatTensor(c_adv).to(self.device)
        r_ret_t = torch.FloatTensor(r_ret).to(self.device)
        c_ret_t = torch.FloatTensor(c_ret).to(self.device)

        r_adv_t = (r_adv_t - r_adv_t.mean()) / (r_adv_t.std() + 1e-8)
        c_adv_t = (c_adv_t - c_adv_t.mean()) / (c_adv_t.std() + 1e-8)

        # 舊策略對數機率（重要性採樣基礎）
        with torch.no_grad():
            old_dist = self.net.get_dist(states)
            old_log_probs = old_dist.log_prob(actions).sum(-1)
            old_surr_r = ((old_log_probs - old_log_probs).exp() * r_adv_t).mean().item()
            old_surr_c = ((old_log_probs - old_log_probs).exp() * c_adv_t).mean().item()

        # --- 計算獎勵策略梯度 g ---
        dist = self.net.get_dist(states)
        log_probs = dist.log_prob(actions).sum(-1)
        surr_r = (log_probs * r_adv_t.detach()).mean()
        g_grads = torch.autograd.grad(surr_r, self._actor_params(), retain_graph=True)
        g = torch.cat([gg.reshape(-1) for gg in g_grads]).detach()

        # --- 計算代價策略梯度 b ---
        surr_c = (log_probs * c_adv_t.detach()).mean()
        b_grads = torch.autograd.grad(surr_c, self._actor_params())
        b = torch.cat([bg.reshape(-1) for bg in b_grads]).detach()

        # --- 自然梯度 x = H⁻¹g（獎勵方向）---
        x = self._cg(g, states)

        # CPO 純量計算
        Hx = self._fvp(x, states)
        p_scalar = float(torch.dot(x, Hx))   # g^T H^{-1} g  ≈ xᵀHx

        # --- CPO 步驟方向 ---
        if c_violation <= 0.0:
            # 安全：純 TRPO 步驟
            scale = math.sqrt(2 * self.max_kl / (p_scalar + 1e-8))
            step_dir = (scale * x).detach()
            update_type = "TRPO"
        else:
            # 違規：CPO 雙重規劃
            y = self._cg(b, states)
            Hy = self._fvp(y, states)
            q_scalar = float(torch.dot(y, Hy))   # b^T H^{-1} b
            r_scalar = float(torch.dot(x, Hy))   # g^T H^{-1} b

            kappa = (c_violation ** 2) / (2 * self.max_kl + 1e-8)
            det = q_scalar * p_scalar - r_scalar ** 2

            if det > 1e-8 and q_scalar > kappa:
                disc = max(0.0, kappa * det / q_scalar)
                nu = max(0.0, (r_scalar + math.sqrt(disc)) / q_scalar)
            elif q_scalar > 1e-8:
                # 退化：純代價恢復步驟
                nu = c_violation / (q_scalar + 1e-8)
            else:
                nu = 0.0

            step_dir_raw = (x - nu * y).detach()
            Hs = self._fvp(step_dir_raw, states)
            sHs = float(torch.dot(step_dir_raw, Hs))
            scale = math.sqrt(2 * self.max_kl / (sHs + 1e-8))
            step_dir = (scale * step_dir_raw).detach()
            update_type = f"CPO(ν={nu:.3f})"

        # --- 回溯線搜尋（同時確認 KL 和代價改善）---
        old_params = self._flat_params().clone()
        final_kl = 0.0
        for i in range(self.ls_iters):
            new_params = old_params + (0.5 ** i) * step_dir
            self._set_flat_params(new_params)

            with torch.no_grad():
                new_dist = self.net.get_dist(states)
                new_lp = new_dist.log_prob(actions).sum(-1)
                kl = torch.distributions.kl_divergence(old_dist, new_dist).sum(-1).mean().item()
                ratio = (new_lp - old_log_probs).exp()
                new_surr_r = (ratio * r_adv_t).mean().item()
                new_surr_c = (ratio * c_adv_t).mean().item()

            kl_ok = kl <= self.max_kl * 1.5  # 允許少量超出
            reward_ok = new_surr_r >= old_surr_r
            cost_ok = (c_violation <= 0) or (new_surr_c <= old_surr_c)

            if kl_ok and reward_ok and cost_ok:
                final_kl = kl
                break
        else:
            self._set_flat_params(old_params)
            final_kl = 0.0

        # --- 評論家更新（多輪 Adam）---
        for _ in range(self.n_critic_epochs):
            rv_pred, cv_pred = self.net.get_values(states)
            critic_loss = (
                nn.functional.mse_loss(rv_pred, r_ret_t.detach())
                + nn.functional.mse_loss(cv_pred, c_ret_t.detach())
            )
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        if np.isnan(critic_loss.item()):
            raise RuntimeError(f"NaN critic loss at step {self.total_steps}")

        self.total_steps += T
        self._buffer.clear()

        safe_flag = "✓" if mean_ep_cost <= self.cost_limit else "✗"
        return {
            "critic_loss": critic_loss.item(),
            "kl": final_kl,
            "mean_ep_cost": mean_ep_cost,
            "c_violation": c_violation,
            "update_type": 0.0 if "TRPO" in update_type else 1.0,  # 0=TRPO, 1=CPO
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        self._ensure_dir(path)
        torch.save({
            "net": self.net.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }, os.path.join(path, "cpo.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "cpo.pt"), map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])

    def save_resume(self, path: str) -> None:
        self.save(path)

    def load_resume(self, path: str) -> None:
        self.load(path)

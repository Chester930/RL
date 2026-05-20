"""
TRPO 代理人 — 信任域策略最佳化 (Trust Region Policy Optimization)。

參考文獻：
    Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015).
    Trust Region Policy Optimization. ICML 2015. arXiv:1502.05477.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn as nn
from typing import List

from common.base_agent import BaseAgent
from network import ActorCriticNetwork


class TRPOAgent(BaseAgent):
    """
    信任域策略最佳化 (Trust Region Policy Optimization)。

    TRPO 透過對策略更新施加 KL 散度約束，保證了策略的單調提升：

        最大化 (maximize)：   E_t [ pi(a_t|s_t) / pi_old(a_t|s_t) * A_t ]
        限制條件 (subject to)： KL(pi_old || pi_new) <= delta

    此問題透過以下方式求解：
    1. 使用共軛梯度法 (Conjugate gradient) 計算自然梯度方向 d。
    2. 使用線性搜尋 (Line search) 尋找滿足 KL 約束的步長大小。

    注意：正確實作 TRPO 非常複雜。此程式碼架構僅展示其核心結構，
    關鍵方法（如 conjugate_gradient、line_search）標有 TODO 註釋。

    引數：
        max_kl:     每次更新允許的最大 KL 散度。
        damping:    Fisher-向量積的阻尼因子 (Damping factor)。
        cg_iters:   共軛梯度法的疊代次數。
        line_search_iters: 回溯線性搜尋的步數。
        gae_lambda: GAE 平滑引數。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.97,
        max_kl: float = 0.01,
        damping: float = 0.1,
        cg_iters: int = 10,
        line_search_iters: int = 10,
        device: str = "cpu",
    ):
        super().__init__(state_dim, action_dim, device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_kl = max_kl
        self.damping = damping
        self.cg_iters = cg_iters
        self.line_search_iters = line_search_iters

        self.net = ActorCriticNetwork(state_dim, action_dim).to(self.device)

        # 評論家 (Critic) 擁有獨立的最佳化器（透過梯度下降進行更新）
        self.critic_optimizer = torch.optim.Adam(
            self.net.critic.parameters(), lr=lr_critic
        )

        # 資料取樣儲存區 (Rollout storage)
        self._buffer: List[dict] = []

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist = self.net.get_policy(state_t)
            if evaluate:
                return int(dist.probs.argmax().item())
            return int(dist.sample().item())

    def store(self, state, action, reward, done):
        self._buffer.append({"s": state, "a": action, "r": reward, "done": done})

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def _compute_gae(self, states, rewards, dones, next_value):
        """計算 GAE 優勢函式與回報。"""
        T = len(rewards)
        advantages = torch.zeros(T, device=self.device)
        gae = 0.0

        with torch.no_grad():
            values = self.net.get_value(states).squeeze(1)
        values_np = values.cpu().numpy().tolist() + [next_value]

        for t in reversed(range(T)):
            d = float(dones[t])
            delta = rewards[t] + self.gamma * values_np[t+1] * (1 - d) - values_np[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - d) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def _flat_params(self) -> torch.Tensor:
        """將所有演員 (Actor) 引數獲取為扁平向量。"""
        return torch.cat([p.data.view(-1) for p in self.net.actor.parameters()])

    def _set_flat_params(self, flat_params: torch.Tensor) -> None:
        """從扁平向量設定演員引數。"""
        idx = 0
        for p in self.net.actor.parameters():
            n = p.numel()
            p.data.copy_(flat_params[idx: idx + n].view(p.shape))
            idx += n

    def _fisher_vector_product(self, v: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        計算 Fisher-向量積 (Fisher-vector product，即 KL 的 Hessian 矩陣乘以 v)。

        TODO: 透過雙重反向傳播 (Double-backprop) 實作 Fisher-向量積：
            1. 計算 KL(pi_old || pi_new)
            2. 計算 Jacobian-向量積 kl_grad dot v
            3. 再次求導以獲得 Hessian-向量積
        """
        dist = self.net.get_policy(states)
        with torch.no_grad():
            old_dist = self.net.get_policy(states)

        kl = torch.distributions.kl_divergence(old_dist, dist).mean()
        grads = torch.autograd.grad(kl, self.net.actor.parameters(), create_graph=True)
        flat_grad = torch.cat([g.view(-1) for g in grads])

        gvp = (flat_grad * v).sum()
        hvp_grads = torch.autograd.grad(gvp, self.net.actor.parameters())
        flat_hvp = torch.cat([g.contiguous().view(-1) for g in hvp_grads])
        return flat_hvp + self.damping * v

    def _conjugate_gradient(self, b: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        使用共軛梯度法 (Conjugate gradient) 求解 Hx = b。

        TODO: 標準 CG 演演算法應用於 Fisher 矩陣：
            x = 0, r = b, p = b
            for i in range(cg_iters):
                alpha = r^T r / p^T H p
                x += alpha * p
                r -= alpha * H p
                if ||r|| < eps: break
                beta = r_new^T r_new / r_old^T r_old
                p = r_new + beta * p
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        r_dot = torch.dot(r, r)

        for _ in range(self.cg_iters):
            Hp = self._fisher_vector_product(p, states)
            alpha = r_dot / (torch.dot(p, Hp) + 1e-8)
            x += alpha * p
            r -= alpha * Hp
            r_dot_new = torch.dot(r, r)
            beta = r_dot_new / (r_dot + 1e-8)
            p = r + beta * p
            r_dot = r_dot_new

        return x

    def update(self, next_state=None, last_done=False) -> dict:
        """
        TRPO 更新流程：
        1. 計算 GAE 優勢函式
        2. 計算策略梯度 g
        3. 使用共軛梯度法求解 Hx = g (獲得自然梯度方向)
        4. 使用回溯線性搜尋尋找滿足 KL 約束的最大步長
        5. 透過梯度下降更新評論家 (Critic)
        """
        if not self._buffer:
            return {}

        states = torch.FloatTensor(np.array([x["s"] for x in self._buffer])).to(self.device)
        actions = torch.LongTensor([x["a"] for x in self._buffer]).to(self.device)
        rewards = [x["r"] for x in self._buffer]
        dones = [x["done"] for x in self._buffer]

        if last_done or next_state is None:
            nv = 0.0
        else:
            ns_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                nv = float(self.net.get_value(ns_t).item())

        advantages, returns = self._compute_gae(states, rewards, dones, nv)
        advantages = (advantages - advantages.mean()) / (advantages.std(correction=0) + 1e-8)

        # Old log probs
        with torch.no_grad():
            old_dist = self.net.get_policy(states)
            old_log_probs = old_dist.log_prob(actions)

        # 策略梯度 (Policy gradient)
        dist = self.net.get_policy(states)
        log_probs = dist.log_prob(actions)
        ratio = (log_probs - old_log_probs).exp()
        surr_loss = -(ratio * advantages.detach()).mean()

        grads = torch.autograd.grad(surr_loss, self.net.actor.parameters())
        flat_grad = torch.cat([g.view(-1) for g in grads])

        # TODO: 自然梯度步長 d = H^{-1} g
        natural_grad = self._conjugate_gradient(-flat_grad, states)

        # TODO: 計算滿足 KL 約束的最大步長 (Max step size)
        sAs = torch.dot(natural_grad, self._fisher_vector_product(natural_grad, states))
        step_size = torch.sqrt(2 * self.max_kl / (sAs + 1e-8))
        max_step = step_size * natural_grad

        # TODO: 回溯線性搜尋 (Backtracking line search)
        old_params = self._flat_params().clone()
        for i in range(self.line_search_iters):
            new_params = old_params + (0.5 ** i) * max_step
            self._set_flat_params(new_params)
            new_dist = self.net.get_policy(states)
            kl = torch.distributions.kl_divergence(old_dist, new_dist).mean()
            if kl < self.max_kl:
                break
        else:
            self._set_flat_params(old_params)  # 若線性搜尋失敗則恢復原始引數

        # Critic update
        values = self.net.get_value(states).squeeze(1)
        critic_loss = nn.functional.mse_loss(values, returns.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.total_steps += len(self._buffer)
        self._buffer.clear()

        return {"critic_loss": float(critic_loss.item())}

    def save(self, path: str) -> None:
        self._ensure_dir(path)
        torch.save({"net": self.net.state_dict()}, os.path.join(path, "trpo.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "trpo.pt"), map_location=self.device)
        self.net.load_state_dict(ckpt["net"])

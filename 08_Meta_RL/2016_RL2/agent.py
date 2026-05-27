"""
RL² 代理人 — 透過 GRU 實現「學習如何學習」。

核心思想：
  普通 RL：策略 π(a|s) 只看當前狀態
  RL²   ：策略 π(a|s, h_t) 同時看當前狀態 + GRU 隱藏狀態 h_t
           h_t 編碼了任務內所有歷史互動 (動作、獎勵、done 訊號)

  訓練：PPO 更新 GRU 參數，讓 RNN 學會快速識別任務結構
  推論：在新任務上，GRU 在前幾集探索後自動切換到最佳動作

輸入向量（每步）：[prev_action_onehot (n_arms), prev_reward (1), prev_done (1)]

參考：
    Wang et al. (2016). Learning to reinforcement learn. arXiv:1611.02779
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict

from network import RL2Network


class RL2Agent:
    """
    基於 GRU 的 Meta-RL 代理人，使用 PPO 更新。

    關鍵差異（相較標準 PPO）：
      - 維護 GRU 隱藏狀態（任務內不重置）
      - 輸入為 [prev_action_onehot, prev_reward, prev_done]，不含直接觀察
      - 更新時以整條任務序列為單位，保留時序結構

    引數：
        n_arms:      Bandit 臂數（= 動作維度）
        hidden_dim:  GRU 隱藏層維度
        clip_eps:    PPO 剪裁 epsilon
        gae_lambda:  GAE λ
        n_epochs:    每批資料的梯度更新次數
    """

    def __init__(
        self,
        n_arms: int,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        gae_lambda: float = 0.95,
        ent_coef: float = 0.05,
        vf_coef: float = 0.5,
        n_epochs: int = 4,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        self.n_arms = n_arms
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.n_epochs = n_epochs
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        # input_dim = prev_action_onehot(n_arms) + prev_reward(1) + prev_done(1)
        input_dim = n_arms + 2
        self.net = RL2Network(input_dim, n_arms, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)

        # 推論時的 GRU 隱藏狀態（每個新任務呼叫 reset_hidden() 重置）
        self._hidden = self.net.init_hidden(1).to(self.device)
        self._prev_action = 0
        self._prev_reward = 0.0
        self._prev_done = 0.0

    # ------------------------------------------------------------------
    # 推論
    # ------------------------------------------------------------------

    def reset_hidden(self) -> None:
        """在新任務開始時重置 GRU 隱藏狀態與前一步資訊。"""
        self._hidden = self.net.init_hidden(1).to(self.device)
        self._prev_action = 0
        self._prev_reward = 0.0
        self._prev_done = 0.0

    def _make_input(self, prev_action: int, prev_reward: float, prev_done: float) -> torch.Tensor:
        """組裝 GRU 輸入向量 [onehot(action), reward, done]，shape (1,1,input_dim)。"""
        onehot = np.zeros(self.n_arms, dtype=np.float32)
        onehot[prev_action] = 1.0
        vec = np.concatenate([onehot, [prev_reward, prev_done]])
        return torch.FloatTensor(vec).unsqueeze(0).unsqueeze(0).to(self.device)

    def select_action(self, evaluate: bool = False):
        """
        選擇動作，同時更新 GRU 隱藏狀態。

        evaluate=True → 取 logits argmax（貪婪）
        evaluate=False → 從 Categorical 採樣
        """
        x = self._make_input(self._prev_action, self._prev_reward, self._prev_done)
        with torch.no_grad():
            logits, value, new_hidden = self.net(x, self._hidden)
        self._hidden = new_hidden

        dist = torch.distributions.Categorical(logits=logits.squeeze(0).squeeze(0))
        if evaluate:
            action = int(logits.squeeze().argmax().item())
        else:
            action = int(dist.sample().item())

        log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()
        return action, log_prob, float(value.item())

    def observe(self, action: int, reward: float, done: bool) -> None:
        """儲存本步結果，供下一步 GRU 輸入使用。"""
        self._prev_action = action
        self._prev_reward = reward
        self._prev_done = float(done)

    # ------------------------------------------------------------------
    # 訓練（整批任務序列 PPO 更新）
    # ------------------------------------------------------------------

    def update(self, task_batches: List[Dict]) -> dict:
        """
        PPO 更新，以多個任務序列為輸入。

        task_batches 每個元素為一個 dict：
          - "inputs":    np.ndarray (T, input_dim)   # GRU 輸入序列
          - "actions":   np.ndarray (T,)
          - "rewards":   np.ndarray (T,)
          - "dones":     np.ndarray (T,)
          - "log_probs": np.ndarray (T,)
          - "values":    np.ndarray (T,)
        """
        if not task_batches:
            return {}

        # 計算 GAE 優勢與回報（每個任務獨立）
        all_inputs, all_actions, all_old_lp = [], [], []
        all_advantages, all_returns = [], []

        for tb in task_batches:
            T = len(tb["rewards"])
            rewards = tb["rewards"]
            values = tb["values"]
            dones = tb["dones"]

            # 末步 bootstrap = 0（任務在 T 步後結束）
            advantages = np.zeros(T, dtype=np.float32)
            gae = 0.0
            values_ext = np.append(values, 0.0)
            for t in reversed(range(T)):
                d = dones[t]
                delta = rewards[t] + self.gamma * values_ext[t+1] * (1-d) - values_ext[t]
                gae = delta + self.gamma * self.gae_lambda * (1-d) * gae
                advantages[t] = gae
            returns = advantages + values

            all_inputs.append(tb["inputs"])
            all_actions.append(tb["actions"])
            all_old_lp.append(tb["log_probs"])
            all_advantages.append(advantages)
            all_returns.append(returns)

        metrics_list = []

        for _ in range(self.n_epochs):
            # 逐任務序列更新（保留時序結構，不跨任務打亂）
            for i in range(len(task_batches)):
                x = torch.FloatTensor(all_inputs[i]).unsqueeze(0).to(self.device)  # (1, T, D)
                actions = torch.LongTensor(all_actions[i]).to(self.device)
                old_lp = torch.FloatTensor(all_old_lp[i]).to(self.device)
                adv = torch.FloatTensor(all_advantages[i]).to(self.device)
                ret = torch.FloatTensor(all_returns[i]).to(self.device)

                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # 從頭重跑整條序列（使 GRU 梯度流通）
                h0 = self.net.init_hidden(1).to(self.device)
                logits, values_pred, _ = self.net(x, h0)
                logits = logits.squeeze(0)        # (T, n_arms)
                values_pred = values_pred.squeeze(0)  # (T,)

                dist = torch.distributions.Categorical(logits=logits)
                new_lp = dist.log_prob(actions)
                entropy = dist.entropy()

                log_ratio = new_lp - old_lp
                ratio = log_ratio.exp()

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.functional.mse_loss(values_pred, ret)
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
                })

        if not metrics_list:
            return {}
        return {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0]}

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save({
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, os.path.join(path, "rl2.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "rl2.pt"), map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

    def save_resume(self, path: str) -> None:
        self.save(path)

    def load_resume(self, path: str) -> None:
        self.load(path)

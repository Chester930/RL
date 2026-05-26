"""
MAPPO — 搭配集中式價值函式 (Centralized value function) 的多代理人 PPO。

CTDE（集中式訓練，分散式執行 / Centralized Training, Decentralized Execution）：
    演員 (Actors)：各代理人的區域性觀測 -> 動作（執行時分散式）
    評論家 (Critics)：全域性狀態 -> V(s)（訓練時集中式）

論文核心發現：在合作型任務中，搭配全域性評論家的簡單 PPO 往往能優於
MADDPG 與 QMIX 等更為複雜的 MARL 演演算法。

參考文獻：
    Yu, C., Velu, A., Vinitsky, E., Wang, Y., Bayen, A., & Wu, Y. (2021).
    The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games.
    arXiv:2103.01955.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from network import AgentActor, CentralizedCritic


class MAPPOAgent:
    """
    MAPPO 管理各代理人的演員 (Actors) 與集中式評論家 (Centralized Critics)。

    引數：
        n_agents:        代理人數量。
        obs_dims:        各代理人的區域性觀測維度列表。
        global_state_dim: 用於集中式評論家的全域性狀態維度。
        action_dims:     各代理人的動作維度列表。
        hidden_dim:      MLP 隱藏層維度。
        lr_actor:        演員學習率。
        lr_critic:       評論家學習率。
        gamma:           折扣因子。
        gae_lambda:      GAE 平滑引數。
        clip_eps:        PPO 截斷 (Clip) 引數。
        n_epochs:        每次更新的 PPO 最佳化小週期數量。
        batch_size:      小批次維度。
        entropy_coef:    熵獎勵係數。
        value_coef:      價值損失係數。
        max_grad_norm:   梯度截斷範數。
        device:          "cpu" 或 "cuda"。
    """

    def __init__(
        self,
        n_agents: int,
        obs_dims: list,
        global_state_dim: int,
        action_dims: list,
        hidden_dim: int = 256,
        lr_actor: float = 5e-4,
        lr_critic: float = 5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        n_epochs: int = 10,
        batch_size: int = 64,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 10.0,
        device: str = "cpu",
    ):
        self.n_agents = n_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        # 各代理人的演員網路 (Per-agent actors)
        self.actors = nn.ModuleList([
            AgentActor(obs_dims[i], action_dims[i], hidden_dim, continuous=False).to(device)
            for i in range(n_agents)
        ])

        # 集中式評論家網路（每個代理人一個，或所有代理人共享 — 這裡為每人一個）
        self.critics = nn.ModuleList([
            CentralizedCritic(global_state_dim, hidden_dim).to(device)
            for _ in range(n_agents)
        ])

        self.actor_optimizers = [
            optim.Adam(self.actors[i].parameters(), lr=lr_actor, eps=1e-5)
            for i in range(n_agents)
        ]
        self.critic_optimizers = [
            optim.Adam(self.critics[i].parameters(), lr=lr_critic, eps=1e-5)
            for i in range(n_agents)
        ]

        # 取樣資料儲存 (Rollout storage) — 由 collect_rollout 填充
        self._rollout = None

    # --- 執行動作 (Acting) ---

    def select_actions(self, obs_list: list, deterministic: bool = False) -> tuple:
        """
        為所有代理人選擇動作並計算對數機率 (Log-probs)。

        引數：
            obs_list: 包含 n_agents 個 (obs_dim,) numpy 陣列的列表。
        回傳：
            actions:   包含 n_agents 個整數/浮動數動作的列表。
            log_probs: 包含 n_agents 個對數機率標量的列表。
        """
        actions, log_probs = [], []
        for i, obs in enumerate(obs_list):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, log_prob = self.actors[i].get_action(obs_t, deterministic)
            actions.append(action.item())
            log_probs.append(log_prob.item())
        return actions, log_probs

    def get_values(self, global_states: list) -> list:
        """
        從集中式評論家獲取價值估計 (Value estimates)。

        引數：
            global_states: 包含 n_agents 個 (global_state_dim,) 陣列的列表，
                           或者是重複的單個共享全域性狀態。
        """
        values = []
        for i, gs in enumerate(global_states):
            gs_t = torch.FloatTensor(gs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                v = self.critics[i](gs_t).item()
            values.append(v)
        return values

    # --- 取樣緩衝區 (Rollout buffer) ---

    def init_rollout(self, rollout_steps: int):
        """為一次取樣 (Rollout) 初始化每步的儲存空間。"""
        self._rollout = {
            "obs": [[] for _ in range(self.n_agents)],
            "global_states": [[] for _ in range(self.n_agents)],
            "actions": [[] for _ in range(self.n_agents)],
            "log_probs": [[] for _ in range(self.n_agents)],
            "rewards": [[] for _ in range(self.n_agents)],
            "values": [[] for _ in range(self.n_agents)],
            "dones": [[] for _ in range(self.n_agents)],
        }

    def store_step(self, obs_list, global_state, actions, log_probs,
                   rewards, values, dones):
        """儲存單步環境互動資料。"""
        for i in range(self.n_agents):
            self._rollout["obs"][i].append(obs_list[i])
            self._rollout["global_states"][i].append(global_state)
            self._rollout["actions"][i].append(actions[i])
            self._rollout["log_probs"][i].append(log_probs[i])
            self._rollout["rewards"][i].append(rewards[i])
            self._rollout["values"][i].append(values[i])
            self._rollout["dones"][i].append(float(dones[i]))

    # --- 學習更新 (Learning) ---

    def _compute_gae(self, rewards, values, dones, last_value):
        """計算 GAE 優勢值與回報 (Returns)。"""
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            nv = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * nv * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + np.array(values)
        return advantages, returns

    def update(self, last_values: list) -> dict:
        """
        使用收集到的取樣資料為所有代理人執行 PPO 更新。

        引數：
            last_values: 取樣結束時的引導價值 (Bootstrap values)。
        """
        if self._rollout is None:
            return {}

        total_metrics = {}

        for i in range(self.n_agents):
            rewards = np.array(self._rollout["rewards"][i], dtype=np.float32)
            values = np.array(self._rollout["values"][i], dtype=np.float32)
            dones = np.array(self._rollout["dones"][i], dtype=np.float32)

            advantages, returns = self._compute_gae(
                rewards, values, dones, last_values[i]
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            obs_t = torch.FloatTensor(
                np.array(self._rollout["obs"][i])
            ).to(self.device)
            gs_t = torch.FloatTensor(
                np.array(self._rollout["global_states"][i])
            ).to(self.device)
            acts_t = torch.LongTensor(
                np.array(self._rollout["actions"][i])
            ).to(self.device)
            old_lp_t = torch.FloatTensor(
                np.array(self._rollout["log_probs"][i])
            ).to(self.device)
            adv_t = torch.FloatTensor(advantages).to(self.device)
            ret_t = torch.FloatTensor(returns).to(self.device)

            T = len(rewards)
            indices = np.arange(T)
            actor_losses, critic_losses = [], []

            for _ in range(self.n_epochs):
                np.random.shuffle(indices)
                for start in range(0, T, self.batch_size):
                    mb = indices[start: start + self.batch_size]
                    mb_obs = obs_t[mb]
                    mb_gs = gs_t[mb]
                    mb_acts = acts_t[mb]
                    mb_adv = adv_t[mb]
                    mb_ret = ret_t[mb]
                    mb_old_lp = old_lp_t[mb]

                    # 演員網路更新 (Actor)
                    dist = self.actors[i](mb_obs)
                    new_lp = dist.log_prob(mb_acts)
                    entropy = dist.entropy().mean()

                    ratio = (new_lp - mb_old_lp).exp()
                    surr1 = ratio * mb_adv
                    surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # 評論家網路更新 (Critic)
                    values_pred = self.critics[i](mb_gs)
                    critic_loss = nn.functional.mse_loss(values_pred, mb_ret)

                    total_loss = (
                        actor_loss
                        + self.value_coef * critic_loss
                        - self.entropy_coef * entropy
                    )

                    self.actor_optimizers[i].zero_grad()
                    self.critic_optimizers[i].zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.actors[i].parameters())
                        + list(self.critics[i].parameters()),
                        self.max_grad_norm,
                    )
                    self.actor_optimizers[i].step()
                    self.critic_optimizers[i].step()

                    actor_losses.append(actor_loss.item())
                    critic_losses.append(critic_loss.item())

            total_metrics[f"agent{i}_actor_loss"] = np.mean(actor_losses)
            total_metrics[f"agent{i}_critic_loss"] = np.mean(critic_losses)

        self._rollout = None
        return total_metrics

    # --- 持久化 (Persistence) ---

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save({
            "actors": [a.state_dict() for a in self.actors],
            "critics": [c.state_dict() for c in self.critics],
        }, os.path.join(path, "mappo_checkpoint.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "mappo_checkpoint.pt"), map_location=self.device)
        for i, sd in enumerate(ckpt["actors"]):
            self.actors[i].load_state_dict(sd)
        for i, sd in enumerate(ckpt["critics"]):
            self.critics[i].load_state_dict(sd)

    def save_resume(self, path: str) -> None:
        """儲存暫停點，供關機後續跑使用。"""
        self.save(path)

    def load_resume(self, path: str) -> None:
        """載入暫停點。"""
        self.load(path)

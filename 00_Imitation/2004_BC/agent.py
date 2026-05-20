"""
Behavioral Cloning 代理人：用監督學習模仿專家策略。

參考文獻：
    Pomerleau, D. A. (1991). Efficient training of artificial neural networks
    for autonomous navigation. Neural Computation, 3(1), 88–97.

    Bain, M., & Sammut, C. (1995). A framework for behavioural cloning.
    Machine Intelligence, 15, 103–129.

核心概念：
    BC 把 RL 問題轉化成監督學習，直接學習「狀態 → 動作」的對映：
        loss = MSE(policy(state), expert_action)

    優點：實作簡單、收斂快、不需要獎勵訊號
    缺點：Distribution Shift（分佈偏移）——見 train.py 的展示
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from common.base_agent import BaseAgent
from network import BCNetwork


class BCAgent(BaseAgent):
    """
    Behavioral Cloning agent。

    訓練方式：epoch-based（每次 update 掃過整個 demo dataset 一遍），
    與其他 RL agent 的 step-based update 不同。

    介面與 BaseAgent 一致，可直接使用 common/utils/evaluator.py 的 evaluate()。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_scale: float = 2.0,
        lr: float = 1e-3,
        hidden_dims: tuple = (256, 256),
        device: str = "cpu",
    ):
        super().__init__(state_dim, action_dim, device)
        self.action_scale = action_scale

        self.policy = BCNetwork(
            state_dim, action_dim,
            hidden_dims=hidden_dims,
            action_scale=action_scale,
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self._dataset: TensorDataset | None = None

    # ------------------------------------------------------------------
    # 示範資料載入
    # ------------------------------------------------------------------

    def load_demos(self, demos_path: str) -> int:
        """從 .npz 檔載入專家示範資料，回傳 transition 數量。"""
        data = np.load(demos_path)
        states = torch.FloatTensor(data["states"])
        actions = torch.FloatTensor(data["actions"])
        self._dataset = TensorDataset(states, actions)

        print(f"[BCAgent] 載入 {len(self._dataset)} 筆 transitions  from {demos_path}")
        print(f"  State  shape : {tuple(states.shape)}  mean={states.mean(0).numpy().round(3)}")
        print(f"  Action shape : {tuple(actions.shape)}  mean={actions.mean(0).numpy().round(3)}")
        return len(self._dataset)

    # ------------------------------------------------------------------
    # 訓練（BaseAgent 介面）
    # ------------------------------------------------------------------

    def update(self, batch_size: int = 256) -> dict:
        """
        對 demo dataset 做一個完整 epoch 的監督學習更新。

        注意：這裡的 update() 是 epoch-based，
        一次呼叫 = 掃過所有 demo 一遍（多個 mini-batch）。
        這與其他 RL agent（DQN/PPO/SAC）的 step-based update 不同。
        """
        assert self._dataset is not None, "請先呼叫 load_demos()"

        loader = DataLoader(self._dataset, batch_size=batch_size, shuffle=True)
        self.policy.train()

        total_loss = 0.0
        n_batches = 0

        for states_b, actions_b in loader:
            states_b = states_b.to(self.device)
            actions_b = actions_b.to(self.device)

            pred = self.policy(states_b)
            loss = nn.functional.mse_loss(pred, actions_b)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        self.total_steps += 1
        return {"bc_loss": total_loss / n_batches}

    # ------------------------------------------------------------------
    # 動作選擇（與 evaluator.evaluate() 相容）
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """BC 策略永遠是確定性的，evaluate 引數僅為介面一致性保留。"""
        self.policy.eval()
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.policy(state_t)
        return action.cpu().numpy()[0]

    # ------------------------------------------------------------------
    # 存取 checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        self._ensure_dir(path)
        torch.save(self.policy.state_dict(), os.path.join(path, "bc.pt"))
        print(f"[BCAgent] Checkpoint saved → {path}/bc.pt")

    def load(self, path: str) -> None:
        ckpt_file = os.path.join(path, "bc.pt")
        self.policy.load_state_dict(
            torch.load(ckpt_file, map_location=self.device)
        )
        print(f"[BCAgent] Checkpoint loaded ← {ckpt_file}")

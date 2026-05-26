"""
MuZero 代理人 — 無需預知規則的 AlphaZero。

參考文獻：
    Schrittwieser, J., et al. (2019). Mastering Atari, Go, Chess and Shogi
    by Planning with a Learned Model. Nature 588, 604-609. arXiv:1911.08265.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import math
from typing import Optional, List

from common.base_agent import BaseAgent
from network import RepresentationNetwork, DynamicsNetwork, PredictionNetwork


class MCTSNode:
    """蒙特卡羅樹搜尋 (Monte Carlo Tree Search) 中的節點。"""

    def __init__(self, prior: float = 0.0):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children = {}  # 動作 -> MCTSNode 對映
        self.hidden_state = None
        self.reward = 0.0

    @property
    def value(self) -> float:
        return self.value_sum / max(1, self.visit_count)

    def expanded(self) -> bool:
        return len(self.children) > 0


class MCTS:
    """
    用於 MuZero 的蒙特卡羅樹搜尋 (MCTS)。

    完全在學習到的潛在空間中執行：
    - 選擇 (Selection)： 使用 UCB 分數選擇動作
    - 擴充套件 (Expansion)： 使用動態模型取得下一個狀態
    - 回傳 (Backup)： 將價值評估沿樹向上傳遞更新
    """

    def __init__(self, action_dim: int, num_simulations: int = 50,
                 c1: float = 1.25, c2: float = 19652, discount: float = 0.99):
        self.action_dim = action_dim
        self.num_simulations = num_simulations
        self.c1 = c1
        self.c2 = c2
        self.discount = discount

    def ucb_score(self, parent: MCTSNode, child: MCTSNode) -> float:
        """結合策略先驗的置信區間上限 (Upper Confidence Bound)。"""
        pb_c = math.log((parent.visit_count + self.c2 + 1) / self.c2) + self.c1
        prior_score = pb_c * child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
        return child.value + prior_score

    def run(self, root_state: torch.Tensor, prediction_net: PredictionNetwork,
            dynamics_net: DynamicsNetwork) -> List[int]:
        """
        從根狀態 (root_state) 開始執行 MCTS。

        流程：
        1. 使用預測網路的策略與價值初始化根節點
        2. 執行 num_simulations 次模擬：
           a. 選擇：使用 UCB 遍歷樹，直到遇到未擴充套件節點
           b. 擴充套件：使用動態網路對所有動作進行擴充套件
           c. 回傳：將價值向樹根方向更新

        回傳：
            action_probs: 基於造訪次數的動作機率分佈
        """
        root = MCTSNode()
        with torch.no_grad():
            policy_logits, value_logits = prediction_net(root_state)
            probs = torch.softmax(policy_logits, dim=-1).squeeze(0)
            value = PredictionNetwork.support_to_scalar(value_logits).item()

        root.value_sum = value
        root.visit_count = 1
        root.hidden_state = root_state
        for a in range(self.action_dim):
            root.children[a] = MCTSNode(prior=probs[a].item())

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            last_action = None

            # Selection
            while node.expanded():
                last_action = max(node.children, key=lambda a: self.ucb_score(node, node.children[a]))
                node = node.children[last_action]
                search_path.append(node)

            # Expansion: use dynamics_net to get next state + reward, prediction_net for policy/value
            parent = search_path[-2] if len(search_path) >= 2 else root
            leaf_value = root.value

            if last_action is not None and parent.hidden_state is not None:
                with torch.no_grad():
                    a_onehot = torch.zeros(1, self.action_dim, device=parent.hidden_state.device)
                    a_onehot[0, last_action] = 1.0
                    next_state, rew_logits = dynamics_net(parent.hidden_state, a_onehot)
                    node.reward = PredictionNetwork.support_to_scalar(rew_logits).item()
                    node_pol, node_val = prediction_net(next_state)
                    node_probs = torch.softmax(node_pol, dim=-1).squeeze(0)
                    leaf_value = PredictionNetwork.support_to_scalar(node_val).item()
                node.hidden_state = next_state
                for a in range(self.action_dim):
                    node.children[a] = MCTSNode(prior=node_probs[a].item())

            # Backup
            for bnode in reversed(search_path):
                bnode.value_sum += leaf_value
                bnode.visit_count += 1
                leaf_value = bnode.reward + self.discount * leaf_value

        # 根據造訪次數進行動作選擇 (Action selection by visit count)
        visit_counts = np.array([root.children[a].visit_count for a in range(self.action_dim)])
        return visit_counts / visit_counts.sum()


class MuZeroAgent(BaseAgent):
    """
    MuZero：透過完全自主學習的模型來進行規劃。

    與 AlphaZero 不同，MuZero 不需要預先知道遊戲規則。
    它學習以下模型：
    - h: 觀測 -> 隱藏狀態 (表示模型 / Representation)
    - g: (隱藏狀態, 動作) -> (下一個隱藏狀態, 獎勵) (動態模型 / Dynamics)
    - f: 隱藏狀態 -> (策略, 價值) (預測模型 / Prediction)

    訓練包含：
    1. 自我對弈 (Self-play)：使用 MCTS 收集對局資料
    2. 展開學習：將模型展開 K 步並計算損失函式
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        num_simulations: int = 50,
        unroll_steps: int = 5,
        td_steps: int = 10,
        gamma: float = 0.997,
        support_size: int = 601,
        device: str = "cpu",
    ):
        super().__init__(state_dim, action_dim, device)
        self.gamma = gamma
        self.unroll_steps = unroll_steps
        self.td_steps = td_steps
        self.hidden_dim = hidden_dim

        # 三個核心網路 (Three networks)
        self.representation = RepresentationNetwork(state_dim, hidden_dim).to(self.device)
        self.dynamics = DynamicsNetwork(hidden_dim, action_dim, hidden_dim, support_size).to(self.device)
        self.prediction = PredictionNetwork(hidden_dim, action_dim, support_size).to(self.device)

        self.optimizer = optim.Adam(
            list(self.representation.parameters()) +
            list(self.dynamics.parameters()) +
            list(self.prediction.parameters()),
            lr=lr, weight_decay=weight_decay,
        )

        self.mcts = MCTS(action_dim, num_simulations=num_simulations, discount=gamma)
        self.replay_buffer = deque(maxlen=100_000)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        在潛在空間執行 MCTS，根據造訪次數選擇動作。
        評估期間：確定性選擇（取造訪次數最大值）。
        訓練期間：從造訪次數分佈中取樣。

        MCTS 造訪次數分佈儲存於 self.last_mcts_probs，
        train.py 可在呼叫後讀取並存入 game_history["mcts_probs"]。
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            hidden = self.representation(state_t)

        action_probs = self.mcts.run(hidden, self.prediction, self.dynamics)
        self.last_mcts_probs = action_probs  # 供 train.py 讀取

        if evaluate:
            return int(np.argmax(action_probs))
        return int(np.random.choice(self.action_dim, p=action_probs))

    def store(self, game_history: list) -> None:
        """儲存完整的對局歷史記錄以供訓練。"""
        self.replay_buffer.append(game_history)

    def update(self) -> dict:
        """
        MuZero 展開訓練流程 (Unroll training)：
        針對每個樣本：
            s0 = 表示模型(觀測)
            針對 k = 0..K：
                pi_k, v_k = 預測模型(s_k)
                r_k, s_{k+1} = 動態模型(s_k, a_k)
            損失 = 策略損失 + 價值損失 + 獎勵損失

        注意：完整實作需要：
            1. 從重播緩衝區中取樣軌跡 (Trajectories)
            2. 計算引導目標 (Bootstrap targets，使用 td_steps 向前看)
            3. 將模型展開 K 步
            4. 計算交叉熵損失 (Cross-entropy losses)
        """
        if len(self.replay_buffer) < 32:
            return {}

        B = 16
        buf = list(self.replay_buffer)
        total_p = torch.tensor(0.0, device=self.device)
        total_v = torch.tensor(0.0, device=self.device)
        total_r = torch.tensor(0.0, device=self.device)
        n = 0

        self.optimizer.zero_grad()

        for _ in range(B):
            game = buf[np.random.randint(len(buf))]
            T = len(game)
            if T < 2:
                continue
            pos = np.random.randint(0, T)

            s = self.representation(
                torch.FloatTensor(game[pos]["obs"]).unsqueeze(0).to(self.device))

            for k in range(self.unroll_steps + 1):
                idx = pos + k
                pol_logits, val_logits = self.prediction(s)

                # Policy target: MCTS 造訪次數分佈 π_mcts = N(s,a) / Σ N(s,a')
                if idx < T and "mcts_probs" in game[idx]:
                    p_tgt = torch.FloatTensor(game[idx]["mcts_probs"]).unsqueeze(0).to(self.device)
                elif idx < T:
                    # 向後相容：舊資料無 mcts_probs 時退回 one-hot
                    p_tgt = torch.zeros(1, self.action_dim, device=self.device)
                    p_tgt[0, game[idx]["action"]] = 1.0
                else:
                    p_tgt = torch.full((1, self.action_dim), 1.0 / self.action_dim,
                                       device=self.device)

                # n-step value target bootstrapped with prediction network
                v_scalar = 0.0
                for i in range(self.td_steps):
                    if idx + i < T:
                        v_scalar += (self.gamma ** i) * game[idx + i]["reward"]
                    else:
                        break
                boot_idx = idx + self.td_steps
                if boot_idx < T:
                    with torch.no_grad():
                        bh = self.representation(
                            torch.FloatTensor(game[boot_idx]["obs"]).unsqueeze(0).to(self.device))
                        _, bv = self.prediction(bh)
                        v_scalar += (self.gamma ** self.td_steps) * \
                            PredictionNetwork.support_to_scalar(bv).item()
                v_tgt = self._scalar_to_support(v_scalar)

                total_p = total_p + (-p_tgt * F.log_softmax(pol_logits, dim=-1)).sum(-1).mean()
                total_v = total_v + (-v_tgt * F.log_softmax(val_logits, dim=-1)).sum(-1).mean()
                n += 1

                if k < self.unroll_steps:
                    a = game[idx]["action"] if idx < T else 0
                    r = game[idx]["reward"] if idx < T else 0.0
                    a_oh = torch.zeros(1, self.action_dim, device=self.device)
                    a_oh[0, a] = 1.0
                    s, r_logits = self.dynamics(s, a_oh)
                    r_tgt = self._scalar_to_support(r)
                    total_r = total_r + (-r_tgt * F.log_softmax(r_logits, dim=-1)).sum(-1).mean()

        if n == 0:
            return {}

        loss = (total_p + total_v + total_r) / n
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.representation.parameters()) +
            list(self.dynamics.parameters()) +
            list(self.prediction.parameters()),
            5.0,
        )
        self.optimizer.step()
        self.total_steps += 1
        return {
            "policy_loss": float((total_p / n).item()),
            "value_loss":  float((total_v / n).item()),
            "reward_loss": float((total_r / n).item()),
        }

    def save(self, path: str) -> None:
        self._ensure_dir(path)
        torch.save({
            "representation": self.representation.state_dict(),
            "dynamics": self.dynamics.state_dict(),
            "prediction": self.prediction.state_dict(),
        }, os.path.join(path, "muzero.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "muzero.pt"), map_location=self.device)
        self.representation.load_state_dict(ckpt["representation"])
        self.dynamics.load_state_dict(ckpt["dynamics"])
        self.prediction.load_state_dict(ckpt["prediction"])

    def _scalar_to_support(self, scalar: float) -> torch.Tensor:
        """Two-hot encode a scalar into the categorical support distribution."""
        support_size = self.dynamics.support_size
        v_min, v_max = -300.0, 300.0
        scalar = max(v_min, min(v_max, float(scalar)))
        step = (v_max - v_min) / (support_size - 1)  # = 1.0 for 601 atoms
        lower_idx = min(int((scalar - v_min) / step), support_size - 2)
        upper_frac = (scalar - v_min) / step - lower_idx
        target = torch.zeros(1, support_size, device=self.device)
        target[0, lower_idx] = 1.0 - upper_frac
        target[0, lower_idx + 1] = upper_frac
        return target

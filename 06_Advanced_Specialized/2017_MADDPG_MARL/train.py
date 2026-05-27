"""
在多代理人環境上訓練 MADDPG。

使用 PettingZoo 風格的並行環境 (Parallel env) API 或簡單的自定義多代理人環境。

參考文獻：
    Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import random
import pickle
import numpy as np
import torch

from agent import MADDPGAgent
from common.utils.logger import Logger

RESUME_DIR = "checkpoints/resume"


class SimpleCoopEnv:
    """
    用於測試的極簡 2-代理人合作環境。

    兩個代理人都觀察同一個共享狀態，並接收合作型獎勵。
    TODO: 替換為 PettingZoo 或正式的自定義多代理人環境。
    """

    def __init__(self, n_agents: int = 2, obs_dim: int = 4, action_dim: int = 2):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.t = 0
        self.max_steps = 50

    def reset(self):
        self.state = np.random.randn(self.obs_dim).astype(np.float32)
        self.t = 0
        return [self.state.copy() for _ in range(self.n_agents)]

    def step(self, actions):
        self.t += 1
        joint_action = np.concatenate(actions)
        self.state = np.clip(self.state + joint_action[:self.obs_dim] * 0.1,
                             -1.0, 1.0).astype(np.float32)
        # 合作獎勵：所有代理人根據與原點的距離受罰 (Cooperative reward)
        reward = -float(np.linalg.norm(self.state))
        rewards = [reward for _ in range(self.n_agents)]
        done = self.t >= self.max_steps
        dones = [done for _ in range(self.n_agents)]
        next_obs = [self.state.copy() for _ in range(self.n_agents)]
        return next_obs, rewards, dones, {}


def train(config: dict) -> MADDPGAgent:
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

    n_agents = config["n_agents"]
    obs_dims = [config["obs_dim"]] * n_agents
    action_dims = [config["action_dim"]] * n_agents

    env = SimpleCoopEnv(n_agents=n_agents,
                        obs_dim=config["obs_dim"],
                        action_dim=config["action_dim"])

    agent = MADDPGAgent(
        obs_dims=obs_dims,
        action_dims=action_dims,
        hidden_dim=config["hidden_dim"],
        lr_actor=config["lr_actor"],
        lr_critic=config["lr_critic"],
        gamma=config["gamma"],
        tau=config["tau"],
        noise_std=config["noise_std"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        device=config["device"],
    )

    logger = Logger(log_dir="runs", run_name="maddpg")

    start_ep = 1

    # 自動偵測暫停點並繼續 (Auto-detect resume checkpoint)
    resume_meta_path = os.path.join(RESUME_DIR, "train_meta.pkl")
    resume_ckpt_path = os.path.join(RESUME_DIR, "maddpg_checkpoint.pt")
    if os.path.exists(resume_ckpt_path) and os.path.exists(resume_meta_path):
        agent.load_resume(RESUME_DIR)
        with open(resume_meta_path, "rb") as f:
            meta = pickle.load(f)
        start_ep = meta["ep"] + 1
        random.setstate(meta["random_state"])
        np.random.set_state(meta["np_state"])
        torch.set_rng_state(meta["torch_state"])
        print(f"[RESUME] 從回合 {meta['ep']} 繼續訓練")

    print(f"正在為 {n_agents} 個代理人進行 MADDPG 訓練，共 {config['total_episodes']} 回合...")

    for ep in range(start_ep, config["total_episodes"] + 1):
        obs_list = env.reset()
        ep_rewards = [0.0] * n_agents
        done = False

        while not done:
            actions = agent.select_actions(obs_list)
            next_obs_list, rewards, dones, _ = env.step(actions)

            agent.buffer.push(obs_list, actions, rewards, next_obs_list, dones)
            obs_list = next_obs_list

            for i in range(n_agents):
                ep_rewards[i] += rewards[i]
            done = all(dones)

            metrics = agent.update()

        if ep % config["log_freq"] == 0:
            mean_reward = np.mean(ep_rewards)
            logger.log_scalar("train/mean_reward", mean_reward, ep)
            print(f"回合 {ep:6d}  平均獎勵: {mean_reward:.2f}")

        if ep % config["save_freq"] == 0:
            agent.save(f"checkpoints/maddpg_ep{ep}")
            agent.save_resume(RESUME_DIR)
            meta = {
                "ep": ep,
                "random_state": random.getstate(),
                "np_state": np.random.get_state(),
                "torch_state": torch.get_rng_state(),
            }
            with open(os.path.join(RESUME_DIR, "train_meta.pkl"), "wb") as f:
                pickle.dump(meta, f)
            print(f"  [RESUME] 暫停點已儲存至 {RESUME_DIR}（回合 {ep}）")

    logger.close()
    return agent


if __name__ == "__main__":
    config = {
        "n_agents": 2,
        "obs_dim": 4,
        "action_dim": 2,
        "total_episodes": 80_000,    # 50k→80k 給更多收斂空間
        "hidden_dim": 128,
        "lr_actor": 1e-4,
        "lr_critic": 5e-4,           # 1e-3→5e-4 降低 Critic 更新幅度
        "gamma": 0.95,
        "tau": 0.005,                # 0.01→0.005 放慢 target network 更新，穩定 Q 估計
        "noise_std": 0.1,            # 0.2→0.1 後期減少探索噪聲
        "buffer_size": 100_000,
        "batch_size": 256,
        "log_freq": 200,
        "save_freq": 2000,
        "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
        "seed": 42,
    }
    train(config)

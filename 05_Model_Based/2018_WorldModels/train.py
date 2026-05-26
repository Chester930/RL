"""
World Models 三階段訓練管線 (Training pipeline)。

階段 1：在隨機取樣的影像幀上訓練 VAE。
階段 2：將所有影像幀編碼為 z；在序列上訓練 MDN-RNN。
階段 3：在夢境（或真實）環境中透過 CMA-ES 最佳化控制器。

參考文獻：
    Ha, D., & Schmidhuber, J. (2018). World Models. arXiv:1803.10122.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import random
import numpy as np
import torch
import torch.optim as optim
# pyrefly: ignore [missing-import]
import gymnasium as gym

from agent import WorldModelsAgent
from common.utils.logger import Logger


# ---------------------------------------------------------------------------
# 階段 1：收集隨機取樣資料 (Collect random rollouts)
# ---------------------------------------------------------------------------

def collect_random_frames(env_id: str, n_episodes: int, img_size: int = 64):
    """
    執行隨機策略並收集 (obs, action, next_obs, done) 資料。

    回傳：
        frames:  (H, W, C) uint8 陣列列表。
        rollouts: 包含每一集 "obs"、"actions"、"dones" 的字典列表。
    """
    env = gym.make(env_id, render_mode="rgb_array")
    frames = []
    rollouts = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_obs, ep_actions, ep_dones = [], [], []

        done = False
        while not done:
            action = env.action_space.sample()
            frame = env.render()  # (H, W, C)
            if frame is not None:
                # 視需要將尺寸調整為 img_size x img_size
                import cv2
                frame = cv2.resize(frame, (img_size, img_size))
                frames.append(frame)
                ep_obs.append(frame)
                ep_actions.append(action)

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_dones.append(float(done))

        rollouts.append({
            "obs": np.array(ep_obs),
            "actions": np.array(ep_actions),
            "dones": np.array(ep_dones),
        })
        if (ep + 1) % 10 == 0:
            print(f"  已收集集數 {ep+1}/{n_episodes}")

    env.close()
    return frames, rollouts


# ---------------------------------------------------------------------------
# 階段 1：訓練 VAE
# ---------------------------------------------------------------------------

def train_vae(agent: WorldModelsAgent, frames: list, config: dict) -> None:
    print("\n=== 階段 1：訓練 VAE ===")
    optimizer = optim.Adam(agent.vae.parameters(), lr=config["vae_lr"])
    frames_arr = np.array(frames)

    for epoch in range(config["vae_epochs"]):
        indices = np.random.permutation(len(frames_arr))
        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(frames_arr), config["vae_batch_size"]):
            batch_idx = indices[start: start + config["vae_batch_size"]]
            batch = frames_arr[batch_idx]
            metrics = agent.update_vae(
                batch, optimizer, kl_weight=config["kl_weight"]
            )
            total_loss += metrics["vae_loss"]
            n_batches += 1

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{config['vae_epochs']}  "
                  f"VAE 損失: {total_loss/n_batches:.4f}")


# ---------------------------------------------------------------------------
# 階段 2：訓練 MDN-RNN
# ---------------------------------------------------------------------------

def encode_rollouts(agent: WorldModelsAgent, rollouts: list) -> list:
    """將所有取樣的影像幀編碼為潛在 z 序列。"""
    encoded = []
    agent.vae.eval()
    with torch.no_grad():
        for ep in rollouts:
            obs = ep["obs"]  # (T, H, W, C)
            x = torch.FloatTensor(obs).permute(0, 3, 1, 2).to(agent.device) / 255.0
            mu, _ = agent.vae.encode(x)
            encoded.append({
                "z": mu.cpu().numpy(),
                "actions": ep["actions"],
                "dones": ep["dones"],
            })
    return encoded


def train_mdnrnn(agent: WorldModelsAgent, encoded_rollouts: list, config: dict) -> None:
    print("\n=== 階段 2：訓練 MDN-RNN ===")
    optimizer = optim.Adam(agent.mdnrnn.parameters(), lr=config["rnn_lr"])
    seq_len = config["rnn_seq_len"]

    for epoch in range(config["rnn_epochs"]):
        np.random.shuffle(encoded_rollouts)
        total_loss = 0.0
        n_batches = 0

        # 從取樣資料中構建固定長度的序列 (Fixed-length sequences)
        batch_z, batch_a, batch_z_next, batch_done = [], [], [], []

        for ep in encoded_rollouts:
            z = ep["z"]        # (T, z_dim)
            a = ep["actions"]  # (T, action_dim) or (T,)
            d = ep["dones"]    # (T,)

            if len(z) < seq_len + 1:
                continue

            # 視需要將純量動作擴充套件為 2-D
            if a.ndim == 1:
                a = a.reshape(-1, 1)

            for start in range(0, len(z) - seq_len, seq_len):
                batch_z.append(z[start: start + seq_len])
                batch_a.append(a[start: start + seq_len])
                batch_z_next.append(z[start + 1: start + seq_len + 1])
                batch_done.append(d[start: start + seq_len])

            if len(batch_z) >= config["rnn_batch_size"]:
                z_t = torch.FloatTensor(np.array(batch_z[: config["rnn_batch_size"]]))
                a_t = torch.FloatTensor(np.array(batch_a[: config["rnn_batch_size"]]))
                zn_t = torch.FloatTensor(np.array(batch_z_next[: config["rnn_batch_size"]]))
                d_t = torch.FloatTensor(np.array(batch_done[: config["rnn_batch_size"]]))

                metrics = agent.update_mdnrnn(z_t, a_t, zn_t, d_t, optimizer)
                total_loss += metrics["mdnrnn_loss"]
                n_batches += 1
                batch_z, batch_a, batch_z_next, batch_done = [], [], [], []

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{config['rnn_epochs']}  "
                  f"MDN-RNN 損失: {total_loss/max(n_batches,1):.4f}")


# ---------------------------------------------------------------------------
# 階段 3：CMA-ES 控制器最佳化 (CMA-ES Controller optimization)
# ---------------------------------------------------------------------------

def rollout_controller(agent: WorldModelsAgent, env, max_steps: int = 1000) -> float:
    """使用目前的控制器執行一集；回傳總獎勵。"""
    obs, _ = env.reset()
    agent.reset_hidden()
    total_reward = 0.0

    for _ in range(max_steps):
        frame = env.render()
        import cv2
        if frame is not None:
            frame = cv2.resize(frame, (64, 64))
        else:
            # 若無法渲染則使用的備案 (Fallback)
            frame = np.zeros((64, 64, 3), dtype=np.uint8)

        action = agent.select_action(frame)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    return total_reward


def train_controller_cmaes(agent: WorldModelsAgent, env, config: dict) -> None:
    """
    對控制器引數進行 CMA-ES 最佳化。

    待辦：安裝 cma 函式庫 (pip install cma) 以使用完整的 CMA-ES 功能。
    此骨架程式碼展示了適應度 (Fitness) 評估迴圈。
    """
    print("\n=== 階段 3：CMA-ES 控制器最佳化 ===")

    try:
        # pyrefly: ignore [missing-import]
        import cma  # pip install cma
    except ImportError:
        print("  [警告] 未找到 'cma' 套件。請使用 pip install cma 進行安裝")
        print("  正在執行隨機搜尋備案...")
        _random_search_controller(agent, env, config)
        return

    x0 = agent.get_controller_params()
    sigma0 = config["cmaes_sigma"]

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        "maxiter": config["cmaes_generations"],
        "popsize": config["cmaes_popsize"],
    })

    best_reward = -np.inf

    while not es.stop():
        solutions = es.ask()
        fitnesses = []

        for params in solutions:
            agent.set_controller_params(params)
            rewards = [
                rollout_controller(agent, env, config["max_steps"])
                for _ in range(config["eval_rollouts"])
            ]
            # CMA-ES 會進行極小化；因此將獎勵取負號 (Negate reward)
            fitnesses.append(-np.mean(rewards))

        es.tell(solutions, fitnesses)
        best_reward = max(best_reward, -min(fitnesses))
        print(f"  世代 {es.result.iterations}  最佳獎勵: {best_reward:.1f}")

    best_params = es.result.xbest
    agent.set_controller_params(best_params)


def _random_search_controller(agent: WorldModelsAgent, env, config: dict) -> None:
    """備案：針對控制器的隨機擾動爬山演演算法。"""
    x0 = agent.get_controller_params()
    best_params = x0.copy()
    best_reward = rollout_controller(agent, env, config["max_steps"])

    for i in range(config["cmaes_generations"]):
        candidate = best_params + np.random.randn(*best_params.shape) * 0.1
        agent.set_controller_params(candidate)
        reward = rollout_controller(agent, env, config["max_steps"])
        if reward > best_reward:
            best_reward = reward
            best_params = candidate.copy()
        if (i + 1) % 20 == 0:
            print(f"  迭代 {i+1}  最佳獎勵: {best_reward:.1f}")

    agent.set_controller_params(best_params)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(config: dict) -> WorldModelsAgent:
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

    agent = WorldModelsAgent(
        obs_channels=config["obs_channels"],
        img_size=config["img_size"],
        latent_dim=config["latent_dim"],
        hidden_dim=config["hidden_dim"],
        action_dim=config["action_dim"],
        n_mixtures=config["n_mixtures"],
        device=config["device"],
    )

    os.makedirs("checkpoints", exist_ok=True)

    # 階段 1：收集影像並訓練 VAE
    print("正在收集隨機取樣資料...")
    frames, rollouts = collect_random_frames(
        config["env_id"], config["n_random_episodes"], config["img_size"]
    )
    print(f"從 {len(rollouts)} 集中收集了 {len(frames)} 幀影像。")

    train_vae(agent, frames, config)
    agent.save("checkpoints/world_models_vae")
    print("VAE 已儲存。")

    # Phase 2: Encode and train MDN-RNN
    encoded = encode_rollouts(agent, rollouts)
    train_mdnrnn(agent, encoded, config)
    agent.save("checkpoints/world_models_rnn")
    print("MDN-RNN 已儲存。")

    # Phase 3: CMA-ES Controller
    ctrl_env = gym.make(config["env_id"], render_mode="rgb_array")
    train_controller_cmaes(agent, ctrl_env, config)
    ctrl_env.close()

    agent.save("checkpoints/world_models_final")
    print("最終的 World Models 代理人已儲存。")
    return agent


if __name__ == "__main__":
    config = {
        "env_id": "CarRacing-v3",  # CarRacing-v2 已棄用，改用 v3
        "obs_channels": 3,
        "img_size": 64,
        "latent_dim": 32,
        "hidden_dim": 256,
        "action_dim": 3,
        "n_mixtures": 5,
        # Phase 1 — CPU demo：100 episodes → 10
        "n_random_episodes": 10,
        "vae_epochs": 10,
        "vae_batch_size": 32,
        "vae_lr": 1e-4,
        "kl_weight": 1.0,
        # Phase 2 — CPU demo：縮短 epochs
        "rnn_epochs": 10,
        "rnn_batch_size": 16,
        "rnn_seq_len": 32,
        "rnn_lr": 1e-4,
        # Phase 3 — CPU demo：100 generations → 30, popsize 16 → 8
        "cmaes_generations": 30,
        "cmaes_popsize": 8,
        "cmaes_sigma": 0.1,
        "eval_rollouts": 2,        # CPU demo：4 → 2
        "max_steps": 300,          # CPU demo：1000 → 300
        "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
        "seed": 42,
    }
    train(config)

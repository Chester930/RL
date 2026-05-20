"""在 DMControl 或 Atari 環境上訓練 Dreamer。"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import numpy as np
# pyrefly: ignore [missing-import]
import gymnasium as gym

from agent import DreamerAgent
from common.utils.logger import Logger


def train(config: dict) -> DreamerAgent:
    """
    Dreamer 訓練迴圈：
    - 將真實經驗收集至重播緩衝區
    - 定期執行世界模型與行為訓練更新
    """
    # Dreamer 需要畫素觀測影像。為了簡化，此處使用包裝後的環境。
    # 在實際應用中，請使用具備畫素觀測功能的 dm_control 或 Atari 環境。
    env = gym.make(config["env_id"], render_mode="rgb_array")

    action_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n

    agent = DreamerAgent(
        state_dim=0,  # Dreamer 使用影像，而非攤平的狀態向量 (Flat state)
        action_dim=action_dim,
        obs_channels=config["obs_channels"],
        deter_dim=config["deter_dim"],
        stoch_dim=config["stoch_dim"],
        embed_dim=config["embed_dim"],
        gamma=config["gamma"],
        lambda_=config["lambda_"],
        imagine_horizon=config["imagine_horizon"],
        device=config["device"],
    )

    logger = Logger(log_dir="runs", run_name=f"dreamer_{config['env_id']}")
    global_step = 0

    print("注意：這是一個骨架實作版本 (Skeleton implementation)。")
    print("如需完整的 Dreamer 實作，請參考：")
    print("  https://github.com/danijar/dreamer")
    print("  https://github.com/zhaoyi11/dreamer-pytorch")

    for episode in range(1, config["n_episodes"] + 1):
        obs, _ = env.reset()
        agent.reset_state()
        ep_return = ep_length = 0
        done = False

        while not done:
            # 渲染影像以取得畫素觀測值 (Pixel observation)
            pixel_obs = env.render()  # (H, W, C)
            if pixel_obs is None:
                # 若無法渲染則使用全黑影像作為備案 (Fallback)
                pixel_obs = np.zeros((64, 64, 3), dtype=np.uint8)

            # 選擇動作 (Select action)
            if global_step < config["seed_steps"]:
                action = env.action_space.sample()
                if not isinstance(action, np.ndarray):
                    action = np.array([action], dtype=np.float32)
            else:
                action = agent.select_action(pixel_obs.transpose(2, 0, 1))

            next_obs, reward, terminated, truncated, _ = env.step(
                action if hasattr(env.action_space, 'shape') else int(action.argmax())
            )
            done = terminated or truncated
            agent.store(pixel_obs, action, reward, done)
            ep_return += reward
            ep_length += 1
            global_step += 1

            # 每隔 C 步進行更新 (Update every C steps)
            if global_step % config["update_every"] == 0 and global_step > config["seed_steps"]:
                for _ in range(config["update_steps"]):
                    metrics = agent.update()
                if metrics and global_step % (config["update_every"] * 10) == 0:
                    logger.log_scalars(metrics, global_step)

        logger.log_episode(ep_return, ep_length, global_step)
        if episode % 10 == 0:
            print(f"集數 {episode:5d} | 步數 {global_step:8d} | 回報: {ep_return:.1f}")

    logger.close()
    env.close()
    return agent


if __name__ == "__main__":
    config = {
        "env_id": "Pendulum-v1",   # Replace with dm_control env for real Dreamer
        "n_episodes": 100,
        "obs_channels": 3,
        "deter_dim": 200,
        "stoch_dim": 30,
        "embed_dim": 1024,
        "gamma": 0.99,
        "lambda_": 0.95,
        "imagine_horizon": 15,
        "seed_steps": 5000,
        "update_every": 50,
        "update_steps": 1,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    train(config)

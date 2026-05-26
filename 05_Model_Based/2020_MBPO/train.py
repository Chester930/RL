"""
MBPO 訓練迴圈。

真實環境步數與模型取樣 (Rollouts) 及 SAC 更新交替執行。

參考文獻：
    Janner et al. (2019). When to Trust Your Model: Model-Based Policy Optimization.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import random
import pickle
import numpy as np
import torch
# pyrefly: ignore [missing-import]
import gymnasium as gym

from agent import MBPOAgent
from common.utils.logger import Logger
from common.utils.evaluator import evaluate

RESUME_DIR = "checkpoints/resume"


def train(config: dict) -> MBPOAgent:
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

    best_return = -float("inf")

    env = gym.make(config["env_id"])
    eval_env = gym.make(config["env_id"])

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = MBPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config["hidden_dim"],
        ensemble_members=config["ensemble_members"],
        n_elite=config["n_elite"],
        rollout_length=config["rollout_length"],
        real_ratio=config["real_ratio"],
        gamma=config["gamma"],
        tau=config["tau"],
        alpha=config.get("sac_alpha", 0.2),
        auto_alpha=config.get("auto_alpha", False),
        lr=config["lr"],
        model_lr=config["model_lr"],
        real_buffer_size=config["real_buffer_size"],
        model_buffer_size=config["model_buffer_size"],
        batch_size=config["batch_size"],
        device=config["device"],
    )

    logger = Logger(log_dir="runs", run_name=f"mbpo_{config['env_id']}")

    obs, _ = env.reset()
    ep_return = 0.0
    ep_length = 0
    start_step = 1

    # 自動偵測暫停點並繼續 (Auto-detect resume checkpoint)
    resume_meta_path = os.path.join(RESUME_DIR, "train_meta.pkl")
    resume_ckpt_path = os.path.join(RESUME_DIR, "mbpo_checkpoint.pt")
    if os.path.exists(resume_ckpt_path) and os.path.exists(resume_meta_path):
        agent.load_resume(RESUME_DIR)
        with open(resume_meta_path, "rb") as f:
            meta = pickle.load(f)
        start_step  = meta["step"] + 1
        ep_return   = meta["ep_return"]
        ep_length   = meta["ep_length"]
        best_return = meta["best_return"]
        obs         = meta["obs"]
        random.setstate(meta["random_state"])
        np.random.set_state(meta["np_state"])
        torch.set_rng_state(meta["torch_state"])
        print(f"[RESUME] 從步數 {meta['step']} 繼續訓練，歷史最佳 {best_return:.1f}")

    print(f"正在 {config['env_id']} 環境上進行 MBPO 訓練，總步數為 {config['total_steps']} 步...")

    for step in range(start_step, config["total_steps"] + 1):
        # 收集真實轉換資料 (Collect real transition)
        if step < config["random_steps"]:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.real_buffer.push(obs, action, reward, next_obs, float(done))
        obs = next_obs
        ep_return += reward
        ep_length += 1

        if done:
            logger.log_episode(ep_return, ep_length, step)
            obs, _ = env.reset()
            ep_return = 0.0
            ep_length = 0

        if step < config["random_steps"]:
            continue

        # rollout_length 慢速排程：(0→1, 20k→3, 40k→5)，避免過早長 rollout 累積誤差
        schedule = config.get("rollout_length_schedule", [(0, 1), (20000, 3), (40000, 5)])
        new_rollout = schedule[0][1]
        for threshold, length in schedule:
            if step >= threshold:
                new_rollout = length
        if new_rollout != agent.rollout_length:
            agent.rollout_length = new_rollout
            print(f"步數 {step}: rollout_length 更新為 {new_rollout}")

        # 定期重新訓練動態模型 (Retrain dynamics model)
        if step % config["model_train_freq"] == 0:
            model_metrics = agent.update_model(n_epochs=config["model_epochs"])
            if model_metrics:
                logger.log_scalars(model_metrics, step)

            # 生成模型取樣 (Generate model rollouts)
            agent.model_rollout(n_rollouts=config["rollout_batch_size"])

        # SAC 梯度更新步數 (SAC gradient steps)
        for _ in range(config["sac_updates_per_step"]):
            metrics = agent.update()
            if metrics and np.isnan(metrics.get("critic_loss", 0)):
                raise RuntimeError(f"NaN loss detected at step {step}, stopping training.")
            if metrics and step % config["log_freq"] == 0:
                logger.log_scalars(metrics, step)

        # 評估 (Evaluation)
        if step % config["eval_freq"] == 0:
            mean_r, std_r = evaluate(agent, eval_env, n_episodes=5)
            logger.log_scalar("eval/mean_return", mean_r, step)
            print(f"步數 {step:8d}  評估回報: {mean_r:.1f} ± {std_r:.1f}")
            if step > 10_000 and mean_r < best_return * 0.3:
                print(f"  [WARNING] eval 崩潰：{mean_r:.1f} vs 峰值 {best_return:.1f}")
            if mean_r > best_return:
                best_return = mean_r
                agent.save("checkpoints/best")
                print(f"  ★ 新最佳：{mean_r:.1f}，已儲存")

        # 檢查點 (Checkpoint)
        if step % config["save_freq"] == 0:
            agent.save(f"checkpoints/mbpo_step{step}")
            # 同時儲存暫停點，供關機後續跑 (Save resume state for pause/resume)
            agent.save_resume(RESUME_DIR)
            meta = {
                "step": step,
                "ep_return": ep_return,
                "ep_length": ep_length,
                "best_return": best_return,
                "obs": obs,
                "random_state": random.getstate(),
                "np_state": np.random.get_state(),
                "torch_state": torch.get_rng_state(),
            }
            with open(os.path.join(RESUME_DIR, "train_meta.pkl"), "wb") as f:
                pickle.dump(meta, f)
            print(f"  [RESUME] 暫停點已儲存至 {RESUME_DIR}（步數 {step}）")

    logger.close()
    env.close()
    eval_env.close()
    return agent


if __name__ == "__main__":
    config = {
        "env_id": "Pendulum-v1",
        "total_steps": 200_000,    # R-3：150k→200k
        "random_steps": 2_000,
        "hidden_dim": 256,
        "ensemble_members": 3,
        "n_elite": 2,
        "rollout_length": 1,       # 起始值；排程每 10k 步 +1，最高 5
        "rollout_batch_size": 400, # 200→400
        "real_ratio": 0.5,
        "sac_alpha": 0.2,          # 固定 alpha，移除自動調整（alpha 爆炸是崩潰主因）
        "auto_alpha": False,
        "rollout_length_schedule": [(0, 1), (20000, 3), (40000, 5)],  # 慢速增長
        "gamma": 0.99,
        "tau": 0.005,
        "lr": 3e-4,                # 恢復標準 SAC lr（real_ratio=0.5 已穩定）
        "model_lr": 1e-3,
        "real_buffer_size": 100_000,
        "model_buffer_size": 400_000,  # 100k→400k，容納長 rollout
        "batch_size": 256,
        "model_train_freq": 500,   # 250→500，累積更多真實資料再訓練
        "model_epochs": 5,         # 3→5，更充分訓練模型
        "sac_updates_per_step": 5,
        "log_freq": 5_000,
        "eval_freq": 10_000,
        "save_freq": 50_000,
        "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
        "seed": 42,
    }
    train(config)

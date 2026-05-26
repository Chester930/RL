"""
GRPO 訓練：群體相對策略最佳化 (Group Relative Policy Optimization)。

參考文獻：
    Shao et al. (2024). DeepSeekMath. arXiv:2402.03300.
    DeepSeek-R1 (2025). arXiv:2501.12948.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import random
import numpy as np
import torch

from agent import GRPOAgent
from common.utils.logger import Logger


def make_math_reward_fn(vocab_size: int, answer_token: int = 42):
    """
    用於數學問題展示的合成獎勵函式。

    生產環境中：將模型輸出與正確答案 (Ground-truth) 進行比較。
    若回應以 'answer_token' 結尾則回傳 1.0，否則回傳 0.0。

    格式獎勵 (應用於 DeepSeek-R1):
        若回應符合 <think>...</think><answer>...</answer> 格式則 +1
    準確度獎勵:
        若答案正確則 +1
    """
    def reward_fn(prompt_ids: torch.Tensor, response_ids: torch.Tensor) -> torch.Tensor:
        # 檢查最後一個 token 是否匹配答案 token（合成的正確性檢查）
        last_tokens = response_ids[:, -1]  # (G,)
        rewards = (last_tokens == answer_token).float()
        # 增加少量的格式獎勵（此處簡化為固定獎勵）
        format_reward = torch.ones(len(response_ids)) * 0.1
        return (rewards + format_reward).to(response_ids.device)

    return reward_fn


def train(config: dict) -> GRPOAgent:
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

    device = config["device"]

    agent = GRPOAgent(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        max_seq_len=config["max_seq_len"],
        group_size=config["group_size"],
        beta=config["beta"],
        clip_eps=config["clip_eps"],
        lr=config["lr"],
        temperature=config["temperature"],
        device=device,
    )

    reward_fn = make_math_reward_fn(config["vocab_size"])
    logger = Logger(log_dir="runs", run_name="grpo")
    os.makedirs("checkpoints", exist_ok=True)

    print(f"正在訓練 GRPO，總步數為 {config['total_steps']} 步...")
    print(f"  群體大小 (group_size)={config['group_size']}, beta={config['beta']}")
    print("注意：生產環境中，請使用真實數學資料集與可驗證的獎勵機制。")

    for step in range(1, config["total_steps"] + 1):
        # 取樣一個提示詞（生產環境中：應從訓練資料集中獲取）
        prompt_ids = torch.randint(
            0, config["vocab_size"],
            (config["prompt_len"],), device=device
        )

        metrics = agent.update(
            prompt_ids=prompt_ids,
            reward_fn=reward_fn,
            max_new_tokens=config["max_new_tokens"],
            n_epochs=config["n_epochs_per_step"],
        )

        if step % config["log_freq"] == 0:
            logger.log_scalars(metrics, step)
            print(f"步數 {step:5d}  "
                  f"損失: {metrics['total_loss']:.4f}  "
                  f"KL: {metrics['kl']:.4f}  "
                  f"平均獎勵: {metrics['mean_reward']:.3f}  "
                  f"標準差: {metrics['reward_std']:.3f}")

        if step % config["save_freq"] == 0:
            agent.save(f"checkpoints/grpo_step{step}")

    logger.close()
    return agent


if __name__ == "__main__":
    config = {
        "vocab_size": 1000,
        "d_model": 128,
        "n_layers": 4,
        "n_heads": 4,
        "max_seq_len": 256,
        "prompt_len": 32,
        "max_new_tokens": 64,
        # GRPO 超引數
        "group_size": 8,      # G — 每個提示詞取樣的回應數量
        "beta": 0.04,         # KL 懲罰項 (DeepSeek-R1 使用 0.001-0.04)
        "clip_eps": 0.2,      # PPO 截斷 (Clip) 引數
        "temperature": 1.0,   # 取樣溫度 (Sampling temperature)
        "lr": 1e-6,           # 用於微調的極小學習率 (Fine-tuning lr)
        "n_epochs_per_step": 1,
        "total_steps": 500,
        "log_freq": 50,
        "save_freq": 250,
        "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
        "seed": 42,
    }
    train(config)

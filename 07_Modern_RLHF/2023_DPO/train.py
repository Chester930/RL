"""
DPO 訓練：直接從偏好對 (Preference pairs) 進行直接偏好最佳化。

參考文獻：
    Rafailov et al. (2023). Direct Preference Optimization. NeurIPS 2023.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import random
import numpy as np
import torch
from agent import DPOAgent
from common.utils.logger import Logger


def make_synthetic_preferences(vocab_size, seq_len, batch_size, device):
    """用於測試的合成偏好對。"""
    chosen = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    rejected = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return chosen, rejected


def train(config: dict) -> DPOAgent:
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

    device = config["device"]

    agent = DPOAgent(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        max_seq_len=config["max_seq_len"],
        beta=config["beta"],
        lr=config["lr"],
        label_smoothing=config["label_smoothing"],
        device=device,
    )

    logger = Logger(log_dir="runs", run_name="dpo")
    os.makedirs("checkpoints", exist_ok=True)

    print(f"正在訓練 DPO，總步數為 {config['total_steps']} 步...")
    print("注意：生產環境中，請使用真實的 SFT 模型與人類偏好資料。")

    for step in range(1, config["total_steps"] + 1):
        chosen_ids, rejected_ids = make_synthetic_preferences(
            config["vocab_size"], config["max_seq_len"],
            config["batch_size"], device
        )
        prompt_len = config["max_seq_len"] // 2  # 前半部分為提示詞 (Prompt)

        metrics = agent.update(chosen_ids, rejected_ids, prompt_len)

        if step % config["log_freq"] == 0:
            logger.log_scalars(metrics, step)
            print(f"步數 {step:5d}  損失: {metrics['dpo_loss']:.4f}  "
                  f"獎勵差距: {metrics['reward_margin']:.4f}  "
                  f"準確率: {metrics['accuracy']:.2%}")

        if step % config["save_freq"] == 0:
            agent.save(f"checkpoints/dpo_step{step}")

    logger.close()
    return agent


if __name__ == "__main__":
    config = {
        "vocab_size": 1000,
        "d_model": 128,
        "n_layers": 4,
        "n_heads": 4,
        "max_seq_len": 64,
        "beta": 0.1,           # KL 正規化強度 (KL regularization strength)
        "lr": 1e-5,
        "label_smoothing": 0.0,
        "batch_size": 16,
        "total_steps": 1000,
        "log_freq": 100,
        "save_freq": 500,
        "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
        "seed": 42,
    }
    train(config)

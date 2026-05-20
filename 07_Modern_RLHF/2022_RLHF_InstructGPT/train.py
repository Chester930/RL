"""
RLHF InstructGPT 三階段訓練流程。

第一階段：SFT — 在演示資料 (Demonstrations) 上執行監督式微調。
第二階段：RM  — 基於人類偏好對比資料 (Preference pairs) 訓練獎勵模型。
第三階段：PPO — 對比獎勵模型進行策略最佳化。

參考文獻：
    Ouyang et al. (2022). Training language models to follow instructions
    with human feedback. NeurIPS 2022. arXiv:2203.02155.

生產環境注意：
    實際的大型語言模型 RLHF 請使用 HuggingFace TRL：
        pip install trl transformers
        from trl import PPOTrainer, RewardTrainer
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn.functional as F

from agent import RLHFAgent
from common.utils.logger import Logger


def make_synthetic_sft_data(vocab_size: int, seq_len: int, n_samples: int, device: str):
    """用於測試的合成演示資料。"""
    input_ids = torch.randint(0, vocab_size, (n_samples, seq_len), device=device)
    labels = input_ids.clone()
    # 將前半部分（提示詞）標記為 -100 以在計算損失時忽略
    labels[:, :seq_len // 2] = -100
    return input_ids, labels


def make_synthetic_preference_data(vocab_size: int, seq_len: int, batch_size: int, device: str):
    """用於獎勵模型訓練的合成偏好對。"""
    chosen = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    rejected = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return chosen, rejected


def make_synthetic_prompts(vocab_size: int, prompt_len: int, response_len: int,
                           batch_size: int, device: str):
    """用於 PPO 生成的合成提示詞 + 回應。"""
    prompt_ids = torch.randint(0, vocab_size, (batch_size, prompt_len), device=device)
    response_ids = torch.randint(0, vocab_size, (batch_size, response_len), device=device)
    return prompt_ids, response_ids


def train(config: dict) -> RLHFAgent:
    device = config["device"]
    agent = RLHFAgent(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        max_seq_len=config["max_seq_len"],
        lr_sft=config["lr_sft"],
        lr_rm=config["lr_rm"],
        lr_ppo=config["lr_ppo"],
        kl_coef=config["kl_coef"],
        clip_eps=config["clip_eps"],
        device=device,
    )

    logger = Logger(log_dir="runs", run_name="rlhf_instructgpt")
    os.makedirs("checkpoints", exist_ok=True)

    # =========================================================================
    # Phase 1: Supervised Fine-Tuning (SFT)
    # =========================================================================
    print("\n=== 第一階段：監督式微調 (SFT) ===")

    for step in range(1, config["sft_steps"] + 1):
        input_ids, labels = make_synthetic_sft_data(
            config["vocab_size"], config["max_seq_len"],
            config["batch_size"], device
        )
        metrics = agent.sft_step(input_ids, labels)

        if step % config["log_freq"] == 0:
            logger.log_scalars(metrics, step)
            print(f"  SFT 步數 {step:5d}  損失: {metrics['sft_loss']:.4f}")

    agent.save("checkpoints/rlhf", phase="sft")
    agent.initialize_rm_from_sft()

    # =========================================================================
    # Phase 2: Reward Model Training
    # =========================================================================
    print("\n=== 第二階段：獎勵模型訓練 (RM) ===")

    for step in range(1, config["rm_steps"] + 1):
        chosen, rejected = make_synthetic_preference_data(
            config["vocab_size"], config["max_seq_len"],
            config["batch_size"], device
        )
        metrics = agent.rm_step(chosen, rejected)

        if step % config["log_freq"] == 0:
            logger.log_scalars({f"rm_{k}": v for k, v in metrics.items()},
                               config["sft_steps"] + step)
            print(f"  RM 步數  {step:5d}  損失: {metrics['rm_loss']:.4f}  "
                  f"獎勵差距: {metrics['reward_margin']:.4f}")

    agent.save("checkpoints/rlhf", phase="rm")
    agent.init_ppo_from_sft()

    # =========================================================================
    # Phase 3: PPO-RLHF
    # =========================================================================
    print("\n=== 第三階段：PPO-RLHF ===")

    for step in range(1, config["ppo_steps"] + 1):
        prompt_ids, response_ids = make_synthetic_prompts(
            config["vocab_size"],
            config["max_seq_len"] // 2,  # 提示詞部分 (Prompt)
            config["max_seq_len"] // 2,  # 回應部分 (Response)
            config["batch_size"], device,
        )

        # 計算目前策略下的舊對數機率 (Old log-probs) — 於更新前計算
        with torch.no_grad():
            logits = agent.policy(response_ids)
            log_probs_all = F.log_softmax(logits, dim=-1)
            token_lp = log_probs_all.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)
            old_log_probs = token_lp.mean(dim=-1)

        # 計算包含 RM 分數與 KL 懲罰的獎勵值 (RM + KL rewards)
        rewards = agent.compute_rewards(prompt_ids, response_ids)

        # PPO update
        metrics = agent.ppo_step(prompt_ids, response_ids, old_log_probs, rewards)

        global_step = config["sft_steps"] + config["rm_steps"] + step
        if step % config["log_freq"] == 0:
            logger.log_scalars({f"ppo_{k}": v for k, v in metrics.items()}, global_step)
            print(f"  PPO 步數 {step:5d}  損失: {metrics['ppo_loss']:.4f}  "
                  f"平均獎勵: {metrics['mean_reward']:.4f}")

    agent.save("checkpoints/rlhf", phase="ppo")
    logger.close()
    return agent


if __name__ == "__main__":
    config = {
        "vocab_size": 1000,
        "d_model": 128,
        "n_layers": 4,
        "n_heads": 4,
        "max_seq_len": 64,
        "batch_size": 16,
        # Phase 1
        "sft_steps": 500,
        "lr_sft": 1e-4,
        # Phase 2
        "rm_steps": 500,
        "lr_rm": 1e-4,
        # Phase 3
        "ppo_steps": 500,
        "lr_ppo": 1e-5,
        "kl_coef": 0.1,
        "clip_eps": 0.2,
        "log_freq": 100,
        "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
    }
    train(config)

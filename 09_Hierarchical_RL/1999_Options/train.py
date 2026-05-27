"""
在 FourRooms GridWorld 上訓練 Options 框架。

訓練目標：
  Agent 學會透過走廊 option 組合，從左上（起點）到右下（目標）。

對比實驗：
  - Flat Q-learning：每步選原始動作（無 option）
  - Options         ：每步選 option，option 內選動作

Options 的優勢：時間抽象讓 agent 能直接思考「去哪個房間」，
而非規劃數百步的原始動作序列。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import random
import pickle
import numpy as np

from env import FourRoomsEnv, H, W, GOAL
from agent import OptionsAgent

RESUME_DIR = "checkpoints/resume"


# ------------------------------------------------------------------
# Flat Q-learning 基線（對比用）
# ------------------------------------------------------------------

class FlatQAgent:
    """標準 Tabular Q-learning，無時間抽象，作為 Options 的對照組。"""

    def __init__(self, n_states: int, n_actions: int = 4,
                 alpha: float = 0.1, gamma: float = 0.99, eps: float = 0.3):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def select_action(self, state: int) -> int:
        if np.random.random() < self.eps:
            return np.random.randint(4)
        return int(np.argmax(self.Q[state]))

    def update(self, s, a, r, s_next, done):
        target = r + (0.0 if done else self.gamma * np.max(self.Q[s_next]))
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])


# ------------------------------------------------------------------
# 評估（貪婪策略成功率）
# ------------------------------------------------------------------

def evaluate(agent, env: FourRoomsEnv, n_episodes: int = 50, use_options: bool = True):
    """
    評估 agent 在 n 集內的成功率（到達目標 = 成功）。

    Options 評估須完整走 act/observe 迴圈（含 option 終止判斷），
    同時暫時將 eps 設為 0（純貪婪）。
    """
    successes = 0
    for _ in range(n_episodes):
        s_int = env.reset()

        if use_options:
            agent.reset_option()
            # 暫時關閉探索
            saved_hi, saved_lo = agent.eps_hi, agent.eps_lo
            agent.eps_hi = 0.0
            agent.eps_lo = 0.0

        for _ in range(env.max_steps):
            s_rc = env.get_state()

            if use_options:
                a = agent.act(s_int, s_rc)
                ns_int, _, done = env.step(a)
                ns_rc = env.get_state()
                agent.observe(s_int, a, ns_int, s_rc, ns_rc, 0.0, done)
            else:
                a = int(np.argmax(agent.Q[s_int]))
                ns_int, _, done = env.step(a)

            if env.get_state() == GOAL:
                successes += 1
                done = True

            s_int = ns_int
            if done:
                break

        if use_options:
            agent.eps_hi, agent.eps_lo = saved_hi, saved_lo
            agent.reset_option()

    return successes / n_episodes


# ------------------------------------------------------------------
# 訓練
# ------------------------------------------------------------------

def train(config: dict):
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    env = FourRoomsEnv(max_steps=config["max_steps_per_ep"])
    n_states = env.n_states

    agent = OptionsAgent(
        n_states=n_states,
        n_options=config["n_options"],
        alpha_hi=config["alpha_hi"],
        alpha_lo=config["alpha_lo"],
        gamma=config["gamma"],
        eps_hi=config["eps_hi"],
        eps_lo=config["eps_lo"],
        option_steps=config["option_steps"],
    )
    flat_agent = FlatQAgent(n_states, alpha=config["alpha_lo"], gamma=config["gamma"])

    best_success_rate = 0.0
    start_ep = 1

    # 接續訓練
    resume_meta = os.path.join(RESUME_DIR, "train_meta.pkl")
    resume_ckpt = os.path.join(RESUME_DIR, "options_agent.pkl")
    if os.path.exists(resume_ckpt) and os.path.exists(resume_meta):
        agent.load_resume(RESUME_DIR)
        with open(resume_meta, "rb") as f:
            meta = pickle.load(f)
        start_ep = meta["episode"] + 1
        best_success_rate = meta["best_success_rate"]
        random.setstate(meta["random_state"])
        np.random.set_state(meta["np_state"])
        print(f"[RESUME] 從集數 {meta['episode']} 繼續，歷史最佳成功率 {best_success_rate:.3f}")

    n_episodes = config["n_episodes"]
    print(f"FourRooms Options | {n_episodes} 集訓練 | option_steps={config['option_steps']}")
    print(f"  4 個 option 子目標（走廊位置）: {[(2,6),(8,6),(5,2),(5,9)]}")

    options_rewards = []
    flat_rewards = []

    for ep in range(start_ep, n_episodes + 1):
        # ---- Options agent ----
        s_int = env.reset()
        agent.reset_option()
        ep_reward = 0.0
        while True:
            s_rc = env.get_state()
            a = agent.act(s_int, s_rc)
            ns_int, r, done = env.step(a)
            ns_rc = env.get_state()
            agent.observe(s_int, a, ns_int, s_rc, ns_rc, r, done)
            ep_reward += r
            s_int = ns_int
            if done:
                break
        options_rewards.append(ep_reward)

        # ---- Flat Q-learning agent ----
        s_int = env.reset()
        ep_reward_flat = 0.0
        while True:
            a = flat_agent.select_action(s_int)
            ns_int, r, done = env.step(a)
            flat_agent.update(s_int, a, r, ns_int, done)
            ep_reward_flat += r
            s_int = ns_int
            if done:
                break
        flat_rewards.append(ep_reward_flat)

        if ep % config["log_freq"] == 0:
            w = config["log_freq"]
            mean_o = np.mean(options_rewards[-w:])
            mean_f = np.mean(flat_rewards[-w:])
            print(
                f"集數 {ep:5d}/{n_episodes} | "
                f"Options 平均獎勵 {mean_o:.3f} | "
                f"Flat Q 平均獎勵 {mean_f:.3f}"
            )

        if ep % config["eval_freq"] == 0:
            sr_opts = evaluate(agent, env, use_options=True)
            sr_flat = evaluate(flat_agent, env, use_options=False)
            print(
                f"  >>> Eval | Options 成功率 {sr_opts:.3f} | Flat Q 成功率 {sr_flat:.3f}"
            )
            if sr_opts > best_success_rate:
                best_success_rate = sr_opts
                agent.save("checkpoints/best")
                print(f"  ★ 新最佳成功率: {best_success_rate:.3f}，已儲存")

        if ep % config["save_freq"] == 0:
            agent.save_resume(RESUME_DIR)
            meta = {
                "episode": ep,
                "best_success_rate": best_success_rate,
                "random_state": random.getstate(),
                "np_state": np.random.get_state(),
            }
            with open(resume_meta, "wb") as f:
                pickle.dump(meta, f)
            print(f"  [RESUME] 暫停點已儲存（集數 {ep}）")

    return agent, flat_agent


# ------------------------------------------------------------------
# 入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    config = {
        # 環境
        "max_steps_per_ep": 500,

        # Options 設定
        "n_options": 4,         # 4 個走廊 option
        "option_steps": 50,     # 每個 option 最多執行 50 步

        # Q-learning 超參數
        "alpha_hi": 0.1,        # 高層學習率
        "alpha_lo": 0.1,        # 低層學習率
        "gamma": 0.99,
        "eps_hi": 0.3,          # 高層探索率
        "eps_lo": 0.1,          # 低層探索率

        # 訓練規模
        "n_episodes": 5000,

        # 日誌 / 存檔
        "log_freq": 200,
        "eval_freq": 500,
        "save_freq": 1000,

        "seed": 42,
    }
    train(config)

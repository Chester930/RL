"""
Behavioral Cloning 訓練指令碼（Pendulum-v1）

流程：
    Step 1  收集（或載入）SAC expert 示範資料 → demos.npz
    Step 2  BC 監督學習訓練（epoch-based）
    Step 3  每 eval_freq epochs 跑標準評估
    Step 4  訓練後進行「分佈偏移測試」—— 課程核心示範

用法：
    python train.py              # 完整流程
    python train.py --skip-collect   # 重用已有的 demos.npz
"""

import sys
import os
import argparse
import numpy as np
import torch
import gymnasium as gym

# 只插入必要路徑；SAC 模組透過 sac_expert.py 的 importlib 載入，不插 SAC_DIR
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.dirname(__file__))

from agent import BCAgent
from collect_demos import collect as collect_demos
from sac_expert import SACExpertAgent
from common.utils.evaluator import evaluate


# ──────────────────────────────────────────────────────────────────────
# 分佈偏移測試工具
# ──────────────────────────────────────────────────────────────────────

def eval_at_angle(agent, env: gym.Env,
                  angle_deg: float, n_episodes: int = 5) -> float:
    """
    固定初始角度，評估 agent 的平均回報。

    Pendulum 內部狀態：unwrapped.state = [theta, theta_dot]
        theta = 0   → 擺錘直立（↑）目標位置
        theta = π/2 → 擺錘水平（→）
        theta = π   → 擺錘朝下（↓）最難恢復

    BC 在 θ ≈ 0 的初始條件表現接近專家（訓練分佈內），
    θ 越大 → 越少見過這種狀態 → 動作失準 → 分佈偏移崩潰。
    """
    angle_rad = np.deg2rad(angle_deg)
    returns = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        # 覆寫初始狀態，略過環境的隨機 init
        env.unwrapped.state = np.array([angle_rad, 0.0], dtype=np.float32)
        obs = env.unwrapped._get_obs()

        ep_return = 0.0
        while True:
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            if terminated or truncated:
                break

        returns.append(ep_return)

    return float(np.mean(returns))


# ──────────────────────────────────────────────────────────────────────
# 主訓練流程
# ──────────────────────────────────────────────────────────────────────

def train(config: dict, skip_collect: bool = False) -> BCAgent:

    # ── Step 1: 準備 demo 資料 ────────────────────────────────────────
    demos_path = os.path.join(os.path.dirname(__file__), config["demos_path"])

    if skip_collect and os.path.exists(demos_path):
        print(f"[Train] 跳過收集，使用現有 demos: {demos_path}")
    else:
        print("=" * 60)
        print("Step 1｜收集 SAC expert 示範資料")
        print("=" * 60)
        collect_demos(
            n_episodes=config["n_demo_episodes"],
            save_path=demos_path,
            device=config["device"],
        )

    # ── Step 2: 建立 BC agent & 載入 demo ────────────────────────────
    env      = gym.make(config["env_id"])
    eval_env = gym.make(config["env_id"])

    state_dim    = env.observation_space.shape[0]
    action_dim   = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])

    agent = BCAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_scale=action_scale,
        lr=config["lr"],
        device=config["device"],
    )
    agent.load_demos(demos_path)

    # ── Step 3: 訓練 ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2｜BC 監督學習訓練")
    print("=" * 60)
    print(f"{'Epoch':>6}  {'BC Loss':>10}  {'Eval Return':>14}")
    print("-" * 38)

    for epoch in range(1, config["n_epochs"] + 1):
        metrics = agent.update(batch_size=config["batch_size"])

        if epoch % config["eval_freq"] == 0 or epoch == 1:
            mean_r, std_r = evaluate(
                agent, eval_env, n_episodes=config["eval_episodes"]
            )
            print(f"{epoch:6d}  {metrics['bc_loss']:10.6f}  "
                  f"{mean_r:8.1f} ± {std_r:.1f}")

    agent.save(os.path.join(os.path.dirname(__file__), "checkpoints"))

    # ── Step 4: 分佈偏移測試（課程核心）─────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3｜分佈偏移測試 (Distribution Shift Demo)")
    print("=" * 60)
    print("""
理論：BC 只學過「專家訪問過的狀態」。
      SAC expert 快速讓擺錘直立（θ→0），
      因此 demo 資料 80%+ 都是 θ ≈ 0 附近的狀態。

      測試：把初始角度從 0° 一路推到 180°，
      BC 在 0° 附近表現接近專家（in-distribution）；
      θ 越大，BC 越少見過這種狀態 → 崩潰（out-of-distribution）。
""")

    sac = SACExpertAgent(device=config["device"])
    test_env = gym.make(config["env_id"])

    test_angles = [0, 30, 60, 90, 120, 150, 180]

    print(f"{'角度':>6}  {'SAC Expert':>12}  {'BC Policy':>12}  {'BC/Expert':>10}")
    print("-" * 50)

    for deg in test_angles:
        sac_r = eval_at_angle(sac, test_env, deg, n_episodes=config["shift_episodes"])
        bc_r  = eval_at_angle(agent, test_env, deg, n_episodes=config["shift_episodes"])

        ratio = (bc_r / sac_r * 100) if sac_r != 0 else float("nan")
        flag  = " ⚠" if ratio < 70 else ""

        print(f"{deg:5d}°  {sac_r:12.1f}  {bc_r:12.1f}  {ratio:9.1f}%{flag}")

    test_env.close()

    print("""
結論：
  • BC 在 0°~30° 表現接近 SAC（訓練分佈內，模仿成功）
  • BC 在 90°+ 開始崩潰（分佈偏移，從未見過這些狀態）
  • SAC 在所有角度都穩定（off-policy 探索覆蓋各種初始條件）

  → 這正是 BC 的根本限制，也是為什麼我們需要 RL。
""")

    env.close()
    eval_env.close()
    return agent


# ──────────────────────────────────────────────────────────────────────
# 進入點
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-collect", action="store_true",
                        help="跳過 demo 收集，直接使用現有 demos.npz")
    args = parser.parse_args()

    config = {
        "env_id":          "Pendulum-v1",
        # demo 收集
        "n_demo_episodes": 50,
        "demos_path":      "demos.npz",
        # BC 訓練
        "n_epochs":        100,
        "batch_size":      256,
        "lr":              1e-3,
        # 標準評估
        "eval_freq":       10,
        "eval_episodes":   10,
        # 分佈偏移測試
        "shift_episodes":  5,
        # 裝置
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    train(config, skip_collect=args.skip_collect)

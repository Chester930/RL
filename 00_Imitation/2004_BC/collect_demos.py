"""
使用訓練好的 SAC expert 收集 Pendulum-v1 示範資料。

輸出：demos.npz
    states  : (N, 3) float32  — [cos(θ), sin(θ), θ̇]
    actions : (N, 1) float32  — 扭矩 ∈ [-2, 2]

用法：
    python collect_demos.py                    # 50 集，存至 demos.npz
    python collect_demos.py --episodes 100     # 100 集
"""

import sys
import os
import argparse
import numpy as np
import torch
import gymnasium as gym

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.dirname(__file__))

from sac_expert import load_sac_expert


def collect(n_episodes: int = 50, save_path: str = "demos.npz",
            device: str = "cpu") -> dict:
    """
    跑 SAC expert n_episodes 集，收集所有 (state, action) 對。

    使用確定性動作（get_deterministic_action），確保 demo 品質穩定。
    隨機動作（get_action）雖然也可用，但雜訊會讓 BC 學到的 target 較吵。
    """
    expert, _ = load_sac_expert(device)
    env = gym.make("Pendulum-v1")

    all_states:  list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    returns:     list[float]      = []

    print(f"\n[Collect] 開始收集 {n_episodes} 集示範資料...")
    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_return = 0.0

        while True:
            state_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action = expert.get_deterministic_action(state_t)
            action_np = action.cpu().numpy()[0]

            all_states.append(obs.copy())
            all_actions.append(action_np.copy())

            obs, reward, terminated, truncated, _ = env.step(action_np)
            ep_return += reward

            if terminated or truncated:
                break

        returns.append(ep_return)
        if (ep + 1) % 10 == 0:
            recent = np.mean(returns[-10:])
            print(f"  Episode {ep+1:3d}/{n_episodes}  "
                  f"Return: {ep_return:8.1f}  "
                  f"Recent-10 avg: {recent:.1f}")

    env.close()

    states_arr  = np.array(all_states,  dtype=np.float32)
    actions_arr = np.array(all_actions, dtype=np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    np.savez(save_path, states=states_arr, actions=actions_arr)

    stats = {
        "n_transitions": int(len(states_arr)),
        "n_episodes":    n_episodes,
        "expert_mean":   float(np.mean(returns)),
        "expert_std":    float(np.std(returns)),
    }
    print(f"\n[Collect] 完成！")
    print(f"  Transitions : {stats['n_transitions']}")
    print(f"  Expert eval : {stats['expert_mean']:.1f} ± {stats['expert_std']:.1f}")
    print(f"  已儲存至    : {save_path}")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--save",     type=str, default="demos.npz")
    args = parser.parse_args()

    collect(n_episodes=args.episodes, save_path=args.save)

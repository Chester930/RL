"""生成 Options vs Flat Q-learning 訓練曲線圖。"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- 訓練獎勵（每 200 集平均）----
ep_log = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000,
          2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000,
          4200, 4400, 4600, 4800, 5000]
opts_r = [0.912, 0.976, 0.976, 0.977, 0.977, 0.978, 0.978, 0.978, 0.979, 0.978,
          0.974, 0.975, 0.976, 0.979, 0.978, 0.977, 0.978, 0.978, 0.977, 0.978,
          0.976, 0.977, 0.972, 0.971, 0.977]
flat_r = [0.918, 0.973, 0.973, 0.973, 0.973, 0.973, 0.973, 0.973, 0.974, 0.973,
          0.973, 0.974, 0.973, 0.975, 0.974, 0.974, 0.974, 0.974, 0.974, 0.973,
          0.974, 0.974, 0.973, 0.973, 0.973]

# ---- 評估成功率（每 500 集）----
eval_ep  = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
opts_sr  = [0.84, 0.80, 0.84, 0.82, 0.74, 0.84, 0.82, 0.80, 0.80, 0.84]
flat_sr  = [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Options vs Flat Q-learning — FourRooms GridWorld", fontsize=13, fontweight="bold")

# 左圖：平均集數獎勵
ax = axes[0]
ax.plot(ep_log, opts_r, label="Options", color="#2196F3", linewidth=1.8)
ax.plot(ep_log, flat_r, label="Flat Q", color="#FF9800", linewidth=1.8, linestyle="--")
ax.set_xlabel("Episode")
ax.set_ylabel("Mean Episode Reward")
ax.set_title("Training Reward (200-ep avg)")
ax.set_ylim(0.85, 1.01)
ax.legend()
ax.grid(alpha=0.3)

# 右圖：評估成功率
ax = axes[1]
ax.plot(eval_ep, opts_sr, "o-", label="Options", color="#2196F3", linewidth=2, markersize=7)
ax.plot(eval_ep, flat_sr, "s--", label="Flat Q", color="#FF9800", linewidth=2, markersize=7)
ax.axhline(y=0.84, color="#2196F3", linestyle=":", alpha=0.5, label="Options best (0.84)")
ax.set_xlabel("Episode")
ax.set_ylabel("Success Rate")
ax.set_title("Evaluation Success Rate (greedy, 50 eps)")
ax.set_ylim(0.0, 1.1)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("training_curve.png", dpi=150, bbox_inches="tight")
print("Saved: training_curve.png")

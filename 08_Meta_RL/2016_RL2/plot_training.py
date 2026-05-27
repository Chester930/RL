"""生成 RL² 訓練曲線圖（後半段最優臂命中率 vs 更新次數）。"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- 訓練中每 20 步的全段命中率 ----
train_updates = list(range(20, 1001, 20))
train_hits = [
    0.215, 0.247, 0.228, 0.233, 0.194,  # 20-100
    0.186, 0.193, 0.261, 0.177, 0.212,  # 120-200
    0.203, 0.184, 0.222, 0.188, 0.230,  # 220-300
    0.208, 0.173, 0.219, 0.204, 0.221,  # 320-400
    0.200, 0.231, 0.215, 0.205, 0.201,  # 420-500
    0.258, 0.162, 0.243, 0.211, 0.230,  # 520-600
    0.198, 0.219, 0.165, 0.205, 0.218,  # 620-700
    0.192, 0.220, 0.218, 0.238, 0.207,  # 720-800
    0.219, 0.228, 0.230, 0.235, 0.237,  # 820-900
    0.282, 0.249, 0.221, 0.242, 0.284,  # 920-1000
]

# ---- Eval 後半段命中率（每 100 步）----
eval_updates = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
eval_hits    = [0.216, 0.287, 0.401, 0.141, 0.248, 0.208, 0.198, 0.283, 0.316, 0.418]
random_baseline = 0.200

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("RL² on 5-Armed Bandit — GRU Meta-RL Training", fontsize=13, fontweight="bold")

# 訓練命中率（全段，淺色）
ax.plot(train_updates, train_hits, color="#90CAF9", linewidth=1.0,
        alpha=0.6, label="Train hit rate (all steps)")

# Eval 後半段命中率（深色，標重點）
ax.plot(eval_updates, eval_hits, "o-", color="#1565C0", linewidth=2.2,
        markersize=8, label="Eval hit rate (late half, greedy)")

# 隨機基線
ax.axhline(y=random_baseline, color="#E53935", linestyle="--", linewidth=1.5,
           label=f"Random baseline ({random_baseline:.3f})")

# 標最佳點
best_idx = int(np.argmax(eval_hits))
ax.annotate(f"Best: {eval_hits[best_idx]:.3f}",
            xy=(eval_updates[best_idx], eval_hits[best_idx]),
            xytext=(eval_updates[best_idx] - 150, eval_hits[best_idx] + 0.04),
            arrowprops=dict(arrowstyle="->", color="#1565C0"),
            fontsize=10, color="#1565C0")

ax.set_xlabel("PPO Updates")
ax.set_ylabel("Best-Arm Hit Rate")
ax.set_ylim(0.0, 0.65)
ax.set_xlim(0, 1050)
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("training_curve.png", dpi=150, bbox_inches="tight")
print("Saved: training_curve.png")

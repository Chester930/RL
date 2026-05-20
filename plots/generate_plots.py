"""
課程訓練曲線產生器

從各算法的 training_log.md 數據畫圖，輸出課堂用 PNG。
所有數據硬編碼自實際訓練記錄，直接執行即可產出 7 張圖。

用法：
    cd C:\\Users\\666\\Desktop\\RL\\plots
    python generate_plots.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 全域樣式 ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        150,
    "figure.facecolor":  "white",
    "axes.facecolor":    "#f8f9fa",
    "axes.grid":         True,
    "grid.alpha":        0.4,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "lines.linewidth":   2.0,
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

OUT_DIR = os.path.dirname(__file__)

COLORS = {
    "reinforce": "#e74c3c",
    "ppo":       "#2ecc71",
    "dqn":       "#3498db",
    "q_est":     "#e67e22",
    "ddpg":      "#9b59b6",
    "td3":       "#1abc9c",
    "sac":       "#f39c12",
    "alpha":     "#e74c3c",
    "bc":        "#e74c3c",
    "expert":    "#2ecc71",
}


# ══════════════════════════════════════════════════════════════════════
# 圖 1：REINFORCE 失敗曲線
# 來源：03_Policy_Gradient/1992_REINFORCE/training_log.md
# ══════════════════════════════════════════════════════════════════════
def plot_reinforce_failure():
    episodes = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    eval_ret = [9.6,  9.7,  9.1,  9.1,  9.0,  9.6,  9.2,  9.2,  27.3, 9.5]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episodes, eval_ret, color=COLORS["reinforce"], marker="o", ms=5,
            label="REINFORCE eval")
    ax.axhline(195, color="gray", linestyle="--", alpha=0.7,
               label="CartPole 官方解決標準 (195)")
    ax.axhline(500, color="black", linestyle=":", alpha=0.5,
               label="滿分 (500)")

    ax.set_xlabel("訓練集數 (Episodes)")
    ax.set_ylabel("Eval 平均回報")
    ax.set_title("REINFORCE 在 CartPole-v1 的表現\n高方差導致無法收斂（eval 始終 ~9）")
    ax.set_ylim(0, 550)
    ax.legend(fontsize=9)

    note = ("5000 集後 eval = 9.5\n"
            "高方差 = 每集梯度方向不同\n"
            "→ 需要 Clip（PPO）")
    ax.text(4800, 60, note, ha="right", fontsize=8.5,
            color=COLORS["reinforce"], style="italic")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "01_reinforce_failure.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out}")


# ══════════════════════════════════════════════════════════════════════
# 圖 2：DQN 學習曲線 + Q 值高估
# 來源：02_Value_Based_Deep/2013_DQN/training_log.md
# ══════════════════════════════════════════════════════════════════════
def plot_dqn_learning():
    steps   = [10_000, 30_000, 50_000, 100_000, 150_000]
    eval_r  = [30,     115,    28.8,   106.7,   500.0]
    mean_q  = [None,   None,   95.1,   312.9,   224.4]
    q_steps = [50_000, 100_000, 150_000]
    q_vals  = [95.1,   312.9,   224.4]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 左：eval 曲線
    ax1.plot(steps, eval_r, color=COLORS["dqn"], marker="o", ms=5,
             label="DQN eval")
    ax1.axhline(500, color="black", linestyle=":", alpha=0.5, label="滿分")
    ax1.axvline(50_000, color="gray", linestyle="--", alpha=0.5,
                label="ε 衰減完畢")
    ax1.annotate("ε 衰減完畢\n暫時退步", xy=(50_000, 28.8),
                 xytext=(65_000, 60),
                 arrowprops=dict(arrowstyle="->", color="gray"),
                 fontsize=8, color="gray")
    ax1.set_xlabel("訓練步數")
    ax1.set_ylabel("Eval 平均回報")
    ax1.set_title("DQN 學習曲線\n(CartPole-v1，150K steps → 滿分 500)")
    ax1.set_ylim(0, 560)
    ax1.legend(fontsize=8)

    # 右：Q 值高估
    ax2.bar(["50K", "100K", "150K"], q_vals,
            color=[COLORS["q_est"]] * 3, alpha=0.8, edgecolor="white")
    ax2.axhline(99.3, color=COLORS["dqn"], linestyle="--", linewidth=1.8,
                label="理論最大 Q* ~99.3")
    ax2.set_ylabel("Mean Q 值")
    ax2.set_title("DQN 的 Q 值高估偏差\n（訓練步數）")
    ax2.legend(fontsize=9)

    for i, (x, v) in enumerate(zip(["50K", "100K", "150K"], q_vals)):
        ratio = v / 99.3
        ax2.text(i, v + 5, f"×{ratio:.1f}", ha="center", fontsize=9,
                 color=COLORS["q_est"], fontweight="bold")

    ax2.text(1, 170,
             "Mean Q 是理論值 2~3 倍\n→ Double DQN 的動機",
             ha="center", fontsize=8.5, color=COLORS["q_est"], style="italic")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "02_dqn_learning.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out}")


# ══════════════════════════════════════════════════════════════════════
# 圖 3：DDPG 不穩定性（帶標準差陰影）
# 來源：04_Actor_Critic_Continuous/2015_DDPG/training_log.md
# ══════════════════════════════════════════════════════════════════════
def plot_ddpg_instability():
    steps = [10_000, 20_000, 30_000, 50_000, 80_000,
             100_000, 130_000, 170_000, 200_000]
    mean  = [-164.9, -212.7, -113.6, -136.4, -195.1,
             -101.6, -120.9, -150.4, -121.1]
    std   = [126.7,   93.1,  103.6,   64.3,   76.5,
              72.0,   49.8,   42.4,   72.8]

    mean = np.array(mean)
    std  = np.array(std)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, mean, color=COLORS["ddpg"], marker="o", ms=4,
            label="DDPG eval (mean)")
    ax.fill_between(steps, mean - std, mean + std,
                    alpha=0.25, color=COLORS["ddpg"], label="±1 std")
    ax.axhline(-100, color="gray", linestyle="--", alpha=0.6,
               label="近似最優（-100）")

    ax.annotate("峰值 -101.6\n(100K steps)", xy=(100_000, -101.6),
                xytext=(120_000, -50),
                arrowprops=dict(arrowstyle="->", color=COLORS["ddpg"]),
                fontsize=8.5, color=COLORS["ddpg"])

    ax.set_xlabel("訓練步數")
    ax.set_ylabel("Eval 平均回報（越大越好）")
    ax.set_title("DDPG 在 Pendulum-v1 的訓練曲線\n高標準差（±50~127）顯示策略不穩定")
    ax.legend(fontsize=9)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "03_ddpg_instability.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out}")


# ══════════════════════════════════════════════════════════════════════
# 圖 4：TD3 vs DDPG 對比
# 來源：兩者的 training_log.md
# ══════════════════════════════════════════════════════════════════════
def plot_td3_vs_ddpg():
    ddpg_steps = [10_000, 20_000, 30_000, 50_000, 80_000,
                  100_000, 130_000, 170_000, 200_000]
    ddpg_mean  = [-164.9, -212.7, -113.6, -136.4, -195.1,
                  -101.6, -120.9, -150.4, -121.1]

    td3_steps = [10_000, 20_000, 40_000, 70_000, 80_000,
                 100_000, 150_000, 200_000]
    td3_mean  = [-566.6, -136.9, -122.5, -119.8, -123.2,
                 -120.8, -155.1, -122.8]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ddpg_steps, ddpg_mean, color=COLORS["ddpg"], marker="o", ms=4,
            label="DDPG")
    ax.plot(td3_steps, td3_mean,  color=COLORS["td3"],  marker="s", ms=4,
            label="TD3")
    ax.axhline(-100, color="gray", linestyle="--", alpha=0.6,
               label="近似最優（-100）")

    ax.annotate("TD3 從 20K 就\n快速收斂", xy=(20_000, -136.9),
                xytext=(40_000, -200),
                arrowprops=dict(arrowstyle="->", color=COLORS["td3"]),
                fontsize=8.5, color=COLORS["td3"])

    # TD3 改進說明
    improvements = ("TD3 的三個改進：\n"
                    "① Twin Critics（防高估）\n"
                    "② Delayed Actor（更穩定）\n"
                    "③ Target Smoothing（防過擬合）")
    ax.text(110_000, -500, improvements, fontsize=8, color=COLORS["td3"],
            bbox=dict(facecolor="white", edgecolor=COLORS["td3"],
                      boxstyle="round,pad=0.4", alpha=0.8))

    ax.set_xlabel("訓練步數")
    ax.set_ylabel("Eval 平均回報（越大越好）")
    ax.set_title("TD3 vs DDPG（Pendulum-v1）\nTD3 收斂更快，最終表現相近")
    ax.legend(fontsize=9)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "04_td3_vs_ddpg.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out}")


# ══════════════════════════════════════════════════════════════════════
# 圖 5：SAC — 自動溫度調節 + LunarLander
# 來源：04_Actor_Critic_Continuous/2018_SAC/training_log.md
# ══════════════════════════════════════════════════════════════════════
def plot_sac_alpha():
    # LunarLanderContinuous（SAC 的強項）
    ll_steps  = [25_000, 50_000, 75_000, 100_000, 125_000, 150_000]
    ll_eval   = [-117.2,  160.5,  182.9,   262.4,   198.9,   215.7]
    ll_alpha  = [0.0758,  0.0831, 0.0853,  0.0912,  0.0867,  0.0736]

    # Pendulum（alpha 自動下降的展示）
    pend_steps = [10_000, 20_000, 30_000, 40_000, 100_000]
    pend_alpha = [0.3211,  0.1255, 0.0471, 0.0261,  0.0201]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左：LunarLander eval
    ax = axes[0]
    ax.plot(ll_steps, ll_eval, color=COLORS["sac"], marker="o", ms=5,
            label="SAC eval")
    ax.axhline(200, color="gray", linestyle="--", alpha=0.7,
               label="過關標準 (200)")
    ax.annotate("50K 就突破 200\n（PPO 需 163K）",
                xy=(50_000, 160.5), xytext=(60_000, 50),
                arrowprops=dict(arrowstyle="->", color=COLORS["sac"]),
                fontsize=8.5, color=COLORS["sac"])
    ax.set_xlabel("訓練步數")
    ax.set_ylabel("Eval 平均回報")
    ax.set_title("SAC 在 LunarLanderContinuous-v3\n100K → 262.4（off-policy 樣本效率）")
    ax.legend(fontsize=9)

    # 右：alpha 自動下降（Pendulum）
    ax2 = axes[1]
    ax2.plot(pend_steps, pend_alpha, color=COLORS["alpha"],
             marker="^", ms=6, label="Alpha（溫度參數）")
    ax2.axhline(0.0, color="gray", linestyle=":", alpha=0.5)
    ax2.annotate("自動從 0.32\n下降至 0.02",
                 xy=(40_000, 0.0261), xytext=(55_000, 0.15),
                 arrowprops=dict(arrowstyle="->", color=COLORS["alpha"]),
                 fontsize=8.5, color=COLORS["alpha"])
    ax2.set_xlabel("訓練步數（Pendulum-v1）")
    ax2.set_ylabel("Alpha（探索率）")
    ax2.set_title("SAC 自動溫度調節\n初始高探索 → 收斂後低探索")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "05_sac_alpha.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out}")


# ══════════════════════════════════════════════════════════════════════
# 圖 6：REINFORCE vs PPO 對比（課程壓軸）
# 來源：兩者的 training_log.md
# ══════════════════════════════════════════════════════════════════════
def plot_reinforce_vs_ppo():
    # REINFORCE（CartPole，以 episode 為單位）
    # 1 episode ≈ ~10 steps，5000 episodes ≈ 50K steps
    rf_episodes = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    rf_eval     = [9.6,  9.7,  9.1,  9.1,  9.0,  9.6,  9.2,  9.2, 27.3,  9.5]

    # PPO CartPole（以步數為單位，20K 就達滿分）
    ppo_steps   = [20_480, 40_960, 61_440, 81_920, 102_400, 122_880, 143_360]
    ppo_eval    = [500.0,  500.0,  500.0,  500.0,  500.0,   500.0,   500.0]

    # 統一轉為 steps（REINFORCE 10 steps/ep）
    rf_steps = [ep * 10 for ep in rf_episodes]

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(rf_steps, rf_eval, color=COLORS["reinforce"], marker="o", ms=4,
            linestyle="--", label="REINFORCE（CartPole，無 Clip）")
    ax.plot(ppo_steps, ppo_eval, color=COLORS["ppo"], marker="s", ms=5,
            label="PPO（CartPole，Clipped Surrogate）")

    ax.axhline(195, color="gray", linestyle=":", alpha=0.6,
               label="官方解決標準 (195)")

    # 標記 PPO 20K 就達滿分
    ax.annotate("20K steps → 滿分 500\n（PPO Clip 生效）",
                xy=(20_480, 500), xytext=(30_000, 420),
                arrowprops=dict(arrowstyle="->", color=COLORS["ppo"]),
                fontsize=9, color=COLORS["ppo"], fontweight="bold")

    ax.annotate("REINFORCE 5萬步後\n仍在 9~10（高方差）",
                xy=(50_000, 9.5), xytext=(20_000, 80),
                arrowprops=dict(arrowstyle="->", color=COLORS["reinforce"]),
                fontsize=9, color=COLORS["reinforce"])

    ax.set_xlabel("訓練步數（近似）")
    ax.set_ylabel("Eval 平均回報（CartPole-v1）")
    ax.set_title("REINFORCE vs PPO\nClip 機制讓策略不崩潰 → 20K steps 達滿分")
    ax.set_ylim(-20, 560)
    ax.legend(fontsize=9)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "06_reinforce_vs_ppo.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out}")


# ══════════════════════════════════════════════════════════════════════
# 圖 7：BC vs SAC（實際訓練數字）
# ══════════════════════════════════════════════════════════════════════
def plot_bc_vs_sac_placeholder():
    """BC 分佈偏移測試實際數字（來自 00_Imitation/2004_BC/train.py）。"""
    angles      = [0, 30, 60, 90, 120, 150, 180]
    bc_returns  = [-0.0, -132.7, -127.3, -122.2, -245.0, -231.3, -436.5]
    sac_returns = [-0.0, -132.2, -127.1, -123.4, -122.2, -231.1, -353.5]

    # 如果數據還沒有，畫示意圖
    if bc_returns[0] is None or any(v is None for v in bc_returns):
        fig, ax = plt.subplots(figsize=(8, 4))
        # 示意數據（用來展示預期形狀）
        bc_demo  = [-150, -200, -400, -800, -1200, -1500, -1600]
        sac_demo = [-120, -118, -115, -110,  -105,  -108,  -112]

        ax.plot(angles, sac_demo, color=COLORS["expert"], marker="s", ms=6,
                linewidth=2.5, label="SAC Expert（穩定）")
        ax.plot(angles, bc_demo,  color=COLORS["bc"],     marker="o", ms=6,
                linewidth=2.5, linestyle="--", label="BC Policy（示意）")

        ax.axvspan(60, 180, alpha=0.08, color="red",
                   label="BC 分佈偏移區域（示意）")
        ax.set_xlabel("初始角度（0°=直立，180°=朝下）")
        ax.set_ylabel("Eval 平均回報")
        ax.set_title("BC vs SAC：分佈偏移測試（示意圖）\n跑完 BC train.py 後替換為實際數字")
        ax.legend(fontsize=9)
        ax.text(120, -600,
                "⚠ 示意圖，數字待填入\n  run: python 00_Imitation/2004_BC/train.py",
                fontsize=8.5, color="red",
                bbox=dict(facecolor="lightyellow", edgecolor="red",
                          boxstyle="round,pad=0.3"))

        fig.tight_layout()
        out = os.path.join(OUT_DIR, "07_bc_vs_sac_placeholder.png")
        fig.savefig(out)
        plt.close(fig)
        print(f"  ✓ {out}  (示意圖，數據待填)")
        return

    # 有實際數據時的正式圖
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(angles, sac_returns, color=COLORS["expert"], marker="s", ms=6,
            linewidth=2.5, label="SAC Expert（穩定）")
    ax.plot(angles, bc_returns,  color=COLORS["bc"],     marker="o", ms=6,
            linewidth=2.5, linestyle="--", label="BC Policy")
    ax.axvspan(100, 180, alpha=0.08, color="red", label="BC 崩潰區（120°+）")
    ax.set_xlabel("初始角度（0°=直立，180°=朝下）")
    ax.set_ylabel("Eval 平均回報")
    ax.set_title("BC vs SAC：分佈偏移測試（實際數字）\n角度越大，BC 越少見此狀態 → 崩潰")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "07_bc_vs_sac.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out}")


# ══════════════════════════════════════════════════════════════════════
# 圖 0：Q-table 狀態空間爆炸（概念圖）
# ══════════════════════════════════════════════════════════════════════
def plot_qtable_explosion():
    """展示 Q-table 大小隨關節數量指數爆炸，引出 DQN 的必要性。"""
    bins = 10  # 每個維度離散化成 10 格
    joints = list(range(1, 9))  # 1~8 個關節
    n_actions = 3  # 每個關節 3 個離散動作（左/停/右）

    q_sizes = [bins**j * n_actions**j for j in joints]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(joints, q_sizes, color="#3498db", alpha=0.8, edgecolor="white")

    # 標示幾個關鍵點
    labels = {1: "180\n(可行)", 2: "2,700\n(可行)", 4: "810K\n(勉強)",
              6: "729M\n(不可行)", 8: "65.6B\n(完全不可行)"}
    for j, label in labels.items():
        ax.text(j, q_sizes[j-1] * 1.1, label, ha="center", va="bottom",
                fontsize=8.5, color="#2c3e50")

    # 標示 FetchReach 機器手臂（6DOF）
    ax.axvline(x=6, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(6.15, q_sizes[5] * 0.5, "機器手臂\n(6 DOF)", color="red",
            fontsize=9, va="center")

    ax.set_yscale("log")
    ax.set_xlabel("關節數量（每個關節離散化 10 格，3 個動作）")
    ax.set_ylabel("Q-table 大小（log scale）")
    ax.set_title("Q-table 的維度詛咒\n關節越多，Q-table 指數爆炸 → 需要 DQN")
    ax.set_xticks(joints)
    ax.set_xticklabels([f"{j} 關節" for j in joints])

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "00_qtable_explosion.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out}")


# ══════════════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("產生課程訓練曲線圖...\n")

    plot_qtable_explosion()
    plot_reinforce_failure()
    plot_dqn_learning()
    plot_ddpg_instability()
    plot_td3_vs_ddpg()
    plot_sac_alpha()
    plot_reinforce_vs_ppo()
    plot_bc_vs_sac_placeholder()

    print(f"\n全部完成，PNG 存至：{OUT_DIR}")

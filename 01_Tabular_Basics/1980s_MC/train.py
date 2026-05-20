"""
在 FrozenLake 環境訓練 MC 代理人，並在同目錄產生 training_log.md。

執行方式：
    python train.py
"""

import os
import numpy as np
# pyrefly: ignore [missing-import]
import gymnasium as gym
from datetime import datetime

from agent import MCAgent

# ── 常數 ──────────────────────────────────────────────────────────────────────
#  0  1  2  3
#  4  5  6  7
#  8  9 10 11
# 12 13 14 15
_HOLES = {5, 7, 11, 12}
_GOAL  = 15
_ARROW = {0: '←', 1: '↓', 2: '→', 3: '↑'}


def _state_info(s):
    """回傳 (row, col, label)；label 為 'G' / 'H' / 'F'。"""
    label = "G" if s == _GOAL else ("H" if s in _HOLES else "F")
    return s // 4, s % 4, label


# ── Markdown 輔助 ───────────────────────────────────────────────────────────────

def _q_best_table(Q):
    """4×4 各格最大 Q 值 Markdown 表格。"""
    rows = ["| 行\\列 | 0 | 1 | 2 | 3 |", "|:---:|:---:|:---:|:---:|:---:|"]
    for row in range(4):
        cells = []
        for col in range(4):
            s = row * 4 + col
            if s == _GOAL:
                cells.append("**G**")
            elif s in _HOLES:
                cells.append("**H**")
            else:
                cells.append(f"{float(np.max(Q[s])):.4f}")
        rows.append(f"| **{row}** | {' | '.join(cells)} |")
    return "\n".join(rows)


def _policy_table(Q):
    """4×4 最佳動作（argmax Q）Markdown 表格。"""
    rows = ["| 行\\列 | 0 | 1 | 2 | 3 |", "|:---:|:---:|:---:|:---:|:---:|"]
    for row in range(4):
        cells = []
        for col in range(4):
            s = row * 4 + col
            if s == _GOAL:
                cells.append("**G**")
            elif s in _HOLES:
                cells.append("**H**")
            else:
                cells.append(_ARROW[int(np.argmax(Q[s]))])
        rows.append(f"| **{row}** | {' | '.join(cells)} |")
    return "\n".join(rows)


# ── 詳細集數日誌 ─────────────────────────────────────────────────────────────────

def _log_episode_detail(ep_num, trajectory, Q_snap, count_snap, gamma):
    """
    產生單集完整計算過程的 Markdown 段落。
    trajectory : agent._episode 的副本（在 agent.update() 前複製）
    Q_snap     : agent.Q 的副本（在 agent.update() 前複製）
    count_snap : agent.returns_count 的副本（在 agent.update() 前複製）
    """
    lines = []
    lines.append(f"### 第 {ep_num} 集（完整計算過程）")
    lines.append("")

    success = any(r > 0 for _, _, r in trajectory)
    result_label = "成功抵達終點 G" if success else "失敗（掉入洞口或超時）"
    lines.append(f"**結果：{result_label}　｜　步數：{len(trajectory)}**")
    lines.append("")

    # ① 行走軌跡 ─────────────────────────────────────────────────────────────
    lines.append("**① 行走軌跡：**")
    lines.append("")
    lines.append("| 步驟 | 格子 | 位置 | 動作 | 獎勵 | 備註 |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|:---|")
    for t, (s, a, r) in enumerate(trajectory):
        row, col, label = _state_info(s)
        note = ""
        if t == 0:
            note = "起點 S"
        if label == "H":
            note = "掉入洞口 H"
        if r > 0:
            note = "**到達終點 G！**"
        lines.append(
            f"| {t + 1} | {s} | ({row},{col}) "
            f"| {_ARROW[a]} | {int(r)} | {note} |"
        )
    lines.append("")

    # ② 倒序計算回報 G ──────────────────────────────────────────────────────────
    n = len(trajectory)
    G_vals = [0.0] * n
    G = 0.0
    for i in reversed(range(n)):
        _, _, r = trajectory[i]
        G = r + gamma * G
        G_vals[i] = G

    lines.append("**② 倒序計算回報 G：**")
    lines.append("")
    lines.append("```")
    lines.append(f"公式：G_t = r_t + {gamma} × G_{{t+1}}")
    lines.append(f"（從最後一步往前推，最後一步沒有下一步，所以 G_{{t+1}} = 0）")
    lines.append("")
    for i in reversed(range(n)):
        _, _, r = trajectory[i]
        if i == n - 1:
            lines.append(
                f"步驟 {i + 1:>2}（最後步）：G = r={int(r)}"
                + " " * 28
                + f"→ G = {G_vals[i]:.4f}"
            )
        else:
            lines.append(
                f"步驟 {i + 1:>2}           ：G = {int(r)} + {gamma} × {G_vals[i + 1]:.4f}"
                f"  → G = {G_vals[i]:.4f}"
            )
    lines.append("```")
    lines.append("")

    # ③ 首次存取 Q 更新 ──────────────────────────────────────────────────────────
    lines.append(
        "**③ 首次存取 Q 更新**（同一集數中，每個 (格子, 動作) 組合只用第一次遇到的 G 值更新）**：**"
    )
    lines.append("")
    lines.append("| 步驟 | 格子 | 動作 | G 值 | Q 更新前 | 第 N 次 | Q 更新後 | 計算式 |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|")

    visited = set()
    Q_sim    = Q_snap.copy()
    cnt_sim  = count_snap.copy()

    for t, (s, a, _) in enumerate(trajectory):
        row, col, _ = _state_info(s)
        G_t = G_vals[t]
        if (s, a) not in visited:
            visited.add((s, a))
            q_before = float(Q_sim[s, a])
            n_count  = int(cnt_sim[s, a]) + 1
            q_after  = q_before + (1.0 / n_count) * (G_t - q_before)
            calc_str = f"{q_before:.4f} + (1/{n_count})×({G_t:.4f}−{q_before:.4f})"
            lines.append(
                f"| {t + 1} | {s} ({row},{col}) | {_ARROW[a]} "
                f"| {G_t:.4f} | {q_before:.4f} | 第 {n_count} 次 "
                f"| {q_after:.4f} | {calc_str} |"
            )
            Q_sim[s, a]   = q_after
            cnt_sim[s, a] = n_count
        else:
            lines.append(
                f"| {t + 1} | {s} ({row},{col}) | {_ARROW[a]} "
                f"| {G_t:.4f} | — | *(非首次，跳過)* | — | |"
            )

    lines.append("")
    return "\n".join(lines)


# ── 進度里程碑日誌 ────────────────────────────────────────────────────────────────

def _log_milestone(episode, window, avg_return, epsilon, Q):
    """產生每 N 集的進度摘要。"""
    lines = [
        f"### 第 {episode:,} 集進度",
        "",
        f"**最近 {window:,} 集平均成功率：{avg_return:.1%}　｜　ε = {epsilon:.4f}**",
        "",
        "**目前最佳策略（argmax Q）：**",
        "",
        _policy_table(Q),
        "",
        "**目前最大 Q 值（max Q per state）：**",
        "",
        _q_best_table(Q),
        "",
    ]
    return "\n".join(lines)


# ── 主程式 ────────────────────────────────────────────────────────────────────

def train(config: dict) -> MCAgent:
    env = gym.make(config["env_id"])

    agent = MCAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        gamma=config["gamma"],
        epsilon=config["epsilon"],
    )

    sections = []

    # ── 標頭 ─────────────────────────────────────────────────────────────────
    sections.append("# 訓練日誌 (Training Log)\n")
    sections.append(
        f"> 環境：`{config['env_id']}`  |  "
        f"γ = `{config['gamma']}`  |  "
        f"ε：`{config['epsilon']}` → `{config['epsilon_end']}`（每集 ×`{config['epsilon_decay']}`）  |  "
        f"集數：`{config['n_episodes']:,}`  |  "
        f"產生時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    sections.append("---\n")

    # ── 說明 ─────────────────────────────────────────────────────────────────
    sections.append("## 一、說明\n")

    eps_at_10k    = config["epsilon"] * (config["epsilon_decay"] ** 10000)
    eps_at_50k    = config["epsilon_end"]
    gamma         = config["gamma"]
    decay         = config["epsilon_decay"]
    eps_init      = config["epsilon"]
    eps_end       = config["epsilon_end"]
    log_freq      = config["log_freq"]

    explanation = "\n".join([
        "### 獎勵機制",
        "",
        "| 事件 | 獎勵 r |",
        "|:---|:---:|",
        "| 普通移動（F 格之間） | 0 |",
        "| 成功抵達終點 G | **+1** |",
        "| 掉入洞口 H（遊戲結束） | 0 |",
        "",
        "**冰面滑動**：選擇一個方向後，實際移動是隨機的——",
        "1/3 機率往選擇的方向、1/3 往左轉 90°、1/3 往右轉 90°。",
        "",
        "---",
        "",
        "### MC 核心公式",
        "",
        "**第一步：走完整一集，倒算每步的回報 G**",
        "",
        "```",
        "G_t = r_t + γ × G_{t+1}",
        "",
        f"  r_t    ：步驟 t 拿到的即時獎勵（只有走進終點才得 1 分）",
        f"  γ={gamma} ：折扣因子（每往後一步，未來獎勵打 {int((1-gamma)*100)}% 折）",
        f"  G_{{t+1}}：下一步起的累計折扣回報（從最後一步 G=0 往前推算）",
        "```",
        "",
        "**第二步：首次存取原則 + 增量平均更新 Q(s, a)**",
        "",
        "```",
        "同一集數內，(格子 s, 動作 a) 這個組合若第一次出現，才用此步的 G 更新",
        "（同一集數中該組合若再次出現，略過——避免重複計算同一段路徑）",
        "",
        "N(s,a) ← N(s,a) + 1",
        "Q(s,a) ← Q(s,a) + (1/N) × (G - Q(s,a))",
        "",
        "  等價於：Q(s,a) = 歷次首次存取所得 G 值的平均",
        "  N 越大 → 平均越穩定 → Q 估計越準確",
        "```",
        "",
        "**第三步：ε-greedy 探索**",
        "",
        "```",
        f"  機率 ε    → 隨機選動作（探索未知）",
        f"  機率 1-ε  → 選 Q 最高的動作（利用已學知識）",
        f"  每集後 ε × {decay}，逐漸減少隨機性",
        "```",
        "",
        "| 集數段 | ε 約值 | 意義 |",
        "|:---:|:---:|:---|",
        f"| 第 1 集 | {eps_init:.2f} | 幾乎全部隨機探索 |",
        f"| 第 10,000 集 | {eps_at_10k:.2f} | {eps_at_10k*100:.0f}% 隨機、{(1-eps_at_10k)*100:.0f}% 利用 |",
        f"| 第 50,000 集 | {eps_at_50k:.2f} | {eps_at_50k*100:.0f}% 隨機、{(1-eps_at_50k)*100:.0f}% 利用 |",
        "",
        f"> 日誌顯示：第一筆成功集數（含完整計算過程，G 值非零）、之後每 {log_freq:,} 集（觀察收斂進度）、最終結果。",
        "",
        "---",
    ])
    sections.append(explanation + "\n")

    # ── 訓練過程 ──────────────────────────────────────────────────────────────
    sections.append("## 二、訓練過程\n")

    returns_history = []
    episode_entries = []
    first_success_logged = False  # 第一筆成功是否已記錄

    for episode in range(1, config["n_episodes"] + 1):
        obs, _ = env.reset()
        done      = False
        ep_return = 0.0

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.store_transition(obs, action, reward)
            ep_return += reward
            obs  = next_obs
            done = terminated or truncated

        # 在 update() 清空 _episode 前先複製（只針對第一次成功）
        is_success = ep_return > 0
        if is_success and not first_success_logged:
            traj_snap  = list(agent._episode)
            Q_snap     = agent.Q.copy()
            count_snap = agent.returns_count.copy()

        agent.update()
        returns_history.append(ep_return)

        # ε 衰減
        agent.epsilon = max(config["epsilon_end"],
                            agent.epsilon * config["epsilon_decay"])

        # 第一次成功：完整計算過程
        if is_success and not first_success_logged:
            first_success_logged = True
            episode_entries.append(
                _log_episode_detail(
                    episode, traj_snap, Q_snap, count_snap, config["gamma"]
                )
            )
            print(f"Episode {episode:6d}  ← 第一次成功！已記錄完整過程")

        # 里程碑摘要
        if episode % config["log_freq"] == 0:
            avg = float(np.mean(returns_history[-config["log_freq"]:]))
            episode_entries.append(
                _log_milestone(episode, config["log_freq"], avg,
                               agent.epsilon, agent.Q)
            )
            print(
                f"Episode {episode:6d}/{config['n_episodes']}  "
                f"Avg Return: {avg:.3f}  Eps: {agent.epsilon:.4f}"
            )

    for e in episode_entries:
        sections.append(e + "\n")

    # ── 最終結果 ──────────────────────────────────────────────────────────────
    sections.append("---\n")
    sections.append("## 三、最終結果\n")

    final_avg = float(np.mean(returns_history[-config["log_freq"]:]))
    sections.append("\n".join([
        f"**訓練完成！最後 {config['log_freq']:,} 集平均成功率：{final_avg:.1%}**",
        "",
        "**最終最佳策略：**",
        "",
        _policy_table(agent.Q),
        "",
        "**最終最大 Q 值表（max Q per state）：**",
        "",
        _q_best_table(agent.Q),
        "",
        "> Q 值代表：站在這個格子，按最佳動作繼續走，預期能拿到多少折扣後的總獎勵。",
        "> 越靠近終點 G 的格子，Q 值越高；洞口和終點本身不顯示（固定）。",
    ]) + "\n")

    # ── 寫入日誌 ─────────────────────────────────────────────────────────────
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_log.md")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(sections))
    print(f"\n日誌已儲存：{log_path}")

    env.close()
    return agent


if __name__ == "__main__":
    config = {
        "env_id":        "FrozenLake-v1",
        "n_episodes":    50_000,
        "gamma":         0.99,
        "epsilon":       1.0,
        "epsilon_end":   0.01,
        "epsilon_decay": 0.9999,
        "log_freq":      5000,
    }
    train(config)

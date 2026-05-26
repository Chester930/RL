"""
在 FrozenLake 環境訓練 Q-Learning 代理人，並在同目錄產生 training_log.md。

執行方式：
    python train.py
"""

import os
import random
import numpy as np
# pyrefly: ignore [missing-import]
import gymnasium as gym
from datetime import datetime

from agent import QLearningAgent

# ── 常數 ──────────────────────────────────────────────────────────────────────
#  0  1  2  3
#  4  5  6  7
#  8  9 10 11
# 12 13 14 15
_HOLES = {5, 7, 11, 12}
_GOAL  = 15
_ARROW = {0: '←', 1: '↓', 2: '→', 3: '↑'}


def _state_info(s):
    label = "G" if s == _GOAL else ("H" if s in _HOLES else "F")
    return s // 4, s % 4, label


# ── Markdown 輔助 ──────────────────────────────────────────────────────────────

def _q_best_table(Q):
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


# ── 詳細集數日誌 ────────────────────────────────────────────────────────────────

def _log_episode_detail(ep_num, steps, alpha, gamma):
    """
    steps: list of (s, a, r, s_next, delta, q_sa, max_q_snext)
      q_sa         : Q[s, a]  更新前
      max_q_snext  : max Q[s', :] 更新前（終止時為 0）
    """
    lines = []
    lines.append(f"### 第 {ep_num} 集（完整計算過程）")
    lines.append("")

    success = any(r > 0 for _, _, r, *_ in steps)
    result_label = "成功抵達終點 G" if success else "失敗（掉入洞口或超時）"
    lines.append(f"**結果：{result_label}　｜　步數：{len(steps)}**")
    lines.append("")

    # ① 行走軌跡
    lines.append("**① 行走軌跡：**")
    lines.append("")
    lines.append("| 步驟 | 格子 | 位置 | 動作 | 獎勵 | 備註 |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|:---|")
    for t, (s, a, r, s_next, delta, q_sa, max_q_sn) in enumerate(steps):
        row, col, label = _state_info(s)
        note = ""
        if t == 0:
            note = "起點 S"
        if label == "H":
            note = "掉入洞口 H"
        if r > 0:
            note = "**到達終點 G！**"
        lines.append(f"| {t+1} | {s} | ({row},{col}) | {_ARROW[a]} | {int(r)} | {note} |")
    lines.append("")

    # ② 每步即時更新
    lines.append("**② 每步即時更新（異策略 TD 更新）：**")
    lines.append("")
    lines.append(
        "> 公式：δ = r + γ × **max** Q(s′) − Q(s,a)　　"
        "目標永遠用下一格的「最大 Q 值」，與實際走哪一步無關——這就是「異策略」。"
    )
    lines.append("")
    lines.append(
        "| 步驟 | 格子 | 動作 | r | 下一格 | max Q(s′) | Q(s,a) | δ = TD誤差 | ΔQ(s,a) |"
    )
    lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    n = len(steps)
    for t, (s, a, r, s_next, delta, q_sa, max_q_sn) in enumerate(steps):
        row, col, _ = _state_info(s)
        rn, cn, _ = _state_info(s_next)
        is_done = (t == n - 1)
        max_q_str = "0.0000（終止）" if is_done else f"{max_q_sn:.4f}"
        dq = alpha * delta
        lines.append(
            f"| {t+1} | {s}({row},{col}) | {_ARROW[a]} | {int(r)} "
            f"| {s_next}({rn},{cn}) | {max_q_str} "
            f"| {q_sa:.4f} | {delta:+.4f} | {dq:+.4f} |"
        )
    lines.append("")

    # ③ 關鍵步驟說明
    final_delta = steps[-1][4]
    final_s, final_a = steps[-1][0], steps[-1][1]
    final_q_sa = steps[-1][5]
    final_row, final_col, _ = _state_info(final_s)

    if success:
        lines.append("**③ 本集更新重點：**")
        lines.append("")
        lines.append(
            f"> Q-Learning 每步只更新一個 (格子, 動作) 的 Q 值——沒有跡，沒有回傳。"
        )
        lines.append(
            f"> 本集唯一有效更新：步驟 {n}，格子 {final_s}({final_row},{final_col}) 動作 {_ARROW[final_a]}"
        )
        lines.append(
            f"> δ = 1 − {final_q_sa:.4f} = **{final_delta:+.4f}**，"
            f"Q({final_s}, {_ARROW[final_a]}) 從 {final_q_sa:.4f} → {final_q_sa + alpha*final_delta:.4f}"
        )
        lines.append("")
        lines.append(
            "> 後續集數中，當代理人走到格子 "
            f"{final_s}({final_row},{final_col}) 的相鄰格子時，"
            "該格子的 TD 目標就會借用這個更新後的 Q 值，信用逐步往起點傳播。"
        )
    lines.append("")
    return "\n".join(lines)


# ── 進度里程碑 ──────────────────────────────────────────────────────────────────

def _log_milestone(episode, window, avg_return, epsilon, Q):
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

def train(config: dict) -> QLearningAgent:
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    env = gym.make(config["env_id"])

    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=config["alpha"],
        gamma=config["gamma"],
        epsilon=config["epsilon_start"],
    )

    sections      = []
    ep_returns    = []
    episode_entries = []
    first_success_logged = False

    # ── 標頭 ─────────────────────────────────────────────────────────────────
    sections.append("# 訓練日誌 (Training Log)\n")
    sections.append(
        f"> 環境：`{config['env_id']}`  |  "
        f"γ = `{config['gamma']}`  |  "
        f"α = `{config['alpha']}`  |  "
        f"ε：`{config['epsilon_start']}` → `{config['epsilon_end']}`（每集 ×`{config['epsilon_decay']}`）  |  "
        f"集數：`{config['n_episodes']:,}`  |  "
        f"產生時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    sections.append("---\n")

    # ── 一、說明 ─────────────────────────────────────────────────────────────
    gamma    = config["gamma"]
    alpha    = config["alpha"]
    eps_init = config["epsilon_start"]
    eps_end  = config["epsilon_end"]
    decay    = config["epsilon_decay"]
    log_freq = config["log_freq"]
    eps_at_10k = eps_init * (decay ** 10_000)

    explanation = "\n".join([
        "## 一、說明",
        "",
        "",
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
        "### Q-Learning 核心公式",
        "",
        "**每走一步立刻更新（異策略）：**",
        "",
        "```",
        "① 計算異策略 TD 目標：",
        f"   目標 = r + γ × max_{{a′}} Q(s′, a′)   （集數未結束）",
        f"   目標 = r                                （集數結束，無下一步）",
        "",
        f"   r                   ：即時獎勵",
        f"   γ = {gamma}         ：折扣因子",
        f"   max Q(s′, a′)       ：下一格所有動作中最大的 Q 值（不管實際走哪步）",
        "",
        "② 計算 TD 誤差 δ：",
        "   δ = 目標 − Q(s, a)",
        "",
        "③ 更新 Q(s, a)：",
        f"   Q(s, a) ← Q(s, a) + α × δ     （α = {alpha}）",
        "```",
        "",
        "**異策略的關鍵**：TD 目標用的是 `max Q(s′)`，",
        "代表「如果下一步做出最優選擇能得多少」——與行為策略實際走哪一步完全無關。",
        "因此 Q-Learning 直接學習最優策略 Q*，探索行為不會汙染學到的 Q 值。",
        "",
        "---",
        "",
        "### ε-greedy 探索",
        "",
        "```",
        "  機率 ε    → 隨機選動作（探索未知）",
        "  機率 1-ε  → 選 Q 最高的動作（利用已學知識）",
        f"  每集後 ε × {decay}，逐漸減少隨機性",
        "```",
        "",
        "| 集數段 | ε 約值 | 意義 |",
        "|:---:|:---:|:---|",
        f"| 第 1 集 | {eps_init:.2f} | 幾乎全部隨機探索 |",
        f"| 第 10,000 集 | {eps_at_10k:.2f} | {eps_at_10k*100:.0f}% 隨機、{(1-eps_at_10k)*100:.0f}% 利用 |",
        f"| 第 20,000 集 | {eps_end:.2f} | {eps_end*100:.0f}% 隨機、{(1-eps_end)*100:.0f}% 利用 |",
        "",
        f"> 日誌顯示：第一筆成功集數（含完整計算過程）、之後每 {log_freq:,} 集（觀察收斂進度）、最終結果。",
        "",
        "---",
    ])
    sections.append(explanation + "\n")

    # ── 二、訓練過程 ──────────────────────────────────────────────────────────
    sections.append("## 二、訓練過程\n")

    for episode in range(1, config["n_episodes"] + 1):
        obs, _ = env.reset()
        ep_return = 0.0
        done = False

        capture = not first_success_logged
        ep_steps = [] if capture else None

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if capture:
                q_sa       = float(agent.Q[obs, action])
                max_q_sn   = float(agent.Q[next_obs].max()) if not done else 0.0

            metrics = agent.update(obs, action, reward, next_obs, done)

            if capture:
                ep_steps.append((
                    obs, action, reward, next_obs,
                    metrics["td_error"], q_sa, max_q_sn,
                ))

            ep_return += reward
            obs = next_obs

        ep_returns.append(ep_return)

        if capture and ep_return > 0 and not first_success_logged:
            first_success_logged = True
            episode_entries.append(
                _log_episode_detail(episode, ep_steps, alpha, gamma)
            )
            print(f"Episode {episode:6d}  ← 第一次成功！已記錄完整過程")

        # 指數 ε 衰減（與 MC、TD(λ) 一致）
        agent.epsilon = max(config["epsilon_end"],
                            agent.epsilon * config["epsilon_decay"])

        if episode % config["log_freq"] == 0:
            avg = float(np.mean(ep_returns[-config["log_freq"]:]))
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

    # ── 三、最終結果 ──────────────────────────────────────────────────────────
    sections.append("---\n")
    sections.append("## 三、最終結果\n")

    final_avg = float(np.mean(ep_returns[-config["log_freq"]:]))
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
        "n_episodes":    20_000,
        "alpha":         0.1,
        "gamma":         0.99,
        "epsilon_start": 1.0,
        "epsilon_end":   0.01,
        "epsilon_decay": 0.9998,
        "log_freq":      5_000,
        "seed":          42,
    }
    train(config)

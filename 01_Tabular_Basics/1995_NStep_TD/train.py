"""
在 FrozenLake 上訓練 N-步時序差分代理人，並在同目錄產生 training_log.md。

訓練迴圈明確管理 n-步緩衝區：
- 每一步：執行 store(s, a, r)，若緩衝區滿了則更新。
- 集數結束時：排乾 (Flush) 緩衝區中剩餘的專案。

執行方式：
    python train.py
"""

import os
import numpy as np
# pyrefly: ignore [missing-import]
import gymnasium as gym
from datetime import datetime

from agent import NStepTDAgent

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

def _log_episode_detail(ep_num, n, ep_steps, ep_updates, alpha, gamma):
    """
    ep_steps:   list of (s, a, r, s_next, a_next)
    ep_updates: list of dicts with keys:
                trigger_1idx (int | "Flush"), target_1idx, s_t, a_t,
                G, q_before, td_error, is_flush
    """
    lines = []
    lines.append(f"### 第 {ep_num} 集（完整計算過程）")
    lines.append("")

    success = any(r > 0 for _, _, r, *_ in ep_steps)
    result_label = "成功抵達終點 G" if success else "失敗（掉入洞口或超時）"
    lines.append(f"**結果：{result_label}　｜　步數：{len(ep_steps)}　｜　n = {n}**")
    lines.append("")

    # ① 行走軌跡
    lines.append("**① 行走軌跡：**")
    lines.append("")
    lines.append("| 步驟 | 格子 | 位置 | 動作 | 獎勵 | 備註 |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|:---|")
    for t, (s, a, r, s_next, a_next) in enumerate(ep_steps):
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

    # ② n-步緩衝區更新
    lines.append(f"**② {n}-步緩衝區更新：**")
    lines.append("")
    lines.append(
        f"> 公式：G_{{t:t+{n}}} = r_{{t+1}} + γ×r_{{t+2}} + ... + "
        f"γ^{n-1}×r_{{t+{n}}} + γ^{n}×Q(s_{{t+{n}}}, a_{{t+{n}}})"
    )
    lines.append(
        f"> 緩衝區累積 {n} 個轉移後觸發正常更新（滑動視窗）；"
        f"集數結束時 Flush 剩餘 {n-1} 個轉移（不對終止狀態引導）。"
    )
    lines.append("")
    lines.append(
        "| 更新 | 觸發 | 目標步驟 | 目標 (s, a) | n-步回報 G | Q(s,a)前 | δ = G−Q | ΔQ | 備註 |"
    )
    lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|")

    for i, upd in enumerate(ep_updates):
        trigger_str = (f"步驟 {upd['trigger_1idx']}"
                       if not upd['is_flush'] else "Flush")
        target_str  = f"步驟 {upd['target_1idx']}"
        s_t, a_t    = upd['s_t'], upd['a_t']
        r_s, c_s, _ = _state_info(s_t)
        note        = "Flush（集數結束）" if upd['is_flush'] else ""
        if upd['G'] > 1e-6:
            note = ("  **非零回報！**" if not note else note + "  **非零回報！**")
        dq = alpha * upd['td_error']
        lines.append(
            f"| {i+1} | {trigger_str} | {target_str} "
            f"| {s_t}({r_s},{c_s}) {_ARROW[a_t]} "
            f"| {upd['G']:+.4f} | {upd['q_before']:.4f} "
            f"| {upd['td_error']:+.4f} | {dq:+.4f} | {note} |"
        )
    lines.append("")

    # ③ 本集更新重點
    nonzero = [u for u in ep_updates if u['G'] > 1e-6]
    flush_updates = [u for u in ep_updates if u['is_flush']]
    lines.append("**③ 本集更新重點：**")
    lines.append("")
    if success:
        lines.append(
            f"> 本集共 {len(ep_updates)} 次更新（{len(ep_updates) - len(flush_updates)} 次正常 + "
            f"{len(flush_updates)} 次 Flush）。"
        )
        if nonzero:
            first_nz = nonzero[0]
            r_nz, c_nz, _ = _state_info(first_nz['s_t'])
            lines.append(
                f"> 大部分更新 G ≈ 0（沿途無獎勵）；最後 {len(nonzero)} 次更新有非零回報："
                f"成功獎勵 r=1 往前傳播 {n} 步。"
            )
            lines.append("")
            lines.append(
                f"> **N-步 TD 的優勢**：n={n} 讓獎勵訊號一次向前傳遞 {n} 步。"
                f"對比 TD(0)（n=1）只能傳遞 1 步，N-步 TD 讓前面 {n} 個 (格子, 動作)"
                f"在同一集數中就能學到有意義的 Q 值。"
            )
    lines.append("")

    return "\n".join(lines)


# ── 進度里程碑 ──────────────────────────────────────────────────────────────────

def _log_milestone(episode, window, avg_return, epsilon, Q, n):
    lines = [
        f"### 第 {episode:,} 集進度",
        "",
        f"**最近 {window:,} 集平均成功率：{avg_return:.1%}　｜　ε = {epsilon:.4f}　｜　n = {n}**",
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

def train(config: dict) -> NStepTDAgent:
    env = gym.make(config["env_id"])
    n   = config["n_steps"]

    agent = NStepTDAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        n=n,
        alpha=config["alpha"],
        gamma=config["gamma"],
        epsilon=config["epsilon_start"],
    )

    sections             = []
    ep_returns           = []
    episode_entries      = []
    first_success_logged = False

    # ── 標頭 ─────────────────────────────────────────────────────────────────
    sections.append("# 訓練日誌 (Training Log)\n")
    sections.append(
        f"> 環境：`{config['env_id']}`  |  "
        f"γ = `{config['gamma']}`  |  "
        f"α = `{config['alpha']}`  |  "
        f"n = `{n}`  |  "
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
        f"### {n}-步 TD 核心公式",
        "",
        f"**n = {n}：累積 {n} 步獎勵後進行引導 (Bootstrap)，平衡偏差與變異數。**",
        "",
        "```",
        f"① 每步：將 (s_t, a_t, r_{{t+1}}) 存入滑動緩衝區",
        "",
        f"② 當緩衝區滿 {n} 個轉移時，計算 n-步回報並更新最舊的 (s_t, a_t)：",
        f"   G_{{t:t+{n}}} = r_{{t+1}} + γ×r_{{t+2}} + ... + γ^{n-1}×r_{{t+{n}}}",
        f"             + γ^{n} × Q(s_{{t+{n}}}, a_{{t+{n}}})   （未結束時引導）",
        "",
        "   δ = G_{t:t+n} − Q(s_t, a_t)",
        f"   Q(s_t, a_t) ← Q(s_t, a_t) + α × δ     （α = {alpha}）",
        "",
        f"③ 集數結束時，Flush 緩衝區剩餘 {n-1} 個轉移（不引導終止狀態）：",
        "   G = 剩餘獎勵的折扣加總（無 Bootstrap 項）",
        "```",
        "",
        "| n 值 | 型別 | 偏差 (Bias) | 變異數 (Variance) | 說明 |",
        "|:---:|:---:|:---:|:---:|:---|",
        "| **1** | TD(0) | 高 | 低 | 立即引導，僅考慮下一步 |",
        f"| **{n}** | N-步 TD | 中 | 中 | **本次訓練**，前看 {n} 步 |",
        "| **∞** | 蒙特卡羅 | 無 | 高 | 使用完整集數回報，不引導 |",
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
        agent._buffer.clear()
        action    = agent.select_action(obs)
        ep_return = 0.0
        done      = False

        capture    = not first_success_logged
        ep_steps   = [] if capture else None
        ep_updates = [] if capture else None

        while not done:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward

            next_action = agent.select_action(next_obs) if not done else 0

            if capture:
                ep_steps.append((obs, action, reward, next_obs, next_action))

            agent.store(obs, action, reward)

            if agent.is_ready():
                if capture:
                    buf_snap = list(agent._buffer)
                    s_t, a_t, _ = buf_snap[0]
                    q_before     = float(agent.Q[s_t, a_t])
                    G_val = sum(agent.gamma ** k * r for k, (_, _, r) in enumerate(buf_snap))
                    G_val += (agent.gamma ** len(buf_snap)) * agent.Q[next_obs, next_action]

                metrics = agent.update(next_obs, next_action, done=False)

                if capture:
                    ep_updates.append({
                        "trigger_1idx": len(ep_steps),
                        "target_1idx":  len(ep_steps) - agent.n + 1,
                        "s_t":          s_t,
                        "a_t":          a_t,
                        "G":            G_val,
                        "q_before":     q_before,
                        "td_error":     metrics.get("td_error", 0.0),
                        "is_flush":     False,
                    })

            obs    = next_obs
            action = next_action

        # Flush：捕捉剩餘緩衝區，然後交給 agent 執行
        if capture:
            remaining    = list(agent._buffer)
            flush_q_pre  = [float(agent.Q[s, a]) for s, a, _ in remaining]
            flush_G      = [
                sum(agent.gamma ** k * r for k, (_, _, r) in enumerate(remaining[i:]))
                for i in range(len(remaining))
            ]

        agent.update(obs, 0, done=True, flush=True)

        if capture:
            base = len(ep_steps) - len(remaining)
            for i, (s, a, _) in enumerate(remaining):
                ep_updates.append({
                    "trigger_1idx": "Flush",
                    "target_1idx":  base + i + 1,
                    "s_t":          s,
                    "a_t":          a,
                    "G":            flush_G[i],
                    "q_before":     flush_q_pre[i],
                    "td_error":     flush_G[i] - flush_q_pre[i],
                    "is_flush":     True,
                })

        ep_returns.append(ep_return)

        if capture and ep_return > 0 and not first_success_logged:
            first_success_logged = True
            episode_entries.append(
                _log_episode_detail(episode, n, ep_steps, ep_updates, alpha, gamma)
            )
            print(f"Episode {episode:6d}  ← 第一次成功！已記錄完整過程")

        agent.epsilon = max(
            config["epsilon_end"],
            agent.epsilon * config["epsilon_decay"],
        )

        if episode % config["log_freq"] == 0:
            avg = float(np.mean(ep_returns[-config["log_freq"]:]))
            episode_entries.append(
                _log_milestone(episode, config["log_freq"], avg,
                               agent.epsilon, agent.Q, n)
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
        f"> Q 值代表：站在這個格子，按當前 ε-greedy 策略繼續走，預期能拿到多少折扣後的總獎勵。",
        f"> N-步 TD (n={n}) 使用前瞻 {n} 步的回報進行更新，在偏差與變異數之間取得平衡。",
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
        "n_steps":       4,
        "alpha":         0.1,
        "gamma":         0.99,
        "epsilon_start": 1.0,
        "epsilon_end":   0.01,
        "epsilon_decay": 0.9998,
        "log_freq":      5_000,
    }
    train(config)

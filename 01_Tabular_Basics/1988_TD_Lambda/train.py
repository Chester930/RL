"""
訓練 TD(lambda) 代理人，比較不同 lambda 值，並產生 training_log.md。

執行方式：
    python train.py
"""

import os
import numpy as np
# pyrefly: ignore [missing-import]
import gymnasium as gym
from datetime import datetime

from agent import TDLambdaAgent

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


# ── 重建跡矩陣 ──────────────────────────────────────────────────────────────────

def _reconstruct_final_traces(steps, gamma, lam, trace_type, n_states, n_actions):
    """
    從軌跡重建最終步的跡矩陣 E（在 ③ 更新後、衰減/重置前）。
    回傳 list of (s, a, e_val, steps_from_end)，只含 E > 1e-6 的對。
    """
    E = np.zeros((n_states, n_actions))
    n = len(steps)

    for t, (s, a, *_) in enumerate(steps):
        if trace_type == "replace":
            E[s, :] = 0.0
            E[s, a] = 1.0
        else:
            E[s, a] += 1.0

        if t < n - 1:
            E *= gamma * lam
        # 最後一步不衰減，保留給呈現

    active = []
    for s_idx in range(n_states):
        for a_idx in range(n_actions):
            if E[s_idx, a_idx] > 1e-6:
                # 找最後一次訪問這個 (s,a) 的步驟
                last_t = max(
                    (t for t, (s2, a2, *_) in enumerate(steps) if s2 == s_idx and a2 == a_idx),
                    default=-1,
                )
                dist = (n - 1) - last_t
                active.append((s_idx, a_idx, E[s_idx, a_idx], dist))

    active.sort(key=lambda x: -x[2])   # 距終點最遠排最前（最早訪問）
    return active


# ── 詳細集數日誌 ────────────────────────────────────────────────────────────────

def _log_episode_detail_td(ep_num, steps, alpha, gamma, lam, trace_type, n_states, n_actions):
    """
    steps: list of (s, a, r, s_next, a_next, delta, q_sa, q_snext_anext)
    """
    lines = []
    lines.append(f"### 第 {ep_num} 集（完整計算過程，λ = {lam}）")
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
    for t, (s, a, r, s_next, a_next, delta, q_sa, q_sn) in enumerate(steps):
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
    lines.append("**② 每步即時更新（TD 誤差 × 資格跡）：**")
    lines.append("")
    lines.append("> 每走一步立刻更新——不等集數結束。公式：δ = r + γ×Q(s′,a′) − Q(s,a)")
    lines.append("")

    n = len(steps)
    lines.append("| 步驟 | 格子 | 動作 | r | 下一格 | 下一動作 | Q(s,a) | Q(s′,a′) | δ = TD誤差 | E(s,a)↑ | ΔQ(s,a) |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for t, (s, a, r, s_next, a_next, delta, q_sa, q_sn) in enumerate(steps):
        row, col, _ = _state_info(s)
        rn, cn, _ = _state_info(s_next)
        is_done = (t == n - 1)
        next_a_str = "—" if is_done else _ARROW[a_next]
        e_sa = 1.0   # replace trace 永遠設為 1
        dq = alpha * delta * e_sa
        lines.append(
            f"| {t+1} | {s}({row},{col}) | {_ARROW[a]} | {int(r)} "
            f"| {s_next}({rn},{cn}) | {next_a_str} "
            f"| {q_sa:.4f} | {q_sn:.4f} "
            f"| {delta:+.4f} | {e_sa:.1f} | {dq:+.4f} |"
        )
    lines.append("")

    # ③ 最終步跡傳播
    final_delta = steps[-1][5]
    active = _reconstruct_final_traces(steps, gamma, lam, trace_type, n_states, n_actions)

    if success and active:
        lines.append(
            f"**③ 最終步 δ 回傳跡更新**（δ = {final_delta:+.4f}，"
            f"正值 → 整條路徑都被強化）**：**"
        )
        lines.append("")
        lines.append(
            f"> 公式：ΔQ(s,a) = α × δ × E(s,a) = {alpha} × {final_delta:.4f} × E"
        )
        lines.append(
            f"> E 值越大代表訪問越近；λ = {lam} 讓跡每步衰減 γλ = {gamma*lam:.3f}，"
            f"遠處步驟仍獲得一定信用。"
        )
        lines.append("")
        lines.append("| 格子 | 位置 | 動作 | 距終步（步） | E 值 | ΔQ = α×δ×E |")
        lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|")
        for s_idx, a_idx, e_val, dist in active:
            row, col, _ = _state_info(s_idx)
            dq = alpha * final_delta * e_val
            lines.append(
                f"| {s_idx} | ({row},{col}) | {_ARROW[a_idx]} "
                f"| {dist} | {e_val:.4f} | {dq:+.4f} |"
            )
        lines.append("")
        lines.append(
            f"> **與 MC 的關鍵差異**：MC 要等集數結束才倒算 G；"
            f"TD(λ) 在終點步驟拿到 δ 後立刻透過跡把信用分配給所有走過的格子。"
        )
    lines.append("")
    return "\n".join(lines)


# ── 進度里程碑（多 λ 對比）────────────────────────────────────────────────────────

def _log_milestone_comparison(episode, window, results_per_lam, Q_per_lam, lambdas):
    """
    results_per_lam : {lam: (avg_return, epsilon)}
    Q_per_lam       : {lam: Q_table}
    """
    _METHOD = {0.0: "TD(0)", 0.5: "中間地帶", 0.9: "接近 MC", 1.0: "MC 等效"}

    lines = [f"### 第 {episode:,} 集進度", ""]

    lines.append(f"**各 λ 值最近 {window:,} 集平均成功率：**")
    lines.append("")
    lines.append("| λ | 對應方法 | 成功率 | ε |")
    lines.append("|:---:|:---:|:---:|:---:|")
    for lam in lambdas:
        avg, eps = results_per_lam[lam]
        lines.append(f"| {lam} | {_METHOD.get(lam, str(lam))} | {avg:.1%} | {eps:.4f} |")
    lines.append("")

    best_lam = max(lambdas, key=lambda l: results_per_lam[l][0])
    Q_best = Q_per_lam[best_lam]

    lines.append(f"**λ = {best_lam} 目前最佳策略（argmax Q）：**")
    lines.append("")
    lines.append(_policy_table(Q_best))
    lines.append("")
    lines.append(f"**λ = {best_lam} 目前最大 Q 值（max Q per state）：**")
    lines.append("")
    lines.append(_q_best_table(Q_best))
    lines.append("")
    return "\n".join(lines)


# ── 主訓練迴圈 ─────────────────────────────────────────────────────────────────

def train_all(config: dict) -> None:
    lambdas    = config["lambdas"]
    detail_lam = config.get("detail_lambda", 0.9)

    log_eps = sorted(range(config["log_freq"], config["n_episodes"] + 1, config["log_freq"]))
    milestone_stats = {ep: {} for ep in log_eps}
    milestone_Q     = {ep: {} for ep in log_eps}
    Q_final         = {}
    final_avg       = {}

    detail_data = None   # (ep_num, steps)

    env_tmp  = gym.make(config["env_id"])
    n_states  = env_tmp.observation_space.n
    n_actions = env_tmp.action_space.n
    env_tmp.close()

    for lam in lambdas:
        print(f"\n--- Lambda = {lam} ---")
        env = gym.make(config["env_id"])

        agent = TDLambdaAgent(
            n_states=n_states,
            n_actions=n_actions,
            alpha=config["alpha"],
            gamma=config["gamma"],
            lam=lam,
            epsilon=config["epsilon_start"],
            trace_type=config.get("trace_type", "replace"),
        )

        ep_returns          = []
        first_success_done  = False

        for episode in range(1, config["n_episodes"] + 1):
            obs, _ = env.reset()
            agent.reset_traces()
            action    = agent.select_action(obs)
            ep_return = 0.0
            done      = False

            capture = (lam == detail_lam and not first_success_done)
            ep_steps = [] if capture else None

            while not done:
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                next_action = agent.select_action(next_obs) if not done else 0

                if capture:
                    q_sa  = float(agent.Q[obs, action])
                    q_sn  = float(agent.Q[next_obs, next_action]) if not done else 0.0

                metrics = agent.update(obs, action, reward, next_obs, next_action, done)

                if capture:
                    ep_steps.append((
                        obs, action, reward, next_obs, next_action,
                        metrics["td_error"], q_sa, q_sn,
                    ))

                ep_return += reward
                obs    = next_obs
                action = next_action

            ep_returns.append(ep_return)

            if capture and ep_return > 0 and not first_success_done:
                first_success_done = True
                detail_data = (episode, ep_steps)
                print(f"  Episode {episode:6d}  ← 第一次成功！已記錄完整過程 (λ={lam})")

            agent.epsilon = max(
                config["epsilon_end"],
                agent.epsilon * config["epsilon_decay"],
            )

            if episode % config["log_freq"] == 0:
                avg = float(np.mean(ep_returns[-config["log_freq"]:]))
                milestone_stats[episode][lam] = (avg, agent.epsilon)
                milestone_Q[episode][lam]     = agent.Q.copy()
                print(
                    f"  [λ={lam}] Episode {episode:6d}  "
                    f"Avg Return: {avg:.3f}  Eps: {agent.epsilon:.4f}"
                )

        Q_final[lam]   = agent.Q.copy()
        final_avg[lam] = float(np.mean(ep_returns[-config["log_freq"]:]))
        env.close()

    # ── 組裝日誌 ──────────────────────────────────────────────────────────────
    sections = []

    # 標頭
    lam_str = " / ".join(str(l) for l in lambdas)
    sections.append("# 訓練日誌 (Training Log)\n")
    sections.append(
        f"> 環境：`{config['env_id']}`  |  "
        f"γ = `{config['gamma']}`  |  "
        f"α = `{config['alpha']}`  |  "
        f"λ：`{lam_str}`  |  "
        f"跡型別：`{config.get('trace_type', 'replace')}`  |  "
        f"ε：`{config['epsilon_start']}` → `{config['epsilon_end']}`"
        f"（×`{config['epsilon_decay']}`）  |  "
        f"集數：`{config['n_episodes']:,}`（每個 λ）  |  "
        f"產生時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    sections.append("---\n")

    # 一、說明
    gamma        = config["gamma"]
    alpha        = config["alpha"]
    eps_init     = config["epsilon_start"]
    eps_end      = config["epsilon_end"]
    decay        = config["epsilon_decay"]
    eps_at_10k   = eps_init * (decay ** 10_000)
    log_freq     = config["log_freq"]

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
        "### TD(λ) 核心公式",
        "",
        "**每走一步立刻更新（三個動作，迴圈執行）：**",
        "",
        "```",
        "① 計算 TD 誤差 δ：",
        f"   δ = r + γ × Q(s′, a′) − Q(s, a)    （集數未結束）",
        f"   δ = r − Q(s, a)                      （集數結束，無下一步）",
        "",
        f"   r      ：即時獎勵（只有走進終點才得 1 分）",
        f"   γ={gamma}：折扣因子",
        "   Q(s,a)  ：目前格子-動作的估計價值",
        "   Q(s′,a′)：下一格-下一動作的估計價值（同策略，SARSA 目標）",
        "",
        "② 更新資格跡 E(s,a)（替換式跡）：",
        "   E(s, :) ← 0     ← 清除 s 所有動作的跡",
        "   E(s, a) ← 1     ← 重置剛訪問的 (s,a) 跡為 1",
        "",
        "③ 用 δ × E 更新所有帶跡的格子，然後衰減跡：",
        f"   Q(s,a) ← Q(s,a) + α × δ × E(s,a)   （對所有 s,a，α={alpha}）",
        f"   E(s,a) ← γ × λ × E(s,a)             （對所有 s,a 衰減）",
        "```",
        "",
        f"| λ 值 | 對應方法 | 含義 |",
        "|:---:|:---:|:---|",
        f"| 0.0 | TD(0) | 每步衰減為 0，僅更新當前 (s,a) |",
        f"| 0.5 | 中間地帶 | 每步衰減 {gamma*0.5:.3f}，近期步驟獲更多信用 |",
        f"| 0.9 | 接近 MC | 每步衰減 {gamma*0.9:.3f}，絕大部分歷史都被更新 |",
        f"| 1.0 | MC 等效 | 每步衰減 {gamma*1.0:.3f}，整條軌跡等比例更新 |",
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
        f"> 日誌顯示：λ={detail_lam} 第一筆成功集（含完整計算過程）、之後每 {log_freq:,} 集四個 λ 對比、最終結果。",
        "",
        "---",
    ])
    sections.append(explanation + "\n")

    # 二、訓練過程
    sections.append("## 二、訓練過程\n")
    entries = []

    if detail_data:
        ep_num, ep_steps = detail_data
        entries.append(
            _log_episode_detail_td(
                ep_num, ep_steps, alpha, gamma, detail_lam,
                config.get("trace_type", "replace"), n_states, n_actions,
            )
        )

    for episode in log_eps:
        data = milestone_stats[episode]
        Q_snap = milestone_Q[episode]
        if data:
            entries.append(
                _log_milestone_comparison(episode, log_freq, data, Q_snap, lambdas)
            )

    for e in entries:
        sections.append(e + "\n")

    # 三、最終結果
    sections.append("---\n")
    sections.append("## 三、最終結果\n")

    _METHOD = {0.0: "TD(0)", 0.5: "中間地帶", 0.9: "接近 MC", 1.0: "MC 等效"}
    _TRAIT  = {
        0.0: "偏差高、變異數低；僅更新當前步驟",
        0.5: "折衷；適中的偏差與變異數",
        0.9: "低偏差；適合稀疏獎勵環境",
        1.0: "無偏差但高變異數；需等集數結束",
    }

    final_lines = [
        f"**訓練完成！各 λ 值最後 {log_freq:,} 集平均成功率：**",
        "",
        "| λ | 對應方法 | 成功率 | 特點 |",
        "|:---:|:---:|:---:|:---|",
    ]
    for lam in lambdas:
        final_lines.append(
            f"| {lam} | {_METHOD.get(lam, str(lam))} "
            f"| {final_avg[lam]:.1%} | {_TRAIT.get(lam, '')} |"
        )

    best_lam = max(lambdas, key=lambda l: final_avg[l])
    Q_best   = Q_final[best_lam]

    final_lines += [
        "",
        f"**最佳 λ = {best_lam} 最終策略：**",
        "",
        _policy_table(Q_best),
        "",
        f"**最佳 λ = {best_lam} 最終 Q 值表（max Q per state）：**",
        "",
        _q_best_table(Q_best),
        "",
        "> Q 值代表：站在這個格子，按最佳動作繼續走，預期能拿到多少折扣後的總獎勵。",
        "> 越靠近終點 G 的格子，Q 值越高；洞口和終點本身不顯示（固定）。",
    ]
    sections.append("\n".join(final_lines) + "\n")

    # 寫入日誌
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_log.md")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(sections))
    print(f"\n日誌已儲存：{log_path}")


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
        "trace_type":    "replace",
        "lambdas":       [0.0, 0.5, 0.9, 1.0],
        "detail_lambda": 0.9,
    }
    train_all(config)

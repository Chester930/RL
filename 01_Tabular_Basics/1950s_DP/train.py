"""
在 FrozenLake (已知模型的 4x4 GridWorld) 上訓練 DP 代理人。

執行方式：
    python train.py

執行後會在同目錄產生 training_log.md，記錄完整的逐步運算過程。
"""

import os
import random
import numpy as np
# pyrefly: ignore [missing-import]
import gymnasium as gym
from datetime import datetime

from agent import DPAgent

# ── 常數 ──────────────────────────────────────────────────────────────────────
# FrozenLake 4x4 預設地圖：洞口與終點的狀態編號
#  0  1  2  3
#  4  5  6  7
#  8  9 10 11
# 12 13 14 15
_HOLES = {5, 7, 11, 12}
_GOAL  = 15
_ARROW = {0: '←', 1: '↓', 2: '→', 3: '↑'}


# ── 詳細計算展開 ───────────────────────────────────────────────────────────────

def _detail_calc(s, P, V_before, n_actions, gamma):
    """
    回傳格子 s 的完整 Q 計算過程（code block 格式）。
    展示每個動作的每條轉移路徑與算式，最後標出 V(s) = max Q。
    """
    row, col = s // 4, s % 4
    v_old = V_before[s]
    lines = []
    lines.append(f"格子 {s} ({row},{col})   V: {v_old:.4f} → ?")
    lines.append("─" * 52)

    q_vals = []
    for a in range(n_actions):
        lines.append(f"動作 {_ARROW[a]} ({a})：")
        q = 0.0
        trans_results = []
        for prob, s_next, r, done in P[s][a]:
            future = 0.0 if done else V_before[s_next]
            contrib = prob * (r + gamma * future)
            q += contrib
            note = ""
            if r > 0:
                note = "  ← 有獎勵!"
            elif done:
                note = "  ← 終止"
            line = (
                f"  格子{s_next:>2} (機率{prob:.3f})："
                f"  {prob:.3f} × (r={r} + {gamma}×{future:.4f})"
                f" = {contrib:.4f}{note}"
            )
            trans_results.append(line)
        lines.extend(trans_results)
        sum_str = " + ".join(f"{prob*(r+gamma*(0.0 if done else V_before[s_next])):.4f}"
                              for prob, s_next, r, done in P[s][a])
        lines.append(f"  Q({_ARROW[a]}) = {sum_str} = {q:.4f}")
        lines.append("")
        q_vals.append(q)

    best_q = max(q_vals)
    best_actions = [_ARROW[a] for a, q in enumerate(q_vals) if abs(q - best_q) < 1e-9]
    q_list = ", ".join(f"{q:.4f}" for q in q_vals)
    lines.append(f"→ V({s}) = max({q_list}) = {best_q:.4f}  "
                 f"（最佳：{'／'.join(best_actions)}）")
    return "```\n" + "\n".join(lines) + "\n```"


# ── Markdown 輔助 ───────────────────────────────────────────────────────────────

def _v_table(V):
    """4×4 V 值 Markdown 表格。"""
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
                cells.append(f"{V[s]:.4f}")
        rows.append(f"| **{row}** | {' | '.join(cells)} |")
    return "\n".join(rows)


def _policy_table(policy):
    """4×4 策略箭頭 Markdown 表格。"""
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
                cells.append(_ARROW[policy[s]])
        rows.append(f"| **{row}** | {' | '.join(cells)} |")
    return "\n".join(rows)


# ── Value Iteration（帶日誌）────────────────────────────────────────────────────

def _run_vi(P, n_states, n_actions, gamma, theta, log_at):
    """
    執行 Value Iteration。
    log_at: 要記錄的 sweep 集合（其餘 sweep 跳過），最後一輪必定記錄。
    回傳 (V, policy, n_sweeps, entries)。
    """
    V = np.zeros(n_states)
    entries = []
    n_sweeps = 0

    while True:
        delta = 0.0
        v_before = V.copy()
        for s in range(n_states):
            q = np.zeros(n_actions)
            for a in range(n_actions):
                for prob, s_next, r, done in P[s][a]:
                    q[a] += prob * (r + gamma * V[s_next] * (1 - done))
            V[s] = np.max(q)
            delta = max(delta, abs(v_before[s] - V[s]))
        n_sweeps += 1
        converged = delta < theta

        if n_sweeps in log_at or converged:
            changed = [(s, v_before[s], V[s])
                       for s in range(n_states)
                       if abs(V[s] - v_before[s]) > 1e-9]

            title = f"### Sweep {n_sweeps}" + ("　✅ 收斂" if converged else "")
            lines = [title, "", f"**最大變化量 δ：** `{delta:.8f}`", "", _v_table(V), ""]

            if changed:
                lines += ["**本輪有更新的格子：**", "",
                          "| 格子 | 行,列 | 更新前 | 更新後 | 變化量 |",
                          "|:---:|:---:|:---:|:---:|:---:|"]
                for s, before, after in changed:
                    r, c = s // 4, s % 4
                    lines.append(
                        f"| 狀態 {s:>2} | ({r},{c}) "
                        f"| {before:.4f} | {after:.4f} | +{after - before:.4f} |"
                    )

                # 前 3 輪：展開每個更新格子的完整 Q 計算過程
                if n_sweeps <= 3:
                    lines += ["", "**完整計算過程：**", ""]
                    for s, _, _ in changed:
                        # 還原計算該格子當下，就地更新的 V 陣列狀態
                        v_at_step = np.zeros(n_states)
                        v_at_step[:s] = V[:s]
                        v_at_step[s:] = v_before[s:]
                        
                        lines.append(_detail_calc(s, P, v_at_step, n_actions, gamma))
                        lines.append("")
            else:
                lines.append("**本輪無更新（已完全收斂）**")

            entries.append("\n".join(lines))

        if converged:
            break

    # 從 V* 推出最佳策略
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        q = np.zeros(n_actions)
        for a in range(n_actions):
            for prob, s_next, r, done in P[s][a]:
                q[a] += prob * (r + gamma * V[s_next] * (1 - done))
        policy[s] = np.argmax(q)

    return V, policy, n_sweeps, entries


# ── Policy Iteration（帶日誌）───────────────────────────────────────────────────

def _run_pi(P, n_states, n_actions, gamma, theta):
    """
    執行 Policy Iteration。
    回傳 (V, policy, n_iters, entries)。
    """
    V = np.zeros(n_states)
    policy = np.zeros(n_states, dtype=int)
    entries = []

    # 記錄初始策略
    entries.append("\n".join([
        "### 初始策略（全部往左 ←）", "", _policy_table(policy)
    ]))

    n_iters = 0
    while True:
        # 策略評估（前 2 輪記錄收斂過程）
        eval_sweep = 0
        eval_log = []   # 每輪評估的內層 sweep 記錄
        while True:
            d = 0.0
            for s in range(n_states):
                v_old = V[s]
                a = policy[s]
                V[s] = sum(
                    prob * (r + gamma * V[s_next] * (1 - done))
                    for prob, s_next, r, done in P[s][a]
                )
                d = max(d, abs(v_old - V[s]))
            eval_sweep += 1
            converged_eval = d < theta

            # 前 2 輪：記錄關鍵 sweep（第 1~3 次、每 10 次、最後一次）
            if n_iters < 2:
                if eval_sweep <= 3 or eval_sweep % 10 == 0 or converged_eval:
                    label = f"第 {eval_sweep} 次掃描　δ = {d:.8f}"
                    if converged_eval:
                        label += "　✅ 收斂"
                    eval_log.append(f"**{label}**\n\n{_v_table(V)}")

            if converged_eval:
                break

        # 策略改進
        changes = []
        stable = True
        for s in range(n_states):
            old_a = policy[s]
            q = np.zeros(n_actions)
            for a in range(n_actions):
                for prob, s_next, r, done in P[s][a]:
                    q[a] += prob * (r + gamma * V[s_next] * (1 - done))
            policy[s] = np.argmax(q)
            if old_a != policy[s]:
                changes.append((s, old_a, policy[s]))
                stable = False

        n_iters += 1
        title = f"### 第 {n_iters} 輪" + ("　✅ 策略穩定，收斂" if stable else "")
        lines = [title, ""]

        # 前 2 輪：插入評估收斂過程
        if eval_log:
            lines += ["**─ 評估收斂過程（每次掃描） ─**", ""]
            lines += ["\n\n---\n\n".join(eval_log), ""]

        lines += ["**─ 評估後 V^π ─**", "", _v_table(V), "",
                  "**─ 改進結果 ─**", ""]

        if changes:
            lines += [f"共 **{len(changes)}** 個格子改變方向：", "",
                      "| 格子 | 行,列 | 原方向 | 新方向 |",
                      "|:---:|:---:|:---:|:---:|"]
            for s, old_a, new_a in changes:
                r, c = s // 4, s % 4
                lines.append(
                    f"| 狀態 {s:>2} | ({r},{c}) "
                    f"| {_ARROW[old_a]} ({old_a}) | {_ARROW[new_a]} ({new_a}) |"
                )

            # 前 3 輪：展開每個改方向格子的 Q 計算，說明為何換方向
            if n_iters <= 3:
                lines += ["", "**完整計算過程（為何改方向）：**", ""]
                for s, old_a, new_a in changes:
                    lines.append(_detail_calc(s, P, V, n_actions, gamma))
                    lines.append("")
        else:
            lines.append("無格子改變方向　→　策略已達最優。")

        lines += ["", "**改進後策略：**", "", _policy_table(policy)]
        entries.append("\n".join(lines))

        if stable:
            break

    return V, policy, n_iters, entries


# ── 評估 ──────────────────────────────────────────────────────────────────────

def _evaluate(env, policy, n_episodes):
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_return = 0.0
        done = False
        while not done:
            action = int(policy[obs])
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated
        returns.append(ep_return)
    return float(np.mean(returns))


# ── 主程式 ────────────────────────────────────────────────────────────────────

def train(config: dict) -> None:
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    env = gym.make(config["env_id"])
    P = env.unwrapped.P
    n_states  = env.observation_space.n
    n_actions = env.action_space.n
    gamma = config["gamma"]
    theta = config["theta"]

    print(f"Environment : {config['env_id']}")
    print(f"States      : {n_states}   Actions: {n_actions}")
    print(f"Gamma       : {gamma}   Theta: {theta}")
    print()

    sections = []
    sections.append(f"# 訓練日誌 (Training Log)\n")
    sections.append(
        f"> 環境：`{config['env_id']}`  |  "
        f"γ = `{gamma}`  |  θ = `{theta}`  |  "
        f"產生時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    sections.append("---\n")

    # ── Value Iteration ──────────────────────────────────────────────────────
    print("=== 價值疊代 ===")
    sections.append("## 一、價值疊代 (Value Iteration)\n")
    sections.append("\n".join([
        "### 說明",
        "",
        "**獎勵機制**",
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
        "**V(s) 的意義**：站在格子 s，按照最佳走法繼續走，預期能得到幾分。",
        "",
        "**更新公式**",
        "",
        "```",
        "Q(s, a) = Σ p(s'|s,a) × [ r  +  γ × V(s') ]",
        "           ↑ 轉移機率      ↑即時獎勵  ↑折扣未來分數",
        "",
        f"  p(s'|s,a)：從格子 s 選動作 a，抵達格子 s' 的機率（含滑動）",
        f"  r         ：這次移動的獎勵（只有走進終點才拿到 1 分）",
        f"  γ = {gamma}  ：折扣因子（每往後一步，獎勵打 {(1-gamma)*100:.0f}% 折）",
        f"  V(s')     ：下一格目前的估計分數",
        "",
        "V(s) = max_a Q(s, a)   ← 試完 4 個方向，取最高分更新",
        "```",
        "",
        f"**δ（最大變化量）**：本輪所有格子中 V 改變最大的量。δ 逐輪縮小，",
        f"低於 θ = {theta} 時停止（代表分數已完全收斂）。",
        "",
        "> 日誌顯示：第 1、2、3 輪（觀察初始擴散）、之後每 50 輪（觀察收斂進度）、最終收斂輪。",
        "",
        "---",
    ]) + "\n")

    log_at = set(range(1, 4)) | set(range(50, 400, 50))
    V_vi, policy_vi, n_sweeps, vi_entries = _run_vi(
        P, n_states, n_actions, gamma, theta, log_at
    )
    for e in vi_entries:
        sections.append(e + "\n")
    sections.append(f"\n**→ 總計在第 {n_sweeps} 輪收斂。**\n")
    sections.append("---\n")

    vi_rate = _evaluate(env, policy_vi, config["eval_episodes"])
    print(f"在 {n_sweeps} 次遍歷後收斂。")
    print(f"V* (重塑維度):\n{V_vi.reshape(4, 4).round(3)}")
    print(f"VI 成功率: {vi_rate:.2%}  ({config['eval_episodes']} 集數)\n")

    # ── Policy Iteration ─────────────────────────────────────────────────────
    print("=== 策略疊代 ===")
    sections.append("## 二、策略疊代 (Policy Iteration)\n")
    sections.append("\n".join([
        "### 說明",
        "",
        "獎勵機制與冰面滑動規則同上。",
        "",
        "**評估公式**（固定走法，如實算出這套走法能得幾分）",
        "",
        "```",
        "V^π(s) = Σ p(s'|s, π(s)) × [ r  +  γ × V^π(s') ]",
        "                  ↑ 只用走法規定的那個動作，不試其他方向",
        "```",
        "",
        "注意：這裡不取 max，只算「目前走法規定的動作」的期望分數。",
        "如果走法很差，算出來的 V 就會很低——這如實反映了這套走法的結果。",
        "",
        "**改進公式**（找出每格更好的方向）",
        "",
        "```",
        "Q(s, a)  = Σ p(s'|s,a) × [ r  +  γ × V^π(s') ]   ← 用評估後的 V^π 代入",
        "π'(s)    = argmax_a Q(s, a)                         ← 換成分數最高的方向",
        "```",
        "",
        "**「改變方向」的意思**：原本規定格子 s 往 A 走，改進後發現往 B 的 Q 更高，",
        "就把規定改成往 B 走。若某輪完全沒有格子改方向 → 走法已是最佳，停止。",
        "",
        "> 每輪均完整記錄：評估後的 V^π、哪些格子改了方向、改進後的完整策略。",
        "",
        "---",
    ]) + "\n")

    V_pi, policy_pi, n_iters, pi_entries = _run_pi(
        P, n_states, n_actions, gamma, theta
    )
    for e in pi_entries:
        sections.append(e + "\n")
    sections.append(f"\n**→ 總計在第 {n_iters} 輪策略改進後收斂。**\n")

    pi_rate = _evaluate(env, policy_pi, config["eval_episodes"])
    print(f"在 {n_iters} 次策略改進步數後收斂。")
    print(f"最佳策略 (0=左, 1=下, 2=右, 3=上):\n{policy_pi.reshape(4, 4)}")
    print(f"PI 成功率: {pi_rate:.2%}  ({config['eval_episodes']} 集數)")

    # ── 寫入日誌 ─────────────────────────────────────────────────────────────
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_log.md")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(sections))
    print(f"\n日誌已儲存：{log_path}")

    env.close()


if __name__ == "__main__":
    config = {
        "env_id":        "FrozenLake-v1",
        "gamma":         0.99,
        "theta":         1e-8,
        "eval_episodes": 1000,
        "seed":          42,
    }
    train(config)

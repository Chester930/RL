import random
import numpy as np
from env import SweeperEnv
from agent import QLearningAgent

def main():
    random.seed(42)
    np.random.seed(42)

    env = SweeperEnv()
    agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions)
    
    episodes = 500
    q_history = []
    
    print("開始訓練掃地機器人...")
    with open("training_log.md", "w", encoding="utf-8") as f:
        f.write("# 掃地機器人 Q-Learning 訓練日誌\n\n")
        f.write("## 📜 規則說明\n\n")
        f.write("- **目標**：從起點 (0) 走到充電座 (15)\n")
        f.write("- **獎勵機制 (Reward)**：\n")
        f.write("  - 抵達充電座：`+10`\n")
        f.write("  - 踩到水坑 (5, 12)：`-5`\n")
        f.write("  - 每移動一步：`-0.1` (日常消耗，迫使找最短路徑)\n")
        f.write("- **學習參數**：學習率 $\\alpha = 0.1$，折扣因子 $\\gamma = 0.9$\n\n")
        
        f.write("## 🗺️ 地圖對照\n\n")
        f.write("| Col 0 | Col 1 | Col 2 | Col 3 |\n")
        f.write("| :---: | :---: | :---: | :---: |\n")
        f.write("| 0 (起點) | 1 | 2 | 3 |\n")
        f.write("| 4 | 5 (水坑) | 6 | 7 |\n")
        f.write("| 8 | 9 | 10 | 11 |\n")
        f.write("| 12 (水坑) | 13 | 14 | 15 (目標) |\n\n")
        
        f.write("## 📖 如何看懂這份日誌\n\n")
        f.write("下方每次輸出的 Q-Table 是一個 `16 x 4` 的矩陣：\n")
        f.write("- **行 (直向 0~15)**：代表機器人所在的狀態 (0是起點，15是目標)\n")
        f.write("- **列 (橫向 0~3)** ：代表採取的動作 `[上, 下, 左, 右]`\n")
        f.write("- **裡面的數字**    ：代表在該狀態下，採取該動作的預期總回報 (數字越大越好)\n\n")
        
        f.write("## 🧮 Q-Table 更新計算範例 (破除 0 的魔咒)\n\n")
        f.write("為什麼訓練初期的表格會出現 `-0.01` 呢？\n")
        f.write("假設機器人第一回合在 **狀態 0**，隨機選擇了往 **右 (動作 3)**，走到了 **狀態 1**。\n")
        f.write("1. 拿到走路消耗獎勵：$r = -0.1$\n")
        f.write("2. 計算 TD Target：預期回報 = $-0.1 + 0.9 \\times \\max(Q(狀態1))$。因為初期 $Q(狀態1)$ 全是 0，所以 TD Target = $-0.1$。\n")
        f.write("3. 計算 TD Error：$-0.1 - Q(狀態0, 右)$ = $-0.1 - 0 = -0.1$。\n")
        f.write("4. 更新公式：$Q(狀態0, 右) = 0 + \\alpha \\times (-0.1) = 0 + 0.1 \\times (-0.1) = -0.01$。\n")
        f.write("這就是為什麼你看到機器人只要走過的地方，Q 值就會先被扣成 `-0.01`！\n\n")
        f.write("---\n\n")
        
        for episode in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # 1. 代理人根據當前狀態選擇動作
                action = agent.select_action(state)
                
                # 2. 環境根據動作給出下一個狀態與獎勵
                next_state, reward, done = env.step(action)
                
                # 3. 代理人根據經驗更新知識庫 (Q-Table)
                agent.update(state, action, reward, next_state, done)
                
                # 移動到下一個狀態
                state = next_state
                total_reward += reward
                
            # 紀錄每一回合結束時的 Q-Table (必須用 copy() 才不會只存到參照)
            q_history.append(agent.Q.copy())
            
            # 將每一次的 Q-Table 寫入 Markdown 檔中
            f.write(f"### Episode {episode + 1}\n")
            f.write(f"- **Total Reward**: `{total_reward:.2f}`\n")
            f.write("- **Q-Table**:\n")
            f.write("```text\n")
            # 使用 np.array2string 來格式化陣列輸出
            q_str = np.array2string(agent.Q, formatter={'float_kind':lambda x: "%.2f" % x})
            f.write(q_str + "\n")
            f.write("```\n\n")
                
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

    # 將歷史紀錄儲存為 npy 檔案
    np.save("q_table_history.npy", np.array(q_history))
    print(f"\n已經將 {episodes} 回合的 Q-Table 歷史紀錄儲存為 q_table_history.npy")
    print("詳細的訓練過程已經輸出到 training_log.md")

    print("\n訓練結束！以下是學到的 Q-Table：")
    # 設定 NumPy 列印格式，讓小數點比較好看
    np.set_printoptions(precision=2, suppress=True)
    print(agent.Q)
    
    print("\n測試機器人學到的最佳路徑 (Greedy Policy):")
    state = env.reset()
    done = False
    path = [state]
    actions_map = {0: "上", 1: "下", 2: "左", 3: "右"}
    
    # 關閉隨機探索來測試
    agent.epsilon = 0.0
    
    step_count = 0
    while not done and step_count < 20:
        action = agent.select_action(state)
        print(f"在狀態 {state}，選擇動作：{actions_map[action]}")
        next_state, reward, done = env.step(action)
        state = next_state
        path.append(state)
        step_count += 1
        
    print(f"最終路徑: {path}")
    if state == env.goal:
        print("成功抵達充電座！")
    else:
        print("機器人迷失了...")

if __name__ == "__main__":
    main()

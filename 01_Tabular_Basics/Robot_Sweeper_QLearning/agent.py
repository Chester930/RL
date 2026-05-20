import numpy as np

class QLearningAgent:
    def __init__(self, n_states: int, n_actions: int, alpha: float=0.1, gamma: float=0.9, epsilon: float=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        
        # 學習率 alpha: 每次更新要改變多少
        self.alpha = alpha
        
        # 折扣因子 gamma: 對未來獎勵的重視程度
        self.gamma = gamma
        
        # 探索率 epsilon: 有多高的機率會隨機行動
        self.epsilon = epsilon
        
        # 知識庫 Q-Table 初始化為 0
        self.Q = np.zeros((n_states, n_actions))
        
    def select_action(self, state: int) -> int:
        """ Epsilon-Greedy 策略 """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions) # 隨機探索
        else:
            # 利用已知最好的動作
            return int(np.argmax(self.Q[state]))
            
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """
        Q-Learning 核心更新公式實作
        """
        # 1. 計算預期最高回報 (TD Target)
        # 公式: TD Target = r + gamma * max Q(s', a')
        if done:
            # 如果已經結束了，未來就沒有回報了，所以預期回報就是當前拿到的 reward
            td_target = reward
        else:
            # 如果還沒結束，預期回報 = 當前的 reward + (折扣因子 * 下一個狀態所有可能動作中最高的 Q 值)
            td_target = reward + self.gamma * np.max(self.Q[next_state])
            
        # 2. 計算現實與理想的落差 (TD Error)
        # 公式: TD Error = TD Target - Q(s, a)
        # td_error 越大，代表我們之前的認知 ( self.Q[state, action] ) 越不準確
        td_error = td_target - self.Q[state, action]
        
        # 3. 更新知識庫 (Q-Table)
        # 公式: Q(s, a) <- Q(s, a) + alpha * TD Error
        # 用學習率 alpha 控制更新的步伐，把之前的認知朝著 td_target 的方向拉一點點
        self.Q[state, action] = self.Q[state, action] + self.alpha * td_error

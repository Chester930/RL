class SweeperEnv:
    """
    掃地機器人 4x4 迷宮環境。
    
    網格：
    0  1  2  3
    4  5  6  7
    8  9 10 11
   12 13 14 15
   
    起點：0
    目標（充電座）：15 (Reward: +10)
    陷阱（水坑）：5, 12 (Reward: -5)
    每步消耗：-0.1
    
    動作空間：
    0: 上, 1: 下, 2: 左, 3: 右
    """
    def __init__(self):
        self.n_states = 16
        self.n_actions = 4
        self.state = 0
        
        self.goal = 15
        self.traps = [5, 12]
        
    def reset(self):
        self.state = 0
        return self.state
        
    def step(self, action):
        row = self.state // 4
        col = self.state % 4
        
        # 0: 上, 1: 下, 2: 左, 3: 右
        if action == 0:
            row = max(0, row - 1)
        elif action == 1:
            row = min(3, row + 1)
        elif action == 2:
            col = max(0, col - 1)
        elif action == 3:
            col = min(3, col + 1)
            
        next_state = row * 4 + col
        self.state = next_state
        
        # 決定獎勵與是否結束
        done = False
        reward = -0.1  # 日常消耗
        
        if next_state == self.goal:
            reward = 10.0
            done = True
        elif next_state in self.traps:
            reward = -5.0
            done = True
            
        return next_state, reward, done

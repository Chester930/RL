"""
FourRooms GridWorld 環境。

經典 Options 論文（Sutton et al. 1999）使用的標準環境。
4 個房間由走廊相連，Agent 須從左上房間到達右下房間的目標點。

地圖（13×13，X=牆，.=空格，S=起點，G=終點，D=走廊）：

  XXXXXXXXXXXXX
  X.....X.....X
  X.....D.....X
  X.....X.....X
  X.....X.....X
  XXDXXXXXXXXXX   ← 下走廊在 row=5, col=2
  X.....X.....X
  X.....X.....X
  X.....D.....X
  X.....X.....X
  XXXXXXXXXXXXX

房間定義（row_range, col_range）：
  Room 0 (左上): rows 1-4, cols 1-5
  Room 1 (右上): rows 1-4, cols 7-11
  Room 2 (左下): rows 6-9, cols 1-5
  Room 3 (右下): rows 6-9, cols 7-11

走廊（doorway）位置：
  D0: (2, 6)  — 左上 ↔ 右上
  D1: (8, 6)  — 左下 ↔ 右下
  D2: (5, 2)  — 左上 ↔ 左下
  D3: (5, 8)  — 右上 ↔ 右下（但本地圖無此走廊，以 (5,8) 做替代）
"""

import numpy as np
from typing import Tuple, List


# 地圖：0=空格, 1=牆
GRID = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,1,0,1,1,1,1,1,1,0,1,1,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1],
], dtype=np.int32)

H, W = GRID.shape
GOAL = (9, 11)

# 走廊位置（4 個 option 的子目標）
DOORWAYS = [
    (2, 6),   # option 0: 到達左上↔右上走廊
    (8, 6),   # option 1: 到達左下↔右下走廊
    (5, 2),   # option 2: 到達左上↔左下走廊
    (5, 9),   # option 3: 到達右上↔右下走廊
]

# 4 個移動動作
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
N_ACTIONS = 4


class FourRoomsEnv:
    """
    FourRooms GridWorld。

    observation: (row, col) — 整數座標
    action:      int 0-3 (up/down/left/right)
    reward:      +1 到達目標，其他步 -0.001（鼓勵快速）
    max_steps:   500（避免無限循環）
    """

    def __init__(self, max_steps: int = 500):
        self.max_steps = max_steps
        self.state = None
        self.steps = 0

    @property
    def n_states(self) -> int:
        return H * W

    def _flat(self, r: int, c: int) -> int:
        return r * W + c

    def reset(self, start: Tuple[int, int] = (1, 1)) -> int:
        self.state = start
        self.steps = 0
        return self._flat(*self.state)

    def step(self, action: int):
        r, c = self.state
        dr, dc = ACTIONS[action]
        nr, nc = r + dr, c + dc

        if 0 <= nr < H and 0 <= nc < W and GRID[nr, nc] == 0:
            self.state = (nr, nc)

        self.steps += 1
        done = (self.state == GOAL) or (self.steps >= self.max_steps)
        reward = 1.0 if self.state == GOAL else -0.001
        return self._flat(*self.state), reward, done

    def get_state(self) -> Tuple[int, int]:
        return self.state

    def is_free(self, r: int, c: int) -> bool:
        return 0 <= r < H and 0 <= c < W and GRID[r, c] == 0

    def free_states(self) -> List[int]:
        return [self._flat(r, c) for r in range(H) for c in range(W) if GRID[r, c] == 0]

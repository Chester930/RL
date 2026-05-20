"""
SAC expert loader，供 collect_demos.py 與 train.py 共用。

用 importlib.util 明確載入 SAC 的 network.py，
避免與 BC 自己的 network.py / agent.py 同名衝突。
"""

import importlib.util
import os
import torch
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
_SAC_NET_PATH = os.path.join(
    ROOT, '04_Actor_Critic_Continuous/2018_SAC/network.py'
)
_SAC_CKPT_PATH = os.path.join(
    ROOT, '04_Actor_Critic_Continuous/2018_SAC/checkpoints_pendulum/sac.pt'
)


def _import_sac_network():
    """用 importlib 以獨立模組名稱載入 SAC 的 network.py，不影響 sys.modules['network']。"""
    spec = importlib.util.spec_from_file_location("_sac_network", _SAC_NET_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_sac_expert(device: str = "cpu"):
    """
    載入 SAC policy 作為示範專家。

    只需要 PolicyNetwork（不需要 Critic），
    回傳 .get_deterministic_action(state_tensor) 的物件。

    回傳：(policy, action_scale)
    """
    import gymnasium as gym
    sac_net = _import_sac_network()
    SACPolicy = sac_net.PolicyNetwork

    env = gym.make("Pendulum-v1")
    state_dim    = env.observation_space.shape[0]   # 3
    action_dim   = env.action_space.shape[0]         # 1
    action_scale = float(env.action_space.high[0])  # 2.0
    env.close()

    policy = SACPolicy(state_dim, action_dim, action_scale=action_scale).to(device)
    ckpt = torch.load(_SAC_CKPT_PATH, map_location=device, weights_only=True)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    print(f"[SACExpert] Loaded from {_SAC_CKPT_PATH}")
    return policy, action_scale


class SACExpertAgent:
    """
    薄包裝，讓 SAC expert 具有與 BCAgent 相同的 select_action 介面，
    方便在 distribution shift 測試中統一呼叫。
    """

    def __init__(self, device: str = "cpu"):
        self.policy, self.action_scale = load_sac_expert(device)
        self.device = torch.device(device)

    def select_action(self, state: np.ndarray, evaluate: bool = True) -> np.ndarray:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.policy.get_deterministic_action(state_t)
        return action.cpu().numpy()[0]

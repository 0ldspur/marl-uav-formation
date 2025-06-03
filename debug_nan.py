import sys, os, torch, numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.quad_env import MultiUAVEnv
from marl.train_mappo import Policy as PolicyNet   # ← fixed import

env = MultiUAVEnv(render=False, num_agents=4)

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
policy  = PolicyNet(obs_dim, act_dim)
policy.load_state_dict(torch.load("marl/models/best_policy.pth", map_location="cpu"))
policy.eval()

obs = env.reset()
for step in range(2000):
    act = policy(torch.tensor(obs, dtype=torch.float32)).detach().numpy()

    if not np.isfinite(act).all():
        print(f"❌ NaN/Inf in action at step {step}")
        break

    obs, _, _, _ = env.step(act)

else:
    print("✅ 2000 steps with finite actions, no crash")

env.close()

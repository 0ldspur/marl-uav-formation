import os, sys, torch, wandb, argparse
import numpy as np
from torch import nn
from torch.distributions import Normal

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from envs.quad_env import MultiUAVEnv

# ───── CLI Arguments ─────
parser = argparse.ArgumentParser()
parser.add_argument("--render", type=str, default="headless", choices=["2d", "3d", "headless"],
                    help="Choose render mode: 2d (top-down), 3d, or headless (fastest)")
args = parser.parse_args()

# ───── Hyperparameters ─────
lr           = 3e-4
episodes     = 200
steps_ep     = 300
entropy_coef = 0.005
gamma        = 0.99

# ───── Networks ─────
class Policy(nn.Module):
    def __init__(self, obs, act):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act)
        )
    def forward(self, x):
        return self.net(x)

class Value(nn.Module):
    def __init__(self, obs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)

# ───── Training Loop ─────
def train():
    wandb.init(project="marl-uav-formation",
               name=f"debug_{args.render}",
               config=dict(episodes=episodes, steps=steps_ep, lr=lr, render=args.render))

    env = MultiUAVEnv(render_mode=args.render)
    obs_d, act_d = env.observation_space.shape[0], env.action_space.shape[0]
    pi, vf = Policy(obs_d, act_d), Value(obs_d)
    opt_p, opt_v = torch.optim.Adam(pi.parameters(), lr=lr), torch.optim.Adam(vf.parameters(), lr=lr)

    # Save initial weights for update detection
    initial_params = [p.clone().detach() for p in pi.parameters()]
    best = -1e9

    for ep in range(episodes):
        obs = env.reset()
        ep_rew = 0.0

        for step in range(steps_ep):
            ot = torch.tensor(obs, dtype=torch.float32)
            act = pi(ot).detach().numpy()

            # 🔒 Clamp action to avoid large unrealistic values
            act = np.clip(act, -1.0, 1.0)

            if ep < 5 or ep % 25 == 0:
                print(f"\n[Ep {ep} Step {step}]")
                print("→ Action sample:", act[:6])
                print("→ Obs sample:", ot[:6])

            obs, r, _, _ = env.step(act)

            val = vf(ot)
            n_val = vf(torch.tensor(obs, dtype=torch.float32))
            adv = r + gamma * n_val.item() - val.item()
            if ep < 5 or ep % 25 == 0:
                print(f"→ Reward: {r:.2f}, Advantage: {adv:.4f}")
                print(f"→ Value: {val.item():.2f} → {n_val.item():.2f}")

            ent = Normal(pi(ot), torch.ones_like(pi(ot))).entropy().mean()
            loss_p = -(adv * (pi(ot)-ot).pow(2).sum() + entropy_coef * ent)
            loss_v = (val - (r + gamma * n_val.item()))**2

            opt_p.zero_grad(); loss_p.backward(); opt_p.step()
            opt_v.zero_grad(); loss_v.backward(); opt_v.step()
            ep_rew += r

        wandb.log({"episode": ep + 1, "reward": ep_rew})
        print(f"[{args.render.upper()}] Ep {ep+1:4d} | Reward: {ep_rew:8.2f}")

        if ep % 25 == 0:
            changed = any((not torch.equal(ip, p.detach())) for ip, p in zip(initial_params, pi.parameters()))
            print(f"🔁 Policy parameters changed since start? {'✅ YES' if changed else '❌ NO'}")

        if ep_rew > best:
            best = ep_rew
            os.makedirs("marl/models", exist_ok=True)
            torch.save(pi.state_dict(), "marl/models/best_policy.pth")

    env.close()
    wandb.finish()

if __name__ == "__main__":
    train()

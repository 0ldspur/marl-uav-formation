import sys, os, torch, wandb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from envs.quad_env import MultiUAVEnv
from torch import nn
from torch.distributions import Normal

# global settings
steps_ep   = 400
entropy_c  = 0.005
gamma      = 0.99

# ----------------------------------------------------------------------
class Policy(nn.Module):
    def __init__(self, obs, act):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU(),
            nn.Linear(128,act)
        )
    def forward(self,x): return self.net(x)

class Value(nn.Module):
    def __init__(self, obs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU(),
            nn.Linear(128,1)
        )
    def forward(self,x): return self.net(x)
# ----------------------------------------------------------------------

def train_phase(name, goal_weight, episodes, lr, load=None):
    wandb.init(project="marl-uav-formation",
               name=name,
               resume="allow",
               config=dict(goal_weight=goal_weight, episodes=episodes, lr=lr))

    env  = MultiUAVEnv(render=False, goal_weight=goal_weight)
    obs_d, act_d = env.observation_space.shape[0], env.action_space.shape[0]
    pi, vf = Policy(obs_d, act_d), Value(obs_d)
    if load: pi.load_state_dict(torch.load(load))

    opt_p = torch.optim.Adam(pi.parameters(), lr=lr)
    opt_v = torch.optim.Adam(vf.parameters(), lr=lr)

    best = -1e9
    for ep in range(episodes):
        obs, ep_rew = env.reset(), 0.0
        for _ in range(steps_ep):
            ot = torch.tensor(obs, dtype=torch.float32)
            act = pi(ot).detach().numpy()
            obs, r, _, _ = env.step(act)

            val = vf(ot)
            n_val = vf(torch.tensor(obs, dtype=torch.float32))
            adv = r + gamma * n_val.item() - val.item()

            ent = Normal(pi(ot), torch.ones_like(pi(ot))).entropy().mean()
            loss_p = -(adv * (pi(ot)-ot).pow(2).sum() + entropy_c * ent)
            loss_v = (val - (r + gamma * n_val.item()))**2

            opt_p.zero_grad(); loss_p.backward(); opt_p.step()
            opt_v.zero_grad(); loss_v.backward(); opt_v.step()
            ep_rew += r

        wandb.log({"episode": ep+1, "reward": ep_rew})
        print(f"{name}  Ep {ep+1:4d}  Reward {ep_rew:8.2f}")

        if ep_rew > best:
            best = ep_rew
            os.makedirs("marl/models", exist_ok=True)
            torch.save(pi.state_dict(), "marl/models/best_policy.pth")

    env.close(); wandb.finish()
    return "marl/models/best_policy.pth"

if __name__ == "__main__":
    # Phase A : hold formation, no goal bonus
    ckpt = train_phase("phaseA_static_formation",
                       goal_weight=0.0,
                       episodes=600,
                       lr=3e-4)

    # Phase B : fine-tune for A→B with strong goal bonus, lower LR
    train_phase("phaseB_AtoB_goal",
                goal_weight=6.0,
                episodes=800,
                lr=1e-4,
                load=ckpt)

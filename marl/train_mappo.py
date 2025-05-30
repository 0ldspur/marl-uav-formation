import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import wandb

# Allow imports from envs/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from envs.quad_env import MultiUAVEnv

# ─── ARGUMENT PARSING ──────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--render", choices=["2d", "3d", "headless"], default="headless")
parser.add_argument("--episodes", type=int, default=400)
parser.add_argument("--max-steps", type=int, default=300)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--ent-coef", type=float, default=0.01)
parser.add_argument("--clip-range", type=float, default=0.2)
parser.add_argument("--explore-std", type=float, default=0.2)
# Add reward shaping params
parser.add_argument("--formation-penalty", type=float, default=-1.0)
parser.add_argument("--mean-penalty", type=float, default=-0.4)
parser.add_argument("--cohesion-bonus-weight", type=float, default=10.0)
parser.add_argument("--progress-weight", type=float, default=0.3)
parser.add_argument("--alive-reward", type=float, default=0.5)
parser.add_argument("--formation-tol", type=float, default=15.0)
parser.add_argument("--goal-bonus", type=float, default=1500.0)
args = parser.parse_args()

# ─── WANDB INIT ─────────────────────────────────────
wandb.init(
    project="marl-uav-formation",
    name=f"mappo-tuned-{args.render}",
    config=vars(args)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── POLICY AND VALUE MODELS ────────────────────────
class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim)
        )
    def forward(self, x): return self.net(x)

class Value(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x): return self.net(x)

# ─── TRAIN FUNCTION ─────────────────────────────────
def train():
    env = MultiUAVEnv(
        render_mode=args.render,
        num_agents=4,
        action_scales=(0.5, 0.2, 0.1),
        max_steps=args.max_steps,
        goal_tol=0.5,
        formation_tol=args.formation_tol,
        goal_bonus=args.goal_bonus,
        formation_penalty=args.formation_penalty,
        mean_penalty=args.mean_penalty,
        cohesion_bonus_weight=args.cohesion_bonus_weight,
        progress_weight=args.progress_weight,
        alive_reward=args.alive_reward,
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    pi = Policy(obs_dim, act_dim).to(device)
    vf = Value(obs_dim).to(device)
    opt_p = torch.optim.Adam(pi.parameters(), lr=args.lr)
    opt_v = torch.optim.Adam(vf.parameters(), lr=args.lr)

    best_reward = -np.inf
    success_cnt = 0

    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        ep_rew = 0.0
        within_tol = False

        for t in range(args.max_steps):
            ot = torch.tensor(obs, dtype=torch.float32, device=device)
            mean_act = pi(ot)
            dist = Normal(mean_act, args.explore_std)
            action = dist.sample()
            logp = dist.log_prob(action).sum()
            ent = dist.entropy().sum()

            next_obs, r, done, info = env.step(action.cpu().numpy())
            ep_rew += r

            v = vf(ot).squeeze()
            v_next = vf(torch.tensor(next_obs, dtype=torch.float32, device=device)).squeeze().detach()
            target = r + args.gamma * v_next * (1 - done)
            adv = (target - v).detach()

            ratio = torch.exp(logp - logp.detach())
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1 - args.clip_range, 1 + args.clip_range) * adv
            loss_p = -torch.min(s1, s2).mean() - args.ent_coef * ent
            loss_v = (v - target).pow(2).mean()

            opt_p.zero_grad(); loss_p.backward(); opt_p.step()
            opt_v.zero_grad(); loss_v.backward(); opt_v.step()

            obs = next_obs

            if done:
                if info.get("is_success", False): success_cnt += 1
                within_tol = info.get("max_formation_error", 999) <= env.formation_tol
                break

        succ_rate = success_cnt / ep
        wandb.log({
            "episode": ep,
            "reward": ep_rew,
            "success_rate": succ_rate,
            "centroid_x": info.get("centroid_x", 0),
            "mean_formation_error": info.get("mean_formation_error", 999),
            "max_formation_error": info.get("max_formation_error", 999),
            "within_formation_tol": float(within_tol),
        })

        print(f"[{args.render}] Ep {ep:3d} | Rew {ep_rew:8.2f} | Succ {info.get('is_success')} |"
              f" CentroidX: {info.get('centroid_x'):.2f} | MaxErr: {info.get('max_formation_error'):.2f}")

        if ep_rew > best_reward:
            best_reward = ep_rew
            os.makedirs("marl/models", exist_ok=True)
            torch.save(pi.state_dict(), "marl/models/best_policy.pth")
            torch.save(vf.state_dict(), "marl/models/best_value.pth")

    env.close()
    wandb.finish()

# ─── RUN TRAINING ───────────────────────────────────
if __name__ == "__main__":
    train()

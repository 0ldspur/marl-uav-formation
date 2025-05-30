import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

# allow imports from envs/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from envs.single_uav_env import SingleUAVEnv

parser = argparse.ArgumentParser()
parser.add_argument("--episodes",  type=int,   default=500)
parser.add_argument("--max-steps", type=int,   default=300)
parser.add_argument("--lr",        type=float, default=3e-4)
parser.add_argument("--gamma",     type=float, default=0.99)
parser.add_argument("--std",       type=float, default=0.2)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim,128), nn.ReLU(),
            nn.Linear(128,128),    nn.ReLU(),
            nn.Linear(128,act_dim)
        )
    def forward(self,x): return self.net(x)

def discount_rewards(rewards, gamma):
    R = 0
    discounted = []
    for r in reversed(rewards):
        R = r + gamma * R
        discounted.insert(0, R)
    return discounted

def train():
    env = SingleUAVEnv(render_mode="headless")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = Policy(obs_dim, act_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    for ep in range(1, args.episodes+1):
        obs = env.reset()
        log_probs = []
        rewards   = []

        for t in range(args.max_steps):
            state = torch.tensor(obs, dtype=torch.float32, device=device)
            mean  = policy(state)
            dist  = Normal(mean, args.std)
            action = dist.sample()
            log_p  = dist.log_prob(action).sum()

            obs, r, done, info = env.step(action.cpu().numpy())
            log_probs.append(log_p)
            rewards.append(r)

            if done:
                break

        # compute returns and normalize
        returns = discount_rewards(rewards, args.gamma)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # policy loss
        policy_loss = []
        for log_p, R in zip(log_probs, returns):
            policy_loss.append(-log_p * R)
        policy_loss = torch.stack(policy_loss).sum()

        # update
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        total_reward = sum(rewards)
        succ = info.get("is_success", False)
        print(f"Ep {ep:3d}  Reward {total_reward:8.2f}  Success {succ}")

    env.close()

if __name__ == "__main__":
    train()

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from torch import nn
from envs.quad_env import MultiUAVEnv
import wandb

# ───────────────────────────────
# 🧠 Hyperparameters
# ───────────────────────────────
learning_rate = 3e-4
gamma = 0.99
num_episodes = 300
timesteps_per_episode = 200
num_agents = 4

# ───────────────────────────────
# 🧠 Policy Network
# ───────────────────────────────
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )

    def forward(self, x):
        return self.net(x)

# ───────────────────────────────
# 🧠 Value Network
# ───────────────────────────────
class ValueNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# ───────────────────────────────
# 🚀 Training Loop
# ───────────────────────────────
def train():
    wandb.init(project="marl-uav-formation", name="ppo-shared-policy", config={
        "learning_rate": learning_rate,
        "gamma": gamma,
        "timesteps_per_episode": timesteps_per_episode,
        "num_agents": num_agents
    })

    env = MultiUAVEnv(render=False, num_agents=num_agents)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy_net = PolicyNet(obs_dim, act_dim)
    value_net = ValueNet(obs_dim)

    optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
    optimizer_value = torch.optim.Adam(value_net.parameters(), lr=learning_rate)

    best_reward = float("-inf")

    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0

        for t in range(timesteps_per_episode):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action = policy_net(obs_tensor).detach().numpy()
            next_obs, reward, done, _ = env.step(action)

            value = value_net(obs_tensor)
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32)
            next_value = value_net(next_obs_tensor)
            advantage = reward + gamma * next_value.item() - value.item()

            pred_action = policy_net(obs_tensor)
            policy_loss = -advantage * torch.sum((pred_action - obs_tensor) ** 2)

            value_target = reward + gamma * next_value.item()
            value_loss = (value - value_target) ** 2

            optimizer_policy.zero_grad()
            policy_loss.backward()
            optimizer_policy.step()

            optimizer_value.zero_grad()
            value_loss.backward()
            optimizer_value.step()

            obs = next_obs
            total_reward += reward

        print(f"Episode {episode + 1} | Total Reward: {total_reward:.2f}")
        wandb.log({"episode": episode + 1, "reward": total_reward})

        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            os.makedirs("marl/models", exist_ok=True)
            torch.save(policy_net.state_dict(), "marl/models/best_policy.pth")
            torch.save(value_net.state_dict(), "marl/models/best_value.pth")
            print(f"✅ New best model saved at episode {episode + 1} | Reward: {total_reward:.2f}")

    env.close()
    wandb.finish()

# ───────────────────────────────
# 🏁 Run Training
# ───────────────────────────────
if __name__ == "__main__":
    train()

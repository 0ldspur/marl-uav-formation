import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import optuna
import traceback
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from envs.quad_env import MultiUAVEnv

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

def objective(trial):
    try:
        print(f"Starting trial {trial.number}...")

        lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        formation_penalty = trial.suggest_float('formation_penalty', -2.0, -0.5)
        mean_penalty = trial.suggest_float('mean_penalty', -1.0, -0.2)
        cohesion_bonus_weight = trial.suggest_float('cohesion_bonus_weight', 5.0, 20.0)
        progress_weight = trial.suggest_float('progress_weight', 0.1, 0.5)
        alive_reward = trial.suggest_float('alive_reward', 0.1, 1.0)
        formation_tol = trial.suggest_float('formation_tol', 10.0, 25.0)
        explore_std = trial.suggest_float('explore_std', 0.05, 0.3)

        env = MultiUAVEnv(
            render_mode="headless",
            num_agents=4,
            action_scales=(0.5, 0.2, 0.1),
            max_steps=200,
            formation_tol=formation_tol,
            goal_bonus=1500.0,
            formation_penalty=formation_penalty,
            mean_penalty=mean_penalty,
            cohesion_bonus_weight=cohesion_bonus_weight,
            progress_weight=progress_weight,
            alive_reward=alive_reward,
        )

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        pi = Policy(obs_dim, act_dim)
        vf = Value(obs_dim)
        opt_p = torch.optim.Adam(pi.parameters(), lr=lr)
        opt_v = torch.optim.Adam(vf.parameters(), lr=lr)

        episodes = 30   # Fast trial, increase for more reliability
        total_reward = 0
        total_success = 0

        for ep in range(episodes):
            obs = env.reset()
            ep_rew = 0.0
            for t in range(env.max_steps):
                ot = torch.tensor(obs, dtype=torch.float32)
                mean_act = pi(ot)
                dist = Normal(mean_act, explore_std)
                action = dist.sample()
                logp = dist.log_prob(action).sum()
                next_obs, r, done, info = env.step(action.numpy())
                v = vf(ot).squeeze()
                v_next = vf(torch.tensor(next_obs, dtype=torch.float32)).squeeze().detach()
                target = r + 0.99 * v_next * (1 - done)
                adv = (target - v).detach()
                loss_p = -(logp * adv).mean()
                loss_v = (v - target).pow(2).mean()
                opt_p.zero_grad(); loss_p.backward(); opt_p.step()
                opt_v.zero_grad(); loss_v.backward(); opt_v.step()
                obs = next_obs
                ep_rew += r
                if done:
                    break
            total_reward += ep_rew
            if info.get("is_success", False):
                total_success += 1

        env.close()
        avg_reward = total_reward / episodes
        avg_success = total_success / episodes
        print(f"Trial {trial.number}: avg_reward={avg_reward}, avg_success={avg_success}")
        # Prefer solutions with both high reward and high success rate
        return -avg_reward + (1 - avg_success) * 100
    except Exception as e:
        print(f"Exception in trial {trial.number}:\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    try:
        print("Launching Optuna hyperparameter search...")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=40)

        print("\nBest trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        # Save to file for your records
        with open("optuna_best_params.json", "w") as f:
            json.dump(trial.params, f, indent=2)
        print("\nBest hyperparameters saved to optuna_best_params.json")

    except Exception as e:
        print("Main Optuna loop crashed!")
        print(traceback.format_exc())

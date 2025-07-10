import os
import sys
import csv
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from envs.quad_env import MultiUAVEnv

# ───────── ARGUMENTS ─────────
parser = argparse.ArgumentParser()
parser.add_argument("--render", choices=["2d", "3d", "headless"], default="headless")
parser.add_argument("--episodes", type=int, default=2000)
parser.add_argument("--max-steps", type=int, default=300)
parser.add_argument("--lr", type=float, default=2.5e-5)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--ent-coef", type=float, default=0.05)
parser.add_argument("--clip-range", type=float, default=0.2)
parser.add_argument("--explore-std", type=float, default=0.07)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

# ───────── CURRICULUM CONFIG ─────────
curriculum = [
    {"goal_tol": 2.5, "formation_tol": 5.0, "goal_bonus": 500.0, "progress_weight": 2.0, "alive_reward": 2.0, "dist_goal_bonus_weight": 1500.0},
    {"goal_tol": 1.0, "formation_tol": 2.5, "goal_bonus": 900.0, "progress_weight": 1.0, "alive_reward": 0.5, "dist_goal_bonus_weight": 900.0},
    {"goal_tol": 0.5, "formation_tol": 1.7, "goal_bonus": 1500.0, "progress_weight": 0.3, "alive_reward": 0.3, "dist_goal_bonus_weight": 600.0}
]
consecutive_success_needed = 4
curr_stage = 0

# ───────── WANDB INIT ─────────
wandb.init(
    project="marl-uav-formation",
    name=f"mappo-run-{args.render}",
    config=vars(args),
    resume="allow"
)
wandb_url = wandb.run.url
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────── NETWORKS ─────────
class SharedPolicyRNN(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.ln = nn.LayerNorm(obs_dim)
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.gru = nn.GRU(128, 64, batch_first=True)
        self.out = nn.Linear(64, act_dim)
    def forward(self, x, h=None):
        x = self.ln(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = x.unsqueeze(1)
        if h is not None:
            h = h.detach()
        y, h_new = self.gru(x, h)
        y = y.squeeze(1)
        return self.out(y), h_new

class ValueNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x): return self.fc(x)

# ───────── TRAIN LOOP ─────────
def train():
    global curr_stage
    stage = curriculum[curr_stage]
    env = MultiUAVEnv(
        render_mode=args.render,
        num_agents=4,
        action_scales=(0.5, 0.2, 0.1),
        max_steps=args.max_steps,
        goal_tol=stage["goal_tol"],
        formation_tol=stage["formation_tol"],
        goal_bonus=stage["goal_bonus"],
        formation_penalty=-1.5,
        mean_penalty=-0.5,
        cohesion_bonus_weight=10.0,
        progress_weight=stage["progress_weight"],
        alive_reward=stage["alive_reward"],
        dist_goal_bonus_weight=stage["dist_goal_bonus_weight"],
    )

    obs_dim = 9
    act_dim = 3
    n_agents = env.num_agents

    policy = SharedPolicyRNN(obs_dim, act_dim).to(device)
    value = ValueNet(obs_dim).to(device)
    opt_pol = torch.optim.Adam(policy.parameters(), lr=args.lr)
    opt_val = torch.optim.Adam(value.parameters(), lr=args.lr)

    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    last_episode = 1
    best_reward = -np.inf
    patience, patience_limit = 0, 300
    success_streak = 0

    if args.resume and os.path.exists(f"{ckpt_dir}/last_episode.txt"):
        with open(f"{ckpt_dir}/last_episode.txt", "r") as f:
            last_episode = int(f.read()) + 1
        policy.load_state_dict(torch.load(f"{ckpt_dir}/shared_policy.pth"))
        value.load_state_dict(torch.load(f"{ckpt_dir}/value.pth"))
        print(f"✅ Resumed from episode {last_episode}")

    csv_file = "training_logs.csv"
    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "episode", "reward", "success", "max_err", "mean_err", "centroid_x", "wandb_url"
            ])

    for ep in range(last_episode, args.episodes + 1):
        obs = env.reset(randomize=True)
        obs_split = np.split(obs, n_agents)
        ep_rew, success = 0.0, False

        entropy_coef = args.ent_coef * max(0.3, 1 - (ep / (args.episodes * 0.8)))

        ep_obs = [[] for _ in range(n_agents)]
        ep_actions = [[] for _ in range(n_agents)]
        ep_logps = [[] for _ in range(n_agents)]
        ep_ents = [[] for _ in range(n_agents)]
        ep_vals = [[] for _ in range(n_agents)]
        ep_rewards = []
        ep_dones = []

        h_pol = [torch.zeros(1, 1, 64, device=device) for _ in range(n_agents)]

        for step in range(args.max_steps):
            actions, logps, ents, vals = [], [], [], []
            for i in range(n_agents):
                ot = torch.tensor(obs_split[i], dtype=torch.float32, device=device).unsqueeze(0)
                mu, h_new = policy(ot, h_pol[i])
                h_pol[i] = h_new.detach()
                dist = Normal(mu, args.explore_std)
                act = dist.sample()
                actions.append(act.squeeze(0).cpu().numpy())
                logps.append(dist.log_prob(act).sum())
                ents.append(dist.entropy().sum())
                vals.append(value(ot).squeeze())

            for i in range(n_agents):
                ep_obs[i].append(obs_split[i])
                ep_actions[i].append(actions[i])
                ep_logps[i].append(logps[i])
                ep_ents[i].append(ents[i])
                ep_vals[i].append(vals[i])

            flat_action = np.concatenate(actions)
            next_obs, r, done, info = env.step(flat_action)
            obs_split = np.split(next_obs, n_agents)
            ep_rew += r
            ep_rewards.append(r)
            ep_dones.append(done)

            if done:
                success = info.get("is_success", False)
                last_info = info  # for logging rewards
                break

        # ----- MAPPO-style batch update -----
        policy_losses = []
        value_losses = []

        for i in range(n_agents):
            obs_batch = torch.tensor(np.array(ep_obs[i]), dtype=torch.float32, device=device)
            act_batch = torch.tensor(np.array(ep_actions[i]), dtype=torch.float32, device=device)
            logp_batch = torch.stack(ep_logps[i])
            val_batch = torch.stack(ep_vals[i])

            returns = []
            ret = 0.0
            for rew in reversed(ep_rewards):
                ret = rew + args.gamma * ret
                returns.insert(0, ret)
            returns = torch.tensor(returns, dtype=torch.float32, device=device)

            adv = returns - val_batch.detach()

            mu_batch, _ = policy(obs_batch, None)
            dist = Normal(mu_batch, args.explore_std)
            new_logp = dist.log_prob(act_batch).sum(-1)
            entropy = dist.entropy().sum(-1)

            ratio = torch.exp(new_logp - logp_batch.detach())
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1 - args.clip_range, 1 + args.clip_range) * adv
            loss_p = -torch.min(s1, s2).mean() - entropy_coef * entropy.mean()
            loss_v = (val_batch - returns).pow(2).mean()

            policy_losses.append(loss_p)
            value_losses.append(loss_v)

        opt_pol.zero_grad()
        sum(policy_losses).backward()
        opt_pol.step()
        opt_val.zero_grad()
        sum(value_losses).backward()
        opt_val.step()

        # Logging and curriculum
        wandb.log({
            "episode": ep,
            "total_reward": ep_rew,
            "success": float(success),
            "max_err": last_info.get("max_formation_error", 999) if 'last_info' in locals() else 999,
            "mean_err": last_info.get("mean_formation_error", 999) if 'last_info' in locals() else 999,
            "centroid_x": last_info.get("centroid_x", 0.0) if 'last_info' in locals() else 0.0,
            "entropy": float(entropy_coef),
            "progress_reward": last_info.get("progress_reward", 0.0) if 'last_info' in locals() else 0.0,
            "cohesion_bonus": last_info.get("cohesion_bonus", 0.0) if 'last_info' in locals() else 0.0,
            "form_penalty": last_info.get("form_penalty", 0.0) if 'last_info' in locals() else 0.0,
            "dist_goal_bonus": last_info.get("dist_goal_bonus", 0.0) if 'last_info' in locals() else 0.0,
            "direction_bonus": last_info.get("direction_bonus", 0.0) if 'last_info' in locals() else 0.0,
        })

        timestamp = datetime.datetime.now().isoformat()
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, ep, ep_rew, int(success),
                last_info.get("max_formation_error", 0) if 'last_info' in locals() else 0,
                last_info.get("mean_formation_error", 0) if 'last_info' in locals() else 0,
                last_info.get("centroid_x", 0.0) if 'last_info' in locals() else 0.0,
                wandb_url
            ])

        if ep_rew > best_reward:
            best_reward = ep_rew
            os.makedirs("marl/models", exist_ok=True)
            torch.save(policy.state_dict(), f"marl/models/best_shared_policy.pth")

        # ---- Curriculum logic ----
        if success:
            success_streak += 1
            patience = 0
        else:
            success_streak = 0
            patience += 1

        if success_streak >= consecutive_success_needed and curr_stage < len(curriculum) - 1:
            curr_stage += 1
            print(f"\n🎯 Curriculum advancing to stage {curr_stage+1}: {curriculum[curr_stage]}")
            # Update env with new curriculum params
            stage = curriculum[curr_stage]
            env.goal_tol = stage["goal_tol"]
            env.formation_tol = stage["formation_tol"]
            env.goal_bonus = stage["goal_bonus"]
            env.progress_weight = stage["progress_weight"]
            env.alive_reward = stage["alive_reward"]
            env.dist_goal_bonus_weight = stage["dist_goal_bonus_weight"]
            success_streak = 0

        if patience >= patience_limit:
            print("🛑 Early stopping (no successes in patience window)")
            break

        if ep % 50 == 0:
            torch.save(policy.state_dict(), f"{ckpt_dir}/shared_policy.pth")
            torch.save(value.state_dict(), f"{ckpt_dir}/value.pth")
            with open(f"{ckpt_dir}/last_episode.txt", "w") as f:
                f.write(str(ep))

        # ----------- FIXED PRINT BLOCK -----------
        if ep % 25 == 0 or success:
            centroid_x = float(last_info.get('centroid_x', 0)) if 'last_info' in locals() else 0.0
            mean_err = float(last_info.get('mean_formation_error', 0)) if 'last_info' in locals() else 0.0
            print(f"[Ep {ep:4d}] Reward: {ep_rew:8.2f} | Success: {success} | Centroid_x: {centroid_x:.2f} | Formation Err: {mean_err:.2f}")

    env.close()
    wandb.finish()

if __name__ == "__main__":
    train()

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import torch
import numpy as np
from torch.distributions import Normal
from envs.quad_env import MultiUAVEnv
from train_mappo import SharedPolicyRNN  # Make sure this import matches your train script

# ----------- CONFIG ----------------
NUM_EVAL_EPISODES = 50
MODEL_PATH = "marl/models/best_shared_policy.pth"
RENDER = "3d"   # Use "3d" for GUI
SAVE_VIDEO = True    # Set True and RENDER="3d" for video
VIDEO_PATH = "eval_run.mp4"

def eval_playback():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = MultiUAVEnv(render_mode=RENDER)
    obs_dim = 9
    act_dim = 3
    n_agents = env.num_agents

    policy = SharedPolicyRNN(obs_dim, act_dim).to(device)
    policy.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    policy.eval()

    successes, all_max_errs, all_mean_errs, all_cx = 0, [], [], []
    videos = []

    for ep in range(NUM_EVAL_EPISODES):
        obs = env.reset(randomize=True)
        obs_split = np.split(obs, n_agents)
        done, ep_rew = False, 0.0
        h_pol = [torch.zeros(1, 1, 64, device=device) for _ in range(n_agents)]
        frames = []

        for step in range(env.max_steps):
            actions = []
            with torch.no_grad():
                for i in range(n_agents):
                    ot = torch.tensor(obs_split[i], dtype=torch.float32, device=device).unsqueeze(0)
                    mu, h_new = policy(ot, h_pol[i])
                    h_pol[i] = h_new.detach()
                    actions.append(mu.squeeze(0).cpu().numpy())  # DETERMINISTIC: no noise!
            flat_action = np.concatenate(actions)
            next_obs, r, done, info = env.step(flat_action)
            obs_split = np.split(next_obs, n_agents)
            ep_rew += r

            if SAVE_VIDEO and RENDER != "headless":
                import pybullet as p
                frame = p.getCameraImage(640, 480)[2]
                import cv2
                frame = np.array(frame)
                videos.append(frame)

            if done:
                break

        # Episode metrics
        success = info.get("is_success", False)
        successes += int(success)
        all_max_errs.append(info.get("max_formation_error", 0))
        all_mean_errs.append(info.get("mean_formation_error", 0))
        all_cx.append(info.get("centroid_x", 0))

        print(f"Episode {ep+1}/{NUM_EVAL_EPISODES} | Success: {success} | Max Err: {info.get('max_formation_error', 0):.3f} | Mean Err: {info.get('mean_formation_error', 0):.3f} | Centroid_x: {info.get('centroid_x', 0):.2f}")

    print("\n========== EVAL SUMMARY ==========")
    print(f"Success Rate: {successes} / {NUM_EVAL_EPISODES} ({successes/NUM_EVAL_EPISODES*100:.1f}%)")
    print(f"Mean Max Formation Error: {np.mean(all_max_errs):.3f}")
    print(f"Mean Formation Error: {np.mean(all_mean_errs):.3f}")
    print(f"Mean Centroid X: {np.mean(all_cx):.2f}")

    if SAVE_VIDEO and len(videos) > 0:
        import cv2
        out = cv2.VideoWriter(VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), 15, (640, 480))
        for frame in videos:
            out.write(frame)
        out.release()
        print(f"Saved video: {VIDEO_PATH}")

    env.close()

if __name__ == "__main__":
    eval_playback()

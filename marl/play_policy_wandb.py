import sys
import os
import time
import numpy as np
import cv2
import torch
import pybullet as p
import pybullet_data
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from envs.quad_env import MultiUAVEnv
from train_mappo import PolicyNet

# ───── Configuration ─────
MODEL_PATH    = "marl/models/best_policy.pth"
NUM_AGENTS    = 4
FPS           = 30
WIDTH, HEIGHT = 640, 480
RECORD_DIR    = "videos"
RECORD_FILE   = "wandb_playback.avi"   # AVI container
# ─────────────────────────

def play_and_log():
    # 1) Initialize WandB
    run = wandb.init(
        project="marl-uav-formation",
        name="playback-with-wandb-video"
    )

    # 2) Setup PyBullet
    env = MultiUAVEnv(render=True, num_agents=NUM_AGENTS)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetDebugVisualizerCamera(4, 45, -30, [0.5, 0.5, 1.0])
    cam_info = p.getDebugVisualizerCamera()
    view_mat, proj_mat = cam_info[2], cam_info[3]

    # 3) Load policy on CPU
    device = torch.device("cpu")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = PolicyNet(obs_dim, act_dim).to(device)
    policy.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    policy.eval()

    # 4) Setup video writer (XVID codec in AVI)
    os.makedirs(RECORD_DIR, exist_ok=True)
    video_path = os.path.join(RECORD_DIR, RECORD_FILE)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(video_path, fourcc, FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        print("⚠️  VideoWriter failed to open. Continuing without recording.")
        writer = None

    # 5) Optional: draw target markers
    for t in env.formation_targets:
        p.loadURDF("sphere2.urdf", t, globalScaling=0.2)

    # 6) Playback + capture loop (10s)
    obs = env.reset()
    for _ in range(FPS * 10):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        action = policy(obs_tensor).detach().cpu().numpy()
        obs, _, _, _ = env.step(action)

        if writer:
            img = p.getCameraImage(WIDTH, HEIGHT,
                                   viewMatrix=view_mat,
                                   projectionMatrix=proj_mat)
            rgba = np.reshape(img[2], (HEIGHT, WIDTH, 4))
            bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            writer.write(bgr)

        time.sleep(1.0 / FPS)

    # 7) Cleanup
    if writer:
        writer.release()
    env.close()

    # 8) Log to WandB (if recording succeeded)
    if writer:
        run.log({"video": wandb.Video(video_path, fps=FPS)})
    else:
        print("⚠️  No video to upload to WandB.")

    wandb.finish()
    print(f"✅ Completed playback. Video (if any) at: {video_path}")

if __name__ == "__main__":
    play_and_log()

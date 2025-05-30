import os
import time
import cv2
import torch
import numpy as np
import pybullet as p
from torch import nn
import argparse
import sys

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="marl/models/best_policy.pth", help="Path to model file")
    args = parser.parse_args()
    MODEL_FILE = args.model

    RENDER_MODE = "3d"
    VIDEO_FILE = "videos/eval_policy.mp4"
    os.makedirs("videos", exist_ok=True)

    env = MultiUAVEnv(render_mode=RENDER_MODE)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = Policy(obs_dim, act_dim)
    model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
    model.eval()
    print(f"Loaded policy from {MODEL_FILE}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(VIDEO_FILE, fourcc, 30, (1280, 720))

    obs = env.reset()
    for _ in range(100):
        with torch.no_grad():
            act = model(torch.tensor(obs, dtype=torch.float32)).numpy()
        obs, _, done, _ = env.step(act)

        centroid = np.mean([p.getBasePositionAndOrientation(d)[0] for d in env.drones], axis=0)
        p.resetDebugVisualizerCamera(15, 90, -30, centroid)

        try:
            view, proj = p.getDebugVisualizerCamera()[2:4]
            img = p.getCameraImage(1280, 720, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
        except:
            img = p.getCameraImage(1280, 720, view, proj, renderer=p.ER_TINY_RENDERER)[2]

        frame = cv2.cvtColor(np.reshape(img, (720, 1280, 4)), cv2.COLOR_RGBA2BGR)
        out.write(frame)
        time.sleep(1.0 / 30)

    out.release()
    env.close()
    print(f"✅ Saved to {VIDEO_FILE}")
    os.system(f'xdg-open "{VIDEO_FILE}" &')

if __name__ == "__main__":
    main()

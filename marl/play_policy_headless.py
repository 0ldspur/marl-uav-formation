import sys, os, time, cv2, torch, numpy as np, pybullet as p, pybullet_data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from envs.quad_env import MultiUAVEnv
from marl.train_mappo import Policy as PolicyNet

MODEL = "marl/models/best_policy.pth"
NUM   = 4
W, H  = 640, 480
FPS   = 30
OUT   = "videos/flight_A_to_B.mp4"

def main():
    os.makedirs("videos", exist_ok=True)
    video = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    # head-less env
    env = MultiUAVEnv(render=False, num_agents=NUM)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]

    policy = PolicyNet(obs_dim, act_dim)
    policy.load_state_dict(torch.load(MODEL, map_location="cpu"))
    policy.eval()

    # static camera along corridor
    view = p.computeViewMatrix(cameraEyePosition=[2,-6,3],
                               cameraTargetPosition=[2, 2, 1.2],
                               cameraUpVector=[0,0,1])
    proj = p.computeProjectionMatrixFOV(fov=60, aspect=W/H, nearVal=0.1, farVal=10)

    renderer_id = p.ER_BULLET_HARDWARE_OPENGL   # try EGL; will fallback if unavailable

    obs = env.reset()
    for i in range(FPS * 20):           # 20-s clip
        act = policy(torch.tensor(obs, dtype=torch.float32)).detach().numpy()
        obs, _, _, _ = env.step(act)

        if i % 2 == 0:                  # capture every 2nd frame → 15 fps recording
            img = p.getCameraImage(W, H, view, proj, renderer=renderer_id)[2]
            frame = cv2.cvtColor(np.reshape(img, (H, W, 4)), cv2.COLOR_RGBA2BGR)
            video.write(frame)

        time.sleep(1 / FPS)

    video.release(); env.close()
    print(f"✅ Saved {OUT}")
    os.system(f'xdg-open "{OUT}" >/dev/null 2>&1 &')  # preview if desktop available

if __name__ == "__main__":
    main()

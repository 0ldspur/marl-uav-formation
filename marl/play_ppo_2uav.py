import sys
import os
import argparse

# ─── Make sure the project root is on PYTHONPATH ───────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.formation_env_2uav import TwoUAVFormationEnv
from stable_baselines3 import PPO


def evaluate(model_path: str, n_episodes: int = 20, render: bool = False):
    # Instantiate env with or without GUI
    env = TwoUAVFormationEnv(render=render, formation_tol=5.0)
    # Load the trained model, binding it to our env
    model = PPO.load(model_path, env=env)

    successes = 0
    for ep in range(1, n_episodes + 1):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = env.step(action)
        # Check if this episode met both goal and formation tolerance
        if bool(info['centroid_x'] >= env.goal_x and
                info['max_formation_error'] <= env.formation_tol):
            successes += 1
        print(f"Episode {ep:2d} → centroid_x={info['centroid_x']:.2f}, "
              f"max_err={info['max_formation_error']:.2f}, "
              f"{'SUCCESS' if bool(info['centroid_x'] >= env.goal_x and info['max_formation_error'] <= env.formation_tol) else 'FAIL'}")

    print(f"\n✅ Success rate: {successes}/{n_episodes} episodes")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play back PPO policy on 2-UAV env")
    parser.add_argument(
        "--model-path", type=str, default="ppo_2uav_model.zip",
        help="Path to the saved PPO model"
    )
    parser.add_argument(
        "--episodes", "-n", type=int, default=20,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Enable PyBullet GUI rendering"
    )
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        n_episodes=args.episodes,
        render=args.render
    )

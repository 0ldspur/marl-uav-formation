import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.formation_env_2uav import TwoUAVFormationEnv
from stable_baselines3 import PPO

def evaluate(n_episodes=20, render=False):
    env = TwoUAVFormationEnv(render=render)
    model = PPO.load("ppo_2uav_minimal_model", env=env)
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        print(f"Episode {ep+1}: Steps={info['steps']}  Final x={info['centroid_x']:.2f}  Return={total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    evaluate(n_episodes=10, render=False)

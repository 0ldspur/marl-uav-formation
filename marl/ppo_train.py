import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../envs")))
from uav_formation_env import UAVFormationEnv


import gym
from stable_baselines3 import PPO
from uav_formation_env import UAVFormationEnv
from stable_baselines3.common.env_checker import check_env

# Sanity check your environment
env = UAVFormationEnv(n_agents=2, goal_x=10.0, max_steps=600)
check_env(env)

# Train PPO agent
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_uav_tensorboard/"
)

# Train for 100,000 steps (adjust as needed)
model.learn(total_timesteps=100_000)
model.save("ppo_uav_formation")

# Test playback
obs = env.reset()
done = False
step = 0
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(f"Step {step}: Centroid_x={info['centroid_x']:.2f}")
    step += 1

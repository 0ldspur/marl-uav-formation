# random_test.py
import numpy as np
from envs.quad_env import MultiUAVEnv

def test_random(env, steps=100):
    obs = env.reset()
    # reshape obs to (num_agents, obs_per_agent)
    per_agent = obs.shape[0] // env.num_agents
    obs_agents = obs.reshape(env.num_agents, per_agent)
    # initial centroid X
    initial = np.mean(obs_agents[:, 0])

    for _ in range(steps):
        action = env.action_space.sample()
        obs, _, _, _ = env.step(action)

    obs_agents = obs.reshape(env.num_agents, per_agent)
    final = np.mean(obs_agents[:, 0])
    print(f"Centroid X moved from {initial:.2f} → {final:.2f} (Δ = {final - initial:.2f})")

if __name__ == "__main__":
    env = MultiUAVEnv(render_mode="headless")
    test_random(env, steps=200)
    env.close()

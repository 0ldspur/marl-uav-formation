import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.quad_env import MultiUAVEnv

def test_env_step():
    env = MultiUAVEnv(render=False)
    obs = env.reset()
    print("Initial Observation:", obs)

    for t in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {t+1} Observation:", obs)

    env.close()

if __name__ == "__main__":
    test_env_step()


# This test function initializes the MultiUAVEnv environment, resets it, and takes a few steps while printing the observations.

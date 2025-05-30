import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.formation_env_2uav import TwoUAVFormationEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def main():
    env = DummyVecEnv([lambda: TwoUAVFormationEnv(render=False, formation_tol=5.0)])
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        policy_kwargs=dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])])
    )
    model.learn(total_timesteps=100_000)
    model.save("ppo_2uav_minimal_model")
    print("✅ PPO minimal baseline trained and saved as ppo_2uav_minimal_model.zip")

if __name__ == "__main__":
    main()

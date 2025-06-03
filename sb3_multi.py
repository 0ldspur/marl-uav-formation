# sb3_multi.py
from stable_baselines3 import PPO
from envs.quad_env import MultiUAVEnv

env = MultiUAVEnv(render_mode="headless")
model = PPO("MlpPolicy", env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=4096,
            batch_size=256,
            ent_coef=0.01,
            gamma=0.99)
model.learn(total_timesteps=100_000)
model.save("sb3_multi_forward")
print("✅ Done training multi-UAV forward baseline")

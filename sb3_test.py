from stable_baselines3 import PPO
from envs.quad_env import MultiUAVEnv

# 1) Instantiate your environment
env = MultiUAVEnv(render_mode="headless")

# 2) Create a PPO model (MlpPolicy = fully connected)
model = PPO("MlpPolicy", env, verbose=1, 
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            ent_coef=0.01,
            gamma=0.99)

# 3) Train for a short while (e.g. 50k timesteps)
model.learn(total_timesteps=50_000)

# 4) Save and exit
model.save("sb3_quad")
print("✅ SB3 PPO training complete.")

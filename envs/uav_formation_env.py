import gym
import numpy as np

class UAVFormationEnv(gym.Env):
    def __init__(self, n_agents=2, goal_x=10.0, max_steps=600):
        super(UAVFormationEnv, self).__init__()
        self.n_agents = n_agents
        self.goal_x = goal_x
        self.max_steps = max_steps
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(n_agents * 3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(n_agents * 3,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.positions = np.zeros((self.n_agents, 3))
        self.steps = 0
        self.prev_centroid_x = 0
        obs = self.positions.flatten()
        return obs

    def step(self, action):
        action = np.clip(action, -1, 1).reshape((self.n_agents, 3)) * 0.3  # Allow slightly larger moves
        self.positions += action
        self.steps += 1
        obs = self.positions.flatten()
        centroid_x = np.mean(self.positions[:, 0])
        reward = 0

        # Reward for forward progress
        progress = centroid_x - self.prev_centroid_x
        reward += progress
        self.prev_centroid_x = centroid_x

        # Small alive bonus (encourage moving)
        reward += 0.05

        # Penalty for camping just before the goal
        if centroid_x > self.goal_x - 0.5 and centroid_x < self.goal_x:
            reward -= 1

        # Huge bonus for reaching goal (forces completion)
        done = False
        if centroid_x >= self.goal_x:
            reward += 1000
            done = True

        # Episode ends if too many steps
        if self.steps >= self.max_steps:
            done = True

        info = {
            "centroid_x": centroid_x,
            "steps": self.steps
        }
        return obs, reward, done, info

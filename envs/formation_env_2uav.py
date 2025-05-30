import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
import random

class TwoUAVFormationEnv(gym.Env):
    def __init__(self, render=False, formation_tol=5.0, max_steps=1200):
        super().__init__()
        self.seed()
        self.render = render
        self.client = p.connect(p.GUI if self.render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.formation_tol = formation_tol
        self.n_agents = 2
        self.dt = 1. / 60.
        self.start_pos = [[0, 0, 1], [1, 0, 1]]  # two agents, 1m apart
        self.goal_x = 10.0
        self.max_steps = max_steps

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.n_agents * 3,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_agents * 6,), dtype=np.float32
        )

        self.reset()

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)

    def reset(self):
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        self.uavs = []
        for pos in self.start_pos:
            uav = p.loadURDF("sphere2.urdf", pos, globalScaling=0.1)
            self.uavs.append(uav)
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        obs = []
        for uav in self.uavs:
            pos, _ = p.getBasePositionAndOrientation(uav, self.client)
            vel, _ = p.getBaseVelocity(uav, self.client)
            obs.extend(list(pos) + list(vel))
        return np.array(obs, dtype=np.float32)
        # To normalize:
        # obs = np.array(obs, dtype=np.float32)
        # obs[:self.n_agents*3] /= 15.0  # position normalization
        # obs[self.n_agents*3:] /= 5.0   # velocity normalization
        # return obs

    def _centroid_x(self):
        positions = [p.getBasePositionAndOrientation(u, self.client)[0] for u in self.uavs]
        centroid = np.mean(positions, axis=0)
        return centroid[0]

    def step(self, action):
        action = np.clip(action, -1, 1).reshape((self.n_agents, 3)) * 0.2  # adjust scale if needed
        new_positions = []
        for i, uav in enumerate(self.uavs):
            pos, _ = p.getBasePositionAndOrientation(uav, self.client)
            new_pos = [pos[j] + action[i][j] for j in range(3)]
            new_pos = [
                np.clip(new_pos[0], 0, 15),
                np.clip(new_pos[1], -5, 5),
                np.clip(new_pos[2], 0.5, 2.0)
            ]
            new_positions.append(new_pos)
            p.resetBasePositionAndOrientation(uav, new_pos, [0, 0, 0, 1], self.client)

        # Uncomment for sanity checking:
        # print(f"Step {self.steps}: Drone 0 x={new_positions[0][0]:.2f}")

        p.stepSimulation(self.client)
        self.steps += 1

        obs = self._get_obs()
        centroid_x = self._centroid_x()

        # --- Reward Calculation ---
        reward = (centroid_x - 0)  # progress reward
        reward += 1                # alive bonus

        # --- Goal Bonus ---
        goal_bonus_given = False
        if centroid_x >= self.goal_x:
            reward += 20   # can increase for faster learning
            goal_bonus_given = True

        # --- Formation Penalty ---
        positions = [p.getBasePositionAndOrientation(u, self.client)[0] for u in self.uavs]
        dist = np.linalg.norm(np.array(positions[0]) - np.array(positions[1]))
        formation_err = abs(dist - 1.0)  # 1.0m is target separation
        reward -= 0.05 * formation_err   # start gentle!

        # --- Done logic ---
        done = bool(centroid_x >= self.goal_x or self.steps >= self.max_steps)

        # --- Info dict for logging/debugging ---
        info = {
            'centroid_x': centroid_x,
            'steps': self.steps,
            'goal_bonus_given': goal_bonus_given,
            'formation_error': formation_err
        }

        return obs, reward, done, info

    def close(self):
        p.disconnect(self.client)

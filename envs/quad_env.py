import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces

class MultiUAVEnv(gym.Env):
    def __init__(
        self,
        render_mode="headless",
        num_agents=4,
        action_scales=(0.5, 0.2, 0.1),
        max_steps=300,
        goal_tol=0.5,
        formation_tol=15.0,    # Looser for easier learning, tighten later
        goal_bonus=1500.0,
        formation_penalty=-1.0,   # Range: -0.5 to -1.5
        mean_penalty=-0.4,        # Range: -0.2 to -1.0
        cohesion_bonus_weight=10.0, # Range: 5.0 to 15.0
        progress_weight=0.3,      # Range: 0.1 to 0.5
        alive_reward=0.5          # Reduce if agents only "survive"
    ):
        super().__init__()
        self.num_agents = num_agents
        self.action_scales = action_scales
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.goal_tol = goal_tol
        self.formation_tol = formation_tol
        self.goal_bonus = goal_bonus
        self.formation_penalty = formation_penalty
        self.mean_penalty = mean_penalty
        self.cohesion_bonus_weight = cohesion_bonus_weight
        self.progress_weight = progress_weight
        self.alive_reward = alive_reward

        self.client = p.connect(p.GUI if render_mode in ["2d", "3d"] else p.DIRECT)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        self.center_A = np.array([0.0, 0.0, 1.2])
        self.center_B = np.array([15.0, 0.0, 1.2])
        self.offsets = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
        self.offsets_centered = self.offsets - self.offsets.mean(axis=0)

        self.drones = []
        for off in self.offsets:
            pos = (self.center_A + off).tolist()
            self.drones.append(p.loadURDF("sphere2.urdf", pos, globalScaling=0.15))

        if render_mode in ("2d", "3d"):
            green = p.createVisualShape(p.GEOM_SPHERE, 0.1, rgbaColor=[0, 1, 0, 1])
            red = p.createVisualShape(p.GEOM_SPHERE, 0.1, rgbaColor=[1, 0, 0, 1])
            p.createMultiBody(0, green, basePosition=self.center_A)
            p.createMultiBody(0, red, basePosition=self.center_B)
            cam_pos = (self.center_A + self.center_B) / 2 + np.array([0, -5, 2])
            cam_pitch = -30 if render_mode == "3d" else -89
            p.resetDebugVisualizerCamera(20, 90, cam_pitch, cam_pos.tolist())

        dim = 9 * self.num_agents
        self.observation_space = spaces.Box(-np.inf, np.inf, (dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (3 * self.num_agents,), dtype=np.float32)

        self.step_count = 0
        self.prev_centroid_x = 0
        self.got_goal_bonus = False

    def reset(self):
        self.step_count = 0
        self.got_goal_bonus = False

        for d, off in zip(self.drones, self.offsets_centered + self.center_A):
            p.resetBasePositionAndOrientation(d, off.tolist(), [0, 0, 0, 1])
            p.resetBaseVelocity(d, [0, 0, 0], [0, 0, 0])

        p.stepSimulation()
        poses = [p.getBasePositionAndOrientation(d)[0] for d in self.drones]
        xs = [pos[0] for pos in poses]
        self.prev_centroid_x = np.mean(xs)
        return self._get_obs()

    def step(self, action):
        self.step_count += 1
        act = np.clip(action, -1, 1).reshape(self.num_agents, 3)
        sx, sy, sz = self.action_scales

        for (vx, vy, vz), d in zip(act, self.drones):
            pos, orn = p.getBasePositionAndOrientation(d)
            newp = [pos[0] + vx * sx, pos[1] + vy * sy, np.clip(pos[2] + vz * sz, 0.6, 5.0)]
            p.resetBasePositionAndOrientation(d, newp, orn)

        p.stepSimulation()
        poses = [p.getBasePositionAndOrientation(d)[0] for d in self.drones]
        centroid = np.mean(poses, axis=0)
        centroid_x = centroid[0]

        # --- Formation error and bonuses ---
        targets = [centroid + off for off in self.offsets_centered]
        errors = [np.linalg.norm(np.array(pose) - tgt) for pose, tgt in zip(poses, targets)]
        max_err = max(errors)
        mean_err = np.mean(errors)

        r_form = self.formation_penalty * max_err + self.mean_penalty * mean_err

        # --- Cohesion bonus ---
        cohesion_bonus = 0.0
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                dist = np.linalg.norm(np.array(poses[i]) - np.array(poses[j]))
                if dist < 2.0:
                    cohesion_bonus += (2.0 - dist) * self.cohesion_bonus_weight
        r_form += cohesion_bonus

        # --- Progress reward ---
        r_prog = self.progress_weight * (centroid_x - self.prev_centroid_x)
        self.prev_centroid_x = centroid_x

        # --- Alive reward (keep low or 0 for now) ---
        r_alive = self.alive_reward

        r_goal = 0.0
        is_success = False

        # --- Goal and formation check ---
        if (not self.got_goal_bonus
                and centroid_x >= (self.center_B[0] - self.goal_tol)
                and max_err <= self.formation_tol):
            r_goal = self.goal_bonus
            self.got_goal_bonus = True
            is_success = True

        # --- Bonus for being within formation tolerance at any time ---
        formation_in_tol_bonus = 200.0 if max_err <= self.formation_tol else 0.0

        done = self.step_count >= self.max_steps or is_success

        reward = r_form + r_prog + r_alive + r_goal + formation_in_tol_bonus

        info = {
            "is_success": is_success,
            "centroid_x": centroid_x,
            "max_formation_error": max_err,
            "mean_formation_error": mean_err,
            "within_formation_tol": float(max_err <= self.formation_tol)
        }
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        obs = []
        for d in self.drones:
            pos = np.array(p.getBasePositionAndOrientation(d)[0])
            vel = np.array(p.getBaseVelocity(d)[0])
            rel = self.center_B - pos
            obs.extend(pos.tolist())
            obs.extend(rel.tolist())
            obs.extend(vel.tolist())
        return np.array(obs, dtype=np.float32)

    def close(self):
        p.disconnect(self.client)

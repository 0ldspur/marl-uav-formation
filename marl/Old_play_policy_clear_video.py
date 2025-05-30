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
        action_scales=(0.1, 0.1, 0.05),
        max_steps=300,
        randomize_reset=False,
        goal_tol=0.05
    ):
        super().__init__()
        self.num_agents = num_agents
        self.action_scales = action_scales
        self.max_steps = max_steps
        self.randomize_reset = randomize_reset
        self.goal_tol = goal_tol
        self.render_mode = render_mode.lower()

        if self.render_mode in ("2d", "3d"):
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        self.center_A = np.array([0.0, 0.0, 1.2])
        self.center_B = np.array([10.0, 0.0, 1.2])
        self.path_len = np.linalg.norm(self.center_B - self.center_A)

        self.offsets = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        self.drones = []
        for off in self.offsets:
            pos = self.center_A + off
            self.drones.append(p.loadURDF("sphere2.urdf", pos, globalScaling=0.15))
        self.formation_targets = [(self.center_A + o).copy() for o in self.offsets]

        if self.render_mode in ("2d", "3d"):
            green = p.createVisualShape(p.GEOM_SPHERE, 0.1, rgbaColor=[0, 1, 0, 1])
            red = p.createVisualShape(p.GEOM_SPHERE, 0.1, rgbaColor=[1, 0, 0, 1])
            p.createMultiBody(0, green, basePosition=self.center_A)
            p.createMultiBody(0, red, basePosition=self.center_B)

        if self.render_mode == "3d":
            mid = (self.center_A + self.center_B) / 2 + np.array([0, -5, 2])
            p.resetDebugVisualizerCamera(20, 90, -30, mid.tolist())
        elif self.render_mode == "2d":
            p.resetDebugVisualizerCamera(20, 90, -89, ((self.center_A + self.center_B) / 2).tolist())

        dim = 6 * self.num_agents  # each drone: pos (3) + rel_goal (3)
        self.observation_space = spaces.Box(-np.inf, np.inf, (dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (3 * self.num_agents,), dtype=np.float32)

        self.current_step = 0
        self.prev_dist_to_goal = None
        self.goal_reached = False

    def reset(self):
        self.current_step = 0
        self.goal_reached = False
        self.formation_targets = [(self.center_A + o).copy() for o in self.offsets]
        for d, pos in zip(self.drones, self.formation_targets):
            p.resetBasePositionAndOrientation(d, pos.tolist(), [0, 0, 0, 1])
        p.stepSimulation()

        centroid = np.mean(self.formation_targets, axis=0)
        self.prev_dist_to_goal = np.linalg.norm(centroid - self.center_B)
        return self._get_obs()

    def step(self, action):
        self.current_step += 1
        cen = np.mean(self.formation_targets, axis=0)
        dv = self.center_B - cen
        dist = np.linalg.norm(dv)
        if not self.goal_reached and dist > self.goal_tol:
            self.formation_targets = [t + (dv / dist) * 0.05 for t in self.formation_targets]
        elif dist <= self.goal_tol:
            self.goal_reached = True

        sx, sy, sz = self.action_scales
        act = np.clip(np.nan_to_num(action), -1.0, 1.0)
        for (vx, vy, vz), drone in zip(np.split(act, self.num_agents), self.drones):
            pos, orn = p.getBasePositionAndOrientation(drone)
            newp = [pos[0] + vx * sx, pos[1] + vy * sy, np.clip(pos[2] + vz * sz, 0.6, 5.0)]
            p.resetBasePositionAndOrientation(drone, newp, orn)
        p.stepSimulation()

        dists = [np.linalg.norm(np.array(p.getBasePositionAndOrientation(d)[0]) - t)
                 for d, t in zip(self.drones, self.formation_targets)]
        form_err = max(dists)
        r_form = -form_err * 10

        centroid = np.mean([p.getBasePositionAndOrientation(d)[0] for d in self.drones], axis=0)
        dist_now = np.linalg.norm(centroid - self.center_B)
        r_goal = 100.0 * (1.0 - dist_now / self.path_len) if (form_err < 0.25 and dist_now < self.goal_tol) else 0.0
        r_prog = 10.0 * ((self.prev_dist_to_goal - dist_now) / self.path_len)
        r_step = 2.0 * ((self.prev_dist_to_goal - dist_now) / self.path_len)
        reward = r_form + r_goal + r_prog + r_step - 0.001
        self.prev_dist_to_goal = dist_now

        done = self.goal_reached or (self.current_step >= self.max_steps)
        info = {"is_success": bool(self.goal_reached)}
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        obs = []
        for d, t in zip(self.drones, self.formation_targets):
            pos = np.array(p.getBasePositionAndOrientation(d)[0])
            rel_goal = self.center_B - pos
            obs.extend(pos)
            obs.extend(rel_goal)
        return np.array(obs, dtype=np.float32)

    def close(self):
        p.disconnect(self.physics_client)

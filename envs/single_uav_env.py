import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces

class SingleUAVEnv(gym.Env):
    """
    One UAV flies from A→B.
    Obs: [x,y,z, dx,dy,dz, vx,vy,vz]  (9-dim)
    Reward per step:
      • r_prog  = +10 × Δ(x)
      • r_alive = +1
    Success when x ≥ B_x – 0.5
    """
    def __init__(
        self,
        render_mode="headless",
        action_scales=(0.5,0.2,0.1),
        max_steps=300
    ):
        super().__init__()
        self.render_mode   = render_mode
        self.action_scales = action_scales
        self.max_steps     = max_steps

        # PyBullet
        if render_mode in ("2d","3d"):
            self.pc = p.connect(p.GUI)
        else:
            self.pc = p.connect(p.DIRECT)
        p.setGravity(0,0,-9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        # Point A & B
        self.A = np.array([0.0,0.0,1.2])
        self.B = np.array([10.0,0.0,1.2])

        # Spawn drone
        self.drone = p.loadURDF("sphere2.urdf", self.A, globalScaling=0.15)

        # Visual markers
        if render_mode in ("2d","3d"):
            g = p.createVisualShape(p.GEOM_SPHERE,0.1,rgbaColor=[0,1,0,1])
            r = p.createVisualShape(p.GEOM_SPHERE,0.1,rgbaColor=[1,0,0,1])
            p.createMultiBody(0,g,basePosition=self.A)
            p.createMultiBody(0,r,basePosition=self.B)

        # Camera centered on path midpoint
        mid = (self.A + self.B)/2 + np.array([0,-5,2])
        if render_mode=="3d":
            p.resetDebugVisualizerCamera(20,90,-30, mid.tolist())
        else:
            p.resetDebugVisualizerCamera(20,90,-89, mid.tolist())

        # Spaces
        self.observation_space = spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0,1.0,(3,), dtype=np.float32)

        # Internal trackers
        self.step_count      = 0
        self.prev_x          = None

    def reset(self):
        self.step_count = 0
        p.resetBasePositionAndOrientation(self.drone, self.A.tolist(), [0,0,0,1])
        p.resetBaseVelocity(self.drone, [0,0,0], [0,0,0])
        p.stepSimulation()
        pos, _ = p.getBasePositionAndOrientation(self.drone)
        self.prev_x = pos[0]
        return self._get_obs()

    def step(self, action):
        self.step_count += 1

        # Apply action
        sx,sy,sz = self.action_scales
        act = np.clip(action, -1.0, 1.0)
        pos,orn = p.getBasePositionAndOrientation(self.drone)
        newp = [
            pos[0] + act[0]*sx,
            pos[1] + act[1]*sy,
            np.clip(pos[2] + act[2]*sz, 0.6,5.0)
        ]
        p.resetBasePositionAndOrientation(self.drone, newp, orn)
        p.stepSimulation()

        # Observe
        pos, _ = p.getBasePositionAndOrientation(self.drone)
        vel,_  = p.getBaseVelocity(self.drone)
        rel     = self.B - pos

        # Rewards
        r_prog  = 10.0 * (pos[0] - self.prev_x)
        r_alive = 1.0
        self.prev_x = pos[0]

        reward = r_prog + r_alive

        # Done & success
        done    = (self.step_count >= self.max_steps)
        success = pos[0] >= (self.B[0] - 0.5)
        info    = {"is_success": bool(success)}

        return np.array([*pos, *rel, *vel], dtype=np.float32), reward, done, info

    def _get_obs(self):
        pos,_ = p.getBasePositionAndOrientation(self.drone)
        vel,_ = p.getBaseVelocity(self.drone)
        rel    = self.B - pos
        return np.array([*pos, *rel, *vel], dtype=np.float32)

    def close(self):
        p.disconnect(self.pc)

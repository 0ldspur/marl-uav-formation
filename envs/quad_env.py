import pybullet as p
import pybullet_data
import numpy as np
import gym
from gym import spaces
import time

class MultiUAVEnv(gym.Env):
    def __init__(self, render=False, num_agents=4):
        super(MultiUAVEnv, self).__init__()
        self.render = render
        self.num_agents = num_agents

        if self.render:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        # Clean GUI setup
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

        # Set up simulation world
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = p.loadURDF("plane.urdf")

        # Define start positions for 4 UAVs
        self.start_positions = [
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ] 
        self.formation_target = [
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]        

        self.drones = []
        for i in range(self.num_agents):
            drone = p.loadURDF("sphere2.urdf", self.start_positions[i])
            self.drones.append(drone)

        # Reset camera for visibility
        p.resetDebugVisualizerCamera(cameraDistance=4, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0.5, 0.5, 1])

        # Define observation and action spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents * 3,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_agents * 3,), dtype=np.float32)

    def reset(self):
        for i, drone in enumerate(self.drones):
            p.resetBasePositionAndOrientation(drone, self.start_positions[i], [0, 0, 0, 1])
        return self._get_obs()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        for i, drone in enumerate(self.drones):
            vx, vy, vz = action[i*3:(i+1)*3]
            pos, _ = p.getBasePositionAndOrientation(drone)
            new_pos = [pos[0] + vx * 0.05, pos[1] + vy * 0.05, pos[2] + vz * 0.05]
            p.resetBasePositionAndOrientation(drone, new_pos, [0, 0, 0, 1])
        p.stepSimulation()
        obs = self._get_obs()
        
        reward = 0.0
        for i, drone in enumerate(self.drones):
            pos, _ = p.getBasePositionAndOrientation(drone)
            target = np.array(self.formation_target[i])
            reward = np.linalg.norm(np.array(pos) -target)
            
        done = False
        return obs, -reward, done, {}


    def _get_obs(self):
        obs = []
        for drone in self.drones:
            pos, _ = p.getBasePositionAndOrientation(drone)
            obs.append(np.array(pos, dtype=np.float32))  # Each agent's position
        flat_obs = np.concatenate(obs)
        print("Agent-wise positions:", obs)
        return flat_obs

    def close(self):
        p.disconnect()

# Run a test rollout
if __name__ == "__main__":
    env = MultiUAVEnv(render=True)
    obs = env.reset()
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        time.sleep(1./60.)
    env.close()

# 📡 MARL-Based Formation Control – Project Log 

This document tracks the full development of my project on the 3D UAV formation control project using MAPPO — from simulation setup to deployment.

---
## Stage 1 - Python Environment & Installation/Gym Env Setup and Formation Reward Initial Design. 
- Created Conda environment `uav-marl` (Python 3.10)
- Installed required packages: `pybullet`, `gym`, `torch`, `cflib`
- Tested imports (`pybullet`, `gym`, `torch`)
- Set up `.github/workflows/ci.yml` for testing `test_env.py`
- Built `quad_env.py` using PyBullet with 4 agents
- Loaded `sphere2.urdf` as placeholder drone models
- Created GUI with camera and visualizer settings
- Defined `MultiUAVEnv` class compatible with Gym
- Implemented `reset()` and `step()` with position updates
- Connected `action_space` and `observation_space`
- Logged agent-wise positions in `_get_obs()`
- Created `tests/test_env.py` for unit testing environment
- Verified environment behavior over 10 steps
- Defined static target positions (formation layout)
- Calculated reward = negative distance to formation target
- Printed debug info for drone positions
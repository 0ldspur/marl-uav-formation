# MARL-Based Formation Control for Multi-UAV Systems

This project explores the use of Multi-Agent Reinforcement Learning (MARL) to control a fleet of UAVs flying in formation. The goal is to train agents that can independently navigate toward a goal while maintaining a safe and stable formation, without relying on centralized control during execution.

The work is being done gradually in clear, focused phases.


## Phase 1 – Project Setup & Environment
- Set up project structure
- Installed required packages and dependencies
- Defined a custom UAV environment using PyBullet
- Integrated Gym interface for training compatibility
- Added rendering options (2D and 3D modes)

> At this stage, the drones are modeled with simplified dynamics and can be visualized in both 2D and 3D spaces.


## Phase 2 – PPO Baseline (2 UAVs)
- Implemented Proximal Policy Optimization (PPO) using CleanRL-style runner
- Trained a 2-UAV system to reach goal positions while maintaining spacing
- Reward design included goal proximity and loose formation error
- Achieved over 95% success rate and consistent convergence

> This served as a good starting point to establish a stable, single-policy baseline before moving to multi-agent coordination.

## Phase 3 – MAPPO Extension (4 UAVs)
- Switched to Multi-Agent PPO (MAPPO) for 4-agent coordination
- Adopted Centralized Training with Decentralized Execution (CTDE)
- Shared actor across agents, centralized critic using joint observations
- Reward structure enhanced with inter-agent cohesion, collision penalty, and entropy scheduling

> Results show emerging coordination behaviors like synchronized motion, mid-air corrections, and partial self-recovery.


## Phase 4 – Evaluation & Visualization
- Created playback script for policy visualization
- Logged metrics: reward progression, collision count, entropy, formation error
- Evaluations run in both 2D and 3D rendering modes
- Metrics tracked using Weights & Biases (WandB)

## Installation

```bash
git clone https://github.com/Oldspur/marl-uav-formation.git
cd marl-uav-formation
pip install -r requirements.txt

To run training (for PPO or MAPPO):
```bash
python marl/train_mappo.py --render 3d

To evaluate trained policy:
```bash
python marl/eval_playback.py --render 2d

## Status

This is an active research project and still a work in progress.

Commits are dated to reflect the actual progression of the work—newer updates appear as the project advances through each phase.

While the codebase contains working components, some runs might not produce final or stable results yet, as improvements are ongoing across training stability, environment dynamics, and policy tuning.

More updates will be shared as the work progresses.


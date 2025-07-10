# MARL-Based Formation Control for Multi-UAV Systems

This project explores the use of Multi-Agent Reinforcement Learning (MARL) to control a fleet of UAVs flying in formation. The goal is to train agents that can independently navigate toward a goal while maintaining a safe and stable formation, without relying on centralized control during execution.

> **Status:** Active research project — core training and evaluation scripts are included for reproducibility. Ongoing improvements and additional features are expected.

---

## Table of Contents

- [Project Overview](#marl-based-formation-control-for-multi-uav-systems)
- [Phases](#phases)
- [Installation](#installation)
- [Usage](#usage)
- [Notes](#notes)
- [Reproducibility](#reproducibility)
- [License](#license)
- [Contact](#contact)

---

## Phases

### Phase 1 – Project Setup & Environment
- Structured the project and installed dependencies
- Defined a custom UAV environment with PyBullet and Gym interface
- Implemented flexible rendering (headless, 2D, 3D)

### Phase 2 – PPO Baseline (2 UAVs)
- Implemented Proximal Policy Optimization (PPO)
- Trained a 2-UAV system with custom rewards for goal and formation
- Achieved >95% success rate in baseline scenarios

### Phase 3 – MAPPO Extension (4 UAVs)
- Switched to Multi-Agent PPO (MAPPO) for 4 UAVs
- Centralized Training with Decentralized Execution (CTDE)
- Enhanced reward: inter-agent cohesion, collision penalty, entropy scheduling

### Phase 4 – Evaluation & Visualization
- Developed playback scripts for policy visualization
- Tracked reward, formation error, and agent metrics
- Evaluations support both 2D and 3D rendering
- Metrics logged via Weights & Biases (WandB)

---

## Installation

```bash
git clone https://github.com/Oldspur/marl-uav-formation.git
cd marl-uav-formation
pip install -r requirements.txt


### Usage
To train (MAPPO, 4 UAVs, 3D rendering):
python marl/train_mappo.py --render 3d

To evaluate a trained policy:
python marl/eval_playback.py --render 2d

- Change --render to headless, 2d, or 3d as needed.
- The default scripts will use the checkpoint in marl/models/best_shared_policy.pth.
- Sample demo videos and logs are included in the appropriate folders.

###Notes
- Intermediate checkpoints, large logs, and Weights & Biases (wandb/) artifacts are excluded for clarity and repo size.
- Only the final/best model checkpoint and a sample evaluation video are included for demonstration.
- For more results and logs, you can write me for the WandB dashboard.
```


### Reproducibility
- All necessary scripts and environment files are provided to reproduce main results from scratch.
- Training and evaluation scripts are self-contained and documented.
- See comments in each script for advanced usage and options.

### License
This project is licensed under the MIT License. See LICENSE for details.

### Contact
For questions, feedback, or collaboration opportunities, please just write me through my email. \


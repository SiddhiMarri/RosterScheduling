# RosterScheduling

## Overview

This repository contains implementations of two reinforcement learning algorithms, DQN and PPO, used to create air traffic controller rosters.

## Directory Structure

- `DQN/`: Contains the implementation of the Deep Q-Network algorithm.
- `PPO/`: Contains the implementation of the Proximal Policy Optimization algorithm.
  - `HybridModel/`: Combines PPO with linear programming.
  - `PPO1/` to `PPO13/`: Different approaches and iterations of the PPO algorithm. The higher the number, the more recent the approach (e.g., `PPO13` is more recent than `PPO12`).

## Installation

To install the required dependencies for the PPO implementation, navigate to the `PPO` directory and use conda to install the packages listed in the `requirements.txt` file:

```sh
cd PPO
conda install --file requirements.txt
```

## Usage

Detailed usage instructions for each algorithm can be found in their respective directories. Please refer to the README files within the `DQN` and `PPO` directories for more information.

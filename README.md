# Target Gridworld
A simple grid environment with support for obstacles, visibility, and dynamic agents

Includes sensor scheduling environments:
- `target_grid/SensorSchedulingLinear-v0`
- `target_grid/SensorSchedulingGrid-v0`


## Installation

To install run the following commands.  After cloning the repository, move into the base folder,

```{shell}
cd target_grid
```

Install polycheck -- a small library for visibility checking
```{shell}
pip install ./polycheck
```

Finally, install the target grid environments.
```{shell}
pip install ./target_grid
```

## Sensor Scheduling Quick Start

```python
import gymnasium as gym
import target_grid  # registers envs

env = gym.make("target_grid/SensorSchedulingLinear-v0")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

A fuller runnable demo is available in:
- `python/sensor_world/examples/sensor_scheduling_demo.py`

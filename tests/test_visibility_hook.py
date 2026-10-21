import subprocess
import sys

def test_import_and_environment_without_cuda():
    code = """
import sys
import numpy as np
from target_grid.envs import TargetWorldEnv
assert 'polycheck' not in sys.modules
assert 'pycuda' not in sys.modules
calls = []
def cpu_visibility(**kwargs):
    calls.append(kwargs)
    return np.ones(len(kwargs['starts']) * len(kwargs['ends']))
env = TargetWorldEnv(size=4, visibility_fn=cpu_visibility, world_parameters={
    'grid_data': np.zeros((4, 4)), 'agent_start_pos': (0, 0),
    'goal_pos': [(3, 3)], 'edge_probability': 1.0,
})
assert env.visibility_fn is cpu_visibility
env._visibility_fn((0, 0))
assert calls
assert 'pycuda' not in sys.modules
env.close()
"""
    subprocess.run([sys.executable, '-c', code], check=True)

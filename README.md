# Target Gridworld
A simple grid environment with support for obstacles, visibility, and dynamic agents


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


## CPU and CUDA visibility

Base installation requires no CUDA toolkit. Pass `visibility_fn` to
`TargetWorldEnv` for CPU execution. The callable receives keyword arguments
`data`, `starts`, `ends`, and `max_range` and returns a flat start-major array
of visibility probabilities. It must preserve grid Bresenham/range semantics.
Without the hook the original polycheck CUDA implementation is imported lazily
when constructing the environment. Install CUDA dependencies with
`pip install '.[cuda]'`; pin polycheck separately for reproducible experiments.

# MCTS-OCE Implementation Guide

This guide walks through `python/sensor_world/examples/sensor_selection/mcts_oce.py` and explains how the selector works, how it maps to the OCE node specification, and how to tune it.

## 1) Purpose and Interface

Entry point:
- `select_sensors(...)` in `python/sensor_world/examples/sensor_selection/mcts_oce.py:301`

What it does:
- Builds an OCE partition-based belief representation.
- Runs an action-only MCTS search over a receding horizon.
- Returns a binary action vector (`0/1` per sensor) for the current step.

Key selector inputs:
- `state_distribution`: current state belief.
- `coverage_matrix`: sensor-by-state visibility matrix.
- `max_active_sensors`: hard cap on awake sensors.
- `hmm`: provides multi-class transition models and current joint belief (`alphas`) when available.
- `planning_horizon`: lookahead depth.
- `mcts_iterations`: MCTS simulation budget.
- `mcts_max_actions`: cap on candidate actions considered by MCTS.
- `lambda_energy`, `sensor_energy_costs`: energy regularization.


## 2) Core Data Representation

### 2.1 Alpha partitions
Functions:
- `_joint_alpha_from_inputs(...)` at `python/sensor_world/examples/sensor_selection/mcts_oce.py:27`
- `_state_marginal_from_partitions(...)` at `python/sensor_world/examples/sensor_selection/mcts_oce.py:67`

Representation:
- A partition is an unnormalized joint mass matrix with shape `(K, M)`:
  - `K`: number of modes/classes.
  - `M`: number of states.
- The node stores a list of partitions (`list[np.ndarray]`), not a sampled observation branch.

This is the OCE compression mechanism: observation uncertainty is carried inside the node, so tree branching is over actions only.

### 2.2 Node structure
Dataclass:
- `_MCTSNode` at `python/sensor_world/examples/sensor_selection/mcts_oce.py:229`

Fields:
- `depth`: tree depth.
- `partitions`: OCE belief partitions after this action prefix.
- `b_state`: node-local state marginal.
- `candidate_actions`: node-local action set (computed for this node only).
- `next_expand_idx`: ordered expansion cursor for iterative widening.
- `visits`, `value_sum`: standard MCTS stats.
- `children`, `edge_reward`: action-indexed child node and immediate reward cache.


## 3) OCE Transition Update (Action Application)

Function:
- `_apply_action_oce(...)` at `python/sensor_world/examples/sensor_selection/mcts_oce.py:91`

This is the direct implementation of the spec’s propagate/split/merge logic:

1. **Predict each partition under each mode**
- `alpha_pred = np.einsum("km,kmn->kn", alpha, p_modes, ...)`
- This computes `alpha(c,:) @ P^(c)` for each mode `c`.

2. **Occluded continuation (kept per partition)**
- `alpha_occ = alpha_pred * occ_mask`
- If mass > 0, append as a child partition.

3. **Revealed merge by state**
- Accumulate all predicted visible-state mass into `revealed_acc[:, visible]`.
- For each visible state `x`, create one merged revealed partition with all mass concentrated at `x`.

4. **Renormalize and prune**
- Renormalize all partitions via `_renormalize_partitions(...)` at `python/sensor_world/examples/sensor_selection/mcts_oce.py:82`.
- Optional partition cap (`max_partitions`) keeps largest-mass partitions.


## 4) Candidate Action Set

Function:
- `_generate_candidate_actions(...)` at `python/sensor_world/examples/sensor_selection/mcts_oce.py:151`

Why this exists:
- Enumerating all sensor subsets is intractable.
- MCTS operates on a bounded candidate set per node.

How actions are generated:
- Always include all-sleep action.
- Include top-`r` actions by:
  - coverage score (`coverage @ belief`)
  - net score (`coverage score - lambda * energy`)
- Include randomized subsets for exploration.
- Include positive-net-gain subset.
- If generated actions exceed budget, rank and truncate to `max_actions`.

Control:
- `mcts_max_actions` from CLI feeds this `max_actions` cap.
- The action set is recomputed at each node from that node's current belief
  (`node.b_state`) inside `select_sensors(...)`.


## 5) Reward Model

Function:
- `_step_from_action(...)` at `python/sensor_world/examples/sensor_selection/mcts_oce.py:241`

Reward used by MCTS:
- `reward = -(entropy_cost + energy_cost)`

Where:
- `entropy_cost` is the partition-weighted conditional entropy across child partitions.
- `energy_cost = lambda_energy * sum(active_sensor_costs)` (if lambda is set).

Interpretation:
- MCTS maximizes cumulative reward, equivalent to minimizing entropy + energy.


## 6) MCTS Loop

Main search body:
- `select_sensors(...)` at `python/sensor_world/examples/sensor_selection/mcts_oce.py:301`.

### 6.1 Selection / Expansion
- Uses **iterative/progressive widening** to control horizontal spread:
  - `_progressive_widening_limit(...)` at `python/sensor_world/examples/sensor_selection/mcts_oce.py:288`
- Expands actions in ranked order (`next_expand_idx`) so the most likely candidates are considered first.
- When widening limit is reached, selects child with max UCT:
  - `_uct_score(...)` at `python/sensor_world/examples/sensor_selection/mcts_oce.py:278`

### 6.2 No Rollout (Forward Search)
- There is no random rollout phase.
- Node evaluation uses forward-search backup only (see comment at `python/sensor_world/examples/sensor_selection/mcts_oce.py:429`).

### 6.3 Backpropagation
- Accumulates rewards backward along the selected path.
- Updates `visits` and `value_sum` for each traversed node.

### 6.4 Final action choice
- Picks root child with best mean value (tie-break by visits).
- Returns corresponding binary action vector.


## 7) Runtime Controls

From `multiclass_hmm_sensor_experiment.py`:
- `--planning-horizon`
- `--mcts-iterations`
- `--mcts-max-actions`

How they map:
- `planning_horizon` controls lookahead depth (`horizon`) in `select_sensors`.
- `mcts_iterations` controls the number of search iterations in `select_sensors`.
- `mcts_max_actions` limits candidate branching in `_generate_candidate_actions`.

Practical tuning:
- Increase `mcts_iterations` first for better policy quality.
- Increase `mcts_max_actions` only if action diversity is too limited.
- Increase `planning_horizon` cautiously; cost grows quickly with depth.


## 8) Mapping to OCE Node Spec

Direct correspondences:
- **Node partitions**: `_MCTSNode.partitions` (`K x M` alphas).
- **ApplyActionOCE**: `_apply_action_oce(...)`.
- **Action-only branching**: tree expands over candidate action indices only.
- **Within-node observation uncertainty**: encoded in partition list.
- **Node scoring**: partition-weighted entropy + energy from `_step_from_action(...)`.


## 9) End-to-End Call Flow (Single Decision Step)

1. `select_sensors` receives current belief + transitions.
2. Build root partition(s): `_joint_alpha_from_inputs`.
3. Build candidate actions: `_generate_candidate_actions`.
4. Run MCTS iterations:
   - build node-local candidate actions (if not already cached)
   - apply progressive widening to decide whether to expand or select
   - expand/select child action
   - apply OCE update (`_apply_action_oce`)
   - score immediate reward
   - backpropagate value
5. Choose best root action and return binary sensor vector.


## 10) Debugging Checklist

If behavior looks wrong:
- Verify `coverage_matrix` shape is `(num_sensors, num_states)`.
- Check `sensor_energy_costs` length equals `num_sensors`.
- Confirm `hmm.transition_matrices` shape is `(K, M, M)`.
- Start with:
  - `planning_horizon=2`
  - `mcts_iterations=32`
  - `mcts_max_actions=8`
- Inspect action timing and entropy trends in episode CSV output.

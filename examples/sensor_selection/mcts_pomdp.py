from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


EPS = 1e-12
DEFAULT_MAX_ACTIONS = 12
DEFAULT_MAX_HORIZON = 3
DEFAULT_EXPLORATION = math.sqrt(2.0)


def _normalize_distribution(p: np.ndarray) -> np.ndarray:
    q = np.asarray(p, dtype=np.float64).reshape(-1)
    s = float(np.sum(q))
    if s <= EPS:
        if q.size == 0:
            return q
        return np.full((q.size,), 1.0 / float(q.size), dtype=np.float64)
    return q / s


def _joint_alpha_from_inputs(
    *,
    state_distribution: np.ndarray,
    hmm,
) -> tuple[np.ndarray, np.ndarray]:
    if hmm is None:
        state = _normalize_distribution(state_distribution)
        m = int(state.shape[0])
        alpha = state.reshape(1, -1).copy()
        p_modes = np.eye(m, dtype=np.float64).reshape(1, m, m)
        return alpha, p_modes

    p_modes = np.asarray(hmm.transition_matrices, dtype=np.float64)
    k, m, _ = p_modes.shape
    if hmm.alphas is not None:
        alpha = np.asarray(hmm.alphas, dtype=np.float64).copy()
    else:
        state = _normalize_distribution(np.asarray(hmm.state_distribution, dtype=np.float64))
        mode = _normalize_distribution(np.asarray(hmm.mode_distribution, dtype=np.float64))
        if mode.shape[0] != k:
            mode = np.full((k,), 1.0 / float(k), dtype=np.float64)
        if state.shape[0] != m:
            state = _normalize_distribution(state_distribution)
            if state.shape[0] != m:
                state = np.full((m,), 1.0 / float(m), dtype=np.float64)
        alpha = mode[:, None] * state[None, :]

    z = float(np.sum(alpha))
    if z <= EPS:
        alpha = np.full((k, m), 1.0 / float(k * m), dtype=np.float64)
    else:
        alpha /= z
    return alpha, p_modes


def _predict_alpha(alpha: np.ndarray, p_modes: np.ndarray) -> np.ndarray:
    k = int(p_modes.shape[0])
    pred = np.zeros_like(alpha, dtype=np.float64)
    for c in range(k):
        pred[c, :] = alpha[c, :] @ p_modes[c, :]
    return pred


def _predict_state_from_alpha(alpha: np.ndarray, p_modes: np.ndarray) -> np.ndarray:
    pred = _predict_alpha(alpha, p_modes)
    state = np.sum(pred, axis=0)
    return _normalize_distribution(state)


def _infer_absorbing_states_from_modes(
    p_modes: np.ndarray, tol: float = 1e-10
) -> np.ndarray:
    k, m, _ = p_modes.shape
    absorbing = np.ones((m,), dtype=bool)
    for s in range(m):
        for c in range(k):
            row = p_modes[c, s, :]
            if abs(float(np.sum(row)) - 1.0) > tol:
                absorbing[s] = False
                break
            if (1.0 - float(row[s])) > tol:
                absorbing[s] = False
                break
            if float(np.sum(np.abs(np.delete(row, s)))) > tol:
                absorbing[s] = False
                break
    return absorbing


def _build_action_vector(num_sensors: int, active_indices: np.ndarray) -> np.ndarray:
    action = np.zeros((num_sensors,), dtype=np.int8)
    if active_indices.size > 0:
        action[active_indices] = 1
    return action


def _generate_candidate_actions(
    *,
    coverage_matrix: np.ndarray,
    predicted_state: np.ndarray,
    max_active_sensors: int,
    rng: np.random.Generator,
    lambda_energy: float | None,
    sensor_energy_costs: np.ndarray,
    max_actions: int,
) -> list[np.ndarray]:
    num_sensors = int(coverage_matrix.shape[0])
    k = max(0, min(int(max_active_sensors), num_sensors))

    scores = coverage_matrix.astype(np.float64) @ predicted_state
    if lambda_energy is None:
        net_scores = scores.copy()
    else:
        net_scores = scores - float(lambda_energy) * sensor_energy_costs

    sizes = sorted(set([0, 1, 2, 3, 4, 8, k]))
    sizes = [s for s in sizes if 0 <= s <= k]
    if k <= 4:
        sizes = list(range(0, k + 1))

    actions: list[np.ndarray] = []
    seen: set[bytes] = set()

    def add_action(action: np.ndarray) -> None:
        key = action.tobytes()
        if key in seen:
            return
        seen.add(key)
        actions.append(action)

    add_action(np.zeros((num_sensors,), dtype=np.int8))
    for r in sizes:
        if r <= 0:
            continue
        top_idx = np.argsort(-scores, kind="stable")[:r]
        add_action(_build_action_vector(num_sensors, top_idx))
        top_net_idx = np.argsort(-net_scores, kind="stable")[:r]
        add_action(_build_action_vector(num_sensors, top_net_idx))
        for _ in range(2):
            rnd_idx = np.sort(rng.choice(num_sensors, size=r, replace=False))
            add_action(_build_action_vector(num_sensors, rnd_idx))

    if k > 0:
        positive = np.flatnonzero(net_scores > 0.0)
        if positive.size > 0:
            if positive.size > k:
                keep = positive[np.argsort(-net_scores[positive], kind="stable")[:k]]
            else:
                keep = positive
            add_action(_build_action_vector(num_sensors, np.sort(keep)))

    def action_rank(a: np.ndarray) -> float:
        active = np.flatnonzero(a > 0)
        if active.size == 0:
            return -1e9
        covered = np.any(coverage_matrix[active, :], axis=0)
        covered_mass = float(np.sum(predicted_state[covered]))
        energy_term = (
            float(lambda_energy) * float(np.sum(sensor_energy_costs[active]))
            if lambda_energy is not None
            else 0.0
        )
        return covered_mass - energy_term

    ranked = sorted(actions, key=action_rank, reverse=True)
    if len(ranked) <= max_actions:
        return ranked

    required_sizes = [s for s in (0, 1, 2, 3) if s <= k]
    kept: list[np.ndarray] = []
    seen: set[bytes] = set()
    for req in required_sizes:
        for a in ranked:
            if int(np.sum(a)) != req:
                continue
            key = a.tobytes()
            if key in seen:
                continue
            kept.append(a)
            seen.add(key)
            break
        if len(kept) >= max_actions:
            return kept[:max_actions]

    for a in ranked:
        key = a.tobytes()
        if key in seen:
            continue
        kept.append(a)
        seen.add(key)
        if len(kept) >= max_actions:
            break
    return kept[:max_actions]


def _progressive_widening_limit(
    visits: int,
    num_candidates: int,
    *,
    c_pw: float = 1.5,
    alpha_pw: float = 0.6,
) -> int:
    if num_candidates <= 0:
        return 0
    allowed = int(math.ceil(c_pw * (max(1, visits + 1) ** alpha_pw)))
    return max(1, min(num_candidates, allowed))


@dataclass
class _BeliefNode:
    depth: int
    alpha: np.ndarray
    predicted_state: np.ndarray
    candidate_actions: list[np.ndarray] | None = None
    next_expand_idx: int = 0
    visits: int = 0
    action_children: dict[int, "_ActionNode"] = field(default_factory=dict)


@dataclass
class _ActionNode:
    visits: int = 0
    value_sum: float = 0.0
    obs_children: dict[int, _BeliefNode] = field(default_factory=dict)


def _observation_from_state(
    state: int,
    *,
    visible_mask: np.ndarray,
    erasure_obs: int,
) -> int:
    if bool(visible_mask[state]):
        return int(state)
    return int(erasure_obs)


def _update_belief_given_obs(
    *,
    alpha: np.ndarray,
    p_modes: np.ndarray,
    visible_mask: np.ndarray,
    observation: int,
    erasure_obs: int,
) -> np.ndarray:
    pred = _predict_alpha(alpha, p_modes)
    m = int(pred.shape[1])
    likelihood = np.zeros((m,), dtype=np.float64)
    if int(observation) == int(erasure_obs):
        likelihood[~visible_mask] = 1.0
    elif 0 <= int(observation) < m and bool(visible_mask[int(observation)]):
        likelihood[int(observation)] = 1.0
    else:
        # impossible observation under this action; fall back to prediction
        return pred / max(float(np.sum(pred)), EPS)

    post = pred * likelihood[None, :]
    z = float(np.sum(post))
    if z <= EPS:
        return pred / max(float(np.sum(pred)), EPS)
    return post / z


def _sample_mode_state(alpha: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    mode_marg = np.sum(alpha, axis=1)
    mode_marg = _normalize_distribution(mode_marg)
    mode = int(rng.choice(mode_marg.shape[0], p=mode_marg))
    state_cond = _normalize_distribution(alpha[mode, :])
    state = int(rng.choice(state_cond.shape[0], p=state_cond))
    return mode, state


def select_sensors(
    state_distribution: np.ndarray,
    coverage_matrix: np.ndarray,
    max_active_sensors: int,
    rng: np.random.Generator,
    *,
    hmm=None,
    step_index: int | None = None,
    initial_true_state: int | None = None,
    planning_horizon: int | None = None,
    mcts_iterations: int | None = None,
    mcts_max_actions: int | None = None,
    lambda_energy: float | None = None,
    sensor_energy_costs: np.ndarray | None = None,
) -> np.ndarray:
    """
    Monte-Carlo MCTS with explicit action and observation branching.
    """
    del step_index
    del initial_true_state

    num_sensors = int(coverage_matrix.shape[0])
    num_states = int(coverage_matrix.shape[1])
    erasure_obs = int(num_states)
    k = max(0, min(int(max_active_sensors), num_sensors))
    zero_action = np.zeros((num_sensors,), dtype=np.int8)
    if k == 0:
        return zero_action

    if sensor_energy_costs is None:
        costs = np.ones((num_sensors,), dtype=np.float64)
    else:
        costs = np.asarray(sensor_energy_costs, dtype=np.float64).reshape(-1)
        if costs.shape[0] != num_sensors:
            raise ValueError("sensor_energy_costs length must match num_sensors")
        costs = np.clip(costs, 0.0, np.inf)

    alpha0, p_modes = _joint_alpha_from_inputs(
        state_distribution=np.asarray(state_distribution, dtype=np.float64),
        hmm=hmm,
    )
    predicted0 = _predict_state_from_alpha(alpha0, p_modes)
    root = _BeliefNode(depth=0, alpha=alpha0, predicted_state=predicted0)
    coverage_bool = np.asarray(coverage_matrix, dtype=bool)
    absorbing_mask = _infer_absorbing_states_from_modes(p_modes)

    horizon = int(planning_horizon) if planning_horizon is not None else DEFAULT_MAX_HORIZON
    horizon = max(1, horizon)
    iterations = int(mcts_iterations) if mcts_iterations is not None else int(max(64, min(512, 24 * horizon)))
    iterations = max(1, iterations)
    max_actions = int(mcts_max_actions) if mcts_max_actions is not None else DEFAULT_MAX_ACTIONS
    max_actions = max(1, max_actions)

    def _simulate(node: _BeliefNode, mode: int, state: int) -> float:
        if node.depth >= horizon:
            node.visits += 1
            return 0.0

        if node.candidate_actions is None:
            node.candidate_actions = _generate_candidate_actions(
                coverage_matrix=coverage_bool,
                predicted_state=node.predicted_state,
                max_active_sensors=k,
                rng=rng,
                lambda_energy=lambda_energy,
                sensor_energy_costs=costs,
                max_actions=max_actions,
            )
        num_candidates = len(node.candidate_actions)
        if num_candidates == 0:
            node.visits += 1
            return 0.0

        expand_limit = _progressive_widening_limit(node.visits, num_candidates)
        can_expand = (
            len(node.action_children) < expand_limit
            and node.next_expand_idx < num_candidates
        )
        if can_expand:
            action_idx = int(node.next_expand_idx)
            node.next_expand_idx += 1
            action_node = _ActionNode()
            node.action_children[action_idx] = action_node
        else:
            action_idx = max(
                node.action_children.keys(),
                key=lambda idx: (
                    (node.action_children[idx].value_sum / float(node.action_children[idx].visits))
                    if node.action_children[idx].visits > 0
                    else -np.inf
                )
                + DEFAULT_EXPLORATION
                * math.sqrt(math.log(max(1.0, float(node.visits + 1))) / float(max(1, node.action_children[idx].visits))),
            )
            action_node = node.action_children[action_idx]

        action = node.candidate_actions[action_idx]
        active = np.flatnonzero(action > 0)
        if active.size > 0:
            visible_mask = np.any(coverage_bool[active, :], axis=0)
        else:
            visible_mask = np.zeros((num_states,), dtype=bool)

        transition_row = p_modes[mode, state, :]
        next_state = int(rng.choice(num_states, p=transition_row))
        obs = _observation_from_state(
            next_state,
            visible_mask=visible_mask,
            erasure_obs=erasure_obs,
        )

        tracking_cost = 0.0
        if not bool(absorbing_mask[next_state]):
            tracking_cost = 1.0 if obs == erasure_obs else 0.0
        energy_cost = (
            float(lambda_energy) * float(np.sum(costs[active]))
            if lambda_energy is not None
            else 0.0
        )
        immediate_reward = -float(tracking_cost + energy_cost)

        child = action_node.obs_children.get(int(obs))
        if child is None:
            child_alpha = _update_belief_given_obs(
                alpha=node.alpha,
                p_modes=p_modes,
                visible_mask=visible_mask,
                observation=int(obs),
                erasure_obs=erasure_obs,
            )
            child_pred = _predict_state_from_alpha(child_alpha, p_modes)
            child = _BeliefNode(
                depth=node.depth + 1,
                alpha=child_alpha,
                predicted_state=child_pred,
            )
            action_node.obs_children[int(obs)] = child

        total_return = immediate_reward + _simulate(child, mode, next_state)

        action_node.visits += 1
        action_node.value_sum += total_return
        node.visits += 1
        return total_return

    for _ in range(iterations):
        mode0, state0 = _sample_mode_state(root.alpha, rng)
        _simulate(root, mode0, state0)

    if not root.action_children:
        if root.candidate_actions is None:
            root.candidate_actions = _generate_candidate_actions(
                coverage_matrix=coverage_bool,
                predicted_state=root.predicted_state,
                max_active_sensors=k,
                rng=rng,
                lambda_energy=lambda_energy,
                sensor_energy_costs=costs,
                max_actions=max_actions,
            )
        return root.candidate_actions[0] if root.candidate_actions else zero_action

    best_action_idx = max(
        root.action_children.keys(),
        key=lambda idx: (
            (root.action_children[idx].value_sum / float(root.action_children[idx].visits))
            if root.action_children[idx].visits > 0
            else -np.inf,
            root.action_children[idx].visits,
        ),
    )
    if root.candidate_actions is None:
        root.candidate_actions = _generate_candidate_actions(
            coverage_matrix=coverage_bool,
            predicted_state=root.predicted_state,
            max_active_sensors=k,
            rng=rng,
            lambda_energy=lambda_energy,
            sensor_energy_costs=costs,
            max_actions=max_actions,
        )
    return root.candidate_actions[best_action_idx]

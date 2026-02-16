from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

try:
    from numba import njit

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    _NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore
        def _decorator(func):
            return func

        return _decorator


EPS = 1e-12
DEFAULT_MAX_ACTIONS = 12
DEFAULT_MAX_HORIZON = 3


def _normalize_distribution(p: np.ndarray) -> np.ndarray:
    q = np.asarray(p, dtype=np.float64).reshape(-1)
    s = float(np.sum(q))
    if s <= EPS:
        if q.size == 0:
            return q
        return np.full((q.size,), 1.0 / float(q.size), dtype=np.float64)
    return q / s


def _entropy(p: np.ndarray) -> float:
    q = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0)
    return float(-np.sum(q * np.log(q)))


def _joint_alpha_from_inputs(
    *,
    state_distribution: np.ndarray,
    hmm,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
    - initial alpha partition (K, M), normalized
    - mode-conditioned transitions P_modes with shape (K, M, M)
    """
    if hmm is None:
        state = _normalize_distribution(state_distribution)
        alpha = state.reshape(1, -1).copy()
        m = int(alpha.shape[1])
        p_modes = np.eye(m, dtype=np.float64).reshape(1, m, m)
        return alpha, p_modes

    p_modes = np.asarray(hmm.transition_matrices, dtype=np.float64)
    k, m, _ = p_modes.shape
    if hmm.alphas is not None:
        alpha = np.asarray(hmm.alphas, dtype=np.float64).copy()
    else:
        state = _normalize_distribution(
            np.asarray(hmm.state_distribution, dtype=np.float64)
        )
        mode = _normalize_distribution(
            np.asarray(hmm.mode_distribution, dtype=np.float64)
        )
        if mode.shape[0] != k:
            mode = np.full((k,), 1.0 / float(k), dtype=np.float64)
        if state.shape[0] != m:
            state = _normalize_distribution(state_distribution)
            if state.shape[0] != m:
                state = np.full((m,), 1.0 / float(m), dtype=np.float64)
        alpha = mode[:, None] * state[None, :]

    mass = float(np.sum(alpha))
    if mass <= EPS:
        alpha = np.full((k, m), 1.0 / float(k * m), dtype=np.float64)
    else:
        alpha /= mass
    return alpha, p_modes


def _predict_state_marginal_from_partitions(
    *,
    partitions: list[np.ndarray],
    p_modes: np.ndarray,
    num_states: int,
) -> np.ndarray:
    """
    One-step predicted state marginal from the current partition set.

    This is used for action ranking so candidate sets are aligned with where
    the target will be at the next sensing step.
    """
    if not partitions:
        return np.full((num_states,), 1.0 / float(num_states), dtype=np.float64)

    k = int(p_modes.shape[0])
    predicted = np.zeros((num_states,), dtype=np.float64)
    total = 0.0
    for alpha in partitions:
        alpha_pred = np.zeros_like(alpha)
        for c in range(k):
            alpha_pred[c, :] = alpha[c, :] @ p_modes[c, :]
        predicted += np.sum(alpha_pred, axis=0)
        total += float(np.sum(alpha_pred))
    if total <= EPS:
        return np.full((num_states,), 1.0 / float(num_states), dtype=np.float64)
    return predicted / total


def _renormalize_partitions(partitions: list[np.ndarray]) -> list[np.ndarray]:
    if not partitions:
        return partitions
    total = float(sum(float(np.sum(p)) for p in partitions))
    if total <= EPS:
        return partitions
    return [p / total for p in partitions]


@njit(cache=True)
def _apply_action_oce_fast_core(
    partitions: np.ndarray,
    p_modes: np.ndarray,
    visible: np.ndarray,
    max_partitions: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Numba core for OCE propagation.

    Returns:
    - child_parts: packed child partitions (count x K x M)
    - child_is_occ: packed flags (count,), 1 for occluded family
    - count: number of valid children in the packed arrays
    """
    p_count = int(partitions.shape[0])
    k = int(partitions.shape[1])
    m = int(partitions.shape[2])

    is_visible = False
    for s in range(m):
        if visible[s]:
            is_visible = True
            break

    max_children = p_count + m
    child_parts_all = np.zeros((max_children, k, m), dtype=np.float64)
    child_mass_all = np.zeros((max_children,), dtype=np.float64)
    child_is_occ_all = np.zeros((max_children,), dtype=np.int8)
    revealed_acc = np.zeros((k, m), dtype=np.float64)

    alpha_pred = np.zeros((k, m), dtype=np.float64)
    count = 0

    for p_idx in range(p_count):
        # alpha_pred[c, j] = sum_i alpha[c, i] * P[c, i, j]
        for c in range(k):
            for j in range(m):
                acc = 0.0
                for i in range(m):
                    acc += partitions[p_idx, c, i] * p_modes[c, i, j]
                alpha_pred[c, j] = acc

        occ_mass = 0.0
        for c in range(k):
            for j in range(m):
                if not visible[j]:
                    v = alpha_pred[c, j]
                    child_parts_all[count, c, j] = v
                    occ_mass += v
                else:
                    child_parts_all[count, c, j] = 0.0

        if occ_mass > eps:
            child_mass_all[count] = occ_mass
            child_is_occ_all[count] = 1
            count += 1

        if is_visible:
            for c in range(k):
                for j in range(m):
                    if visible[j]:
                        revealed_acc[c, j] += alpha_pred[c, j]

    if is_visible:
        for x in range(m):
            if not visible[x]:
                continue
            mass_x = 0.0
            for c in range(k):
                mass_x += revealed_acc[c, x]
            if mass_x <= eps:
                continue
            for c in range(k):
                child_parts_all[count, c, x] = revealed_acc[c, x]
            child_mass_all[count] = mass_x
            child_is_occ_all[count] = 0
            count += 1

    if count <= 0:
        return (
            np.zeros((0, k, m), dtype=np.float64),
            np.zeros((0,), dtype=np.int8),
            0,
        )

    keep = count
    if keep > max_partitions:
        keep = int(max_partitions)
        order = np.argsort(child_mass_all[:count])[::-1]
        child_parts = np.zeros((keep, k, m), dtype=np.float64)
        child_is_occ = np.zeros((keep,), dtype=np.int8)
        for t in range(keep):
            idx = int(order[t])
            child_parts[t, :, :] = child_parts_all[idx, :, :]
            child_is_occ[t] = child_is_occ_all[idx]
        return child_parts, child_is_occ, keep

    child_parts = np.zeros((keep, k, m), dtype=np.float64)
    child_is_occ = np.zeros((keep,), dtype=np.int8)
    for t in range(keep):
        child_parts[t, :, :] = child_parts_all[t, :, :]
        child_is_occ[t] = child_is_occ_all[t]
    return child_parts, child_is_occ, keep


def _apply_action_oce_fast(
    *,
    partitions: list[np.ndarray],
    p_modes: np.ndarray,
    visible_mask: np.ndarray,
    max_partitions: int = 512,
) -> tuple[list[np.ndarray], int]:
    """
    Numba-accelerated OCE action update.

    Falls back to `_apply_action_oce` when numba is unavailable.
    """
    if not _NUMBA_AVAILABLE:
        return _apply_action_oce(
            partitions=partitions,
            p_modes=p_modes,
            visible_mask=visible_mask,
            max_partitions=max_partitions,
        )
    if not partitions:
        return partitions, 0

    stacked = np.stack(partitions, axis=0).astype(np.float64, copy=False)
    visible = np.asarray(visible_mask, dtype=np.bool_).reshape(-1)
    child_parts_arr, child_is_occ_arr, count = _apply_action_oce_fast_core(
        stacked,
        np.asarray(p_modes, dtype=np.float64),
        visible,
        int(max_partitions),
        float(EPS),
    )
    if count <= 0:
        return [], 0

    child_parts = [child_parts_arr[i].copy() for i in range(int(count))]
    num_occ_partitions = int(np.sum(child_is_occ_arr[: int(count)]))
    return child_parts, num_occ_partitions


def _apply_action_oce(
    *,
    partitions: list[np.ndarray],
    p_modes: np.ndarray,
    visible_mask: np.ndarray,
    max_partitions: int = 512,
) -> tuple[list[np.ndarray], int]:
    """
    OCE partition propagation under ideal reveal/occlusion sensing.

    Returns:
    - child_partitions: propagated child partitions (occluded + revealed families)
    - num_occ_partitions: number of leading entries in child_partitions that
      correspond to y_k = occ branches

    If there are no remaining partitions, returns an empty list and zero.
    Otherwise, returns a list of child partitions representing the updated belief state after applying the action and observing
    the revealed states.  The child partitions are not normalized, however, they retain their mass relative to each other
    (i.e. they sum to the same total mass as the input partitions).  The number of child partitions is at most max_partitions;
    if there are more, only the most probable ones are kept.
    """
    if not partitions:
        return partitions, 0

    visible = np.asarray(visible_mask, dtype=bool).reshape(-1)
    is_visible = np.any(visible)
    occ = ~visible
    k = int(p_modes.shape[0])
    m = int(p_modes.shape[1])

    child_partitions: list[np.ndarray] = []
    revealed_acc = np.zeros((k, m), dtype=np.float64)

    for alpha in partitions:
        # alpha_pred = np.einsum("km,kmn->kn", alpha, p_modes, optimize=True)
        # Faster for small K
        alpha_pred = np.zeros_like(alpha)
        for c in range(k):
            alpha_pred[c, :] = alpha[c, :] @ p_modes[c, :]

        alpha_occ = alpha_pred * occ[None, :]
        if float(np.sum(alpha_occ)) > EPS:
            child_partitions.append(alpha_occ)
        if is_visible:
            revealed_acc[:, visible] += alpha_pred[:, visible]

    num_occ_partitions = len(child_partitions)

    # if is_visible:
    #     for x in np.flatnonzero(visible):
    #         mass_x = float(np.sum(revealed_acc[:, x]))
    #         if mass_x <= EPS:
    #             continue
    #         alpha_rev = np.zeros((k, m), dtype=np.float64)
    #         alpha_rev[:, x] = revealed_acc[:, x]
    #         child_partitions.append(alpha_rev)

    if is_visible:
        visible_indices = np.flatnonzero(visible)
        # Compute all masses at once
        masses = np.sum(revealed_acc[:, visible_indices], axis=0)
        valid = masses > EPS

        # Only create partitions for valid states
        for i, x in enumerate(visible_indices):
            if valid[i]:
                alpha_rev = np.zeros((k, m), dtype=np.float64)
                alpha_rev[:, x] = revealed_acc[:, x]
                child_partitions.append(alpha_rev)

    if len(child_partitions) > max_partitions:
        original_occ_count = num_occ_partitions
        masses = np.asarray(
            [float(np.sum(p)) for p in child_partitions], dtype=np.float64
        )
        keep_idx = np.argsort(-masses, kind="stable")[:max_partitions]
        child_partitions = [child_partitions[int(i)] for i in keep_idx]
        num_occ_partitions = int(np.sum(keep_idx < original_occ_count))

    return child_partitions, num_occ_partitions


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
    max_actions: int = 24,
) -> list[np.ndarray]:
    num_sensors = int(coverage_matrix.shape[0])
    num_states = int(coverage_matrix.shape[1])
    k = max(0, min(int(max_active_sensors), num_sensors))
    k_eff = k
    if lambda_energy is not None and k > 0:
        # Coverage mass is upper-bounded by 1.0; if even the cheapest r sensors
        # cost more than that bound, those large-cardinality actions are poor
        # candidates in high-lambda regimes.
        sorted_costs = np.sort(np.asarray(sensor_energy_costs, dtype=np.float64))
        cum_costs = np.cumsum(sorted_costs)
        affordable = np.flatnonzero(float(lambda_energy) * cum_costs <= 1.0 + 1e-9)
        if affordable.size == 0:
            k_eff = 0
        else:
            k_eff = min(k, int(affordable[-1]) + 1)

    scores = coverage_matrix.astype(np.float64) @ predicted_state
    if lambda_energy is None:
        net_scores = scores.copy()
    else:
        net_scores = scores - float(lambda_energy) * sensor_energy_costs

    sizes = sorted(set([0, 1, 2, 3, 4, 8, k_eff]))
    sizes = [s for s in sizes if 0 <= s <= k_eff]
    if k_eff <= 4:
        sizes = list(range(0, k_eff + 1))

    actions: list[np.ndarray] = []
    seen: set[bytes] = set()

    def add_action(action: np.ndarray) -> None:
        key = action.tobytes()
        if key in seen:
            return
        seen.add(key)
        actions.append(action)

    def action_rank(a: np.ndarray) -> float:
        active = np.flatnonzero(a > 0)
        if active.size == 0:
            return 0.0
        covered = np.any(coverage_matrix[active, :], axis=0)
        covered_mass = float(np.sum(predicted_state[covered]))
        energy_term = (
            float(lambda_energy) * float(np.sum(sensor_energy_costs[active]))
            if lambda_energy is not None
            else 0.0
        )
        return covered_mass - energy_term

    def greedy_action_of_size(r: int) -> np.ndarray:
        if r <= 0:
            return np.zeros((num_sensors,), dtype=np.int8)
        covered = np.zeros((num_states,), dtype=bool)
        remaining = np.ones((num_sensors,), dtype=bool)
        selected: list[int] = []
        for _ in range(r):
            best_idx = -1
            best_gain = -np.inf
            candidate_indices = np.flatnonzero(remaining)
            if candidate_indices.size == 0:
                break
            for idx in candidate_indices:
                new_cov = coverage_matrix[int(idx), :] & (~covered)
                marginal_gain = float(np.sum(predicted_state[new_cov]))
                energy_term = (
                    float(lambda_energy) * float(sensor_energy_costs[int(idx)])
                    if lambda_energy is not None
                    else 0.0
                )
                gain = marginal_gain - energy_term
                if gain > best_gain:
                    best_gain = gain
                    best_idx = int(idx)
            if best_idx < 0:
                break
            selected.append(best_idx)
            covered |= coverage_matrix[best_idx, :]
            remaining[best_idx] = False

        if len(selected) < r:
            leftovers = np.flatnonzero(remaining)
            if leftovers.size > 0:
                need = min(r - len(selected), int(leftovers.size))
                filler = leftovers[np.argsort(-net_scores[leftovers], kind="stable")[:need]]
                selected.extend([int(i) for i in filler])
        return _build_action_vector(num_sensors, np.asarray(sorted(selected), dtype=np.int32))

    add_action(np.zeros((num_sensors,), dtype=np.int8))

    for r in sizes:
        if r <= 0:
            continue
        top_idx = np.argsort(-scores, kind="stable")[:r]
        add_action(_build_action_vector(num_sensors, top_idx))

        top_net_idx = np.argsort(-net_scores, kind="stable")[:r]
        add_action(_build_action_vector(num_sensors, top_net_idx))
        add_action(greedy_action_of_size(r))

        for _ in range(2):
            rnd_idx = np.sort(rng.choice(num_sensors, size=r, replace=False))
            add_action(_build_action_vector(num_sensors, rnd_idx))

    # Add a richer bank of low-cardinality candidates so high-lambda regimes
    # can still discover good 1/2-sensor solutions.
    top_small = min(num_sensors, 10)
    sensor_order = np.argsort(-net_scores, kind="stable")[:top_small]
    if k_eff >= 1:
        for idx in sensor_order:
            add_action(
                _build_action_vector(num_sensors, np.asarray([idx], dtype=np.int32))
            )

    pair_pool = min(top_small, 8)
    pair_candidates: list[tuple[float, np.ndarray]] = []
    if k_eff >= 2 and pair_pool >= 2:
        top_pair_sensors = sensor_order[:pair_pool]
        for i in range(pair_pool):
            si = int(top_pair_sensors[i])
            for j in range(i + 1, pair_pool):
                sj = int(top_pair_sensors[j])
                pair_idx = np.asarray([si, sj], dtype=np.int32)
                a = _build_action_vector(num_sensors, pair_idx)
                pair_candidates.append((action_rank(a), a))
        pair_candidates.sort(key=lambda t: t[0], reverse=True)
        for _, a in pair_candidates[: min(12, len(pair_candidates))]:
            add_action(a)

    if k_eff > 0:
        positive = np.flatnonzero(net_scores > 0.0)
        if positive.size > 0:
            if positive.size > k_eff:
                keep = positive[np.argsort(-net_scores[positive], kind="stable")[:k_eff]]
            else:
                keep = positive
            add_action(_build_action_vector(num_sensors, np.sort(keep)))

    ranked = sorted(actions, key=action_rank, reverse=True)
    if len(ranked) <= max_actions:
        return ranked

    # Preserve cardinality diversity with round-robin truncation, otherwise
    # progressive widening can collapse to a narrow size band.
    by_size: dict[int, list[np.ndarray]] = {}
    for a in ranked:
        s = int(np.sum(a))
        by_size.setdefault(s, []).append(a)

    # Include a broad range of sizes first; then fill by score.
    base_sizes = [s for s in (0, 1, 2, 3, 4, 5, 6, 8, k_eff) if s in by_size]
    extra_sizes = [s for s in sorted(by_size.keys()) if s not in base_sizes]
    size_order = base_sizes + extra_sizes

    kept: list[np.ndarray] = []
    kept_keys: set[bytes] = set()
    ptr = {s: 0 for s in size_order}
    while len(kept) < max_actions:
        progressed = False
        for s in size_order:
            i = ptr[s]
            if i >= len(by_size[s]):
                continue
            a = by_size[s][i]
            ptr[s] += 1
            key = a.tobytes()
            if key in kept_keys:
                continue
            kept.append(a)
            kept_keys.add(key)
            progressed = True
            if len(kept) >= max_actions:
                break
        if not progressed:
            break
    return kept[:max_actions]


@dataclass
class _MCTSNode:
    depth: int
    partitions: list[np.ndarray]
    predicted_state: np.ndarray
    candidate_actions: list[np.ndarray] | None = None
    next_expand_idx: int = 0
    visits: int = 0
    value_sum: float = 0.0
    children: dict[int, "_MCTSNode"] = field(default_factory=dict)
    edge_reward: dict[int, float] = field(default_factory=dict)


def _step_from_action(
    *,
    partitions: list[np.ndarray],
    action: np.ndarray,
    coverage_matrix: np.ndarray,
    p_modes: np.ndarray,
    lambda_energy: float | None,
    sensor_energy_costs: np.ndarray,
) -> tuple[list[np.ndarray], float]:
    active = np.flatnonzero(action > 0)
    if active.size > 0:
        visible_mask = np.any(coverage_matrix[active, :], axis=0)
    else:
        visible_mask = np.zeros((coverage_matrix.shape[1],), dtype=bool)

    child_partitions, num_occ_partitions = _apply_action_oce_fast(
        partitions=partitions,
        p_modes=p_modes,
        visible_mask=visible_mask,
    )

    entropy_cost = 0.0
    if num_occ_partitions > 0:
        for p in child_partitions[:num_occ_partitions]:
            # OCE state entropy: marginalize over latent mode/class first.
            state_marginal = np.sum(p, axis=0)
            mass = float(np.sum(state_marginal))
            if mass > EPS:
                entropy_cost += mass * _entropy(state_marginal / mass)

    energy_cost = (
        float(lambda_energy) * float(np.sum(sensor_energy_costs[active]))
        if lambda_energy is not None
        else 0.0
    )
    reward = -float(entropy_cost + energy_cost)
    return child_partitions, reward


def _uct_score(parent_visits: int, child: _MCTSNode, exploration: float) -> float:
    if child.visits <= 0:
        return float("inf")
    mean = child.value_sum / float(child.visits)
    bonus = exploration * math.sqrt(
        math.log(max(1.0, float(parent_visits))) / float(child.visits)
    )
    return mean + bonus


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
    MCTS-OCE selector with action-only branching and OCE partition propagation.
    """
    del step_index
    del initial_true_state

    num_sensors = int(coverage_matrix.shape[0])
    num_states = int(coverage_matrix.shape[1])
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
    root_partitions = [alpha0]
    root_predicted_state = _predict_state_marginal_from_partitions(
        partitions=root_partitions,
        p_modes=p_modes,
        num_states=num_states,
    )

    horizon = (
        int(planning_horizon) if planning_horizon is not None else DEFAULT_MAX_HORIZON
    )
    horizon = max(1, horizon)
    max_actions = (
        int(mcts_max_actions) if mcts_max_actions is not None else DEFAULT_MAX_ACTIONS
    )
    max_actions = max(1, max_actions)
    if mcts_iterations is None:
        iterations = int(max(64, min(512, 24 * horizon)))
    else:
        iterations = max(1, int(mcts_iterations))
    exploration = math.sqrt(2.0)
    coverage_bool = np.asarray(coverage_matrix, dtype=bool)

    root = _MCTSNode(
        depth=0,
        partitions=root_partitions,
        predicted_state=root_predicted_state,
    )

    for _ in range(iterations):
        node = root
        depth = int(node.depth)
        trajectory: list[tuple[_MCTSNode, float]] = []

        while depth < horizon:
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
                break

            child_limit = _progressive_widening_limit(
                node.visits, num_candidates=num_candidates
            )
            can_expand = (
                len(node.children) < child_limit
                and node.next_expand_idx < num_candidates
            )
            if can_expand:
                action_idx = int(node.next_expand_idx)
                node.next_expand_idx += 1
                action = node.candidate_actions[action_idx]
                child_partitions, edge_reward = _step_from_action(
                    partitions=node.partitions,
                    action=action,
                    coverage_matrix=coverage_matrix,
                    p_modes=p_modes,
                    lambda_energy=lambda_energy,
                    sensor_energy_costs=costs,
                )
                child_predicted_state = _predict_state_marginal_from_partitions(
                    partitions=child_partitions,
                    p_modes=p_modes,
                    num_states=num_states,
                )
                child = _MCTSNode(
                    depth=depth + 1,
                    partitions=child_partitions,
                    predicted_state=child_predicted_state,
                )
                node.children[action_idx] = child
                node.edge_reward[action_idx] = edge_reward
                trajectory.append((node, edge_reward))
                node = child
            else:
                if not node.children:
                    break

                best_action_idx = max(
                    node.children.keys(),
                    key=lambda idx: _uct_score(
                        node.visits, node.children[idx], exploration
                    ),
                )
                trajectory.append((node, node.edge_reward[best_action_idx]))
                node = node.children[best_action_idx]

            depth += 1

        # No random rollout: OCE partitions encode the node's expected
        # information state, so we treat this as a forward-search backup.
        g = 0.0
        node.visits += 1
        node.value_sum += g
        for parent, edge_reward in reversed(trajectory):
            g += edge_reward
            parent.visits += 1
            parent.value_sum += g

    if not root.children:
        root_actions = _generate_candidate_actions(
            coverage_matrix=coverage_bool,
            predicted_state=root.predicted_state,
            max_active_sensors=k,
            rng=rng,
            lambda_energy=lambda_energy,
            sensor_energy_costs=costs,
            max_actions=max_actions,
        )
        return root_actions[0] if root_actions else zero_action

    best_action_idx = max(
        root.children.keys(),
        key=lambda idx: (
            (
                (root.children[idx].value_sum / float(root.children[idx].visits))
                if root.children[idx].visits > 0
                else -np.inf
            ),
            root.children[idx].visits,
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

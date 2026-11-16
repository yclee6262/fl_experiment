"""Explicit budgeted coalition objective with exact and greedy solvers."""

from itertools import combinations

import numpy as np


def residual_dissimilarity(left, right, eps=1e-8):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    denominator = np.linalg.norm(left) * np.linalg.norm(right) + eps
    similarity = abs(float(np.dot(left, right) / denominator))
    return max(0.0, 1.0 - similarity)


def coalition_value(indices, candidates, diversity_eta, max_size):
    """One explicit set objective shared by exact and greedy selection.

    Quality is the sum of normalized Stage 0 quality shares. Diversity is the
    additive pairwise residual dissimilarity normalized by the maximum number
    of pairs allowed by the coalition-size constraint.
    """
    indices = tuple(indices)
    quality = sum(float(candidates[index]["quality_share"]) for index in indices)
    pair_denominator = max(1, max_size * (max_size - 1) / 2)
    diversity_sum = sum(
        residual_dissimilarity(
            candidates[left]["residual"], candidates[right]["residual"]
        )
        for left, right in combinations(indices, 2)
    )
    diversity = diversity_sum / pair_denominator
    return quality + float(diversity_eta) * diversity


def feasible(indices, candidates, budget, max_size):
    return (
        0 < len(indices) <= max_size
        and sum(float(candidates[index]["bid"]) for index in indices)
        <= budget + 1e-12
    )


def exact_select(candidates, budget, max_size, diversity_eta):
    if len(candidates) > 20:
        raise ValueError("Exact selection is restricted to at most 20 candidates.")
    best = None
    for size in range(1, min(max_size, len(candidates)) + 1):
        for indices in combinations(range(len(candidates)), size):
            if not feasible(indices, candidates, budget, max_size):
                continue
            value = coalition_value(indices, candidates, diversity_eta, max_size)
            key = (value, -sum(candidates[i]["bid"] for i in indices),
                   tuple(-candidates[i]["agent_id"] for i in indices))
            if best is None or key > best[0]:
                best = (key, indices)
    if best is None:
        raise ValueError("No non-empty coalition is feasible under the budget.")
    return selection_report(best[1], candidates, budget, max_size, diversity_eta)


def greedy_select(candidates, budget, max_size, diversity_eta):
    selected = []
    while len(selected) < max_size:
        current_value = (
            coalition_value(selected, candidates, diversity_eta, max_size)
            if selected else 0.0
        )
        choices = []
        for index in range(len(candidates)):
            if index in selected:
                continue
            proposal = selected + [index]
            if not feasible(proposal, candidates, budget, max_size):
                continue
            value = coalition_value(proposal, candidates, diversity_eta, max_size)
            marginal = value - current_value
            choices.append((marginal, value, -candidates[index]["bid"],
                            -candidates[index]["agent_id"], index))
        if not choices:
            break
        selected.append(max(choices)[-1])
    if not selected:
        raise ValueError("No non-empty coalition is feasible under the budget.")
    return selection_report(selected, candidates, budget, max_size, diversity_eta)


def selection_report(indices, candidates, budget, max_size, diversity_eta):
    indices = tuple(sorted(indices))
    bid_sum = float(sum(candidates[index]["bid"] for index in indices))
    return {
        "indices": indices,
        "agent_ids": [int(candidates[index]["agent_id"]) for index in indices],
        "objective": float(coalition_value(
            indices, candidates, diversity_eta, max_size
        )),
        "bid_sum": bid_sum,
        "budget": float(budget),
        "budget_feasible": bid_sum <= budget + 1e-12,
        "max_size": int(max_size),
    }

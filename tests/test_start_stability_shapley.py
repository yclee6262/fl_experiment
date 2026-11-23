import itertools
import numpy as np
from experiments.contribution.start_stability.historical_shapley import (
    exact_shapley, permutation_shapley, allocation, compare)


def game(players, baseline=0.):
    values = {frozenset(): 0.}
    for size in range(1, len(players)+1):
        for subset in itertools.combinations(players, size):
            values[frozenset(subset)] = baseline + sum(i for i in subset) + (3 if {1, 2} <= set(subset) else 0)
    return values


def test_exact_formula_permutations_and_efficiency_for_interactions():
    players = [1, 2, 3]
    values = game(players)
    phi = exact_shapley(players, values)
    np.testing.assert_allclose(phi, [2.5, 3.5, 3.])
    np.testing.assert_allclose(phi, permutation_shapley(players, values), atol=1e-12)
    assert abs(phi.sum()-values[frozenset(players)]) < 1e-12


def test_empty_baseline_changes_shapley_but_not_full_loo():
    players = [1, 2, 3]
    initial, shifted = game(players), game(players, .3)
    np.testing.assert_allclose(exact_shapley(players, shifted)-exact_shapley(players, initial), .1)
    full = frozenset(players)
    for person in players:
        assert np.isclose(initial[full]-initial[full-{person}], shifted[full]-shifted[full-{person}])


def test_reserve_and_undefined_rank_are_preserved():
    np.testing.assert_array_equal(allocation([-1., 0., -2.]), [0., 0., 0., 1.])
    result = compare(np.ones(3), np.array([-1., 0., -2.]))
    assert result['spearman'] is None
    assert result['allocation_TV'] == 1.
    assert result['loo_reserved'] == 1

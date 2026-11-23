import numpy as np
from experiments.contribution.start_stability.engine import (
    exact_shapley, allocation, integer_pool, shadow_payment)
from experiments.contribution.start_stability.run import coefficients


def test_shapley_additive_and_unanimity_negative_game():
    values = [sum([2., -1., 4.][i] for i in range(3) if mask & (1 << i)) for mask in range(8)]
    np.testing.assert_allclose(exact_shapley(values, 3), [2., -1., 4.])
    np.testing.assert_allclose(exact_shapley([0.]*7+[-6.], 3), [-2., -2., -2.])


def test_solver_start_coefficients_are_fixed_finite_convex_combinations():
    c = dict(seed=2026101610, ids=[6, 8, 7])
    w = coefficients(c)
    assert w.shape == (3, 3) and np.min(w) >= 0
    np.testing.assert_allclose(w.sum(1), 1.)
    np.testing.assert_array_equal(coefficients(c), w)
    anchors = np.array([[1., -1.], [-.5, .5], [.1, -.2]])
    assert np.max(abs(w@anchors)) <= 1


def test_shadow_payment_cancellation_and_reserve_conserve_budget():
    parts = [dict(tauA=1e-8, scores={'F':dict(A=-1., M=-2.)}),
             dict(tauA=1e-8, scores={'F':dict(A=0., M=0.)}),
             dict(tauA=1e-8, scores={'F':dict(A=0., M=0.)})]
    p = shadow_payment(parts, [6, 8, 7], 'F')
    assert p['basic'] == [0, 100, 100] and p['reserved'] == 1100
    assert sum(p['total'])+p['reserved'] == p['budget'] == 1300
    assert sum(p['normalized_total']) == 1.
    pay, reserve = integer_pool(allocation([1., 1., 1.], 1e-10), 500, [6, 8, 7])
    assert pay == [167, 166, 167] and reserve == 0

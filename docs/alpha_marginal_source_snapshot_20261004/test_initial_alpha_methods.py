import numpy as np
import pytest

from initial_alpha_methods import METHODS, initial_alpha


class Agent:
    def __init__(self, agent_id, prediction):
        self.agent_id = agent_id
        self.prediction = prediction

    def api_predict(self, x):
        x = np.asarray(x)
        return np.full(len(x) if x.ndim == 2 else 1, self.prediction)


def inputs():
    agents = [Agent(1, 0.), Agent(2, 1.), Agent(3, 2.)]
    return agents, [0.01, .04, .16], np.zeros((20, 2)), np.zeros(20), np.zeros((3, 2))


@pytest.mark.parametrize("method", METHODS)
def test_weights_align_and_form_simplex(method):
    agents, mse, x, y, proposals = inputs()
    alpha, audit = initial_alpha(method, agents, mse, x, y, proposals, 0.)
    assert audit["agent_ids"] == [1, 2, 3]
    assert len(alpha) == 3 and np.all(alpha >= 0)
    assert alpha.sum() == pytest.approx(1.)
    if method in ("quality_shrink", "ensemble_shrink", "target_peer_shrink"):
        assert np.min(alpha) >= .5 / 3 - 1e-10


def test_quality_shrink_is_exact_predeclared_blend():
    agents, mse, x, y, proposals = inputs()
    alpha, _ = initial_alpha("quality_shrink", agents, mse, x, y, proposals, 0.)
    quality = 1 / (np.asarray(mse) + 1e-5)
    quality /= quality.sum()
    np.testing.assert_allclose(alpha, .5 / 3 + .5 * quality)


def test_peer_invalid_proposals_trigger_explicit_fallback():
    agents, mse, x, y, proposals = inputs()
    alpha, audit = initial_alpha("target_peer_shrink", agents, mse, x, y,
                                 np.full_like(proposals, 2.), 0.)
    np.testing.assert_allclose(alpha, np.ones(3) / 3)
    assert audit["fallback_reason"] == "no_valid_peer_predictions"


def test_invalid_mse_and_misalignment():
    agents, mse, x, y, proposals = inputs()
    alpha, audit = initial_alpha("B_stage0", agents, [np.nan, .04, .16], x, y, proposals, 0.)
    np.testing.assert_allclose(alpha, np.ones(3) / 3)
    assert audit["fallback_reason"] == "invalid_stage0_mse"
    with pytest.raises(ValueError, match="align"):
        initial_alpha("B_stage0", agents, mse[:2], x, y, proposals, 0.)


def test_one_participant():
    alpha, audit = initial_alpha("target_peer_shrink", [Agent(7, .1)], [.01],
                                 np.zeros((2, 2)), np.zeros(2), np.zeros((1, 2)), 0.)
    np.testing.assert_array_equal(alpha, [1.])
    assert audit["fallback_reason"] == "single_participant"


def test_target_local_precision_uses_near_target_blind_error_and_prior():
    agents = [Agent(1, 0.), Agent(2, 1.)]
    x = np.zeros((4, 2))
    y = np.asarray([0., 0., 1., 1.])
    mse = np.asarray([.5, .5])
    alpha, audit = initial_alpha("target_local_precision", agents, mse, x, y,
                                 np.zeros((2, 2)), 0.)
    kernel = np.exp(-np.abs(y) / .5)
    residual = np.column_stack([-y, 1-y])
    local = ((residual**2 * kernel[:, None]).sum(axis=0) + 5*mse) / (kernel.sum()+5)
    scores = 1/(local+1e-5)
    np.testing.assert_allclose(alpha, scores/scores.sum())
    assert alpha[0] > alpha[1]
    assert audit["target_kernel_mass"] == pytest.approx(float(kernel.sum()))


def test_best_agent_uses_stage0_winner_only():
    agents, _, x, y, proposals = inputs()
    alpha, audit = initial_alpha("best_agent", agents, [.04, .01, .16],
                                 x, y, proposals, 0.)
    np.testing.assert_array_equal(alpha, [0., 1., 0.])
    assert audit["winner_agent_id"] == 2


def test_local_bias_risk_matches_ridge_bias_plus_residual_variance():
    agents, mse, x, y, proposals = inputs()
    alpha, audit = initial_alpha("local_bias_risk", agents, mse, x, y,
                                 proposals, 0.)
    prediction = np.column_stack([agent.api_predict(x) for agent in agents])
    residual = prediction - y[:, None]
    design = np.column_stack([np.ones(len(x)), x])
    coefficients = np.linalg.solve(design.T @ design + np.diag([.01, 1., 1.]),
                                   design.T @ residual)
    bias = np.r_[1., proposals.mean(axis=0)] @ coefficients
    variance = np.mean((residual - design @ coefficients)**2, axis=0)
    scores = 1/(bias**2 + variance + 1e-5)
    np.testing.assert_allclose(alpha, scores/scores.sum())
    np.testing.assert_allclose(audit["estimated_local_bias"], bias)
    np.testing.assert_allclose(audit["estimated_residual_variance"], variance)


def test_marginal_information_weights_positive_blind_leave_one_out_gain():
    agents, mse, x, y, proposals = inputs()
    alpha, audit = initial_alpha("marginal_information_50", agents, mse, x, y,
                                 proposals, 0.)
    quality = 1 / (np.asarray(mse) + 1e-5)
    quality /= quality.sum()
    predictions = np.column_stack([agent.api_predict(x) for agent in agents])
    full_loss = np.mean((predictions @ quality-y)**2)
    gains = []
    for index in range(len(agents)):
        remaining = quality.copy()
        remaining[index] = 0
        remaining /= remaining.sum()
        gains.append(np.mean((predictions @ remaining-y)**2)-full_loss)
    positive = np.maximum(gains, 0)
    expected = .5*quality + .5*positive/positive.sum()
    np.testing.assert_allclose(alpha, expected)
    np.testing.assert_allclose(audit["blind_leave_one_out_gain"], gains)
    assert alpha[0] > quality[0]


def test_marginal_information_has_explicit_no_positive_gain_fallback():
    agents = [Agent(1, 1.), Agent(2, 1.)]
    x, y, proposals = np.zeros((20, 2)), np.ones(20), np.zeros((2, 2))
    alpha, audit = initial_alpha("marginal_information_50", agents, [.02, .03],
                                 x, y, proposals, 0.)
    quality = 1/(np.asarray([.02, .03])+1e-5)
    np.testing.assert_allclose(alpha, quality/quality.sum())
    assert audit["fallback_reason"] == "no_positive_marginal_gain_use_stage0"

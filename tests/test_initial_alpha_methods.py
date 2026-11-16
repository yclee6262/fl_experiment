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


def test_bootstrap_reuses_qualification_predictions_without_changing_alpha():
    agents, mse, x, y, proposals = inputs()
    predictions = np.column_stack([a.api_predict(x) for a in agents])
    fresh, fresh_audit = initial_alpha("bootstrap_quality", agents, mse, x, y,
                                      proposals, 0., bootstrap_seed=7)
    for agent in agents:
        def no_api(_):
            raise AssertionError("Qualification reuse must not query again")
        agent.api_predict = no_api
    reused, reused_audit = initial_alpha("bootstrap_quality", agents, mse, x, y,
        proposals, 0., bootstrap_seed=7, cached_blind_predictions=predictions)
    np.testing.assert_array_equal(fresh, reused)
    assert reused_audit["blind_prediction_source"] == "qualification_reuse"
    assert fresh_audit["blind_prediction_source"] == "fresh_api"
    with pytest.raises(ValueError, match="align"):
        initial_alpha("bootstrap_quality", agents, mse, x, y, proposals, 0.,
                      cached_blind_predictions=predictions[:, :2])


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
    np.testing.assert_allclose(np.asarray(audit["blind_row_leave_one_out_gain"]).mean(axis=0),
                               gains)
    assert alpha[0] > quality[0]


def test_marginal_information_has_explicit_no_positive_gain_fallback():
    agents = [Agent(1, 1.), Agent(2, 1.)]
    x, y, proposals = np.zeros((20, 2)), np.ones(20), np.zeros((2, 2))
    alpha, audit = initial_alpha("marginal_information_50", agents, [.02, .03],
                                 x, y, proposals, 0.)
    quality = 1/(np.asarray([.02, .03])+1e-5)
    np.testing.assert_allclose(alpha, quality/quality.sum())
    assert audit["fallback_reason"] == "no_positive_marginal_gain_use_stage0"


def test_target_marginal_blend_preserves_both_stage0_components():
    agents, mse, x, y, proposals = inputs()
    target, _ = initial_alpha("target_local_precision", agents, mse, x, y,
                              proposals, 0.)
    marginal, _ = initial_alpha("marginal_information_50", agents, mse, x, y,
                                proposals, 0.)
    combined, audit = initial_alpha("target_marginal_blend_50", agents, mse,
                                    x, y, proposals, 0.)
    np.testing.assert_allclose(combined, .5*target+.5*marginal)
    np.testing.assert_allclose(audit["target_local_component"], target)
    np.testing.assert_allclose(audit["marginal_component"], marginal)


def test_bootstrap_constant_errors_equal_original_quality_and_query_once_per_model():
    class CountedAgent(Agent):
        calls = 0

        def api_predict(self, x):
            self.calls += 1
            return super().api_predict(x)

    agents = [CountedAgent(7, .1), CountedAgent(3, .2), CountedAgent(9, .4)]
    x, y = np.zeros((20, 2)), np.zeros(20)
    alpha, audit = initial_alpha("bootstrap_quality", agents, [.01, .04, .16],
                                 x, y, np.zeros((3, 2)), 0.)
    expected = 1 / (np.asarray([.01, .04, .16]) + 1e-5)
    np.testing.assert_allclose(alpha, expected / expected.sum())
    np.testing.assert_allclose(audit["bootstrap_weight_std"], 0, atol=1e-13)
    assert [a.calls for a in agents] == [1, 1, 1]


def test_bootstrap_varies_on_heterogeneous_errors_and_is_reproducible_and_aligned():
    agents = [Agent(1, 0.), Agent(2, 1.)]
    x, y = np.zeros((20, 2)), np.r_[np.zeros(15), np.ones(5)]
    kwargs = dict(bootstrap_seed=19, bootstrap_replicates=1000)
    state = np.random.get_state()
    alpha, audit = initial_alpha("bootstrap_quality", agents, [.25, .75],
                                 x, y, np.zeros((2, 2)), 0., **kwargs)
    repeated, _ = initial_alpha("bootstrap_quality", agents, [.25, .75],
                                x, y, np.zeros((2, 2)), 2., **kwargs)
    reversed_alpha, _ = initial_alpha("bootstrap_quality", agents[::-1], [.75, .25],
                                      x, y, np.zeros((2, 2)), 0., **kwargs)
    np.testing.assert_array_equal(alpha, repeated)
    np.testing.assert_allclose(alpha, reversed_alpha[::-1])
    assert all(v > 0 for v in audit["bootstrap_weight_std"])
    assert alpha[0] > alpha[1]
    np.testing.assert_array_equal(np.random.get_state()[1], state[1])
    assert np.random.get_state()[2:] == state[2:]


def test_exponential_weights_temperature_limits_ties_and_numerical_stability():
    agents, _, x, y, proposals = inputs()
    uniform, audit = initial_alpha("exponential_quality", agents, [.2, .2, .2],
                                   x, y, proposals, 0.)
    np.testing.assert_allclose(uniform, np.ones(3)/3)
    assert audit["temperature"] == pytest.approx(.2)
    cold, _ = initial_alpha("exponential_quality", agents, [.01, .04, .16],
                            x, y, proposals, 0., exponential_temperature_ratio=.01)
    hot, _ = initial_alpha("exponential_quality", agents, [.01, .04, .16],
                           x, y, proposals, 0., exponential_temperature_ratio=1000)
    assert cold[0] > .999
    np.testing.assert_allclose(hot, np.ones(3)/3, atol=.002)
    extreme, _ = initial_alpha("exponential_quality", agents, [1e9, 1e9, 1e9],
                               x, y, proposals, 0.)
    np.testing.assert_allclose(extreme, np.ones(3)/3)


def test_new_weight_methods_invalid_config_and_prediction_fallback():
    agents, mse, x, y, proposals = inputs()
    with pytest.raises(ValueError, match="two replicates"):
        initial_alpha("bootstrap_quality", agents, mse, x, y, proposals, 0.,
                      bootstrap_replicates=1)
    with pytest.raises(ValueError, match="positive"):
        initial_alpha("exponential_quality", agents, mse, x, y, proposals, 0.,
                      exponential_temperature_ratio=0)
    alpha, audit = initial_alpha("bootstrap_quality", [Agent(1, np.nan), agents[1]],
                                 [.01, .04], x, y, proposals[:2], 0.)
    expected = 1/(np.asarray([.01, .04])+1e-5)
    np.testing.assert_allclose(alpha, expected/expected.sum())
    assert audit["fallback_reason"] == "invalid_blind_predictions_use_stage0"


def test_bootstrap_cache_must_preserve_qualification_batch_shape():
    class BatchSensitive:
        def __init__(self,ident):self.agent_id=ident
        def api_predict(self,x):
            # Reproduce shape-dependent rounding with distinct engine effects.
            return np.asarray(x)[:,0]+self.agent_id*.01+len(x)*self.agent_id*1e-7
    agents=[BatchSensitive(1),BatchSensitive(2),BatchSensitive(3)]
    x=np.arange(200,dtype=float).reshape(100,2)/200
    y=np.zeros(20)
    prediction20=np.column_stack([a.api_predict(x[:20]) for a in agents])
    prediction100=np.column_stack([a.api_predict(x) for a in agents])
    mse=((prediction20-y[:,None])**2).mean(axis=0)
    kwargs=dict(bootstrap_seed=2,bootstrap_replicates=1000)
    fresh,_=initial_alpha('bootstrap_quality',agents,mse,x[:20],y,np.zeros((3,2)),0.,**kwargs)
    matched,_=initial_alpha('bootstrap_quality',agents,mse,x[:20],y,np.zeros((3,2)),0.,
                          cached_blind_predictions=prediction20,**kwargs)
    wrong,_=initial_alpha('bootstrap_quality',agents,mse,x[:20],y,np.zeros((3,2)),0.,
                        cached_blind_predictions=prediction100[:20],**kwargs)
    np.testing.assert_array_equal(fresh,matched)
    assert not np.array_equal(fresh,wrong)

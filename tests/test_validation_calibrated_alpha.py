import json
import numpy as np
import pytest
from types import SimpleNamespace

from run_validation_calibrated_alpha import _meta_result, TRIO_CANDIDATES, TRIO_METHOD
from validation_calibrated_alpha import (fit_residual_calibrator,
                                         select_candidate,
                                         select_candidate_local_bias,
                                         diverse_four_methods, CANDIDATE_METHODS,
                                         conditional_target_trigger,
                                         select_candidate_jackknife_q90)


def test_conditional_trigger_uses_internal_error_relative_to_blind_residual_only():
    scores = {"A_uniform": {"internal_target_error": .02},
              "B_stage0": {"internal_target_error": .04}}
    trigger, audit = conditional_target_trigger(scores, .02**2)
    assert trigger is True
    assert audit["ratio"] == pytest.approx(1.0)
    trigger, audit = conditional_target_trigger(scores, .04**2)
    assert trigger is False
    assert audit["ratio"] == pytest.approx(.5)
    with pytest.raises(ValueError, match="Invalid"):
        conditional_target_trigger(scores, float("nan"))


class Agent:
    def __init__(self, agent_id, scale, offset):
        self.agent_id = agent_id
        self.scale = scale
        self.offset = offset

    def api_predict(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            return np.asarray([self.scale*x[0]+self.offset])
        return self.scale*x[:,0]+self.offset


def test_calibrated_selection_uses_only_supplied_blind_labels_and_predictions():
    agents = [Agent(1, 1., .1), Agent(2, 1., .2)]
    x = np.column_stack([np.linspace(-1, 1, 20), np.zeros(20)])
    y = x[:,0]
    alpha = [0.5,0.5]
    coefficients, audit = fit_residual_calibrator(agents, x, y, alpha)
    assert audit["calibration_samples"] == 20
    assert coefficients[0] == pytest.approx(.15, abs=.005)
    solutions = {"A_uniform":[0.,0.], "B_stage0":[.5,0.],
                 "mse_power_1p5":None, "corr_precision_0p5":None,
                 "target_local_precision":[2.,0.]}
    chosen, scores = select_candidate(agents, alpha, coefficients, solutions, 0.)
    assert chosen == "A_uniform"
    assert "target_local_precision" not in scores  # box violation cannot be selected


def test_selector_requires_a_valid_candidate():
    agents = [Agent(1,1.,0.)]
    with pytest.raises(ValueError, match="No feasible"):
        select_candidate(agents,[1.],[0.,0.,0.],{},0.)


def test_jackknife_selector_uses_stage0_data_and_rejects_invalid_solution():
    agents = [Agent(1, 1., .1), Agent(2, 1., .2)]
    x = np.column_stack([np.linspace(-1, 1, 20), np.zeros(20)])
    y = x[:, 0]
    solutions = {"A_uniform": [0., 0.], "B_stage0": [.5, 0.],
                 "target_local_precision": [2., 0.]}
    chosen, scores = select_candidate_jackknife_q90(
        agents, [.5, .5], x, y, solutions, 0.)
    assert chosen == "A_uniform"
    assert "target_local_precision" not in scores
    assert scores["A_uniform"]["jackknife_refits"] == 20
    with pytest.raises(ValueError, match="No feasible"):
        select_candidate_jackknife_q90(agents, [.5, .5], x, y, {}, 0.)


def test_local_selector_uses_blind_residuals_and_rejects_box_violations():
    agents = [Agent(1, 1., .1), Agent(2, 1., .2)]
    x = np.column_stack([np.linspace(-1, 1, 20), np.zeros(20)])
    solutions = {"A_uniform": [0., 0.], "B_stage0": [.5, 0.],
                 "mse_power_1p5": None, "corr_precision_0p5": None,
                 "target_local_precision": [2., 0.]}
    chosen, scores = select_candidate_local_bias(
        agents, [.5, .5], x, x[:, 0], solutions, 0.)
    assert chosen == "A_uniform"
    assert "target_local_precision" not in scores
    assert scores["A_uniform"]["local_kernel_mass"] > 0
    with pytest.raises(ValueError, match="No feasible"):
        select_candidate_local_bias(agents, [.5, .5], x, x[:, 0], {}, 0.)


def test_selector_can_score_specialist_option_without_changing_five_candidate_order():
    agents = [Agent(1, 1., .1), Agent(2, 1., .2)]
    x = np.column_stack([np.linspace(-1, 1, 20), np.zeros(20)])
    coefficients, _ = fit_residual_calibrator(agents, x, x[:, 0], [.5, .5])
    solutions = {"A_uniform": [.5, 0.], "B_stage0": [.4, 0.],
                 "best_agent": [0., 0.]}
    chosen, scores = select_candidate(agents, [.5, .5], coefficients, solutions,
                                      0., method_order=("A_uniform", "B_stage0", "best_agent"))
    assert chosen == "best_agent"
    assert set(scores) == set(solutions)


@pytest.mark.parametrize("selector_rule", ["global", "local_bias"])
def test_meta_selection_does_not_use_saved_pooled_scores(selector_rule):
    methods = ("A_uniform", "B_stage0", "mse_power_1p5",
               "corr_precision_0p5", "target_local_precision")
    x = np.column_stack([np.linspace(-1, 1, 20), np.zeros(20)])
    def run(external_errors):
        agents = [Agent(1, 1., .1), Agent(2, 1., .2)]
        base = SimpleNamespace(trusted_agents=agents, test_X=x, test_y=x[:, 0],
                               target_T=0., n_features=2, feasible_lower=-1.,
                               feasible_upper=1., feasibility_tolerance=1e-8)
        candidates = {}
        for index, method in enumerate(methods):
            candidates[method] = {
                "status": "valid", "delivery_status": "valid",
                "initial_alpha": "[0.5, 0.5]", "final_solution": f"[{index/10}, 0]",
                "external_global_error": external_errors[index],
                "initialization_requests": 0, "initialization_input_samples": 0,
                "initialization_seconds": 0., "stage3_requests": 1,
                "stage3_input_samples": 1, "stage3_seconds": 0.,
            }
        return _meta_result(candidates, base, "artifacts/unused", 0,
                            selector_rule=selector_rule)

    first = run([0., 1., 2., 3., 4.])
    second = run([400., 300., 200., 100., 0.])
    assert first["initial_alpha"] == second["initial_alpha"]
    assert first["final_solution"] == second["final_solution"]
    assert first["alpha_audit"] == second["alpha_audit"]
    assert first["initialization_requests"] == second["initialization_requests"] == 12


def test_three_candidate_selector_charges_only_three_solves_and_ignores_pooled_scores():
    x = np.column_stack([np.linspace(-1, 1, 20), np.zeros(20)])
    def run(external_errors):
        base = SimpleNamespace(trusted_agents=[Agent(1, 1., .1), Agent(2, 1., .2)],
                               test_X=x, test_y=x[:, 0], target_T=0., n_features=2,
                               feasible_lower=-1., feasible_upper=1.,
                               feasibility_tolerance=1e-8)
        candidates = {method: {
            "status": "valid", "delivery_status": "valid",
            "initial_alpha": "[0.5, 0.5]", "final_solution": f"[{index/10}, 0]",
            "external_global_error": external_errors[index],
            "initialization_requests": 1, "initialization_input_samples": 20,
            "initialization_seconds": 0., "stage3_requests": 10,
            "stage3_input_samples": 10, "stage3_seconds": 0.,
        } for index, method in enumerate(TRIO_CANDIDATES)}
        return _meta_result(candidates, base, "unused", 0,
                            method_order=TRIO_CANDIDATES,
                            method_name=TRIO_METHOD)
    first = run([0., 1., 2.])
    second = run([2., 1., 0.])
    assert first["final_solution"] == second["final_solution"]
    assert json.loads(first["alpha_audit"])["candidate_method_order"] == list(TRIO_CANDIDATES)
    assert first["stage3_requests"] == second["stage3_requests"] == 30
    assert first["initialization_requests"] == second["initialization_requests"] == 11


def test_diverse_portfolio_charges_omitted_alpha_initialization():
    agents = [Agent(1, 1., .1), Agent(2, 1., .2)]
    x = np.column_stack([np.linspace(-1, 1, 20), np.zeros(20)])
    base = SimpleNamespace(trusted_agents=agents, test_X=x, test_y=x[:, 0],
                           target_T=0., n_features=2, feasible_lower=-1.,
                           feasible_upper=1., feasibility_tolerance=1e-8)
    candidates = {}
    for index, method in enumerate(CANDIDATE_METHODS):
        candidates[method] = {
            "status": "valid", "delivery_status": "valid",
            "initial_alpha": "[0.5, 0.5]", "final_solution": f"[{index/10}, 0]",
            "external_global_error": 0.,
            "initialization_requests": 1, "initialization_input_samples": 20,
            "initialization_seconds": 0., "stage3_requests": 1,
            "stage3_input_samples": 1, "stage3_seconds": 0.,
        }
    subset = CANDIDATE_METHODS[:4]
    control = _meta_result(candidates, base, "unused", 0, method_order=subset)
    diverse = _meta_result(candidates, base, "unused", 0, method_order=subset,
                           charge_all_initializations=True)
    assert diverse["initialization_requests"] == control["initialization_requests"] + 1
    assert diverse["initialization_input_samples"] == control["initialization_input_samples"] + 20
    assert diverse["stage3_requests"] == control["stage3_requests"] == 4
    assert json.loads(diverse["alpha_audit"])["candidate_method_order"] == list(subset)


def test_diverse_portfolio_keeps_ex_ante_anchors_and_four_options():
    values = {
        "A_uniform": [1/3, 1/3, 1/3],
        "B_stage0": [.8, .1, .1],
        "mse_power_1p5": [.95, .025, .025],
        "corr_precision_0p5": [.1, .8, .1],
        "target_local_precision": [.8, .15, .05],
    }
    chosen = diverse_four_methods(values)
    assert len(chosen) == 4
    assert chosen[:2] == ("A_uniform", "B_stage0")
    assert "corr_precision_0p5" in chosen
    with pytest.raises(ValueError, match="valid and aligned"):
        diverse_four_methods({**values, "A_uniform": [float("nan"), .5, .5]})

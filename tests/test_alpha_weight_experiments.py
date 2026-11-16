import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from alpha_weight_experiments import (
    apply_exploration_floor,
    agent_data_controls,
    calibrate_one_condition,
    calibrate_stabilized_condition,
    contribution_distribution,
    comparison_conditions,
    configure_controlled_coalition,
    initialize_optimization_weights,
    raw_contribution_distribution,
    stage3_prediction_cache,
    settle_with_separated_weights,
    box_feasible,
    build_parser,
    run,
    write_run_manifest,
)
from agent_client import AgentNode
from host_server import HostServer, payment_contributions
from dataset import generate_agent_datasets
from models import PretrainedModel
from research_evaluation import (
    load_evaluator_artifact,
    save_scenario_artifacts,
    score_solution,
)
from analyze_p2_results import interpret_comparison
from p3_contribution_validation import full_participant_removal
from recruitment_selection import (
    coalition_value,
    exact_select,
    greedy_select,
)


class FakeAgent:
    def __init__(self, agent_id, prediction, bid=1.0):
        self.agent_id = agent_id
        self.prediction = float(prediction)
        self.bid = float(bid)

    def api_predict(self, _):
        return np.asarray([self.prediction])

    def get_minimum_bid(self):
        return self.bid

    def infer_parameters_D(self, _target, steps=500):
        return np.zeros(1)


class ControlledFakeAgent(FakeAgent):
    def api_predict(self, X):
        X = np.asarray(X)
        return np.zeros(len(X) if X.ndim > 1 else 1)


class GuardedCalibrationServer:
    """Minimal server that makes the last guarded proposal fail L_eval."""

    def __init__(self):
        self.target_T = 0.0
        self.total_budget = 10.0
        self.trusted_agents = [FakeAgent(1, 0.0), FakeAgent(2, 0.0)]
        self.alphas = [0.5, 0.5]
        self.last_subspace_optimization = None

    @staticmethod
    def _agent_bid(agent):
        return agent.get_minimum_bid()

    def _compute_exclusion_reports(self, _solution, **_kwargs):
        first_alpha = float(self.alphas[0])
        if np.isclose(first_alpha, 0.5):
            evaluation_loss = 1.0
            positive_contributions = (1.0, 0.0)
        elif np.isclose(first_alpha, 0.75):
            evaluation_loss = 0.5
            positive_contributions = (1.0, 0.0)
        else:
            evaluation_loss = 2.0
            positive_contributions = (0.0, 1.0)
        reports = [
            {
                "agent_id": 1,
                "marginal_contribution": positive_contributions[0],
                "positive_contribution": positive_contributions[0],
                "restricted_optimization_loss": evaluation_loss + 1.0,
                "restricted_engine": "custom",
                "restricted_engine_losses": {"custom": evaluation_loss + 1.0},
            },
            {
                "agent_id": 2,
                "marginal_contribution": positive_contributions[1],
                "positive_contribution": positive_contributions[1],
                "restricted_optimization_loss": evaluation_loss,
                "restricted_engine": "custom",
                "restricted_engine_losses": {"custom": evaluation_loss},
            },
        ]
        summary = {
            "delivered_solution_eval_loss": evaluation_loss,
            "full_solution_source": "stage3_l_opt",
            "loo_search_objective": "l_opt_current_alpha",
        }
        return evaluation_loss, reports, summary


class RecoveryCalibrationServer(GuardedCalibrationServer):
    """First full solve is dominated by a LOO solve; warm start fixes it."""

    def _compute_exclusion_reports(self, solution, **_kwargs):
        recovered = np.isclose(float(np.asarray(solution)[0]), 5.0)
        evaluation_loss = 0.5 if recovered else 1.0
        contributions = (1.0, 0.0) if recovered else (-0.2, -0.5)
        reports = []
        for index, (contribution, opt_loss, loo_solution) in enumerate(
            zip(contributions, (0.5, 1.5), (5.0, 6.0))
        ):
            reports.append(
                {
                    "index": index,
                    "agent": self.trusted_agents[index],
                    "agent_id": index + 1,
                    "marginal_contribution": contribution,
                    "positive_contribution": max(contribution, 0.0),
                    "restricted_optimization_loss": opt_loss,
                    "restricted_solution": np.asarray([loo_solution]),
                    "restricted_engine": "custom",
                    "restricted_engine_losses": {"custom": opt_loss},
                }
            )
        summary = {
            "delivered_solution_eval_loss": evaluation_loss,
            "full_solution_source": "stage3_l_opt",
            "loo_search_objective": "l_opt_current_alpha",
        }
        return evaluation_loss, reports, summary


class AlphaWeightExperimentTests(unittest.TestCase):
    def test_stage3_cache_uses_model_visible_float32_and_restores_api(self):
        agent = AgentNode(1, None, n_features=2)
        original = agent.api_predict
        calls = []

        def measured(values):
            calls.append(np.asarray(values).copy())
            return original(values)

        agent.api_predict = measured
        first_x = np.array([0.1, 0.2], dtype=np.float64)
        same_model_input = np.array([0.1 + 1e-10, 0.2], dtype=np.float64)
        self.assertTrue(np.array_equal(first_x.astype(np.float32),
                                       same_model_input.astype(np.float32)))
        with stage3_prediction_cache([agent]) as stats:
            first = agent.api_predict(first_x)
            second = agent.api_predict(same_model_input)
            self.assertTrue(np.array_equal(first, second))
            first[0] = 999.0
            self.assertNotEqual(float(agent.api_predict(first_x)[0]), 999.0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(stats, {"hits": 2, "misses": 1})
        self.assertIs(agent.api_predict, measured)
        agent.api_predict(first_x)
        self.assertEqual(len(calls), 2)
        with self.assertRaisesRegex(RuntimeError, "forced failure"):
            with stage3_prediction_cache([agent]):
                raise RuntimeError("forced failure")
        self.assertIs(agent.api_predict, measured)

    def test_payment_fallback_uses_opt_and_does_not_modify_eval_reports(self):
        reports = [
            {'marginal_contribution': -0.1, 'restricted_optimization_loss': 0.7},
            {'marginal_contribution': 0.0, 'restricted_optimization_loss': 0.3},
            {'marginal_contribution': -0.2, 'restricted_optimization_loss': 0.05},
        ]
        q, positives, source, reason = payment_contributions(reports, lambda: 0.1)
        np.testing.assert_allclose(q, [0.75, 0.25, 0])
        self.assertEqual(source, 'opt_fallback')
        self.assertEqual(reports[0]['marginal_contribution'], -0.1)

    def test_payment_primary_does_not_query_fallback(self):
        q, _, source, _ = payment_contributions(
            [{'marginal_contribution': 3}, {'marginal_contribution': 1}],
            lambda: self.fail('Primary payment must not evaluate fallback'),
        )
        np.testing.assert_allclose(q, [0.75, 0.25])
        self.assertEqual(source, 'eval')

    def test_payment_rejects_both_nonpositive_and_invalid_scores(self):
        reports = [{'marginal_contribution': 0, 'restricted_optimization_loss': 0.1}]
        q, _, source, _ = payment_contributions(reports, lambda: 0.2)
        self.assertIsNone(q)
        self.assertEqual(source, 'none')
        with self.assertRaises(ValueError):
            payment_contributions([{'marginal_contribution': float('nan')}], lambda: 0)

    def test_stage4_fallback_pays_only_remaining_positive_agents(self):
        server = HostServer(target_T=0, n_features=1, total_budget=10)
        server.trusted_agents = [FakeAgent(2, 0), FakeAgent(3, 0)]
        server.I_list = [np.zeros(1), np.ones(1)]
        server.alphas = [0.5, 0.5]
        reports = [dict(agent_id=i, alpha=0.5, bid=1.0,
                        marginal_contribution=-0.1, positive_contribution=0.0,
                        restricted_optimization_loss=loss)
                   for i, loss in [(2, 0.3), (3, 0.05)]]
        server._compute_exclusion_reports = lambda S: (0.2, reports)
        server._consensus_loss = lambda S, **kwargs: 0.1
        result = server.phase4_profit_sharing(np.zeros(1))
        self.assertEqual(result['payment_contribution_source'], 'opt_fallback')
        self.assertEqual(result['active_agent_ids'], [2])
        self.assertAlmostEqual(result['paid_total'], 10)

    def test_best_of_selects_the_lower_objective_engine(self):
        server = HostServer(target_T=0.0, n_features=1)
        server.trusted_agents = [FakeAgent(1, 0.0)]
        server.alphas = [1.0]
        server.I_list = [np.asarray([1.0])]

        server._solve_subspace_bfgs = lambda *args, **kwargs: {
            "solution": np.asarray([1.0]),
            "loss": 2.0,
            "history": [2.0],
            "states": ["bfgs"],
            "success": True,
            "message": "",
        }
        server._solve_subspace_custom = lambda *args, **kwargs: {
            "solution": np.asarray([0.0]),
            "loss": 1.0,
            "history": [1.0],
            "states": ["custom"],
            "success": True,
            "message": "",
        }

        result = server.optimize_candidate_subspace(strategy="best_of")
        self.assertEqual(result["chosen_engine"], "custom")
        self.assertEqual(result["engine_losses"], {"bfgs": 2.0, "custom": 1.0})

    def test_best_of_forwards_same_warm_start_to_both_engines(self):
        server = HostServer(target_T=0.0, n_features=1)
        server.trusted_agents = [FakeAgent(1, 0.0), FakeAgent(2, 0.0)]
        server.alphas = [0.5, 0.5]
        server.I_list = [np.asarray([1.0]), np.asarray([2.0])]
        seen = {}

        def fake_engine(name, loss):
            def solve(_candidates, _objective, **kwargs):
                seen[name] = np.asarray(kwargs["initial_solution"]).copy()
                return {
                    "solution": np.asarray([1.25]),
                    "loss": loss,
                    "history": [loss],
                    "states": [name],
                    "success": True,
                    "message": "",
                }
            return solve

        server._solve_subspace_bfgs = fake_engine("bfgs", 1.0)
        server._solve_subspace_custom = fake_engine("custom", 0.9)
        server.optimize_candidate_subspace(
            strategy="best_of", initial_solution=np.asarray([1.25])
        )
        np.testing.assert_allclose(seen["bfgs"], [1.25])
        np.testing.assert_allclose(seen["custom"], [1.25])

    def test_exclusion_reruns_l_opt_and_scores_delivered_solutions_with_l_eval(self):
        server = HostServer(target_T=0.0, n_features=1)
        server.trusted_agents = [
            FakeAgent(1, 0.0),
            FakeAgent(2, 0.0),
            FakeAgent(3, 0.0),
        ]
        server.alphas = [1.0 / 3.0] * 3
        server.I_list = [np.asarray([1.0]), np.asarray([2.0]), np.asarray([3.0])]
        calls = []

        server._consensus_loss = lambda S, **kwargs: float(np.asarray(S)[0])

        def fake_optimize(candidates, **kwargs):
            calls.append((len(candidates), kwargs["strategy"], kwargs["mode"]))
            solution_value = 10.0 + len(calls)
            return {
                "solution": np.asarray([solution_value]),
                "loss": 0.5,
                "history": [0.5],
                "states": ["bfgs"],
                "chosen_engine": "bfgs",
                "engine_losses": {"bfgs": 0.5, "custom": 0.75},
            }

        server.optimize_candidate_subspace = fake_optimize
        base_loss, reports, summary = server._compute_exclusion_reports(
            np.asarray([10.0]),
            verbose=False,
            evaluation_mode="uniform",
            return_summary=True,
        )

        self.assertEqual(base_loss, 10.0)
        self.assertEqual([call[0] for call in calls], [2, 2, 2])
        self.assertTrue(all(call[1] == "best_of" for call in calls))
        self.assertTrue(all(call[2] == "optimization" for call in calls))
        self.assertEqual(
            [row["marginal_contribution"] for row in reports], [1.0, 2.0, 3.0]
        )
        self.assertTrue(
            all(row["restricted_optimization_loss"] == 0.5 for row in reports)
        )
        self.assertEqual(summary["delivered_solution_eval_loss"], 10.0)
        self.assertEqual(summary["full_solution_source"], "stage3_l_opt")
        self.assertEqual(summary["loo_search_objective"], "l_opt_current_alpha")

    def test_initialization_keeps_reputation_separate(self):
        reputation = np.asarray([0.6, 0.3, 0.1])
        rng = np.random.default_rng(7)
        np.testing.assert_allclose(
            initialize_optimization_weights("stage0", reputation, rng), reputation
        )
        np.testing.assert_allclose(
            initialize_optimization_weights("uniform", reputation, rng),
            np.ones(3) / 3,
        )
        np.testing.assert_allclose(
            initialize_optimization_weights("reversed", reputation, rng),
            [0.1, 0.3, 0.6],
        )

    def test_contribution_distribution_marks_uninformative_round(self):
        reports = [
            {"positive_contribution": 0.0},
            {"positive_contribution": 0.0},
        ]
        self.assertIsNone(contribution_distribution(reports))

    def test_contribution_distribution_rejects_nonfinite_scores(self):
        with self.assertRaises(ValueError):
            raw_contribution_distribution(
                [{"positive_contribution": float("nan")}]
            )
        np.testing.assert_allclose(
            raw_contribution_distribution(
                [{"positive_contribution": float("inf")}]
            ),
            [1.0],
        )

    def test_exploration_mass_preserves_a_probability_distribution(self):
        reports = [
            {"positive_contribution": 3.0},
            {"positive_contribution": 1.0},
            {"positive_contribution": 0.0},
        ]
        result = contribution_distribution(reports, exploration_mass=0.12)
        self.assertAlmostEqual(float(np.sum(result)), 1.0)
        self.assertTrue(np.all(result > 0))

    def test_raw_and_calibrated_contribution_are_kept_separate(self):
        reports = [
            {"positive_contribution": 3.0},
            {"positive_contribution": 1.0},
            {"positive_contribution": 0.0},
        ]
        raw = raw_contribution_distribution(reports)
        calibrated = apply_exploration_floor(raw, exploration_mass=0.12)
        np.testing.assert_allclose(raw, [0.75, 0.25, 0.0])
        np.testing.assert_allclose(calibrated, [0.70, 0.26, 0.04])

    def test_solver_audit_selects_lowest_lopt_loo_for_warm_start(self):
        reports = [
            {
                "index": 0,
                "agent_id": 1,
                "restricted_optimization_loss": 0.7,
                "restricted_solution": np.asarray([7.0]),
            },
            {
                "index": 1,
                "agent_id": 2,
                "restricted_optimization_loss": 0.4,
                "restricted_solution": np.asarray([4.0]),
            },
        ]
        audit = HostServer.audit_uninformative_exclusion(
            1.0, reports, epsilon=0.01
        )
        self.assertEqual(audit["status"], "loo_dominates")
        self.assertEqual(audit["agent_id"], 2)
        np.testing.assert_allclose(audit["warm_start_solution"], [4.0])

    def test_pruning_candidate_uses_most_negative_raw_contribution(self):
        reports = [
            {"agent_id": 1, "marginal_contribution": -0.1},
            {"agent_id": 2, "marginal_contribution": -0.4},
        ]
        candidate = HostServer.negative_contributor_candidate(
            reports, epsilon=1e-3
        )
        self.assertEqual(candidate["agent_id"], 2)
        self.assertIsNone(
            HostServer.negative_contributor_candidate(
                [{"agent_id": 1, "marginal_contribution": -1e-7}],
                epsilon=1e-6,
            )
        )

    def test_uninformative_round_recovers_from_best_loo_warm_start(self):
        server = RecoveryCalibrationServer()
        config = {
            "condition_seed": 7,
            "initialization": "uniform",
            "evaluator": "uniform",
            "update_rate": 0.0,
            "update_policy": "guarded",
            "minimum_update_rate": 0.1,
            "evaluation_tolerance": 0.1,
            "exploration_mass": 0.1,
            "trim_fraction": 0.0,
            "optimizer": "best_of",
            "custom_iterations": 2,
            "max_rounds": 1,
            "tolerance": 1e-9,
            "payment_reputation_mix": 0.5,
            "solver_audit_tolerance": 1e-6,
            "max_solver_recoveries": 1,
            "identity": {"experiment": "solver_recovery"},
        }
        starts = []

        def fake_stage3(fake_server, _optimizer, _iterations, initial_solution=None):
            starts.append(
                None
                if initial_solution is None
                else np.asarray(initial_solution).copy()
            )
            solution = np.asarray([10.0]) if initial_solution is None else np.asarray(initial_solution)
            fake_server.last_subspace_optimization = {
                "chosen_engine": "custom",
                "engine_losses": {"custom": 2.0 if initial_solution is None else 0.4},
                "loss": 2.0 if initial_solution is None else 0.4,
            }
            return solution, [], []

        with patch("alpha_weight_experiments.run_stage3", side_effect=fake_stage3):
            summary, rounds, agents, _payments = calibrate_one_condition(
                server,
                config,
                reputation=np.asarray([0.5, 0.5]),
                poisoned_ids=set(),
            )

        self.assertEqual(len(starts), 2)
        self.assertIsNone(starts[0])
        np.testing.assert_allclose(starts[1], [5.0])
        np.testing.assert_allclose(server.alphas, [0.5, 0.5])
        self.assertEqual([agent.agent_id for agent in server.trusted_agents], [1, 2])
        self.assertEqual(rounds[0]["solver_recovery_attempts"], 1)
        self.assertTrue(rounds[0]["solver_recovery_success"])
        self.assertEqual(summary["final_raw_contribution_weights"], "[1.0, 0.0]")
        self.assertEqual(summary["final_contribution_weights"], "[0.9500000000000001, 0.05]")
        self.assertEqual(agents[1]["raw_contribution_share"], 0.0)
        self.assertEqual(agents[1]["calibrated_contribution_share"], 0.05)

    def test_stage3b_removes_one_agent_then_restarts_full_stage3a(self):
        server = HostServer(target_T=0.0, n_features=1, total_budget=10.0)
        server.trusted_agents = [
            FakeAgent(1, 0.0),
            FakeAgent(2, 0.0),
            FakeAgent(3, 0.0),
        ]
        server.I_list = [np.asarray([1.0]), np.asarray([2.0]), np.asarray([3.0])]
        server.alphas = [1.0 / 3.0] * 3
        config = {
            "identity": {"experiment": "nested_stage3"},
            "condition_seed": 5,
            "payment_reputation_mix": 0.5,
            "pruning_tolerance": 1e-6,
            "max_coalition_rounds": 2,
        }
        calls = []

        def fake_calibrate(fake_server, local_config, local_reputation, *_args, **_kwargs):
            calls.append(
                (
                    [agent.agent_id for agent in fake_server.trusted_agents],
                    np.asarray(local_reputation).copy(),
                    local_config["identity"]["coalition_round"],
                )
            )
            if len(calls) == 1:
                contributions = [-0.1, -0.5, -0.2]
            else:
                contributions = [0.2, 0.1]
            reports = [
                {
                    "index": index,
                    "agent_id": agent.agent_id,
                    "marginal_contribution": contributions[index],
                }
                for index, agent in enumerate(fake_server.trusted_agents)
            ]
            fake_server.last_stage3a_terminal_state = {
                "exclusion_reports": reports,
                "solver_audit": {"status": "audited_uninformative"},
                "current_state_accepted": True,
            }
            fake_server.last_alpha_calibration_state = {"solution": np.zeros(1)}
            fake_server.last_query_counter = {
                "stage": "contribution",
                "requests": {"payment": 0},
            }
            return (
                {"stop_reason": "audited_uninformative", "stage3_requests": 3,
                 "contribution_requests": 4},
                [{"coalition_round": local_config["identity"]["coalition_round"]}],
                [],
                [],
            )

        with patch(
            "alpha_weight_experiments.calibrate_one_condition",
            side_effect=fake_calibrate,
        ), patch(
            "alpha_weight_experiments.settle_calibration_state",
            return_value=({"payment_status": "ok", "paid_total": 10.0}, []),
        ):
            summary, _rounds, _agents, _payments, pruning = (
                calibrate_stabilized_condition(
                    server,
                    config,
                    reputation=np.asarray([0.5, 0.3, 0.2]),
                    poisoned_ids=set(),
                )
            )

        self.assertEqual([call[0] for call in calls], [[1, 2, 3], [1, 3]])
        self.assertEqual([call[2] for call in calls], [0, 1])
        np.testing.assert_allclose(calls[1][1], [5.0 / 7.0, 2.0 / 7.0])
        self.assertEqual(pruning[0]["removed_agent_id"], 2)
        self.assertEqual(pruning[1]["status"], "stable")
        self.assertEqual(summary["final_coalition_ids"], "[1, 3]")
        self.assertEqual(summary["stage3_requests"], 6)

    def test_stage3b_uses_rejected_audited_uninformative_terminal_for_pruning(self):
        server = HostServer(target_T=0.0, n_features=1, total_budget=10.0)
        server.trusted_agents = [
            FakeAgent(1, 0.0),
            FakeAgent(2, 0.0),
            FakeAgent(3, 0.0),
        ]
        server.I_list = [np.asarray([1.0]), np.asarray([2.0]), np.asarray([3.0])]
        server.alphas = [1.0 / 3.0] * 3
        config = {
            "identity": {"experiment": "audited_terminal_pruning"},
            "condition_seed": 11,
            "payment_reputation_mix": 0.5,
            "pruning_tolerance": 1e-6,
            "max_coalition_rounds": 1,
        }
        calls = []

        def reports_for(fake_server, contributions):
            return [
                {
                    "index": index,
                    "agent_id": agent.agent_id,
                    "marginal_contribution": contributions[index],
                }
                for index, agent in enumerate(fake_server.trusted_agents)
            ]

        def fake_calibrate(fake_server, local_config, *_args, **_kwargs):
            calls.append([agent.agent_id for agent in fake_server.trusted_agents])
            if len(calls) == 1:
                terminal_reports = reports_for(fake_server, [-0.1, -0.5, 0.0])
                accepted_reports = reports_for(fake_server, [0.2, 0.1, 0.3])
                terminal_accepted = False
                stop_reason = "audited_uninformative"
            else:
                terminal_reports = reports_for(fake_server, [0.2, 0.1])
                accepted_reports = terminal_reports
                terminal_accepted = True
                stop_reason = "fixed_point_tolerance"
            fake_server.last_stage3a_terminal_state = {
                "exclusion_reports": terminal_reports,
                "solver_audit": {
                    "status": (
                        "audited_uninformative"
                        if len(calls) == 1 else "not_needed"
                    )
                },
                "current_state_accepted": terminal_accepted,
            }
            fake_server.last_alpha_calibration_state = {
                "solution": np.zeros(1),
                "exclusion_reports": accepted_reports,
            }
            fake_server.last_query_counter = {
                "stage": "contribution",
                "requests": {"payment": 0},
            }
            return (
                {
                    "stop_reason": stop_reason,
                    "stage3_requests": 3,
                    "contribution_requests": 4,
                },
                [],
                [],
                [],
            )

        with patch(
            "alpha_weight_experiments.calibrate_one_condition",
            side_effect=fake_calibrate,
        ), patch(
            "alpha_weight_experiments.settle_calibration_state",
            return_value=({"payment_status": "ok", "paid_total": 10.0}, []),
        ):
            summary, _rounds, _agents, _payments, pruning = (
                calibrate_stabilized_condition(
                    server,
                    config,
                    reputation=np.asarray([0.5, 0.3, 0.2]),
                    poisoned_ids=set(),
                )
            )

        self.assertEqual(calls, [[1, 2, 3], [1, 3]])
        self.assertEqual(
            pruning[0]["pruning_state_source"],
            "terminal_audited_uninformative",
        )
        self.assertEqual(pruning[0]["removed_agent_id"], 2)
        self.assertEqual(pruning[1]["pruning_state_source"], "terminal_accepted")
        self.assertEqual(summary["final_coalition_ids"], "[1, 3]")

    def test_stage3b_uses_accepted_snapshot_for_other_rejected_terminal(self):
        server = HostServer(target_T=0.0, n_features=1, total_budget=10.0)
        server.trusted_agents = [FakeAgent(1, 0.0), FakeAgent(2, 0.0)]
        server.I_list = [np.asarray([1.0]), np.asarray([2.0])]
        server.alphas = [0.5, 0.5]
        config = {
            "identity": {"experiment": "guard_rejection_pruning"},
            "condition_seed": 13,
            "payment_reputation_mix": 0.5,
            "pruning_tolerance": 1e-6,
            "max_coalition_rounds": 1,
        }
        terminal_reports = [
            {"index": 0, "agent_id": 1, "marginal_contribution": -0.5},
            {"index": 1, "agent_id": 2, "marginal_contribution": 0.0},
        ]
        accepted_reports = [
            {"index": 0, "agent_id": 1, "marginal_contribution": 0.2},
            {"index": 1, "agent_id": 2, "marginal_contribution": 0.1},
        ]

        def fake_calibrate(fake_server, *_args, **_kwargs):
            fake_server.last_stage3a_terminal_state = {
                "exclusion_reports": terminal_reports,
                "solver_audit": {"status": "not_needed"},
                "current_state_accepted": False,
            }
            fake_server.last_alpha_calibration_state = {
                "solution": np.zeros(1),
                "exclusion_reports": accepted_reports,
            }
            fake_server.last_query_counter = {
                "stage": "contribution",
                "requests": {"payment": 0},
            }
            return (
                {
                    "stop_reason": "max_rounds",
                    "stage3_requests": 3,
                    "contribution_requests": 4,
                },
                [],
                [],
                [],
            )

        with patch(
            "alpha_weight_experiments.calibrate_one_condition",
            side_effect=fake_calibrate,
        ), patch(
            "alpha_weight_experiments.settle_calibration_state",
            return_value=({"payment_status": "ok", "paid_total": 10.0}, []),
        ):
            summary, _rounds, _agents, _payments, pruning = (
                calibrate_stabilized_condition(
                    server,
                    config,
                    reputation=np.asarray([0.6, 0.4]),
                    poisoned_ids=set(),
                )
            )

        self.assertEqual(pruning[0]["pruning_state_source"], "accepted_snapshot")
        self.assertEqual(pruning[0]["status"], "stable")
        self.assertIsNone(pruning[0]["removed_agent_id"])
        self.assertEqual(summary["final_coalition_ids"], "[1, 2]")

    def test_consensus_evaluators_are_independent_when_requested(self):
        server = HostServer(target_T=0.0, n_features=1)
        server.trusted_agents = [
            FakeAgent(1, 1.0),
            FakeAgent(2, 3.0),
            FakeAgent(3, 100.0),
        ]
        server.alphas = [0.8, 0.1, 0.1]
        self.assertAlmostEqual(server._consensus_loss([0.0]), 11.1)
        self.assertAlmostEqual(
            server._consensus_loss([0.0], mode="uniform"), 104.0 / 3.0
        )
        self.assertAlmostEqual(server._consensus_loss([0.0], mode="median"), 3.0)
        self.assertAlmostEqual(
            server._consensus_loss([0.0], mode="trimmed", trim_fraction=0.34),
            3.0,
        )

    def test_payment_uses_reputation_and_contribution_not_optimization_alpha(self):
        server = HostServer(target_T=0.0, n_features=1, total_budget=10.0)
        server.trusted_agents = [FakeAgent(1, 0.0), FakeAgent(2, 0.0)]
        server.alphas = [0.99, 0.01]
        status, rows, total = settle_with_separated_weights(
            server,
            reputation=np.asarray([0.25, 0.75]),
            contribution=np.asarray([0.75, 0.25]),
            positive_contribution=np.asarray([3.0, 1.0]),
            reputation_mix=0.5,
        )
        self.assertEqual(status, "ok")
        self.assertAlmostEqual(total, 10.0)
        self.assertAlmostEqual(rows[0]["surplus_share"], 0.5)
        self.assertAlmostEqual(rows[1]["surplus_share"], 0.5)
        self.assertNotEqual(rows[0]["optimization_weight"], rows[0]["surplus_share"])

    def test_guarded_final_rejection_keeps_last_accepted_state(self):
        server = GuardedCalibrationServer()
        config = {
            "condition_seed": 7,
            "initialization": "uniform",
            "evaluator": "uniform",
            "update_rate": 0.5,
            "update_policy": "guarded",
            "minimum_update_rate": 0.1,
            "evaluation_tolerance": 0.1,
            "exploration_mass": 0.0,
            "trim_fraction": 0.0,
            "optimizer": "best_of",
            "custom_iterations": 2,
            "max_rounds": 3,
            "tolerance": 1e-9,
            "payment_reputation_mix": 0.5,
            "identity": {"experiment": "guarded_regression"},
        }

        def fake_stage3(fake_server, _optimizer, _iterations):
            fake_server.last_subspace_optimization = {
                "chosen_engine": "custom",
                "engine_losses": {"custom": 0.0},
            }
            return np.asarray([fake_server.alphas[0]]), [], []

        with patch("alpha_weight_experiments.run_stage3", side_effect=fake_stage3):
            summary, rounds, _agents, payments = calibrate_one_condition(
                server,
                config,
                reputation=np.asarray([0.5, 0.5]),
                poisoned_ids=set(),
            )

        np.testing.assert_allclose(server.alphas, [0.75, 0.25])
        self.assertEqual(summary["final_accepted_round"], 1)
        self.assertEqual(summary["updates_accepted"], 1)
        self.assertEqual(summary["gate_triggers"], 1)
        self.assertAlmostEqual(summary["final_evaluation_loss"], 0.5)
        self.assertAlmostEqual(summary["final_target_error"], 0.75)
        self.assertEqual(json.loads(summary["final_solution"]), [0.75])
        self.assertEqual(json.loads(rounds[-1]["candidate_solution"]), [0.875])
        self.assertEqual(summary["final_contribution_weights"], "[1.0, 0.0]")
        self.assertFalse(rounds[-1]["current_state_accepted"])
        self.assertFalse(rounds[-1]["update_accepted"])
        self.assertEqual(payments[0]["agent_id"], 1)
        self.assertAlmostEqual(payments[0]["optimization_weight"], 0.75)

    def test_guarded_no_progress_stops_repeated_floor_rejections(self):
        server = GuardedCalibrationServer()
        config = {
            "condition_seed": 7,
            "initialization": "uniform",
            "evaluator": "uniform",
            "update_rate": 0.5,
            "update_policy": "guarded",
            "minimum_update_rate": 0.1,
            "evaluation_tolerance": 0.1,
            "guarded_no_progress_patience": 2,
            "exploration_mass": 0.0,
            "trim_fraction": 0.0,
            "optimizer": "best_of",
            "custom_iterations": 2,
            "max_rounds": 20,
            "tolerance": 1e-9,
            "payment_reputation_mix": 0.5,
            "identity": {"experiment": "guarded_no_progress"},
        }

        def fake_stage3(fake_server, _optimizer, _iterations):
            fake_server.last_subspace_optimization = {
                "chosen_engine": "custom",
                "engine_losses": {"custom": 0.0},
            }
            return np.asarray([fake_server.alphas[0]]), [], []

        with patch("alpha_weight_experiments.run_stage3", side_effect=fake_stage3):
            summary, rounds, _agents, _payments = calibrate_one_condition(
                server,
                config,
                reputation=np.asarray([0.5, 0.5]),
                poisoned_ids=set(),
            )

        self.assertEqual(len(rounds), 6)
        self.assertEqual(summary["stop_reason"], "guarded_no_progress")
        self.assertFalse(summary["converged"])
        self.assertTrue(summary["no_progress_triggered"])
        self.assertEqual(summary["final_accepted_round"], 1)
        np.testing.assert_allclose(server.alphas, [0.75, 0.25])
        self.assertEqual(rounds[-1]["floor_rejection_streak"], 2)
        self.assertTrue(rounds[-1]["no_progress_triggered"])
        self.assertFalse(rounds[-1]["proposal_generated"])

    def test_guarded_no_progress_patience_defaults_to_formally_validated_value(self):
        self.assertEqual(
            build_parser().parse_args([]).guarded_no_progress_patience,
            2,
        )

    def test_no_update_keeps_each_initialization_and_does_not_claim_convergence(self):
        for initialization, expected in (("uniform", [0.5, 0.5]),
                                         ("stage0", [0.75, 0.25])):
            server = GuardedCalibrationServer()
            config = dict(condition_seed=7, initialization=initialization,
                          evaluator="uniform", update_rate=0.0, update_policy="fixed",
                          minimum_update_rate=0.05, evaluation_tolerance=0.0,
                          exploration_mass=0.0, trim_fraction=0.0, optimizer="best_of",
                          custom_iterations=2, max_rounds=20, tolerance=1e-9,
                          payment_reputation_mix=0.5, identity={},
                          feasible_lower=-1.0, feasible_upper=1.0)
            def solve(s, *_args):
                s.last_subspace_optimization = {
                    "chosen_engine": "custom", "engine_losses": {"custom": 0.0}}
                return np.asarray([s.alphas[0]]), [], []
            with patch("alpha_weight_experiments.run_stage3", side_effect=solve) as solver:
                summary, rounds, _, _ = calibrate_one_condition(
                    server, config, np.asarray([0.75, 0.25]), set())
            self.assertEqual(solver.call_count, 1)
            np.testing.assert_allclose(server.alphas, expected)
            self.assertEqual(len(rounds), 1)
            self.assertFalse(rounds[0]["proposal_generated"])
            self.assertFalse(summary["converged"])
            self.assertEqual(summary["stop_reason"], "no_update_baseline")
            self.assertTrue(summary["final_box_feasible"])

    def test_box_check_rejects_nonfinite_and_out_of_bounds(self):
        bounds = {"feasible_lower": -1, "feasible_upper": 1}
        self.assertTrue(box_feasible([-1, 1], bounds))
        for x in ([1.001], [float("nan")], [float("inf")]):
            self.assertFalse(box_feasible(x, bounds))
        self.assertIsNone(box_feasible([0], {}))

    def test_manifest_and_output_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args(["--output-dir", directory])
            write_run_manifest(directory, args, "running")
            path = Path(directory) / "run_manifest.json"
            first = json.loads(path.read_text())
            self.assertIn("host_server.py", first["source_sha256"])
            write_run_manifest(directory, args, "complete")
            last = json.loads(path.read_text())
            self.assertEqual(first["started_at_utc"], last["started_at_utc"])
            self.assertEqual(last["status"], "complete")
            with self.assertRaises(FileExistsError):
                run(args)
            self.assertEqual(last, json.loads(path.read_text()))

    def test_no_update_rejects_coalition_stabilization(self):
        args = build_parser().parse_args([
            "--no-alpha-update", "--enable-coalition-stabilization"])
        with self.assertRaisesRegex(ValueError, "fixed coalition"):
            run(args)

    def test_clean_counterfactual_is_preserved_for_poisoned_agents(self):
        records = generate_agent_datasets(
            num_agents=2, n_samples_per_agent=8, poison_ratio=0.5,
            n_features=2, seed=19)
        np.testing.assert_allclose(
            records[0]["observed_y"], records[0]["clean_y"])
        self.assertTrue(records[1]["is_poisoned"])
        self.assertFalse(np.allclose(
            records[1]["observed_y"], records[1]["clean_y"]))

    def test_frozen_evaluator_and_artifacts_reload_reproducibly(self):
        records = generate_agent_datasets(
            num_agents=3, n_samples_per_agent=20, poison_ratio=1 / 3,
            n_features=2, seed=3)
        agents = [SimpleNamespace(agent_id=i + 1, model=PretrainedModel(2))
                  for i in range(3)]
        identity = {"seed": 3, "dimension": 2, "poison_ratio": 1 / 3,
                    "target": 0.0}
        with tempfile.TemporaryDirectory() as directory:
            evaluator, artifact_dir, metrics = save_scenario_artifacts(
                directory, identity, agents, records, evaluator_epochs=3)
            saved = np.load(artifact_dir / "pooled_data.npz")
            split_sets = [set(saved[name].tolist()) for name in
                          ("train_indices", "validation_indices", "test_indices")]
            self.assertFalse(split_sets[0] & split_sets[1])
            self.assertFalse(split_sets[0] & split_sets[2])
            self.assertFalse(split_sets[1] & split_sets[2])
            self.assertEqual(sum(map(len, split_sets)), len(saved["X"]))
            self.assertTrue((artifact_dir / "local_models.pt").exists())
            loaded, payload = load_evaluator_artifact(artifact_dir)
            original_score = score_solution(evaluator, [0.2, -0.1], 0.0)
            loaded_score = score_solution(loaded, [0.2, -0.1], 0.0)
            np.testing.assert_allclose(original_score, loaded_score)
            self.assertEqual(payload["metrics"], metrics)

    def test_external_evaluator_argument_validation(self):
        for arguments in (
            ["--external-evaluator-epochs", "0"],
            ["--external-validation-fraction", "0"],
            ["--external-validation-fraction", "0.6",
             "--external-test-fraction", "0.5"],
            ["--guarded-no-progress-patience", "-1"],
        ):
            args = build_parser().parse_args(arguments)
            with self.assertRaises(ValueError):
                from alpha_weight_experiments import validate_args
                validate_args(args)

    def test_p2_suite_is_exactly_the_four_preregistered_methods(self):
        args = build_parser().parse_args([
            "--comparison-suite", "p2", "--evaluators", "uniform"])
        conditions = comparison_conditions(args)
        self.assertEqual([row["method_label"] for row in conditions], [
            "A_uniform_no_update", "B_stage0_no_update",
            "C_uniform_fixed", "D_uniform_guarded"])
        self.assertEqual(
            [(row["initialization"], row["update_rate"], row["update_policy"])
             for row in conditions],
            [("uniform", 0.0, "fixed"), ("stage0", 0.0, "fixed"),
             ("uniform", 0.5, "fixed"), ("uniform", 0.5, "guarded")],
        )

    def test_p5_suite_uses_static_primary_and_guarded_comparator(self):
        args = build_parser().parse_args([
            "--comparison-suite", "p5", "--evaluators", "uniform"])
        conditions = comparison_conditions(args)
        self.assertEqual([row["method_label"] for row in conditions], [
            "A_uniform_no_update", "D_uniform_guarded"])
        self.assertEqual(conditions[0]["update_rate"], 0.0)
        self.assertEqual(conditions[1]["update_rate"], 0.5)
        self.assertEqual(conditions[1]["update_policy"], "guarded")

    def test_p5_data_controls_are_deterministic_and_keep_mean_sample_scale(self):
        args = build_parser().parse_args([
            "--num-agents", "6", "--samples-per-agent", "100",
            "--data-heterogeneity", "0.6",
            "--sample-imbalance-ratio", "4",
            "--low-quality-ratio", "0.5",
            "--low-quality-noise-std", "0.3",
        ])
        counts, shifts, noise = agent_data_controls(args)
        self.assertEqual(len(counts), 6)
        self.assertLessEqual(abs(np.mean(counts) - 100), 1)
        self.assertAlmostEqual(min(shifts), -0.6)
        self.assertAlmostEqual(max(shifts), 0.6)
        self.assertEqual(noise, [0.05, 0.05, 0.05, 0.3, 0.3, 0.3])

    def test_p5_dataset_controls_preserve_counts_bounds_and_metadata(self):
        records = generate_agent_datasets(
            num_agents=2,
            n_samples_per_agent=10,
            poison_ratio=0.0,
            n_features=2,
            seed=4,
            sample_counts=[6, 14],
            feature_shifts=[-0.6, 0.6],
            noise_stds=[0.05, 0.3],
        )
        self.assertEqual([len(row["X"]) for row in records], [6, 14])
        self.assertEqual([row["sample_count"] for row in records], [6, 14])
        self.assertEqual([row["feature_shift"] for row in records], [-0.6, 0.6])
        self.assertEqual([row["noise_std"] for row in records], [0.05, 0.3])
        self.assertTrue(np.all(records[0]["X"] <= -0.2))
        self.assertTrue(np.all(records[1]["X"] >= 0.2))
        self.assertTrue(all(np.all(np.abs(row["X"]) <= 1.0) for row in records))

    def test_p6_exact_and_greedy_share_objective_and_respect_budget(self):
        candidates = [
            {"agent_id": 1, "quality_share": 0.45, "bid": 2.0,
             "residual": np.asarray([1.0, 0.0])},
            {"agent_id": 2, "quality_share": 0.35, "bid": 2.0,
             "residual": np.asarray([0.0, 1.0])},
            {"agent_id": 3, "quality_share": 0.20, "bid": 1.5,
             "residual": np.asarray([1.0, 1.0])},
        ]
        exact = exact_select(candidates, budget=4.0, max_size=2,
                             diversity_eta=0.5)
        greedy = greedy_select(candidates, budget=4.0, max_size=2,
                               diversity_eta=0.5)
        for result in (exact, greedy):
            self.assertTrue(result["budget_feasible"])
            self.assertLessEqual(result["bid_sum"], 4.0)
            self.assertAlmostEqual(
                result["objective"],
                coalition_value(result["indices"], candidates, 0.5, 2),
            )
        self.assertGreaterEqual(exact["objective"], greedy["objective"])

    def test_p6_exact_selection_rejects_an_infeasible_budget(self):
        candidates = [{
            "agent_id": 1,
            "quality_share": 1.0,
            "bid": 2.0,
            "residual": np.asarray([1.0]),
        }]
        with self.assertRaisesRegex(ValueError, "No non-empty coalition"):
            exact_select(candidates, budget=1.0, max_size=1,
                         diversity_eta=0.5)

    def test_controlled_coalition_forces_poisoned_agent(self):
        server = HostServer(target_T=0.0, n_features=1, n_test=2)
        agents = [ControlledFakeAgent(agent_id, 0.0) for agent_id in range(1, 6)]
        server.all_agents = agents
        server.trusted_agents = [agents[0], agents[1]]
        configure_controlled_coalition(
            server,
            poisoned_ids={4, 5},
            coalition_size=3,
            poison_count=1,
        )
        self.assertEqual([agent.agent_id for agent in server.trusted_agents], [4, 1, 2])
        self.assertEqual(sum(agent.agent_id in {4, 5} for agent in server.trusted_agents), 1)
        self.assertEqual(len(server.I_list), 3)

    def test_formal_comparison_with_undefined_metric_is_not_called_pilot(self):
        self.assertEqual(
            interpret_comparison(30, 29, -0.1, 0.1),
            "incomplete_pairs",
        )
        self.assertEqual(
            interpret_comparison(5, 5, -0.1, 0.1),
            "pilot_only",
        )

    def test_full_participant_removal_removes_all_three_roles_and_restores_state(self):
        agents = [FakeAgent(1, 0.0), FakeAgent(2, 0.0), FakeAgent(3, 0.0)]
        candidates = [np.asarray([1.0]), np.asarray([2.0]), np.asarray([3.0])]
        server = SimpleNamespace(
            trusted_agents=agents,
            I_list=candidates,
            alphas=[0.2, 0.3, 0.5],
            last_subspace_optimization=None,
        )

        def fake_stage3(active, optimizer, iterations, initial_solution=None):
            self.assertEqual([a.agent_id for a in active.trusted_agents], [1, 3])
            np.testing.assert_allclose(active.alphas, [2 / 7, 5 / 7])
            np.testing.assert_allclose(active.I_list, [[1.0], [3.0]])
            np.testing.assert_allclose(initial_solution, [1.5])
            active.last_subspace_optimization = {
                "loss": 0.25, "chosen_engine": "custom"
            }
            return np.asarray([0.75]), [], []

        with patch("p3_contribution_validation.run_stage3", side_effect=fake_stage3):
            result = full_participant_removal(server, 1, np.asarray([1.5]))

        self.assertEqual(result["remaining_agent_ids"], [1, 3])
        np.testing.assert_allclose(result["remaining_alphas"], [2 / 7, 5 / 7])
        self.assertIs(server.trusted_agents, agents)
        self.assertIs(server.I_list, candidates)
        self.assertEqual(server.alphas, [0.2, 0.3, 0.5])

    def test_settlement_default_preserves_legacy_and_research_can_disable_it(self):
        self.assertTrue(build_parser().parse_args([]).settle_payment)
        research_args = build_parser().parse_args(["--no-settle-payment"])
        self.assertFalse(research_args.settle_payment)


if __name__ == "__main__":
    unittest.main()

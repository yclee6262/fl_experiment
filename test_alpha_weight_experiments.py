import unittest
from unittest.mock import patch

import numpy as np

from alpha_weight_experiments import (
    apply_exploration_floor,
    calibrate_one_condition,
    calibrate_stabilized_condition,
    contribution_distribution,
    configure_controlled_coalition,
    initialize_optimization_weights,
    raw_contribution_distribution,
    settle_with_separated_weights,
)
from host_server import HostServer, payment_contributions


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
        self.assertEqual(summary["final_contribution_weights"], "[1.0, 0.0]")
        self.assertFalse(rounds[-1]["current_state_accepted"])
        self.assertFalse(rounds[-1]["update_accepted"])
        self.assertEqual(payments[0]["agent_id"], 1)
        self.assertAlmostEqual(payments[0]["optimization_weight"], 0.75)

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


if __name__ == "__main__":
    unittest.main()

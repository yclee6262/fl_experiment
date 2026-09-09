import unittest
from unittest.mock import patch

import numpy as np

from alpha_weight_experiments import (
    calibrate_one_condition,
    contribution_distribution,
    configure_controlled_coalition,
    initialize_optimization_weights,
    settle_with_separated_weights,
)
from host_server import HostServer


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


class AlphaWeightExperimentTests(unittest.TestCase):
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

    def test_exploration_mass_preserves_a_probability_distribution(self):
        reports = [
            {"positive_contribution": 3.0},
            {"positive_contribution": 1.0},
            {"positive_contribution": 0.0},
        ]
        result = contribution_distribution(reports, exploration_mass=0.12)
        self.assertAlmostEqual(float(np.sum(result)), 1.0)
        self.assertTrue(np.all(result > 0))

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

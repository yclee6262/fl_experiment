"""Regression cases retained from the original calibration study."""
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from experiments.alpha_calibration.experiment import (
    initialize_optimization_weights, contribution_distribution,
    raw_contribution_distribution, apply_exploration_floor,
    settle_with_separated_weights, calibrate_one_condition,
    build_parser, run, write_run_manifest,
)
from core.host_server import HostServer

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

class CalibrationTests(unittest.TestCase):
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

        with patch("experiments.alpha_calibration.experiment.run_stage3", side_effect=fake_stage3):
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

        with patch("experiments.alpha_calibration.experiment.run_stage3", side_effect=fake_stage3):
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
            with patch("experiments.alpha_calibration.experiment.run_stage3", side_effect=solve) as solver:
                summary, rounds, _, _ = calibrate_one_condition(
                    server, config, np.asarray([0.75, 0.25]), set())
            self.assertEqual(solver.call_count, 1)
            np.testing.assert_allclose(server.alphas, expected)
            self.assertEqual(len(rounds), 1)
            self.assertFalse(rounds[0]["proposal_generated"])
            self.assertFalse(summary["converged"])
            self.assertEqual(summary["stop_reason"], "no_update_baseline")
            self.assertTrue(summary["final_box_feasible"])

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

    def test_settlement_default_preserves_legacy_and_research_can_disable_it(self):
        self.assertTrue(build_parser().parse_args([]).settle_payment)
        research_args = build_parser().parse_args(["--no-settle-payment"])
        self.assertFalse(research_args.settle_payment)

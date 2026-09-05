import unittest

import numpy as np

from alpha_weight_experiments import (
    contribution_distribution,
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


class AlphaWeightExperimentTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

"""Verify sparse initial weights and zero-weight query skipping."""

from types import SimpleNamespace

import numpy as np
import pytest

from explore_sparse_alpha import (sparse_ensemble_objective,
                                  top_two_stage0_alpha)


class StubAgent:
    def __init__(self, ident, output):
        self.agent_id = ident
        self.output = output
        self.calls = 0

    def api_predict(self, _solution):
        self.calls += 1
        return np.array([self.output])


def test_top_two_weights_are_normalized_and_stably_tied():
    alpha, selected = top_two_stage0_alpha([0.01, 0.01, 0.5])
    assert selected == [0, 1]
    assert np.allclose(alpha, [0.5, 0.5, 0.0])


def test_zero_weight_agent_is_not_queried():
    agents = [StubAgent(1, 2.0), StubAgent(2, -1.0), StubAgent(3, 100.0)]
    server = SimpleNamespace(trusted_agents=agents, alphas=[0.75, 0.25, 0.0],
                             target_T=1.0, sparse_agent_calls={1: 0, 2: 0, 3: 0})
    loss = sparse_ensemble_objective(server, np.array([0.1, 0.2]))
    assert loss == pytest.approx(abs(0.75*2.0 + 0.25*(-1.0) - 1.0))
    assert [agent.calls for agent in agents] == [1, 1, 0]
    assert server.sparse_agent_calls == {1: 1, 2: 1, 3: 0}


def test_invalid_sparse_inputs_are_rejected():
    with pytest.raises(ValueError):
        top_two_stage0_alpha([0.1, -0.1, 0.2])
    with pytest.raises(ValueError):
        top_two_stage0_alpha([0.1])

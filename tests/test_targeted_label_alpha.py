"""Check the proposal-local initialization before using saved models."""

import numpy as np
import pytest

from explore_targeted_label_alpha import (proposal_local_alpha,
                                          proposal_local_points)


def test_proposal_local_points_are_deterministic_and_within_box():
    proposals = np.array([[-1.0, 1.0], [0.0, 0.0], [1.0, -1.0]])
    first = proposal_local_points(proposals, seed=42, lower=-1.0, upper=1.0)
    assert first.shape == (9, 2)
    assert np.array_equal(first, proposal_local_points(proposals, 42, -1.0, 1.0))
    assert not np.array_equal(first, proposal_local_points(proposals, 43, -1.0, 1.0))
    assert np.all(first >= -1.0) and np.all(first <= 1.0)


def test_local_error_changes_weight_with_auditable_shrinkage():
    mse = [0.1, 0.1]
    labels = [0.0, 0.0]
    predicted = [[0.0, 1.0], [0.0, 1.0]]
    alpha, audit = proposal_local_alpha(mse, predicted, labels)
    assert alpha[0] > alpha[1]
    assert np.isclose(alpha.sum(), 1.0)
    assert audit["proposal_local_mse"] == [0.0, 1.0]
    assert audit["stage0_prior_pseudocount"] == 5.0


def test_bad_evidence_is_rejected():
    with pytest.raises(ValueError):
        proposal_local_alpha([0.1, -0.1], [[0.0, 0.0]], [0.0])
    with pytest.raises(ValueError):
        proposal_local_alpha([0.1, 0.1], [[0.0, np.nan]], [0.0])

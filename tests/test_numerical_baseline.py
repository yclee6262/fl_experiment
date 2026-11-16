"""Guard the numerical comparison against silently scoring only two coordinates."""
import numpy as np
import pytest
import fl_comparison_numerical as baseline
from run_experiments import true_function


@pytest.mark.parametrize("point", [[.1, .2], [.1, .2, .3, -.4, .5], [.3]])
def test_baseline_error_uses_full_decision_dimension(point):
    x = np.asarray(point)
    assert baseline.evaluate_S_with_ground_truth(x) == pytest.approx(abs(true_function(x) - baseline.TARGET_T))

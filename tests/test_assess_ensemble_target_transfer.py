"""Focused coverage for the transfer study's acceptance calculation."""

from assess_ensemble_target_transfer import _scenario_analysis


def _row(seed, old, new, old_requests=100, new_requests=90):
    return {"seed": seed, "status": "valid", "candidate_failures": 0,
            "old_pooled_error": old, "new_pooled_error": new,
            "old_truth_error": old, "new_truth_error": new,
            "old_stage3_requests": old_requests,
            "new_stage3_requests": new_requests}


def test_all_failures_keep_noncomputable_intervals():
    report = _scenario_analysis([
        {"seed": 10, "status": "expected_failure"},
        {"seed": 11, "status": "expected_failure"}], [10, 11], "reference_clean")
    assert report["valid_pairs"] == 0
    assert report["paired_bootstrap_95pct_ci_pooled"] == (None, None)
    assert report["gates"]["scenario_passed"] is False


def test_quality_and_cost_both_required_for_scenario_success():
    good = [_row(seed, 0.05, 0.01) for seed in range(10, 20)]
    report = _scenario_analysis(good, list(range(10, 20)), "reference_clean")
    assert report["gates"]["scenario_passed"] is True
    assert report["paired_bootstrap_97p5pct_simultaneous_ci_pooled"][1] < 0
    expensive = [_row(seed, 0.05, 0.01, new_requests=110)
                 for seed in range(10, 20)]
    report = _scenario_analysis(expensive, list(range(10, 20)), "reference_clean")
    assert report["gates"]["simultaneous_pooled_ci_upper_below_zero"] is True
    assert report["gates"]["scenario_passed"] is False


def test_duplicate_seed_cannot_satisfy_complete_pair_gate():
    report = _scenario_analysis([_row(10, 0.05, 0.01), _row(10, 0.05, 0.01)],
                                [10, 11], "reference_clean")
    assert report["valid_pairs"] == 2
    assert report["gates"]["complete_valid_pairs"] is False

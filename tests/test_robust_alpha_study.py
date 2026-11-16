import csv
import json

import numpy as np

from analyze_robust_alpha_study import analyze, combined_cost, valid
from run_alpha_initialization_study import FIELDS


METHODS = ["A_uniform", "B_stage0", "bootstrap_quality", "exponential_quality"]


def study(tmp_path, rows):
    (tmp_path/"suite_manifest.json").write_text(json.dumps({
        "status": "complete", "scenarios": ["reference_clean"],
        "seeds": [0, 1], "methods": METHODS}))
    directory = tmp_path/"reference_clean"
    directory.mkdir()
    with (directory/"summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return analyze(tmp_path, run_verification=False)


def row(seed, method, error=.1):
    return {"scenario": "reference_clean", "seed": seed, "method": method,
        "delivery_status": "valid", "status": "valid", "final_solution": "[0,0]",
        "final_box_feasible": "True", "external_global_error": error,
        "final_target_error": .1, "initialization_requests": "0", "stage3_requests": "10"}


def test_analysis_detects_all_methods_missing_same_seed(tmp_path):
    report = study(tmp_path, [row(0, m) for m in METHODS])
    assert len(report["coverage"]["missing_keys"]) == 4
    assert not report["acceptance_passed"]
    assert all(c["paired_n"] == 1 and c["excluded"]["missing_row"] == 1
               and not c["confirmed_quality_advantage_over_B"] for c in report["comparisons"])


def test_duplicate_and_nonfinite_scores_never_support_advantage(tmp_path):
    rows = [row(s, m, np.nan if s == 1 else .1) for s in (0, 1) for m in METHODS]
    rows.append(row(0, "A_uniform"))
    report = study(tmp_path, rows)
    assert len(report["coverage"]["duplicate_keys"]) == 1
    assert not report["acceptance_passed"]
    assert all(c["paired_n"] == 1 for c in report["comparisons"])


def test_all_failures_still_analyze_and_ci_unavailable(tmp_path):
    rows = [{"scenario": "reference_clean", "seed": s, "method": m,
             "status": "solver_failure", "delivery_status": "none",
             "initialization_requests": "3", "stage3_requests": "7"}
            for s in (0, 1) for m in METHODS]
    report = study(tmp_path, rows)
    assert report["coverage"]["complete"]
    assert all(c["ci95"] == (None, None) for c in report["comparisons"])
    assert all(s["valid_n"] == 0 and s["method_requests"]["mean"] == 10
               for s in report["summary"])


def test_feasibility_recomputed_and_missing_cost_not_zero():
    good = row(0, "A_uniform")
    assert valid(good)
    assert not valid({**good, "final_solution": "[1.0000001,0]"})
    assert not valid({**good, "final_solution": "[]"})
    assert not valid({**good, "external_global_error": "inf"})
    assert combined_cost({"initialization_requests": "", "stage3_requests": "10"}, "requests") is None
    assert combined_cost({"initialization_requests": "0", "stage3_requests": "10"}, "requests") == 10

"""Preregistered paired assessment of marginal-information alpha replacement."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from research_analysis import paired_bootstrap_mean_ci
from run_validation_calibrated_alpha import MARGINAL_METHOD, META_METHOD


MAX_REQUEST_INCREASE_FRACTION = 0.05


def assess(root, expected_seed_count=180):
    root = Path(root)
    manifest = json.loads((root / "suite_manifest.json").read_text())
    verification = json.loads((root / "selection_verification.json").read_text())
    if manifest["scenarios"] != ["reference_clean"]:
        raise ValueError("Expected one clean basic scenario")
    with (root / "reference_clean" / "summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = Counter((r["scenario"], r["seed"], r["method"]) for r in rows)
    expected = {(scenario, str(seed), method)
                for scenario in manifest["scenarios"]
                for seed in manifest["seeds"] for method in manifest["methods"]}
    complete = (manifest["status"] == "complete" and verification["passed"]
                and set(keys) == expected and all(keys[k] == 1 for k in expected))
    index = {(r["seed"], r["method"]): r for r in rows}
    method_stats = {}
    for method in (META_METHOD, MARGINAL_METHOD):
        group = [index.get((str(seed), method)) for seed in manifest["seeds"]]
        valid = [r for r in group if r is not None
                 and r["delivery_status"] == "valid"
                 and r["final_box_feasible"] == "True"
                 and r["external_global_error"]
                 and np.isfinite(float(r["external_global_error"]))]
        errors = [float(r["external_global_error"]) for r in valid]
        method_stats[method] = {
            "valid": len(valid), "failures": len(group)-len(valid),
            "mean_pooled_error": float(np.mean(errors)) if errors else None,
            "median_pooled_error": float(np.median(errors)) if errors else None,
            "pooled_q1_q3": [float(x) for x in np.quantile(errors, [.25, .75])]
                            if errors else [None, None],
            "mean_truth_error": float(np.mean([float(r["final_target_error"])
                                              for r in valid])) if valid else None,
            "mean_initialization_requests": float(np.mean([
                float(r["initialization_requests"]) for r in valid])) if valid else None,
            "mean_stage3_requests": float(np.mean([
                float(r["stage3_requests"]) for r in valid])) if valid else None,
            "mean_model_requests": float(np.mean([
                float(r["initialization_requests"])+float(r["stage3_requests"])
                for r in valid])) if valid else None,
            "mean_method_seconds": float(np.mean([
                float(r["initialization_seconds"])+float(r["stage3_seconds"])
                for r in valid])) if valid else None,
            "extra_trusted_labels": sorted({int(r["extra_trusted_labels"]) for r in valid}),
        }
    differences, conflicts, choices = [], 0, Counter()
    for seed in manifest["seeds"]:
        old = index.get((str(seed), META_METHOD))
        new = index.get((str(seed), MARGINAL_METHOD))
        if not old or not new or old["delivery_status"] != "valid" or new["delivery_status"] != "valid":
            continue
        delta = float(new["external_global_error"])-float(old["external_global_error"])
        truth_delta = float(new["final_target_error"])-float(old["final_target_error"])
        if not (np.isfinite(delta) and np.isfinite(truth_delta)):
            continue
        differences.append(delta)
        conflicts += delta*truth_delta < 0
        choices[json.loads(new["alpha_audit"])["selected_initialization"]] += 1
    interval = paired_bootstrap_mean_ci(differences)
    old_cost = method_stats[META_METHOD]["mean_model_requests"]
    new_cost = method_stats[MARGINAL_METHOD]["mean_model_requests"]
    request_increase = new_cost-old_cost if new_cost is not None and old_cost is not None else None
    request_increase_fraction = request_increase/old_cost if request_increase is not None and old_cost > 0 else None
    quality_gate = bool(expected_seed_count == 180 and complete
                        and len(manifest["seeds"]) == expected_seed_count
                        and len(differences) == len(manifest["seeds"])
                        and interval[1] is not None and interval[1] < 0
                        and method_stats[MARGINAL_METHOD]["failures"]
                        <= method_stats[META_METHOD]["failures"]
                        and method_stats[MARGINAL_METHOD]["extra_trusted_labels"] == [0])
    cost_gate = bool(request_increase_fraction is not None
                     and request_increase_fraction <= MAX_REQUEST_INCREASE_FRACTION)
    mean_delta = float(np.mean(differences)) if differences else None
    result = {
        "complete_and_verified": complete,
        "seed_count": len(manifest["seeds"]), "paired_n": len(differences),
        "methods": method_stats, "mean_pooled_difference": mean_delta,
        "paired_95pct_ci": interval,
        "wins": sum(d < 0 for d in differences),
        "ties": sum(d == 0 for d in differences),
        "losses": sum(d > 0 for d in differences),
        "pooled_truth_conflicts": conflicts,
        "selected_initializations": dict(choices),
        "mean_extra_model_requests": request_increase,
        "request_increase_fraction": request_increase_fraction,
        "quality_improvement_gate": quality_gate,
        "cost_cap_gate": cost_gate,
        "complex_scenario_gate": quality_gate and cost_gate,
        "mean_break_even_value_per_error_unit_in_request_prices": (
            max(request_increase, 0)/-mean_delta
            if mean_delta is not None and mean_delta < 0 and request_increase is not None
            else None),
        "rule": "180 independent feasible pairs; 10000 seed-bootstrap two-sided 95% CI upper bound for E_marginal-E_five below zero, no higher failure rate, no additional trusted labels, and mean model request increase no greater than 5%. Complex scenarios only if quality and cost gates pass.",
    }
    (root / "marginal_decision.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--expected-seeds", type=int, default=180)
    args = parser.parse_args()
    print(json.dumps(assess(args.root, args.expected_seeds),
                     ensure_ascii=False, indent=2))

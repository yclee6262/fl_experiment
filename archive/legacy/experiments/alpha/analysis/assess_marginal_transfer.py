"""Paired, two-scenario validation of the marginal-information alpha portfolio."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from research_analysis import paired_bootstrap_mean_ci
from run_validation_calibrated_alpha import MARGINAL_METHOD, META_METHOD


SCENARIOS = ("sample_imbalance_4x", "low_quality_25pct")
MAX_REQUEST_INCREASE_FRACTION = 0.05


def _scenario_result(root, scenario, expected_seeds):
    root = Path(root)
    manifest = json.loads((root / "suite_manifest.json").read_text())
    verification = json.loads((root / "selection_verification.json").read_text())
    if manifest["scenarios"] != [scenario]:
        raise ValueError(f"Scenario mismatch: {root}/{scenario}")
    with (root / scenario / "summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = Counter((r["scenario"], r["seed"], r["method"]) for r in rows)
    expected = {(scenario, str(seed), method)
                for seed in expected_seeds for method in manifest["methods"]}
    complete = (manifest["status"] == "complete" and verification["passed"]
                and manifest["seeds"] == list(expected_seeds)
                and set(keys) == expected and all(keys[k] == 1 for k in expected))
    index = {(r["seed"], r["method"]): r for r in rows}
    methods = {}
    for method in (META_METHOD, MARGINAL_METHOD):
        group = [index.get((str(seed), method)) for seed in expected_seeds]
        valid = [r for r in group if r and r["delivery_status"] == "valid"
                 and r["final_box_feasible"] == "True" and r["external_global_error"]
                 and np.isfinite(float(r["external_global_error"]))]
        methods[method] = {
            "valid": len(valid), "failures": len(group)-len(valid),
            "mean_pooled_error": float(np.mean([float(r["external_global_error"])
                                           for r in valid])) if valid else None,
            "mean_truth_error": float(np.mean([float(r["final_target_error"])
                                          for r in valid])) if valid else None,
            "mean_model_requests": float(np.mean([
                float(r["initialization_requests"])+float(r["stage3_requests"])
                for r in valid])) if valid else None,
            "mean_method_seconds": float(np.mean([
                float(r["initialization_seconds"])+float(r["stage3_seconds"])
                for r in valid])) if valid else None,
            "extra_trusted_labels": sorted({int(r["extra_trusted_labels"]) for r in valid}),
        }
    differences, conflicts, choices = [], 0, Counter()
    for seed in expected_seeds:
        old, new = (index.get((str(seed), method))
                    for method in (META_METHOD, MARGINAL_METHOD))
        if not old or not new or old["delivery_status"] != "valid" or new["delivery_status"] != "valid":
            continue
        pooled_delta = float(new["external_global_error"])-float(old["external_global_error"])
        truth_delta = float(new["final_target_error"])-float(old["final_target_error"])
        if not (np.isfinite(pooled_delta) and np.isfinite(truth_delta)):
            continue
        differences.append(pooled_delta)
        conflicts += pooled_delta*truth_delta < 0
        choices[json.loads(new["alpha_audit"])["selected_initialization"]] += 1
    interval95 = paired_bootstrap_mean_ci(differences)
    simultaneous = paired_bootstrap_mean_ci(differences, confidence=0.975)
    old_requests = methods[META_METHOD]["mean_model_requests"]
    new_requests = methods[MARGINAL_METHOD]["mean_model_requests"]
    request_increase_fraction = ((new_requests-old_requests)/old_requests
                                 if old_requests and new_requests is not None else None)
    quality_gate = bool(complete and len(expected_seeds) == 120
                        and len(differences) == len(expected_seeds)
                        and simultaneous[1] is not None and simultaneous[1] < 0
                        and methods[MARGINAL_METHOD]["failures"]
                        <= methods[META_METHOD]["failures"])
    cost_gate = bool(request_increase_fraction is not None
                     and request_increase_fraction <= MAX_REQUEST_INCREASE_FRACTION
                     and methods[MARGINAL_METHOD]["extra_trusted_labels"] == [0])
    return {
        "complete_and_verified": complete,
        "paired_n": len(differences), "methods": methods,
        "mean_pooled_difference": float(np.mean(differences)) if differences else None,
        "paired_95pct_ci": interval95,
        "paired_97p5pct_bonferroni_ci": simultaneous,
        "wins": sum(x < 0 for x in differences),
        "ties": sum(x == 0 for x in differences),
        "losses": sum(x > 0 for x in differences),
        "pooled_truth_conflicts": conflicts,
        "selected_initializations": dict(choices),
        "request_increase_fraction": request_increase_fraction,
        "quality_gate": quality_gate,
        "cost_gate": cost_gate,
        "scenario_transfer_gate": quality_gate and cost_gate,
    }


def assess(imbalance_root, lowquality_root, first_seed=1700, count=120):
    expected = list(range(first_seed, first_seed+count))
    results = {
        "sample_imbalance_4x": _scenario_result(imbalance_root, SCENARIOS[0], expected),
        "low_quality_25pct": _scenario_result(lowquality_root, SCENARIOS[1], expected),
    }
    report = {
        "seeds": expected,
        "results": results,
        "two_scenario_transfer_gate": all(
            r["scenario_transfer_gate"] for r in results.values()),
        "rule": "Two preregistered one-factor scenarios; each uses 120 independent seed pairs. Report per-scenario 95% CI and Bonferroni simultaneous 97.5% CI from 10000 seed bootstraps. Generalization to both scenarios requires both simultaneous upper bounds below zero, full feasible pairs, no more failures, no new labels, and no more than 5% mean request increase in either scenario. Do not pool scenario rows as independent samples.",
    }
    output = (Path(imbalance_root).parent /
              f"alpha_marginal_two_scenario_decision_{first_seed}_{count}.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("imbalance_root")
    parser.add_argument("lowquality_root")
    parser.add_argument("--first-seed", type=int, default=1700)
    parser.add_argument("--count", type=int, default=120)
    args = parser.parse_args()
    print(json.dumps(assess(args.imbalance_root, args.lowquality_root,
                            args.first_seed, args.count), ensure_ascii=False, indent=2))

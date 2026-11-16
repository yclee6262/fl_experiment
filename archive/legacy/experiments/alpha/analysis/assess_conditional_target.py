"""Registered paired analysis of the conditional Target-local safety candidate."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from research_analysis import paired_bootstrap_mean_ci
from run_validation_calibrated_alpha import (
    CONDITIONAL_METHOD, MARGINAL_METHOD, META_METHOD,
)


SCENARIOS = ("reference_clean", "sample_imbalance_4x", "low_quality_25pct")
METHODS = (META_METHOD, MARGINAL_METHOD, CONDITIONAL_METHOD)


def assess_one(root, scenario, seeds):
    root = Path(root)
    manifest = json.loads((root / "suite_manifest.json").read_text())
    verification = json.loads((root / "selection_verification.json").read_text())
    if manifest["scenarios"] != [scenario] or manifest["seeds"] != seeds:
        raise ValueError(f"Manifest scenario/seed mismatch: {root}")
    if not set(METHODS).issubset(manifest["methods"]):
        raise ValueError(f"Missing a registered comparison method: {root}")
    with (root / scenario / "summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = Counter((r["scenario"], r["seed"], r["method"]) for r in rows)
    expected = {(scenario, str(seed), method)
                for seed in seeds for method in manifest["methods"]}
    complete = (manifest["status"] == "complete" and verification["passed"]
                and set(keys) == expected and all(keys[k] == 1 for k in expected))
    by_key = {(r["seed"], r["method"]): r for r in rows}
    stats = {}
    for method in METHODS:
        group = [by_key.get((str(seed), method)) for seed in seeds]
        valid = [r for r in group if r and r["delivery_status"] == "valid"
                 and r["final_box_feasible"] == "True"
                 and r["external_global_error"]
                 and np.isfinite(float(r["external_global_error"]))]
        pooled = [float(r["external_global_error"]) for r in valid]
        stats[method] = {
            "valid": len(valid), "failures": len(seeds)-len(valid),
            "mean_pooled": float(np.mean(pooled)) if valid else None,
            "median_pooled": float(np.median(pooled)) if valid else None,
            "pooled_q1_q3": [float(v) for v in np.quantile(pooled, [.25,.75])]
                            if valid else [None,None],
            "mean_truth": float(np.mean([float(r["final_target_error"])
                                          for r in valid])) if valid else None,
            "mean_requests": float(np.mean([
                float(r["initialization_requests"])+float(r["stage3_requests"])
                for r in valid])) if valid else None,
            "mean_seconds": float(np.mean([
                float(r["initialization_seconds"])+float(r["stage3_seconds"])
                for r in valid])) if valid else None,
            "extra_trusted_labels": sorted({int(r["extra_trusted_labels"])
                                            for r in valid}),
        }
    deltas = []
    conditional_minus_marginal = []
    conflicts = 0
    triggers = 0
    selected = Counter()
    for seed in seeds:
        records = [by_key.get((str(seed), method)) for method in METHODS]
        if any(r is None or r["delivery_status"] != "valid"
               or r["final_box_feasible"] != "True" for r in records):
            continue
        old, marginal, conditional = records
        errors = np.asarray([float(r["external_global_error"]) for r in records])
        truths = np.asarray([float(r["final_target_error"]) for r in records])
        if not (np.isfinite(errors).all() and np.isfinite(truths).all()):
            continue
        deltas.append(float(errors[2]-errors[0]))
        conditional_minus_marginal.append(float(errors[2]-errors[1]))
        conflicts += int((errors[2]-errors[0])*(truths[2]-truths[0]) < 0)
        audit = json.loads(conditional["alpha_audit"])
        triggers += bool(audit["conditional_target"]["triggered"])
        selected[audit["selected_initialization"]] += 1
    old_requests = stats[META_METHOD]["mean_requests"]
    new_requests = stats[CONDITIONAL_METHOD]["mean_requests"]
    request_fraction = ((new_requests-old_requests)/old_requests
                        if old_requests and new_requests is not None else None)
    interval95 = paired_bootstrap_mean_ci(deltas)
    interval_simultaneous = paired_bootstrap_mean_ci(
        deltas, confidence=1-0.05/len(SCENARIOS))
    quality_gate = bool(complete and len(seeds) == 120
                        and len(deltas) == len(seeds)
                        and interval_simultaneous[1] is not None
                        and interval_simultaneous[1] < 0
                        and stats[CONDITIONAL_METHOD]["failures"]
                        <= stats[META_METHOD]["failures"])
    cost_gate = bool(request_fraction is not None and request_fraction <= 0.05
                     and stats[CONDITIONAL_METHOD]["extra_trusted_labels"] == [0])
    return {
        "scenario": scenario, "complete_and_verified": complete,
        "paired_n": len(deltas), "methods": stats,
        "mean_conditional_minus_old": float(np.mean(deltas)) if deltas else None,
        "paired_95pct_ci": [None if v is None else float(v) for v in interval95],
        "paired_bonferroni_simultaneous_ci": [
            None if v is None else float(v) for v in interval_simultaneous],
        "wins_ties_losses": [sum(d < 0 for d in deltas),
                             sum(d == 0 for d in deltas),
                             sum(d > 0 for d in deltas)],
        "largest_regression": max(deltas) if deltas else None,
        "q95_paired_delta": float(np.quantile(deltas, .95)) if deltas else None,
        "mean_conditional_minus_marginal_exploratory": (
            float(np.mean(conditional_minus_marginal))
            if conditional_minus_marginal else None),
        "triggered_seeds": triggers,
        "selected_initializations": dict(selected),
        "pooled_truth_direction_conflicts": conflicts,
        "request_increase_fraction": request_fraction,
        "quality_gate": quality_gate, "cost_gate": cost_gate,
        "joint_gate": quality_gate and cost_gate,
    }


def assess(roots, seeds, output):
    if len(roots) != len(SCENARIOS):
        raise ValueError("Expected one root for each scenario in registered order")
    results = {scenario: assess_one(root, scenario, seeds)
               for root, scenario in zip(roots, SCENARIOS)}
    report = {
        "seeds": seeds, "scenarios": results,
        "cross_scenario_gate": all(r["joint_gate"] for r in results.values()),
        "rule": "120 complete paired seeds per scenario; pooled error primary; 10000 seed bootstraps; Bonferroni simultaneous intervals across three scenarios; no higher failure rate, no extra trusted labels, and <=5% mean independent-method model request increase in each scenario. Truth and conditional-vs-marginal are diagnostics. Batch generation cost is distinct from method-independent cost.",
    }
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs=3)
    parser.add_argument("--first-seed", type=int, default=1900)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(assess(args.roots, list(range(args.first_seed,
                                                 args.first_seed+args.count)),
                            args.output), ensure_ascii=False, indent=2))

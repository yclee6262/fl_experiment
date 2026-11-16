"""Preregistered low-quality paired assessment of target-marginal alpha blend."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from research_analysis import paired_bootstrap_mean_ci
from run_validation_calibrated_alpha import BLEND_METHOD, META_METHOD


SCENARIO = "low_quality_25pct"


def assess(root, seeds, output):
    root = Path(root)
    manifest = json.loads((root / "suite_manifest.json").read_text())
    verification = json.loads((root / "selection_verification.json").read_text())
    if (manifest["scenarios"] != [SCENARIO] or manifest["seeds"] != seeds
            or not {META_METHOD, BLEND_METHOD}.issubset(manifest["methods"])):
        raise ValueError("Formal scenario, seeds, or methods differ from registration")
    with (root / SCENARIO / "summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts = Counter((r["scenario"], r["seed"], r["method"]) for r in rows)
    expected = {(SCENARIO, str(seed), method)
                for seed in seeds for method in manifest["methods"]}
    complete = (manifest["status"] == "complete" and verification["passed"]
                and set(counts) == expected
                and all(counts[key] == 1 for key in expected))
    index = {(r["seed"], r["method"]): r for r in rows}
    groups = {}
    for method in (META_METHOD, BLEND_METHOD):
        group = [index.get((str(seed), method)) for seed in seeds]
        valid = [r for r in group if r and r["delivery_status"] == "valid"
                 and r["final_box_feasible"] == "True" and r["external_global_error"]
                 and np.isfinite(float(r["external_global_error"]))]
        errors = np.asarray([float(r["external_global_error"]) for r in valid])
        groups[method] = {
            "valid": len(valid), "failures": len(seeds)-len(valid),
            "mean_pooled": float(errors.mean()) if valid else None,
            "median_pooled": float(np.median(errors)) if valid else None,
            "q1_q3_pooled": [float(v) for v in np.quantile(errors, [.25,.75])]
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
    differences, conflicts, choices = [], 0, Counter()
    for seed in seeds:
        old = index.get((str(seed), META_METHOD))
        blend = index.get((str(seed), BLEND_METHOD))
        if any(r is None or r["delivery_status"] != "valid"
               or r["final_box_feasible"] != "True" for r in (old, blend)):
            continue
        old_pooled, new_pooled = (float(r["external_global_error"])
                                  for r in (old, blend))
        old_truth, new_truth = (float(r["final_target_error"])
                                for r in (old, blend))
        if not all(np.isfinite(v) for v in
                   (old_pooled, new_pooled, old_truth, new_truth)):
            continue
        differences.append(new_pooled-old_pooled)
        conflicts += int((new_pooled-old_pooled)*(new_truth-old_truth) < 0)
        choices[json.loads(blend["alpha_audit"])["selected_initialization"]] += 1
    ci = paired_bootstrap_mean_ci(differences)
    old_requests = groups[META_METHOD]["mean_requests"]
    new_requests = groups[BLEND_METHOD]["mean_requests"]
    request_fraction = ((new_requests-old_requests)/old_requests
                        if old_requests and new_requests is not None else None)
    quality_gate = bool(complete and len(seeds) == 180 and len(differences) == 180
                        and ci[1] is not None and ci[1] < 0
                        and groups[BLEND_METHOD]["failures"]
                        <= groups[META_METHOD]["failures"])
    cost_gate = bool(request_fraction is not None and request_fraction <= .05
                     and groups[BLEND_METHOD]["extra_trusted_labels"] == [0])
    report = {
        "scenario": SCENARIO, "seeds": seeds,
        "complete_and_verified": complete, "paired_n": len(differences),
        "methods": groups,
        "mean_blend_minus_old": float(np.mean(differences)) if differences else None,
        "paired_95pct_ci": [None if v is None else float(v) for v in ci],
        "wins_ties_losses": [sum(v < 0 for v in differences),
                             sum(v == 0 for v in differences),
                             sum(v > 0 for v in differences)],
        "largest_regression": max(differences) if differences else None,
        "q95_paired_delta": float(np.quantile(differences, .95))
                            if differences else None,
        "selected_initializations": dict(choices),
        "pooled_truth_direction_conflicts": conflicts,
        "request_increase_fraction": request_fraction,
        "quality_gate": quality_gate, "cost_gate": cost_gate,
        "low_quality_advantage_gate": quality_gate and cost_gate,
        "rule": "180 fresh seed pairs in low_quality_25pct only; pooled absolute error primary; 10000 seed bootstraps with fixed analysis RNG; 95% CI upper <0, complete feasible pairs, no higher failures, zero extra trusted labels, <=5% mean independent-method model request increase. Other scenarios are out of claim scope.",
    }
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--first-seed", type=int, default=2200)
    parser.add_argument("--count", type=int, default=180)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(assess(args.root, list(range(args.first_seed,
                                                args.first_seed+args.count)),
                            args.output), ensure_ascii=False, indent=2))

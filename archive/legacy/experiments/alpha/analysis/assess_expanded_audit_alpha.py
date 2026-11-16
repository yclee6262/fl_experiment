"""Preregistered quality and resource assessment for 100-label alpha candidate."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from research_analysis import paired_bootstrap_mean_ci
from run_validation_calibrated_alpha import AUDIT_METHOD, META_METHOD


def assess(root, expected_seed_count=180):
    root = Path(root)
    manifest = json.loads((root / "suite_manifest.json").read_text())
    verification = json.loads((root / "selection_verification.json").read_text())
    if manifest["scenarios"] != ["reference_clean"]:
        raise ValueError("Expected one clean basic scenario")
    with (root / "reference_clean" / "summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = Counter((row["scenario"], row["seed"], row["method"]) for row in rows)
    expected = {(scenario, str(seed), method)
                for scenario in manifest["scenarios"]
                for seed in manifest["seeds"] for method in manifest["methods"]}
    complete = (manifest["status"] == "complete" and verification["passed"]
                and set(keys) == expected and all(keys[key] == 1 for key in expected))
    index = {(row["seed"], row["method"]): row for row in rows}
    stats = {}
    for method in (META_METHOD, AUDIT_METHOD):
        group = [index.get((str(seed), method)) for seed in manifest["seeds"]]
        valid = [row for row in group if row is not None
                 and row["delivery_status"] == "valid"
                 and row["final_box_feasible"] == "True"
                 and row["external_global_error"]
                 and np.isfinite(float(row["external_global_error"]))]
        stats[method] = {
            "valid": len(valid), "failures": len(group)-len(valid),
            "mean_pooled_error": float(np.mean([float(row["external_global_error"])
                                           for row in valid])) if valid else None,
            "mean_truth_error": float(np.mean([float(row["final_target_error"])
                                          for row in valid])) if valid else None,
            "mean_model_requests": float(np.mean([float(row["initialization_requests"])
                                                 +float(row["stage3_requests"])
                                                 for row in valid])) if valid else None,
            "mean_method_seconds": float(np.mean([float(row["initialization_seconds"])
                                                 +float(row["stage3_seconds"])
                                                 for row in valid])) if valid else None,
            "extra_trusted_labels": sorted({int(row["extra_trusted_labels"])
                                            for row in valid}),
        }
    differences, request_increases, choices = [], [], Counter()
    conflicts = 0
    for seed in manifest["seeds"]:
        old, new = (index.get((str(seed), method))
                    for method in (META_METHOD, AUDIT_METHOD))
        if not old or not new or old["delivery_status"] != "valid" or new["delivery_status"] != "valid":
            continue
        delta = float(new["external_global_error"])-float(old["external_global_error"])
        truth_delta = float(new["final_target_error"])-float(old["final_target_error"])
        if not (np.isfinite(delta) and np.isfinite(truth_delta)):
            continue
        differences.append(delta)
        request_increases.append(float(new["initialization_requests"])
                                 +float(new["stage3_requests"])
                                 -float(old["initialization_requests"])
                                 -float(old["stage3_requests"]))
        conflicts += delta*truth_delta < 0
        choices[json.loads(new["alpha_audit"])["selected_initialization"]] += 1
    interval = paired_bootstrap_mean_ci(differences)
    mean_delta = float(np.mean(differences)) if differences else None
    mean_extra_requests = float(np.mean(request_increases)) if request_increases else None
    gate = bool(complete and len(manifest["seeds"]) == expected_seed_count
                and len(differences) == len(manifest["seeds"])
                and interval[1] is not None and interval[1] < 0
                and stats[AUDIT_METHOD]["failures"] <= stats[META_METHOD]["failures"]
                and stats[AUDIT_METHOD]["extra_trusted_labels"] == [80])
    result = {
        "complete_and_verified": complete,
        "seed_count": len(manifest["seeds"]), "paired_n": len(differences),
        "methods": stats, "mean_pooled_difference": mean_delta,
        "paired_95pct_ci": interval,
        "wins": sum(d < 0 for d in differences),
        "ties": sum(d == 0 for d in differences),
        "losses": sum(d > 0 for d in differences),
        "pooled_truth_conflicts": conflicts,
        "selected_initializations": dict(choices),
        "mean_extra_model_requests": mean_extra_requests,
        "extra_trusted_labels_per_run": 80,
        "quality_improvement_gate": gate,
        "break_even_value_per_error_unit": (
            {"request_price_coefficient": mean_extra_requests / -mean_delta,
             "extra_label_price_coefficient": 80 / -mean_delta}
            if mean_delta is not None and mean_delta < 0 else None),
        "economic_note": "For value v per unit pooled-error reduction, local-model request price c, and trusted-label price l, observed-mean break-even is v*(-mean_delta)>c*mean_extra_model_requests+80*l. This is not verified monetary net gain.",
        "rule": "Independent 180-seed paired 10000-bootstrap two-sided 95% CI for E_audit-E_five wholly below zero, complete feasible pairs, no more failures; resource and monetary tradeoff reported separately.",
    }
    (root / "expanded_audit_decision.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--expected-seeds", type=int, default=180)
    args = parser.parse_args()
    print(json.dumps(assess(args.root, args.expected_seeds), indent=2,
                     ensure_ascii=False))

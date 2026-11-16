"""Preregistered seed-paired assessment of local versus global alpha selection."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from research_analysis import paired_bootstrap_mean_ci
from run_validation_calibrated_alpha import LOCAL_METHOD, META_METHOD


def assess(root):
    root = Path(root)
    manifest = json.loads((root / "suite_manifest.json").read_text())
    verification = json.loads((root / "selection_verification.json").read_text())
    if manifest["scenarios"] != ["reference_clean"]:
        raise ValueError("Only the preregistered clean scenario can be assessed")
    with (root / "reference_clean" / "summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = Counter((r["scenario"], r["seed"], r["method"]) for r in rows)
    expected = {("reference_clean", str(seed), method)
                for seed in manifest["seeds"] for method in manifest["methods"]}
    complete = (manifest["status"] == "complete" and verification["passed"]
                and set(keys) == expected and all(keys[key] == 1 for key in expected))
    by_key = {(r["seed"], r["method"]): r for r in rows}
    stats = {}
    for method in (META_METHOD, LOCAL_METHOD):
        group = [by_key.get((str(seed), method)) for seed in manifest["seeds"]]
        valid = [r for r in group if r is not None and r["delivery_status"] == "valid"
                 and r["final_box_feasible"] == "True"
                 and np.isfinite(float(r["external_global_error"]))]
        stats[method] = {
            "valid": len(valid), "failures": len(group)-len(valid),
            "mean_pooled_error": float(np.mean([
                float(r["external_global_error"]) for r in valid])) if valid else None,
            "mean_truth_error": float(np.mean([
                float(r["final_target_error"]) for r in valid])) if valid else None,
            "mean_requests": float(np.mean([
                float(r["initialization_requests"])+float(r["stage3_requests"])
                for r in valid])) if valid else None,
            "mean_method_seconds": float(np.mean([
                float(r["initialization_seconds"])+float(r["stage3_seconds"])
                for r in valid])) if valid else None,
        }
    differences = []
    conflicts = 0
    selected = Counter()
    for seed in manifest["seeds"]:
        old = by_key.get((str(seed), META_METHOD))
        new = by_key.get((str(seed), LOCAL_METHOD))
        if not old or not new or old["delivery_status"] != "valid" or new["delivery_status"] != "valid":
            continue
        delta = float(new["external_global_error"])-float(old["external_global_error"])
        truth = float(new["final_target_error"])-float(old["final_target_error"])
        if not np.isfinite(delta) or not np.isfinite(truth):
            continue
        differences.append(delta)
        conflicts += delta*truth < 0
        selected[json.loads(new["alpha_audit"])["selected_initialization"]] += 1
    interval = paired_bootstrap_mean_ci(differences)
    result = {
        "complete_and_verified": complete,
        "seed_count": len(manifest["seeds"]),
        "paired_n": len(differences),
        "methods": stats,
        "mean_pooled_difference": float(np.mean(differences)) if differences else None,
        "paired_95pct_ci": interval,
        "wins": sum(d < 0 for d in differences),
        "ties": sum(d == 0 for d in differences),
        "losses": sum(d > 0 for d in differences),
        "pooled_truth_conflicts": conflicts,
        "selected_initializations": dict(selected),
        "quality_gate": bool(complete and len(differences) == len(manifest["seeds"])
                             and interval[1] is not None and interval[1] < 0
                             and stats[LOCAL_METHOD]["failures"] <= stats[META_METHOD]["failures"]),
        "rule": "60 paired valid seeds, 10000 seed-bootstrap 95% CI for E_local-E_global wholly below zero, no extra failures; cost separately reported",
    }
    (root / "local_selector_decision.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    print(json.dumps(assess(args.root), indent=2, ensure_ascii=False))

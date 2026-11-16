"""Apply the registered two-baseline acceptance rule to a saved clean cohort."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from research_analysis import paired_bootstrap_mean_ci


META = "validation_calibrated_multistart"
BASELINES = ("A_uniform", "B_stage0")


def assess(root):
    root = Path(root)
    manifest = json.loads((root / "suite_manifest.json").read_text())
    verification = json.loads((root / "selection_verification.json").read_text())
    with (root / "reference_clean" / "summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts = Counter((r["seed"], r["method"]) for r in rows)
    expected = {(str(seed), method) for seed in manifest["seeds"]
                for method in (*BASELINES, META)}
    complete = (manifest["status"] == "complete" and verification["passed"]
                and set(counts) == expected and all(counts[key] == 1 for key in expected))
    index = {(r["seed"], r["method"]): r for r in rows}
    summaries = {}
    for method in (*BASELINES, META):
        group = [index.get((str(seed), method)) for seed in manifest["seeds"]]
        valid = [r for r in group if r and r["delivery_status"] == "valid"
                 and r["final_box_feasible"] == "True"
                 and np.isfinite(float(r["external_global_error"]))]
        summaries[method] = {
            "valid": len(valid), "failures": len(group)-len(valid),
            "pooled_mean": float(np.mean([float(r["external_global_error"]) for r in valid])) if valid else None,
            "truth_mean": float(np.mean([float(r["final_target_error"]) for r in valid])) if valid else None,
            "mean_api_requests": float(np.mean([
                float(r["initialization_requests"])+float(r["stage3_requests"])
                for r in valid])) if valid else None,
            "mean_method_seconds": float(np.mean([
                float(r["initialization_seconds"])+float(r["stage3_seconds"])
                for r in valid])) if valid else None,
        }
    comparisons = {}
    for baseline in BASELINES:
        differences = []
        conflicts = 0
        for seed in manifest["seeds"]:
            meta, control = index.get((str(seed), META)), index.get((str(seed), baseline))
            if (not meta or not control or meta["delivery_status"] != "valid"
                    or control["delivery_status"] != "valid"):
                continue
            pooled = float(meta["external_global_error"])-float(control["external_global_error"])
            truth = float(meta["final_target_error"])-float(control["final_target_error"])
            if not (np.isfinite(pooled) and np.isfinite(truth)):
                continue
            differences.append(pooled)
            if pooled*truth < 0:
                conflicts += 1
        interval = paired_bootstrap_mean_ci(differences, confidence=0.975)
        comparisons[baseline] = {
            "paired_n": len(differences),
            "mean_pooled_difference": float(np.mean(differences)) if differences else None,
            "bonferroni_97p5_ci": interval,
            "wins": sum(value < 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "losses": sum(value > 0 for value in differences),
            "pooled_truth_direction_conflicts": conflicts,
            "passes": bool(complete and len(differences) == len(manifest["seeds"])
                           and interval[1] is not None and interval[1] < 0
                           and summaries[META]["failures"] <= summaries[baseline]["failures"]),
        }
    result = {"study": manifest["study"], "seeds": manifest["seeds"],
              "complete_and_verified": complete, "summary": summaries,
              "comparisons": comparisons,
              "passes_registered_two_baseline_rule": all(c["passes"] for c in comparisons.values()),
              "rule": "Both seed-paired 97.5% bootstrap intervals wholly negative, full pairing and no added failures"}
    (root / "decision.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    arguments = parser.parse_args()
    print(json.dumps(assess(arguments.root), indent=2, ensure_ascii=False))

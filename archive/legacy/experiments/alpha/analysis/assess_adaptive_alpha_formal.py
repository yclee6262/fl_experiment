"""Evaluate the registered sequential B-then-Uniform comparison."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_adaptive_alpha_replay import summarize


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assess(prereg_path, baseline_root, adaptive_path, five_path):
    prereg_path = Path(prereg_path)
    prereg = json.loads(prereg_path.read_text())
    expected_seeds = prereg["formal_seeds"]
    if (prereg["study"] != "adaptive_B_then_uniform_low_quality_formal_v1"
            or prereg["scenario"] != "low_quality_25pct"
            or len(expected_seeds) != 180):
        raise ValueError("Wrong formal preregistration")
    for name, digest in prereg["source_sha256"].items():
        if (_sha(name) != digest or _sha(Path(prereg["source_snapshot"])/name) != digest):
            raise ValueError(f"Source or snapshot mismatch: {name}")
    baseline_root = Path(baseline_root)
    manifest = json.loads((baseline_root/"suite_manifest.json").read_text())
    verification = json.loads((baseline_root/"selection_verification.json").read_text())
    if (manifest["status"] != "complete"
            or manifest["scenarios"] != [prereg["scenario"]]
            or manifest["seeds"] != expected_seeds
            or not verification["delivery_verification_passed"]):
        raise ValueError("Baseline provenance or replay check failed")
    pilot = json.loads((Path(prereg["pilot_result_paths"][0])/
                        "suite_manifest.json").read_text())
    ignored = {"seeds", "output_dir"}
    if any(value != manifest["arguments"].get(key)
           for key, value in pilot["arguments"].items() if key not in ignored):
        raise ValueError("Formal scenario settings differ from pilot")
    adaptive = json.loads(Path(adaptive_path).read_text())
    five = json.loads(Path(five_path).read_text())
    recorded_seeds = adaptive["seeds"]
    missing_seeds = sorted(set(expected_seeds)-set(recorded_seeds))
    if (len(recorded_seeds) != len(set(recorded_seeds))
            or not set(recorded_seeds) <= set(expected_seeds)):
        raise ValueError("Unexpected or duplicated adaptive seed")
    summary_rows = list(csv.DictReader((baseline_root/prereg["scenario"]/
                                       "summary.csv").open()))
    shared_rows = {row["seed"]: row for row in map(
        json.loads, (baseline_root/prereg["scenario"]/
                    "shared_costs.jsonl").read_text().splitlines())}
    failure_records = []
    for seed in missing_seeds:
        rows = [row for row in summary_rows if int(row["seed"]) == seed]
        shared = shared_rows[seed]
        if (len(rows) != 3
                or any(row["failure_stage"] != "qualification"
                       or row["delivery_status"] != "none" for row in rows)
                or not shared.get("missing_fixed_ids")):
            raise ValueError(f"Missing seed is not a recorded qualification failure: {seed}")
        failure_records.append({
            "seed": seed, "status": "qualification_failure",
            "missing_fixed_ids": shared["missing_fixed_ids"],
            "failure_stage": "qualification",
            "preparation": shared["preparation"],
            "stage3_requests": 0,
        })
    if (adaptive["status"] != "complete"
            or adaptive["scenario"] != prereg["scenario"]
            or five["scenario"] != prereg["scenario"]
            or five["seeds"] != recorded_seeds
            or [row["seed"] for row in adaptive["rows"]] != recorded_seeds
            or [row["seed"] for row in five["rows"]] != recorded_seeds
            or adaptive["threshold"] != 0.01):
        raise ValueError("Formal condition set or threshold differs")
    summary = summarize(prereg["scenario"], [adaptive_path], [five_path])
    rows = adaptive["rows"]
    five_by_seed = {row["seed"]: row for row in five["rows"]}
    b_valid = all(row["candidates"]["B_stage0"]["status"] == "valid"
                  for row in rows)
    complete_valid = (not missing_seeds and summary["valid"] == 180
                      and summary["failures"] == 0
                      and b_valid)
    pooled = np.array([
        row["pooled_error"] - five_by_seed[row["seed"]]["candidates"][
            "B_stage0"]["pooled_error"]
        for row in rows if row["delivery_status"] == "valid"], dtype=float)
    truth = np.array([
        row["truth_error_diagnostic"] - five_by_seed[row["seed"]][
            "candidates"]["B_stage0"]["truth_error"]
        for row in rows if row["delivery_status"] == "valid"], dtype=float)
    ci = summary["pooled_delta_adaptive_minus_B_descriptive_95pct_ci"]
    quality_gate = complete_valid and ci[1] < 0
    cost_ratio = (summary["mean_stage3_plus_selection_requests"] /
                  summary["mean_five_stage3_plus_selection_requests"])
    observed_cost_under_half = cost_ratio < 0.5
    cost_gate = complete_valid and observed_cost_under_half
    return {
        "study": prereg["study"], "status": "complete",
        "preregistration_sha256": _sha(prereg_path),
        "scope": "independent 180-seed low-quality synthetic comparison only",
        "summary": summary,
        "expected_conditions": len(expected_seeds),
        "missing_conditions": failure_records,
        "baseline_verification": verification,
        "cost_ratio_adaptive_to_five": cost_ratio,
        "pooled_q95_regression_vs_B": float(np.quantile(pooled, 0.95)),
        "pooled_max_regression_vs_B": float(np.max(pooled)),
        "mean_truth_delta_diagnostic": float(truth.mean()),
        "pooled_truth_direction_conflicts": int(
            (np.sign(pooled)*np.sign(truth) < 0).sum()),
        "gates": {
            "complete_valid_B_adaptive_pairs": bool(complete_valid),
            "pooled_ci_upper_strictly_below_zero": bool(quality_gate),
            "observed_cost_under_half_of_five_on_valid_pairs": bool(
                observed_cost_under_half),
            "adaptive_requests_below_half_of_five": bool(cost_gate),
            "registered_success": bool(quality_gate and cost_gate),
        },
        "economic_limit": "No monetary profitability without query cost / error value",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preregistration")
    parser.add_argument("baseline_root")
    parser.add_argument("adaptive_json")
    parser.add_argument("five_json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(target)
    result = assess(args.preregistration, args.baseline_root,
                    args.adaptive_json, args.five_json)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                 allow_nan=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

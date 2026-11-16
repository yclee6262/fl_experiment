"""Failure-aware, manifest-driven analysis of P5 scenario suites."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from research_analysis import audit_coverage, finite_float, paired_bootstrap_mean_ci, valid_delivery


BASELINE = "A_uniform_no_update"
COMPARATOR = "D_uniform_guarded"


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("\n", encoding="utf-8")
        return
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(
            key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def mean_known(rows, field):
    values = [finite_float(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def analyze(root):
    root = Path(root)
    suite = json.loads((root / "suite_manifest.json").read_text())
    methods = suite.get("comparison", [BASELINE, COMPARATOR])
    seeds = [int(seed) for seed in suite["seeds"]]
    method_summaries, pairs, coverage, shared_costs = [], [], [], []
    for scenario_spec in suite["scenarios"]:
        scenario = scenario_spec["name"]
        rows = read_rows(root / scenario / "summary.csv")
        audit = audit_coverage(rows,
            [(method, str(seed)) for method in methods for seed in seeds],
            ("method_label", "seed"))
        coverage.append({"scenario": scenario, **audit,
            "duplicate_keys": json.dumps(audit["duplicate_keys"]),
            "missing_keys": json.dumps(audit["missing_keys"]),
            "unexpected_keys": json.dumps(audit["unexpected_keys"])})
        by_key = {(row.get("method_label"), int(row["seed"])): row for row in rows}
        for seed in seeds:
            shared = by_key.get((BASELINE, seed))
            if shared is not None:
                shared_costs.append({"scenario": scenario, "seed": seed,
                    **{name: shared.get(name, "") for name in (
                        "shared_training_seconds", "shared_qualification_seconds",
                        "shared_proposal_seconds", "shared_fixed_proposal_seconds",
                        "shared_evaluator_seconds", "shared_qualification_requests",
                        "shared_proposal_requests", "shared_fixed_proposal_requests",
                        "shared_qualification_input_samples", "shared_proposal_input_samples",
                        "shared_fixed_proposal_input_samples")}})
        for method in methods:
            selected = [by_key[(method, seed)] for seed in seeds
                        if (method, seed) in by_key]
            valid = [row for row in selected if valid_delivery(row)]
            errors = [finite_float(row.get("external_global_error")) for row in valid]
            errors = [value for value in errors if value is not None]
            for row in selected:
                method_summaries.append({
                    "scenario": scenario,
                    "method": method,
                    "seed": row.get("seed"),
                    "execution_status": row.get("execution_status", "completed"),
                    "delivery_status": row.get("delivery_status", "unknown"),
                    "feasible": row.get("final_box_feasible", row.get("box_feasible", "")),
                    "failure_stage": row.get("failure_stage", ""),
                    "failure_reason": row.get("failure_reason", row.get("stop_reason", "")),
                    "external_global_error": row.get("external_global_error", ""),
                    "final_target_error": row.get("final_target_error", ""),
                    "stage3_requests": row.get("stage3_requests", ""),
                    "stage3_input_samples": row.get("stage3_input_samples", ""),
                    "stage3_seconds": row.get("stage3_seconds", ""),
                    "contribution_requests": row.get("contribution_requests", ""),
                    "contribution_input_samples": row.get("contribution_input_samples", ""),
                    "contribution_seconds": row.get("contribution_seconds", ""),
                    "payment_status": row.get("payment_status", "unknown"),
                    "payment_requests": row.get("payment_requests", ""),
                    "payment_input_samples": row.get("payment_input_samples", ""),
                    "payment_seconds": row.get("payment_seconds", ""),
                    "alpha_q_l1": row.get("final_consistency_l1", ""),
                    "converged": row.get("converged", ""),
                    "condition_seconds": row.get("condition_seconds", ""),
                })
            # A row-level distribution record carries seed-level summaries in every row.
            summaries = []
            for row in selected:
                if valid_delivery(row):
                    value = finite_float(row.get("external_global_error"))
                    if value is not None:
                        summaries.append(value)
            method_summaries.append({
                "scenario": scenario, "method": method, "record_type": "aggregate",
                "planned_seeds": len(seeds), "observed_runs": len(selected),
                "valid_deliveries": sum(valid_delivery(row) for row in selected),
                "delivery_failures": sum(not valid_delivery(row) for row in selected),
                "delivery_failure_rate": (sum(not valid_delivery(row) for row in selected) / len(seeds)
                                          if seeds else None),
                "delivery_status_counts": json.dumps({status: sum(row.get("delivery_status") == status
                    for row in selected) for status in sorted({row.get("delivery_status", "unknown")
                    for row in selected})}),
                "solver_failures": sum(row.get("delivery_status") == "solver_failure" for row in selected),
                "box_infeasible": sum(row.get("delivery_status") == "box_infeasible" for row in selected),
                "mean_external_error_valid": float(np.mean(summaries)) if summaries else None,
                "median_external_error_valid": float(np.median(summaries)) if summaries else None,
                "q1_external_error_valid": float(np.quantile(summaries, .25)) if summaries else None,
                "q3_external_error_valid": float(np.quantile(summaries, .75)) if summaries else None,
                "mean_stage3_requests_known": mean_known(selected, "stage3_requests"),
                "mean_contribution_requests_known": mean_known(selected, "contribution_requests"),
                "mean_payment_requests_known": mean_known(selected, "payment_requests"),
                "mean_stage3_seconds_known": mean_known(selected, "stage3_seconds"),
                "mean_stage3_input_samples_known": mean_known(selected, "stage3_input_samples"),
                "mean_contribution_seconds_known": mean_known(selected, "contribution_seconds"),
                "mean_contribution_input_samples_known": mean_known(selected, "contribution_input_samples"),
                "mean_payment_seconds_known": mean_known(selected, "payment_seconds"),
                "mean_payment_input_samples_known": mean_known(selected, "payment_input_samples"),
                "unknown_stage3_cost_rows": sum(row.get("stage3_requests", "") == "" for row in selected),
                "analysis_note": "All-run error is diagnostic only; valid deliveries define quality summary.",
            })
        for seed in seeds:
            left = by_key.get((BASELINE, seed))
            right = by_key.get((COMPARATOR, seed))
            left_ok = left is not None and valid_delivery(left)
            right_ok = right is not None and valid_delivery(right)
            a = finite_float(left.get("external_global_error")) if left_ok else None
            d = finite_float(right.get("external_global_error")) if right_ok else None
            delta = d - a if a is not None and d is not None else None
            pairs.append({"scenario": scenario, "seed": seed,
                "A_status": "missing" if left is None else left.get("delivery_status", "unknown"),
                "D_status": "missing" if right is None else right.get("delivery_status", "unknown"),
                "valid_quality_pair": delta is not None,
                "missing_pair_reason": "" if delta is not None else
                    f"A={'valid' if left_ok else 'missing_or_invalid'};D={'valid' if right_ok else 'missing_or_invalid'}",
                "external_error_delta_D_minus_A": delta,
                "truth_error_delta_D_minus_A": (
                    finite_float(right.get("final_target_error")) - finite_float(left.get("final_target_error"))
                    if left_ok and right_ok and finite_float(left.get("final_target_error")) is not None
                    and finite_float(right.get("final_target_error")) is not None else None),
                "requests_delta_D_minus_A": (
                    int(right.get("stage3_requests") or 0) + int(right.get("contribution_requests") or 0)
                    - int(left.get("stage3_requests") or 0) - int(left.get("contribution_requests") or 0)
                    if left is not None and right is not None else None),
            })
    comparisons = []
    for scenario in [spec["name"] for spec in suite["scenarios"]]:
        selected = [row for row in pairs if row["scenario"] == scenario]
        values = [row["external_error_delta_D_minus_A"] for row in selected
                  if row["external_error_delta_D_minus_A"] is not None]
        low, high = paired_bootstrap_mean_ci(values)
        confirmatory = scenario != "hard_target_2p5"
        bonf_low, bonf_high = paired_bootstrap_mean_ci(
            values, confidence=1.0 - 0.05 / 7.0) if confirmatory else (None, None)
        failed_a = sum(row.get("A_status") != "valid" for row in selected)
        failed_d = sum(row.get("D_status") != "valid" for row in selected)
        truth_conflicts = sum(
            (row.get("external_error_delta_D_minus_A") is not None
             and row.get("truth_error_delta_D_minus_A") is not None
             and row["external_error_delta_D_minus_A"] * row["truth_error_delta_D_minus_A"] < 0)
            for row in selected)
        comparisons.append({"scenario": scenario, "planned_pairs": len(seeds),
            "valid_pairs": len(values), "missing_pairs": len(seeds) - len(values),
            "mean_delta_D_minus_A": float(np.mean(values)) if values else None,
            "median_delta_D_minus_A": float(np.median(values)) if values else None,
            "ci95_low": low, "ci95_high": high,
            "family_wise_95_bonferroni_low": bonf_low,
            "family_wise_95_bonferroni_high": bonf_high,
            "confirmatory_scenario": confirmatory,
            "A_observed_failures": failed_a, "D_observed_failures": failed_d,
            "pooled_truth_direction_conflicts": truth_conflicts,
            "formal_advantage_judgment_allowed": (
                confirmatory and len(values) == len(seeds) and failed_d <= failed_a
                and next(item["complete"] for item in coverage if item["scenario"] == scenario)),
            "interpretation": "exploratory_only" if not confirmatory else
                ("pilot_or_incomplete" if len(seeds) < 30 or len(values) != len(seeds)
                 else ("stable_improvement" if bonf_high is not None and bonf_high < 0
                       and failed_d <= failed_a else "no_stable_improvement")),
        })
    return method_summaries, pairs, comparisons, coverage, shared_costs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Analysis output directory must be empty.")
    summaries, pairs, comparisons, coverage, shared_costs = analyze(args.results_root)
    write_csv(output / "method_results.csv", summaries)
    write_csv(output / "paired_results.csv", pairs)
    write_csv(output / "scenario_comparisons.csv", comparisons)
    write_csv(output / "coverage_audit.csv", coverage)
    write_csv(output / "shared_costs_per_seed.csv", shared_costs)
    (output / "analysis_manifest.json").write_text(json.dumps({
        "results_root": str(Path(args.results_root).resolve()),
        "paired_quality": "valid deliveries only; seed is the sampling unit",
        "failure_count_rule": "every delivery_status other than valid counts as a failure, including qualification_failure and interrupted",
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 20260924,
        "formal_claim_requires_complete_manifest_coverage": True,
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

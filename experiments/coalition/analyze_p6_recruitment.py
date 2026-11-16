"""Manifest-driven P6 analysis that retains invalid and missing selections."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from research_analysis import audit_coverage, finite_float, valid_delivery


METHODS = ("exact", "objective_greedy", "legacy_greedy")


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("\n", encoding="utf-8")
        return
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(results_dir):
    root = Path(results_dir)
    manifest = json.loads((root / "run_manifest.json").read_text())
    seeds = manifest.get("seeds", manifest.get("arguments", {}).get("seeds", []))
    budgets = manifest.get("budgets", [2.5, 3.5, 8.0])
    etas = manifest.get("etas", [0.0, 0.5])
    rows = read_rows(root / "selection_results.csv")
    expected = [(str(seed), method, str(budget), str(eta))
                for seed in seeds for budget in budgets for eta in etas for method in METHODS]
    # CSV writers can serialize numerically equivalent floats differently. Normalize identities.
    expected_keys = [(str(seed), method, str(float(budget)), str(float(eta)))
                     for seed in seeds for budget in budgets for eta in etas for method in METHODS]
    normalized_rows = [{**row,
        "seed": str(int(row["seed"])),
        "budget": str(float(row.get("budget", row.get("selection_budget", "nan")))),
        "diversity_eta": str(float(row.get("diversity_eta", 0.5)))} for row in rows]
    coverage = audit_coverage(normalized_rows, expected_keys,
        ("seed", "selection_method", "budget", "diversity_eta"))
    by_identity = {(row["seed"], row["selection_method"], row["budget"], row["diversity_eta"]): row
                   for row in normalized_rows}
    pairs, summaries = [], []
    shared_costs = []
    for seed in seeds:
        source = next((row for row in normalized_rows if row["seed"] == str(seed)), None)
        if source is not None:
            shared_costs.append({"seed": seed, **{field: source.get(field, "") for field in (
                "shared_training_seconds", "shared_qualification_seconds", "shared_proposal_seconds",
                "shared_evaluator_seconds", "selection_precompute_seconds",
                "shared_qualification_requests", "shared_proposal_requests",
                "selection_precompute_requests", "shared_qualification_input_samples",
                "shared_proposal_input_samples", "selection_precompute_input_samples")}})
    for seed in seeds:
        for budget in budgets:
            for eta in etas:
                selected = {method: by_identity.get((str(seed), method, str(float(budget)), str(float(eta))))
                            for method in METHODS}
                exact = selected["exact"]
                for method in METHODS[1:]:
                    row = selected[method]
                    exact_ok = exact is not None and finite_float(exact.get("selection_objective")) is not None
                    row_ok = row is not None and finite_float(row.get("selection_objective")) is not None
                    exact_valid = exact is not None and valid_delivery(exact, "external_global_error")
                    row_valid = row is not None and valid_delivery(row, "external_global_error")
                    pairs.append({"seed": seed, "budget": budget, "diversity_eta": eta,
                        "method": method,
                        "exact_objective_available": exact_ok,
                        "objective_gap_from_exact": (finite_float(exact.get("selection_objective"))-
                            finite_float(row.get("selection_objective")) if exact_ok and row_ok else None),
                        "valid_quality_pair": exact_valid and row_valid,
                        "external_error_delta_from_exact": (
                            finite_float(row.get("external_global_error"))-
                            finite_float(exact.get("external_global_error")) if exact_valid and row_valid else None),
                        "truth_error_delta_from_exact": (
                            finite_float(row.get("true_error"))-finite_float(exact.get("true_error"))
                            if exact_valid and row_valid and finite_float(row.get("true_error")) is not None
                            and finite_float(exact.get("true_error")) is not None else None),
                        "exact_status": "missing" if exact is None else exact.get("delivery_status", "unknown"),
                        "method_status": "missing" if row is None else row.get("delivery_status", "unknown"),
                        "interpretation": "proxy_objective_only; delivered X quality requires valid paired errors"})
                for method in METHODS:
                    row = selected[method]
                    if row is None:
                        summaries.append({"seed": seed, "budget": budget, "diversity_eta": eta,
                            "method": method, "execution_status": "missing", "delivery_status": "missing"})
                    else:
                        summaries.append({"seed": seed, "budget": budget, "diversity_eta": eta,
                            "method": method, "execution_status": row.get("execution_status", "completed"),
                            "delivery_status": row.get("delivery_status", "unknown"),
                            "feasible_delivery": valid_delivery(row),
                            "selection_objective": row.get("selection_objective", ""),
                            "external_global_error": row.get("external_global_error", ""),
                            "true_error": row.get("true_error", ""),
                            "budget_feasible": row.get("budget_feasible", ""),
                            "box_feasible": row.get("box_feasible", ""),
                            "failure_stage": row.get("failure_stage", ""),
                            "failure_reason": row.get("failure_reason", row.get("error", "")),
                            "stage3_requests": row.get("stage3_requests", ""),
                            "stage3_input_samples": row.get("stage3_input_samples", ""),
                            "stage3_seconds": row.get("inversion_seconds", ""),
                            "selection_seconds": row.get("selection_seconds", ""),
                            "payment_status": row.get("payment_status", "unknown"),
                            "scenario_artifact_dir": row.get("scenario_artifact_dir", "")})
    aggregates=[]
    for budget in sorted(set(float(row["budget"]) for row in normalized_rows)):
      for eta in sorted(set(float(row["diversity_eta"]) for row in normalized_rows)):
       for method in METHODS:
        selected=[row for row in normalized_rows if float(row["budget"])==budget
                  and float(row["diversity_eta"])==eta and row["selection_method"]==method]
        quality=[finite_float(row.get("external_global_error")) for row in selected
                 if valid_delivery(row,"external_global_error")]
        quality=[v for v in quality if v is not None]
        objective=[finite_float(row.get("selection_objective")) for row in selected]
        objective=[v for v in objective if v is not None]
        request_values=[finite_float(row.get("stage3_requests")) for row in selected]
        request_values=[v for v in request_values if v is not None]
        sample_values=[finite_float(row.get("stage3_input_samples")) for row in selected]
        sample_values=[v for v in sample_values if v is not None]
        elapsed=[finite_float(row.get("inversion_seconds")) for row in selected]
        elapsed=[v for v in elapsed if v is not None]
        aggregates.append({"budget":budget,"diversity_eta":eta,"method":method,
            "planned_seeds":len(seeds),"observed_rows":len(selected),
            "valid_deliveries":len(quality),"failure_count":sum(row.get("delivery_status")!="valid" for row in selected),
            "failure_rate":(len(selected)-len(quality))/len(seeds) if seeds else None,
            "mean_selection_objective":float(np.mean(objective)) if objective else None,
            "mean_external_error_valid":float(np.mean(quality)) if quality else None,
            "median_external_error_valid":float(np.median(quality)) if quality else None,
            "q1_external_error_valid":float(np.quantile(quality,.25)) if quality else None,
            "q3_external_error_valid":float(np.quantile(quality,.75)) if quality else None,
            "mean_stage3_requests_known":float(np.mean(request_values)) if request_values else None,
            "mean_stage3_input_samples_known":float(np.mean(sample_values)) if sample_values else None,
            "mean_inversion_wall_seconds_known":float(np.mean(elapsed)) if elapsed else None,
            "cost_note":"shared training and preparation cost is in shared_costs_per_seed.csv"})
    objective_violations = [row for row in pairs if row["method"] == "objective_greedy"
                            and row["objective_gap_from_exact"] is not None
                            and row["objective_gap_from_exact"] < -1e-12]
    return summaries, pairs, aggregates, shared_costs, {**coverage,
        "duplicate_keys": json.dumps(coverage["duplicate_keys"]),
        "missing_keys": json.dumps(coverage["missing_keys"]),
        "unexpected_keys": json.dumps(coverage["unexpected_keys"]),
        "objective_greedy_exceeded_exact_count": len(objective_violations)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("P6 analysis directory must be empty.")
    summaries, pairs, aggregates, shared_costs, coverage = analyze(args.results_dir)
    write_csv(output / "method_results.csv", summaries)
    write_csv(output / "paired_results.csv", pairs)
    write_csv(output / "condition_summary.csv", aggregates)
    write_csv(output / "coverage_audit.csv", [coverage])
    write_csv(output / "shared_costs_per_seed.csv", shared_costs)
    (output / "analysis_manifest.json").write_text(json.dumps({
        "results_dir": str(Path(args.results_dir).resolve()),
        "selection_reference": "exact objective, only same-objective comparisons",
        "delivered_X_metric": "pooled evaluator error on valid paired deliveries",
        "formal_judgment_allowed": coverage["complete"],
    }, ensure_ascii=False, indent=2) + "\n")
    if coverage["objective_greedy_exceeded_exact_count"]:
        raise ValueError("Objective greedy exceeded exact on a complete same-objective condition.")


if __name__ == "__main__":
    main()

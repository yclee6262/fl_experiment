"""Paired seed-level analysis for the P2 alpha-strategy comparison."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


BASELINE = "A_uniform_no_update"
METHODS = (
    BASELINE,
    "B_stage0_no_update",
    "C_uniform_fixed",
    "D_uniform_guarded",
)


def read_rows(result_dirs):
    rows = []
    for directory in map(Path, result_dirs):
        with (directory / "summary.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["result_dir"] = str(directory)
                rows.append(row)
    return rows


def bootstrap_mean_ci(differences, seed=20260922, replicates=10000):
    values = np.asarray(differences, dtype=float)
    if not len(values):
        raise ValueError("Cannot bootstrap an empty paired sample.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    estimates = values[indices].mean(axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def optional_float(value):
    if value in (None, ""):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def interpret_comparison(total_scenarios, valid_pairs, low, high):
    if not valid_pairs:
        return "no_valid_pairs"
    if total_scenarios < 30:
        return "pilot_only"
    if valid_pairs < total_scenarios:
        return "incomplete_pairs"
    if high < 0:
        return "stable_improvement"
    if low > 0:
        return "stable_worsening"
    return "insufficient_evidence"


def analyze(result_dirs):
    rows = read_rows(result_dirs)
    grouped = defaultdict(dict)
    for row in rows:
        scenario = (
            row["seed"], row["dimension"], row["poison_ratio"], row["target"]
        )
        method = row.get("method_label")
        if method in grouped[scenario]:
            raise ValueError(f"Duplicate method {method} for scenario {scenario}.")
        grouped[scenario][method] = row
    for scenario, methods in grouped.items():
        missing = set(METHODS) - set(methods)
        if missing:
            raise ValueError(f"Scenario {scenario} is missing methods: {sorted(missing)}")

    metrics = (
        "external_global_error",
        "final_target_error",
        "final_evaluation_loss",
        "final_consistency_l1",
    )
    comparisons = []
    for poison_ratio in sorted({scenario[2] for scenario in grouped}, key=float):
        selected = [methods for scenario, methods in grouped.items()
                    if scenario[2] == poison_ratio]
        for method in METHODS[1:]:
            for metric in metrics:
                pairs = [
                    (optional_float(methods[method].get(metric)),
                     optional_float(methods[BASELINE].get(metric)))
                    for methods in selected
                ]
                differences = [left - right for left, right in pairs
                               if left is not None and right is not None]
                low, high = (
                    bootstrap_mean_ci(differences) if differences else (None, None)
                )
                comparisons.append({
                    "poison_ratio": poison_ratio,
                    "baseline": BASELINE,
                    "method": method,
                    "metric": metric,
                    "total_scenarios": len(selected),
                    "valid_paired_seeds": len(differences),
                    "missing_pairs": len(selected) - len(differences),
                    "mean_delta": (float(np.mean(differences)) if differences else None),
                    "median_delta": (float(np.median(differences)) if differences else None),
                    "ci95_low": low,
                    "ci95_high": high,
                    "improved_seeds": int(np.sum(np.asarray(differences) < 0)),
                    "worsened_seeds": int(np.sum(np.asarray(differences) > 0)),
                    "interpretation": interpret_comparison(
                        len(selected), len(differences), low, high),
                })
    method_rows = []
    for poison_ratio in sorted({scenario[2] for scenario in grouped}, key=float):
        selected = [methods for scenario, methods in grouped.items()
                    if scenario[2] == poison_ratio]
        for method in METHODS:
            method_rows.append({
                "poison_ratio": poison_ratio,
                "method": method,
                "seeds": len(selected),
                "mean_external_error": float(np.mean([
                    float(methods[method]["external_global_error"]) for methods in selected
                ])),
                "median_external_error": float(np.median([
                    float(methods[method]["external_global_error"]) for methods in selected
                ])),
                "mean_true_error": float(np.mean([
                    float(methods[method]["final_target_error"]) for methods in selected
                ])),
                "box_infeasible": sum(
                    methods[method]["final_box_feasible"] != "True" for methods in selected
                ),
                "mean_requests": float(np.mean([
                    int(methods[method]["stage3_requests"])
                    + int(methods[method]["contribution_requests"])
                    + int(methods[method]["payment_requests"])
                    for methods in selected
                ])),
                "converged": sum(methods[method]["converged"] == "True"
                                 for methods in selected),
                "valid_q": sum(
                    methods[method].get("final_contribution_weights", "[]") not in ("", "[]")
                    for methods in selected
                ),
            })
    return method_rows, comparisons


def write_csv(path, rows):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite analysis output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs="+")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Analysis output directory must be empty.")
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries, comparisons = analyze(args.result_dirs)
    write_csv(output_dir / "method_summary.csv", summaries)
    write_csv(output_dir / "paired_comparisons.csv", comparisons)
    (output_dir / "analysis_manifest.json").write_text(json.dumps({
        "result_dirs": args.result_dirs,
        "baseline": BASELINE,
        "bootstrap_replicates": 10000,
        "formal_threshold_seeds": 30,
        "note": (
            "Scenarios with fewer than 30 planned seeds are labeled pilot_only; "
            "formal metrics with undefined values are labeled incomplete_pairs."
        ),
    }, indent=2) + "\n")
    print(f"Analyzed {sum(row['seeds'] for row in summaries) // len(METHODS)} scenarios")


if __name__ == "__main__":
    main()

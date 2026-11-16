"""Analyze P3 candidate-exclusion versus full-participant-removal pilots."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


TOLERANCE = 1e-8


def load_csv(paths, filename):
    rows = []
    for path in map(Path, paths):
        with (path / filename).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["result_dir"] = str(path)
                rows.append(row)
    return rows


def sign(value, tolerance=TOLERANCE):
    value = float(value)
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def safe_spearman(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) < 2 or np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return None
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else None


def analyze(result_dirs):
    effects = load_csv(result_dirs, "participant_effects.csv")
    scenarios = load_csv(result_dirs, "scenario_summary.csv")
    by_scenario = defaultdict(list)
    for row in effects:
        key = (row["result_dir"], row["seed"], row["poison_ratio"])
        by_scenario[key].append(row)

    scenario_results = []
    for key, rows in sorted(by_scenario.items()):
        operational = [float(row["operational_internal_contribution"]) for row in rows]
        candidate_external = [float(row["operational_external_contribution"]) for row in rows]
        full_external = [float(row["full_removal_external_contribution"]) for row in rows]
        op_sign = [sign(value) for value in operational]
        full_sign = [sign(value) for value in full_external]
        nonzero = [index for index in range(len(rows))
                   if op_sign[index] != 0 and full_sign[index] != 0]
        predicted_negative = [index for index, value in enumerate(op_sign) if value < 0]
        scenario_results.append({
            "result_dir": key[0],
            "seed": key[1],
            "poison_ratio": key[2],
            "coalition_size": len(rows),
            "exact_sign_agreement": sum(a == b for a, b in zip(op_sign, full_sign)),
            "nonzero_pairs": len(nonzero),
            "nonzero_direction_agreement": sum(
                op_sign[index] == full_sign[index] for index in nonzero
            ),
            "operational_vs_full_spearman": safe_spearman(operational, full_external),
            "candidate_external_vs_full_spearman": safe_spearman(
                candidate_external, full_external
            ),
            "predicted_negative": len(predicted_negative),
            "predicted_negative_removal_improved": sum(
                full_external[index] < -TOLERANCE for index in predicted_negative
            ),
            "poisoned_agents": sum(row["is_poisoned"] == "True" for row in rows),
            "poisoned_removal_improved": sum(
                row["is_poisoned"] == "True"
                and float(row["full_removal_external_contribution"]) < -TOLERANCE
                for row in rows
            ),
        })

    scenario_by_key = {
        (row["result_dir"], row["seed"], row["poison_ratio"]): row
        for row in scenarios
    }
    aggregate = []
    for poison in sorted({row["poison_ratio"] for row in effects}, key=float):
        selected_effects = [row for row in effects if row["poison_ratio"] == poison]
        selected_scenarios = [row for row in scenario_results if row["poison_ratio"] == poison]
        source_scenarios = [
            scenario_by_key[(row["result_dir"], row["seed"], row["poison_ratio"])]
            for row in selected_scenarios
        ]
        op_sign = [sign(row["operational_internal_contribution"])
                   for row in selected_effects]
        full_sign = [sign(row["full_removal_external_contribution"])
                     for row in selected_effects]
        nonzero = [index for index in range(len(selected_effects))
                   if op_sign[index] != 0 and full_sign[index] != 0]
        predicted_negative = [index for index, value in enumerate(op_sign) if value < 0]
        rank_values = [row["operational_vs_full_spearman"] for row in selected_scenarios
                       if row["operational_vs_full_spearman"] is not None]
        candidate_rank = [row["candidate_external_vs_full_spearman"]
                          for row in selected_scenarios
                          if row["candidate_external_vs_full_spearman"] is not None]
        poisoned = [row for row in selected_effects if row["is_poisoned"] == "True"]
        aggregate.append({
            "poison_ratio": poison,
            "seeds": len(selected_scenarios),
            "participants": len(selected_effects),
            "exact_sign_agreement_rate": (
                sum(a == b for a, b in zip(op_sign, full_sign)) / len(selected_effects)
            ),
            "nonzero_pairs": len(nonzero),
            "nonzero_direction_agreement_rate": (
                sum(op_sign[index] == full_sign[index] for index in nonzero) / len(nonzero)
                if nonzero else None
            ),
            "mean_operational_vs_full_spearman": (
                float(np.mean(rank_values)) if rank_values else None
            ),
            "valid_operational_vs_full_rank_seeds": len(rank_values),
            "mean_candidate_external_vs_full_spearman": (
                float(np.mean(candidate_rank)) if candidate_rank else None
            ),
            "valid_candidate_external_vs_full_rank_seeds": len(candidate_rank),
            "predicted_negative": len(predicted_negative),
            "predicted_negative_removal_improvement_rate": (
                sum(full_sign[index] < 0 for index in predicted_negative)
                / len(predicted_negative) if predicted_negative else None
            ),
            "invalid_q_scenarios": sum(
                row["operational_q_valid"] != "True" for row in source_scenarios
            ),
            "baseline_box_infeasible": sum(
                row.get("baseline_box_feasible") != "True" for row in source_scenarios
            ),
            "candidate_exclusion_box_infeasible": sum(
                row.get("operational_box_feasible") != "True"
                for row in selected_effects
            ),
            "full_removal_box_infeasible": sum(
                row.get("full_removal_box_feasible") != "True"
                for row in selected_effects
            ),
            "mean_candidate_exclusion_requests": float(np.mean([
                int(row["candidate_exclusion_requests"]) for row in source_scenarios
            ])),
            "mean_full_removal_requests": float(np.mean([
                int(row["full_removal_requests"]) for row in source_scenarios
            ])),
            "poisoned_participants": len(poisoned),
            "poisoned_removal_improvement_rate": (
                sum(float(row["full_removal_external_contribution"]) < -TOLERANCE
                    for row in poisoned) / len(poisoned) if poisoned else None
            ),
        })
    return scenario_results, aggregate


def write_csv(path, rows):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs="+")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Analysis output directory must be empty.")
    output.mkdir(parents=True, exist_ok=True)
    scenarios, aggregate = analyze(args.result_dirs)
    write_csv(output / "scenario_metrics.csv", scenarios)
    write_csv(output / "aggregate_metrics.csv", aggregate)
    (output / "analysis_manifest.json").write_text(json.dumps({
        "result_dirs": args.result_dirs,
        "sign_tolerance": TOLERANCE,
        "scope": "5-seed P3 pilot; descriptive evidence only",
        "positive_contribution_meaning": "removal increased pooled-evaluator error",
    }, indent=2) + "\n")
    print(f"Analyzed {len(scenarios)} scenarios")


if __name__ == "__main__":
    main()

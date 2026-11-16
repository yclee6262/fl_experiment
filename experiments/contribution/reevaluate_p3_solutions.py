"""Reload P3 evaluator checkpoints and re-score every persisted solution."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from research_evaluation import load_evaluator_artifact, score_solution
from run_experiments import true_function


SOLUTION_FIELDS = {
    "baseline": (
        "baseline_solution", "baseline_external_error", "baseline_true_error",
        "baseline_box_feasible",
    ),
    "candidate_only": (
        "operational_restricted_solution", "operational_external_error_without", None,
        "operational_box_feasible",
    ),
    "full_removal": (
        "full_removal_solution", "full_removal_external_error_without",
        "full_removal_true_error_without", "full_removal_box_feasible",
    ),
}


def reevaluate(results_dir, tolerance=1e-7):
    results_dir = Path(results_dir)
    manifest = json.loads((results_dir / "run_manifest.json").read_text())
    lower = float(manifest["arguments"]["feasible_lower"])
    upper = float(manifest["arguments"]["feasible_upper"])
    with (results_dir / "participant_effects.csv").open(newline="", encoding="utf-8") as handle:
        effects = list(csv.DictReader(handle))
    evaluators = {}
    output = []
    for row in effects:
        artifact = row["scenario_artifact_dir"]
        if artifact not in evaluators:
            evaluators[artifact], _ = load_evaluator_artifact(results_dir / artifact)
        target = float(row["target"])
        for solution_type, (solution_field, error_field, truth_field, box_field) in (
            SOLUTION_FIELDS.items()
        ):
            solution = np.asarray(json.loads(row[solution_field]), dtype=float)
            _, error = score_solution(evaluators[artifact], solution, target)
            truth_error = abs(true_function(solution) - target)
            saved_error = float(row[error_field])
            saved_truth = float(row[truth_field]) if truth_field else None
            feasible = bool(
                np.all(np.isfinite(solution))
                and np.all(solution >= lower - 1e-8)
                and np.all(solution <= upper + 1e-8)
            )
            max_box_violation = float(max(
                0.0, np.max(lower - solution), np.max(solution - upper)
            )) if np.all(np.isfinite(solution)) else float("inf")
            output.append({
                "seed": row["seed"],
                "poison_ratio": row["poison_ratio"],
                "agent_id": row["agent_id"],
                "solution_type": solution_type,
                "recomputed_external_error": error,
                "saved_external_error": saved_error,
                "external_error_absolute_delta": abs(error - saved_error),
                "external_score_matches": abs(error - saved_error) <= tolerance,
                "recomputed_true_error": truth_error,
                "saved_true_error": "" if saved_truth is None else saved_truth,
                "true_score_matches": (
                    "" if saved_truth is None
                    else abs(truth_error - saved_truth) <= tolerance
                ),
                "recomputed_box_feasible": feasible,
                "recomputed_max_box_violation": max_box_violation,
                "saved_box_feasible": row[box_field],
                "box_matches": str(feasible) == row[box_field],
                "scenario_artifact_dir": artifact,
            })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    args = parser.parse_args()
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    rows = reevaluate(args.results_dir, args.tolerance)
    with output_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mismatches = sum(
        row["external_score_matches"] is not True
        or row["box_matches"] is not True
        or row["true_score_matches"] is False
        for row in rows
    )
    print(f"Re-evaluated {len(rows)} persisted P3 solutions; mismatches={mismatches}")


if __name__ == "__main__":
    main()

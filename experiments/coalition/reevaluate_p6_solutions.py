"""Failure-aware P6 re-evaluation keyed by the full experimental identity."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from research_evaluation import load_evaluator_artifact, score_solution
from run_experiments import true_function


IDENTITY = ("seed", "selection_method", "budget", "diversity_eta")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--box-tolerance", type=float, default=1e-8)
    args = parser.parse_args()
    root, output_path = Path(args.results_dir), Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    with (root / "selection_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    seen, cache, results = set(), {}, []
    for row in rows:
        key = tuple(row.get(field, "") for field in IDENTITY)
        if key in seen:
            raise ValueError(f"Duplicate P6 re-evaluation key: {key}")
        seen.add(key)
        identity = {field: row.get(field, "") for field in IDENTITY}
        identity["budget"] = row.get("budget", row.get("selection_budget", ""))
        solution_text = row.get("final_solution", "")
        if not solution_text or solution_text in {"[]", "null"}:
            results.append({**identity, "reevaluation_status": "not_applicable",
                "reason": row.get("failure_reason", row.get("delivery_status", "no delivered X")),
                "external_score_matches": "not_applicable",
                "true_score_matches": "not_applicable", "recomputed_box_feasible": "not_applicable",
                "scenario_artifact_dir": row.get("scenario_artifact_dir", "")})
            continue
        artifact = row["scenario_artifact_dir"]
        if not (root / artifact).is_dir():
            results.append({**identity, "reevaluation_status": "missing_artifact",
                "reason": artifact, "external_score_matches": "False",
                "true_score_matches": "False", "recomputed_box_feasible": "False",
                "scenario_artifact_dir": artifact})
            continue
        if artifact not in cache:
            cache[artifact], _ = load_evaluator_artifact(root / artifact)
        solution = np.asarray(json.loads(solution_text), dtype=float)
        target = float(row["formal_target"])
        prediction, external_error = score_solution(cache[artifact], solution, target)
        truth_error = abs(true_function(solution) - target)
        saved_external = float(row["external_global_error"])
        saved_truth = float(row["true_error"])
        violation = max(0.0, float(np.max(solution - 1.0)), float(np.max(-1.0 - solution)))
        results.append({**identity, "reevaluation_status": "scored", "reason": "",
            "recomputed_external_prediction": prediction,
            "recomputed_external_error": external_error,
            "saved_external_error": saved_external,
            "external_score_delta": abs(external_error-saved_external),
            "external_score_matches": abs(external_error-saved_external) <= args.tolerance,
            "recomputed_true_error": truth_error, "saved_true_error": saved_truth,
            "true_score_delta": abs(truth_error-saved_truth),
            "true_score_matches": abs(truth_error-saved_truth) <= args.tolerance,
            "recomputed_box_violation": violation,
            "recomputed_box_feasible": bool(np.all(np.isfinite(solution)) and violation <= args.box_tolerance),
            "scenario_artifact_dir": artifact})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", newline="", encoding="utf-8") as handle:
        fields = list(dict.fromkeys(key for row in results for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    mismatches = sum(row.get("external_score_matches") == "False" or
                     row.get("true_score_matches") == "False" or
                     row.get("recomputed_box_feasible") == "False" for row in results)
    print(f"Re-evaluated {len(results)} P6 rows, including non-deliveries; mismatches={mismatches}")


if __name__ == "__main__":
    main()

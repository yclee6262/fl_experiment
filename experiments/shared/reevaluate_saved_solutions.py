"""Failure-aware re-evaluation of P4/P5 delivered X values."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from research_evaluation import load_evaluator_artifact, score_solution
from run_experiments import true_function


KEYS = ("seed", "dimension", "poison_ratio", "target", "method_label")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def reevaluate(results_dir, tolerance=1e-7, box_tolerance=1e-8):
    results_dir = Path(results_dir)
    manifest = json.loads((results_dir / "run_manifest.json").read_text())
    lower = float(manifest["arguments"]["feasible_lower"])
    upper = float(manifest["arguments"]["feasible_upper"])
    evaluator_cache, output, seen = {}, [], set()
    for row in read_csv(results_dir / "summary.csv"):
        identity = tuple(row.get(key, "") for key in KEYS)
        if identity in seen:
            raise ValueError(f"Duplicate re-evaluation identity: {identity}")
        seen.add(identity)
        base = dict(zip(KEYS, identity))
        artifact = row.get("scenario_artifact_dir", "")
        text = row.get("final_solution", "")
        if not text or text in {"[]", "null"}:
            output.append({**base, "reevaluation_status": "not_applicable",
                "reason": row.get("failure_reason", row.get("stop_reason", "no delivered X")),
                "external_score_matches": "not_applicable", "true_score_matches": "not_applicable",
                "recomputed_box_feasible": "not_applicable", "scenario_artifact_dir": artifact})
            continue
        if not (results_dir / artifact).is_dir():
            output.append({**base, "reevaluation_status": "missing_artifact", "reason": artifact,
                "external_score_matches": False, "true_score_matches": False,
                "recomputed_box_feasible": False, "scenario_artifact_dir": artifact})
            continue
        if artifact not in evaluator_cache:
            evaluator_cache[artifact], _ = load_evaluator_artifact(results_dir / artifact)
        solution = np.asarray(json.loads(text), dtype=float)
        target = float(row["target"])
        prediction, external_error = score_solution(evaluator_cache[artifact], solution, target)
        truth_error = abs(true_function(solution) - target)
        saved_external = float(row["external_global_error"])
        saved_truth = float(row["final_target_error"])
        violation = max(0.0, float(np.max(solution-upper)), float(np.max(lower-solution)))
        output.append({**base, "reevaluation_status": "scored", "reason": "",
            "final_solution": text, "recomputed_external_prediction": prediction,
            "recomputed_external_error": external_error, "saved_external_error": saved_external,
            "external_error_absolute_delta": abs(external_error-saved_external),
            "external_score_matches": abs(external_error-saved_external) <= tolerance,
            "external_truth_disagreement": abs(prediction - true_function(solution)),
            "recomputed_true_error": truth_error, "saved_true_error": saved_truth,
            "true_error_absolute_delta": abs(truth_error-saved_truth),
            "true_score_matches": abs(truth_error-saved_truth) <= tolerance,
            "recomputed_box_violation": violation,
            "recomputed_box_feasible": bool(np.all(np.isfinite(solution)) and violation <= box_tolerance),
            "scenario_artifact_dir": artifact})
    return output


def write_csv(path, rows):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing re-evaluation: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--box-tolerance", type=float, default=1e-8)
    args = parser.parse_args()
    rows = reevaluate(args.results_dir, args.tolerance, args.box_tolerance)
    write_csv(args.output, rows)
    mismatches = sum(row.get("external_score_matches") is False or
                     row.get("true_score_matches") is False or
                     row.get("recomputed_box_feasible") is False for row in rows)
    print(f"Re-evaluated {len(rows)} rows, including failed runs; mismatches={mismatches}")


if __name__ == "__main__":
    main()

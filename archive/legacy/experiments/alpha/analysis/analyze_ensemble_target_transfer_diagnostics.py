"""Post-hoc coverage and case diagnostics for the completed transfer study."""

import argparse
import json
from pathlib import Path

import numpy as np

from research_analysis import paired_bootstrap_mean_ci


def diagnose(preregistration, assessment):
    prereg = json.loads(Path(preregistration).read_text())
    assessment = Path(assessment)
    output = {"status": "posthoc_diagnostics_not_success_gates", "scenarios": {}}
    for scenario in prereg["scenarios"]:
        root = Path(prereg["baseline_roots"][scenario]) / scenario
        shared_rows = [json.loads(line) for line in (root / "shared_costs.jsonl").read_text().splitlines()]
        rows = [json.loads(line) for line in
                (assessment / scenario / "seed_rows.jsonl").read_text().splitlines()]
        if ([row["seed"] for row in shared_rows] != prereg["seeds"]
                or [row["seed"] for row in rows] != prereg["seeds"]):
            raise ValueError(f"Incomplete or out-of-order seeds: {scenario}")
        metrics = [json.loads((root / row["artifact_path"] / "metadata.json").read_text())["metrics"]
                   for row in shared_rows]
        near = np.asarray([metric["target_near_samples"] for metric in metrics], dtype=int)
        mse = np.asarray([metric["target_near_truth_mse"] for metric in metrics], dtype=float)
        pooled = np.asarray([row["new_pooled_error"]-row["old_pooled_error"]
                             for row in rows], dtype=float)
        truth = np.asarray([row["new_truth_error"]-row["old_truth_error"]
                            for row in rows], dtype=float)
        if not np.isfinite(mse).all() or not np.isfinite(pooled).all() or not np.isfinite(truth).all():
            raise ValueError(f"Nonfinite diagnostic value: {scenario}")
        best = rows[int(np.argmin(pooled))]
        worst = rows[int(np.argmax(pooled))]
        output["scenarios"][scenario] = {
            "evaluator_artifacts": len(metrics),
            "target_near_samples_min": int(near.min()),
            "target_near_samples_median": float(np.median(near)),
            "target_near_samples_max": int(near.max()),
            "target_near_truth_mse_mean": float(mse.mean()),
            "target_near_sparse_lt30_count": sum(bool(metric["target_near_sparse"])
                                                 for metric in metrics),
            "paired_pooled_median": float(np.median(pooled)),
            "paired_truth_mean": float(truth.mean()),
            "paired_truth_95pct_ci_descriptive": paired_bootstrap_mean_ci(truth),
            "pooled_truth_direction_conflicts_both_gt_0p001": int(np.sum(
                (pooled * truth < 0) & (np.abs(pooled) > 0.001) & (np.abs(truth) > 0.001))),
            "best_improvement": {"seed": best["seed"],
                                 "pooled_delta": float(pooled.min())},
            "worst_regression": {
                "seed": worst["seed"], "pooled_delta": float(pooled.max()),
                "old_selected_initialization": worst["old_selected_initialization"],
                "new_selected_initialization": worst["new_selected_initialization"],
                "old_pooled_error": worst["old_pooled_error"],
                "new_pooled_error": worst["new_pooled_error"],
                "new_candidate_pooled_errors": {method: value["pooled_error"]
                                                for method, value in worst["candidates"].items()},
            },
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preregistration")
    parser.add_argument("assessment_dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = diagnose(args.preregistration, args.assessment_dir)
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

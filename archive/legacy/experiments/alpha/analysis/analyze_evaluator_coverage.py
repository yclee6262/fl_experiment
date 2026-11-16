"""Export global, hard-region, and target-distance evaluator coverage metrics."""

import argparse
import csv
import json
from pathlib import Path


BINS = ("0_to_0p1", "0p1_to_0p25", "0p25_to_0p5", "0p5_to_1", "over_1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.results_root)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    records = []
    for path in sorted(root.glob("**/metadata.json")):
        if "artifacts" not in path.parts:
            continue
        metadata = json.loads(path.read_text())
        scenario = metadata["scenario"]
        metrics = metadata["metrics"]
        row = {
            "seed": scenario["seed"], "dimension": scenario["dimension"],
            "poison_ratio": scenario["poison_ratio"], "target": scenario["target"],
            "scenario_artifact": str(path.parent.relative_to(root)),
            "train_samples": metrics["train_samples"],
            "validation_samples": metrics["validation_samples"],
            "test_samples": metrics["test_samples"],
            "global_test_mse": metrics["test_mse"],
            "global_test_truth_mse": metrics["test_truth_mse"],
            "hard_test_mse": metrics["hard_test_mse"],
            "hard_test_truth_mse": metrics["hard_test_truth_mse"],
            "target_near_samples_within_0p25": metrics.get("target_near_samples"),
            "target_near_truth_mse_within_0p25": metrics.get("target_near_truth_mse"),
            "target_near_sparse_lt30": metrics.get("target_near_sparse"),
            "sparse_rule_interpretation": "descriptive count flag; not a validated reliability threshold or OOD claim",
        }
        for bin_name in BINS:
            for suffix in ("samples", "truth_mse", "sparse_lt30"):
                key = f"target_distance_{bin_name}_{suffix}"
                row[key] = metrics.get(key, "unavailable_in_legacy_artifact")
        records.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        output.write_text("No evaluator artifacts found\n")
        return
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()

"""Posthoc candidate-ranking diagnosis on completed alpha experiments.

Pooled candidate scores are read only to measure ranking error after internal
selection. This is exploratory and must not select or alter a delivered X.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from run_validation_calibrated_alpha import CONDITIONAL_METHOD


def _rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def analyze(root):
    root = Path(root)
    manifest = json.loads((root / "suite_manifest.json").read_text())
    verification = json.loads((root / "selection_verification.json").read_text())
    if manifest["status"] != "complete" or not verification["passed"]:
        raise ValueError(f"Unverified experiment root: {root}")
    scenario, = manifest["scenarios"]
    folder = root / scenario
    candidate_rows = _rows(folder / "candidate_rows.csv")
    summary_rows = _rows(folder / "summary.csv")
    candidate = {(int(r["seed"]), r["method"]): r for r in candidate_rows}
    summary = {(int(r["seed"]), r["method"]): r for r in summary_rows}
    if (len(candidate) != len(candidate_rows)
            or len(summary) != len(summary_rows)):
        raise ValueError(f"Duplicate candidate or summary key: {root}")
    records = []
    for seed in manifest["seeds"]:
        row = summary[seed, CONDITIONAL_METHOD]
        if row["delivery_status"] != "valid" or row["final_box_feasible"] != "True":
            raise ValueError(f"Invalid conditional delivery: {scenario}/{seed}")
        audit = json.loads(row["alpha_audit"])
        methods = tuple(audit["candidate_method_order"])
        internal = np.asarray([float(audit["internal_scores"][m][
            "internal_target_error"]) for m in methods])
        pooled = np.asarray([float(candidate[seed, m]["external_global_error"])
                             for m in methods])
        if not (np.isfinite(internal).all() and np.isfinite(pooled).all()):
            raise ValueError(f"Nonfinite candidate score: {scenario}/{seed}")
        chosen = methods.index(audit["selected_initialization"])
        if abs(float(row["external_global_error"])-pooled[chosen]) > 1e-7:
            raise ValueError(f"Chosen pooled score mismatch: {scenario}/{seed}")
        correlation = float(spearmanr(internal, pooled).statistic)
        if not np.isfinite(correlation):
            correlation = None
        records.append({
            "scenario": scenario, "seed": seed,
            "candidate_count": len(methods),
            "chosen": methods[chosen],
            "triggered": audit["conditional_target"]["triggered"],
            "spearman_internal_vs_pooled_posthoc": correlation,
            "chosen_is_pooled_best_posthoc": bool(pooled[chosen] == pooled.min()),
            "chosen_is_pooled_worst_posthoc": bool(pooled[chosen] == pooled.max()),
            "selector_regret_posthoc": float(pooled[chosen]-pooled.min()),
            "pooled_best_method_posthoc": methods[int(np.argmin(pooled))],
            "pooled_best_error_posthoc": float(pooled.min()),
            "chosen_pooled_error": float(pooled[chosen]),
        })
    correlations = [r["spearman_internal_vs_pooled_posthoc"] for r in records
                    if r["spearman_internal_vs_pooled_posthoc"] is not None]
    regrets = np.asarray([r["selector_regret_posthoc"] for r in records])
    return {
        "scenario": scenario, "seed_count": len(records),
        "evidence_status": "posthoc_exploratory_no_method_selection",
        "median_spearman": float(np.median(correlations)) if correlations else None,
        "negative_rank_correlation_count": sum(x < 0 for x in correlations),
        "undefined_rank_correlation_count": len(records)-len(correlations),
        "chosen_pooled_best_count": sum(r["chosen_is_pooled_best_posthoc"] for r in records),
        "chosen_pooled_worst_count": sum(r["chosen_is_pooled_worst_posthoc"] for r in records),
        "mean_selector_regret_posthoc": float(regrets.mean()),
        "q95_selector_regret_posthoc": float(np.quantile(regrets, .95)),
        "worst_regret_seed": max(records, key=lambda r:r["selector_regret_posthoc"])["seed"],
    }, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    results, rows = [], []
    for root in args.roots:
        result, group = analyze(root)
        results.append(result)
        rows.extend(group)
    (output / "summary.json").write_text(json.dumps(results, ensure_ascii=False,
                                                    indent=2)+"\n")
    with (output / "seed_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

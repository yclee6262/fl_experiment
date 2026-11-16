"""Analyze static-alpha study without hiding missing or failed conditions."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from research_analysis import paired_bootstrap_mean_ci, finite_float
from run_alpha_initialization_study import CONFIRMATORY


def _read(root, manifest):
    rows = []
    for scenario in manifest["scenarios"]:
        path = root / scenario / "summary.csv"
        if path.exists():
            with path.open(newline="") as handle:
                rows.extend(csv.DictReader(handle))
    return rows


def _valid(row):
    return (row["delivery_status"] == "valid" and row["final_box_feasible"] == "True"
            and finite_float(row["external_global_error"]) is not None
            and row["final_solution"] not in ("", "[]"))


def analyze(root):
    root = Path(root)
    manifest = json.loads((root / "suite_manifest.json").read_text())
    rows = _read(root, manifest)
    expected = {(s, str(seed), m) for s in manifest["scenarios"]
                for seed in manifest["seeds"] for m in manifest["methods"]}
    counts = Counter((r["scenario"], r["seed"], r["method"]) for r in rows)
    missing = sorted(expected - counts.keys())
    duplicate = sorted(k for k, n in counts.items() if n > 1)
    unexpected = sorted(counts.keys() - expected)
    index = {(r["scenario"], r["seed"], r["method"]): r for r in rows}
    summary = []
    comparisons = []
    for scenario in manifest["scenarios"]:
        for method in manifest["methods"]:
            group = [index.get((scenario, str(seed), method)) for seed in manifest["seeds"]]
            values = [float(r["external_global_error"]) for r in group if r and _valid(r)]
            failures = Counter("missing" if r is None else r["status"] for r in group if r is None or not _valid(r))
            requests = [float(r["initialization_requests"] or 0) + float(r["stage3_requests"] or 0)
                        for r in group if r and r["initialization_requests"] and r["stage3_requests"]]
            summary.append({"scenario": scenario, "method": method, "expected_n": len(group),
                            "valid_n": len(values), "failure_counts": dict(failures),
                            "pooled_mean": float(np.mean(values)) if values else None,
                            "pooled_median": float(np.median(values)) if values else None,
                            "pooled_q1": float(np.quantile(values, .25)) if values else None,
                            "pooled_q3": float(np.quantile(values, .75)) if values else None,
                            "mean_method_requests": float(np.mean(requests)) if requests else None})
        for method in manifest["methods"][2:]:
            for baseline in manifest["methods"][:2]:
                differences, truth_conflicts, excluded = [], 0, Counter()
                for seed in manifest["seeds"]:
                    candidate = index.get((scenario, str(seed), method))
                    control = index.get((scenario, str(seed), baseline))
                    if not candidate or not control:
                        excluded["missing_row"] += 1
                    elif not _valid(candidate) or not _valid(control):
                        excluded["invalid_or_failed_delivery"] += 1
                    else:
                        delta = float(candidate["external_global_error"]) - float(control["external_global_error"])
                        differences.append(delta)
                        truth_a, truth_b = finite_float(candidate["final_target_error"]), finite_float(control["final_target_error"])
                        if truth_a is not None and truth_b is not None and delta * (truth_a - truth_b) < 0:
                            truth_conflicts += 1
                is_primary = (scenario in CONFIRMATORY and
                              baseline == manifest.get("primary_baseline", "A_uniform"))
                candidate_count = len(manifest["methods"][2:])
                confidence = 1 - .05 / candidate_count if is_primary else .95
                ci = paired_bootstrap_mean_ci(differences, confidence=confidence)
                candidate_failures = len(manifest["seeds"]) - sum(
                    _valid(index[(scenario, str(seed), method)]) for seed in manifest["seeds"]
                    if (scenario, str(seed), method) in index)
                control_failures = len(manifest["seeds"]) - sum(
                    _valid(index[(scenario, str(seed), baseline)]) for seed in manifest["seeds"]
                    if (scenario, str(seed), baseline) in index)
                comparisons.append({"scenario": scenario, "method": method, "baseline": baseline,
                    "paired_n": len(differences), "excluded": dict(excluded),
                    "mean_pooled_difference": float(np.mean(differences)) if differences else None,
                    "ci_lower": ci[0], "ci_upper": ci[1], "ci_confidence": confidence,
                    "pooled_truth_conflicts": truth_conflicts,
                    "candidate_failures": candidate_failures, "baseline_failures": control_failures,
                    "formal_advantage": bool(manifest["status"] == "complete" and
                        len(manifest["seeds"]) == 30 and len(differences) == 30 and
                        not missing and not duplicate and not unexpected and is_primary and
                        ci[1] is not None and ci[1] < 0 and candidate_failures <= control_failures)})
    report = {"suite_status": manifest["status"], "expected_conditions": len(expected),
              "observed_rows": len(rows), "missing_keys": missing,
              "duplicate_keys": duplicate, "unexpected_keys": unexpected,
              "complete": manifest["status"] == "complete" and not (missing or duplicate or unexpected),
              "summary": summary, "comparisons": comparisons,
              "interpretation": "Only valid paired deliveries enter quality CI; failures are separate. Pooled is primary, truth diagnostic."}
    (root / "analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parsed = parser.parse_args()
    print(json.dumps(analyze(parsed.root), ensure_ascii=False, indent=2))

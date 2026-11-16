"""Paired pooled-quality and request-cost assessment of the three-alpha portfolio."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from research_analysis import paired_bootstrap_mean_ci, valid_delivery
from run_validation_calibrated_alpha import META_METHOD, TRIO_METHOD, TRIO_CANDIDATES


METHODS = ("A_uniform", "B_stage0", META_METHOD, TRIO_METHOD)


def _read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _stats(rows, seeds, method):
    group = [rows.get((str(seed), method)) for seed in seeds]
    valid = [row for row in group if row is not None and valid_delivery(row)]
    error = np.asarray([float(row["external_global_error"]) for row in valid])
    return {
        "valid": len(valid), "failures": len(seeds)-len(valid),
        "mean_pooled": float(error.mean()) if len(error) else None,
        "median_pooled": float(np.median(error)) if len(error) else None,
        "q1_q3_pooled": [float(x) for x in np.quantile(error, [.25, .75])]
                        if len(error) else [None, None],
        "mean_truth": float(np.mean([float(r["final_target_error"])
                                      for r in valid])) if valid else None,
        "mean_requests": float(np.mean([
            float(r["initialization_requests"])+float(r["stage3_requests"])
            for r in valid])) if valid else None,
        "mean_method_seconds": float(np.mean([
            float(r["initialization_seconds"])+float(r["stage3_seconds"])
            for r in valid])) if valid else None,
        "extra_trusted_labels": sorted({int(r["extra_trusted_labels"])
                                        for r in valid}),
    }


def assess(root, seeds, scenario, prereg=None):
    root = Path(root)
    manifest = json.loads((root / "suite_manifest.json").read_text())
    verification = json.loads((root / "selection_verification.json").read_text())
    if (manifest["scenarios"] != [scenario] or manifest["seeds"] != seeds
            or tuple(manifest["methods"]) != METHODS
            or tuple(manifest["selection_variants"][TRIO_METHOD]) != TRIO_CANDIDATES
            or manifest["candidate_initializations"] != list(
                manifest["selection_variants"][META_METHOD])):
        raise ValueError("Scenario, seeds, methods, or candidate set mismatch")
    config = manifest["arguments"]
    if (config["dimensions"] != [2] or config["targets"] != [0.0]
            or config["fixed_participant_ids"] != [1, 2, 20]
            or config["constraint_mode"] != "box_constrained"
            or not config["clean_evaluator_noise"]):
        raise ValueError("Formal control settings differ")
    if prereg is not None:
        registration = json.loads(Path(prereg).read_text())
        if (registration["scenario"] != scenario or registration["seeds"] != seeds
                or registration["method"] != TRIO_METHOD
                or registration["baseline"] != META_METHOD):
            raise ValueError("Preregistration mismatch")
        snapshot = root / scenario / "source_snapshot"
        for name, digest in registration["source_sha256"].items():
            source = snapshot / name
            if (not source.is_file() or
                    hashlib.sha256(source.read_bytes()).hexdigest() != digest):
                raise ValueError(f"Preregistered source snapshot mismatch: {name}")
    rows = _read_rows(root / scenario / "summary.csv")
    counts = Counter((r["scenario"], r["seed"], r["method"]) for r in rows)
    expected = {(scenario, str(seed), method)
                for seed in seeds for method in METHODS}
    complete = (manifest["status"] == "complete" and verification["passed"]
                and set(counts) == expected
                and all(counts[key] == 1 for key in expected))
    index = {(r["seed"], r["method"]): r for r in rows}
    stats = {method: _stats(index, seeds, method)
             for method in (META_METHOD, TRIO_METHOD)}
    differences = []
    paired_requests = []
    conflicts = 0
    choices = Counter()
    for seed in seeds:
        old, new = (index.get((str(seed), method))
                    for method in (META_METHOD, TRIO_METHOD))
        if not old or not new or not valid_delivery(old) or not valid_delivery(new):
            continue
        values = [float(row[field]) for row in (old, new)
                  for field in ("external_global_error", "final_target_error",
                                "initialization_requests", "stage3_requests")]
        if not np.isfinite(values).all():
            continue
        old_pooled, new_pooled = (float(row["external_global_error"])
                                  for row in (old, new))
        old_truth, new_truth = (float(row["final_target_error"])
                                for row in (old, new))
        differences.append(new_pooled-old_pooled)
        paired_requests.append((float(old["initialization_requests"])
                                +float(old["stage3_requests"]),
                                float(new["initialization_requests"])
                                +float(new["stage3_requests"])))
        conflicts += int((new_pooled-old_pooled)*(new_truth-old_truth) < 0)
        choices[json.loads(new["alpha_audit"])["selected_initialization"]] += 1
    interval = paired_bootstrap_mean_ci(differences)
    old_requests, new_requests = (stats[method]["mean_requests"]
                                   for method in (META_METHOD, TRIO_METHOD))
    request_reduction = (1-new_requests/old_requests
                         if old_requests is not None and new_requests is not None
                         and old_requests > 0 else None)
    quality_gate = bool(complete and len(differences) == len(seeds)
                        and interval[1] is not None and interval[1] < 0
                        and stats[TRIO_METHOD]["failures"]
                        <= stats[META_METHOD]["failures"])
    cost_gate = bool(request_reduction is not None and request_reduction >= .35
                     and stats[TRIO_METHOD]["extra_trusted_labels"] == [0])
    result = {
        "scenario": scenario, "seeds": seeds,
        "complete_and_verified": complete,
        "preregistered_source_checked": prereg is not None,
        "paired_n": len(differences), "methods": stats,
        "mean_three_minus_five": float(np.mean(differences))
                                 if differences else None,
        "paired_95pct_ci": [None if x is None else float(x) for x in interval],
        "wins_ties_losses": [sum(x < 0 for x in differences),
                             sum(x == 0 for x in differences),
                             sum(x > 0 for x in differences)],
        "q95_paired_delta": float(np.quantile(differences, .95))
                            if differences else None,
        "largest_regression": max(differences) if differences else None,
        "pooled_truth_direction_conflicts": conflicts,
        "selected_initializations": dict(choices),
        "mean_paired_request_savings": float(np.mean([
            old-new for old,new in paired_requests])) if paired_requests else None,
        "request_reduction_fraction": request_reduction,
        "quality_gate": quality_gate, "cost_gate": cost_gate,
        "strict_quality_and_cost_gate": quality_gate and cost_gate,
        "rule": "Pooled absolute error primary; complete feasible seed pairs, "
                "10000 fixed-RNG seed-bootstrap 95% CI upper < 0, no higher "
                "failure rate, >=35% lower method-independent model requests, "
                "and zero extra trusted labels. Truth is diagnostic only.",
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--scenario", default="low_quality_25pct")
    parser.add_argument("--first-seed", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--prereg")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    seeds = list(range(args.first_seed, args.first_seed+args.count))
    result = assess(args.root, seeds, args.scenario, args.prereg)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False,
                                            indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

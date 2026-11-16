"""Assess preregistered transfer of the ensemble-target Stage 3 objective.

The paired unit is a seed within each scenario. Both scenarios are fixed before
formal runs; pooled and synthetic-truth scores are never fed to the selector.
"""

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def _source_workspace(source_file):
    """Find the flat workspace or a standalone historical source snapshot."""
    start = Path(source_file).resolve().parent
    for candidate in (start, *start.parents):
        if all((candidate / name).is_file() for name in
               ("host_server.py", "run_experiments.py", "dataset.py")):
            return candidate
    raise FileNotFoundError(f"Cannot locate source workspace from {source_file}")


import numpy as np

from explore_ensemble_target_objective import analyze as replay_one_seed
from research_analysis import paired_bootstrap_mean_ci


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _baseline_check(root, scenario, seeds, prereg):
    suite = json.loads((root / "suite_manifest.json").read_text())
    verification = json.loads((root / "selection_verification.json").read_text())
    if (suite["status"] != "complete" or suite["scenarios"] != [scenario]
            or suite["seeds"] != seeds
            or suite["candidate_initializations"] != prereg["candidate_initializations"]
            or suite["methods"] != prereg["baseline_methods"]
            or any(suite["arguments"].get(key) != value
                   for key, value in prereg["baseline_arguments_by_scenario"][scenario].items())
            or not verification["passed"]):
        raise ValueError(f"Unverified or incompatible baseline: {scenario}")


def _scenario_analysis(rows, expected_seeds, scenario):
    valid = [row for row in rows if row["status"] == "valid"]
    paired = np.asarray([row["new_pooled_error"] - row["old_pooled_error"]
                         for row in valid], dtype=float)
    truth = np.asarray([row["new_truth_error"] - row["old_truth_error"]
                        for row in valid], dtype=float)
    requests = np.asarray([row["new_stage3_requests"] - row["old_stage3_requests"]
                           for row in valid], dtype=float)
    if (not np.isfinite(paired).all() or not np.isfinite(truth).all()
            or not np.isfinite(requests).all()):
        raise ValueError(f"Nonfinite valid-pair metric: {scenario}")
    complete = ([row["seed"] for row in rows] == expected_seeds
                and len(valid) == len(expected_seeds)
                and all(row["candidate_failures"] == 0 for row in valid))
    simultaneous_ci = paired_bootstrap_mean_ci(paired, confidence=0.975)
    ordinary_ci = paired_bootstrap_mean_ci(paired, confidence=0.95)
    quality = complete and simultaneous_ci[1] is not None and simultaneous_ci[1] < 0
    cost = complete and bool(len(requests)) and float(requests.mean()) <= 0
    return {
        "scenario": scenario, "expected_seeds": len(expected_seeds),
        "recorded_seeds": len(rows), "valid_pairs": len(valid),
        "failure_statuses": dict(Counter(row["status"] for row in rows if row["status"] != "valid")),
        "mean_pooled_error_control": (float(np.mean([row["old_pooled_error"] for row in valid]))
                                      if valid else None),
        "mean_pooled_error_treatment": (float(np.mean([row["new_pooled_error"] for row in valid]))
                                        if valid else None),
        "mean_paired_pooled_delta_treatment_minus_control": (float(paired.mean())
                                                              if len(paired) else None),
        "paired_bootstrap_95pct_ci_pooled": ordinary_ci,
        "paired_bootstrap_97p5pct_simultaneous_ci_pooled": simultaneous_ci,
        "pooled_wins_ties_losses": ([int((paired < -1e-7).sum()),
                                      int((np.abs(paired) <= 1e-7).sum()),
                                      int((paired > 1e-7).sum())] if len(paired) else None),
        "mean_stage3_requests_control": (float(np.mean([row["old_stage3_requests"]
                                                        for row in valid])) if valid else None),
        "mean_stage3_requests_treatment": (float(np.mean([row["new_stage3_requests"]
                                                          for row in valid])) if valid else None),
        "mean_stage3_request_delta": float(requests.mean()) if len(requests) else None,
        "stage3_request_delta_95pct_ci": paired_bootstrap_mean_ci(requests),
        "mean_truth_delta_diagnostic": float(truth.mean()) if len(truth) else None,
        "pooled_truth_direction_conflicts": (int((np.sign(paired) * np.sign(truth) < 0).sum())
                                             if len(truth) else None),
        "pooled_q95_paired_regression": float(np.quantile(paired, 0.95)) if len(paired) else None,
        "pooled_worst_paired_regression": float(paired.max()) if len(paired) else None,
        "gates": {"complete_valid_pairs": bool(complete),
                  "simultaneous_pooled_ci_upper_below_zero": bool(quality),
                  "mean_stage3_requests_nonincreasing": bool(cost),
                  "scenario_passed": bool(quality and cost)},
    }


def assess(preregistration, output):
    preregistration, output = Path(preregistration), Path(output)
    prereg = json.loads(preregistration.read_text())
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Assessment output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "assessment_manifest.json"
    manifest = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "preregistration_sha256": _sha(preregistration),
                "scenarios": prereg["scenarios"], "seeds": prereg["seeds"]}
    _write(manifest_path, manifest)
    mismatches = [name for name, expected in prereg["source_sha256"].items()
                  if _sha(_source_workspace(__file__) / name) != expected]
    if mismatches:
        manifest.update(status="source_mismatch", mismatches=mismatches)
        _write(manifest_path, manifest)
        raise ValueError(f"Preregistered source changed: {mismatches}")
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for name in prereg["source_sha256"]:
        (snapshot / name).write_bytes((_source_workspace(__file__) / name).read_bytes())
    (output / "preregistration.json").write_bytes(preregistration.read_bytes())
    analyses = {}
    try:
        for scenario in prereg["scenarios"]:
            root = Path(prereg["baseline_roots"][scenario])
            _baseline_check(root, scenario, prereg["seeds"], prereg)
            scenario_dir = output / scenario
            scenario_dir.mkdir()
            rows = []
            with (scenario_dir / "seed_rows.jsonl").open("w") as handle:
                for seed in prereg["seeds"]:
                    try:
                        row = replay_one_seed(root, [seed])["rows"][0]
                        row["status"] = ("valid" if row["candidate_failures"] == 0
                                         else "candidate_failure")
                    except ValueError as exc:
                        if any(message in str(exc) for message in (
                                "Invalid saved candidate", "Missing coalition",
                                "No feasible candidate could be internally scored")):
                            row = {"scenario": scenario, "seed": seed,
                                   "status": "expected_failure", "failure_reason": str(exc)}
                        else:
                            raise
                    handle.write(json.dumps(row, ensure_ascii=False,
                                            allow_nan=False) + "\n")
                    handle.flush()
                    rows.append(row)
            analyses[scenario] = _scenario_analysis(rows, prereg["seeds"], scenario)
            if all(row["status"] == "valid" for row in rows):
                _write(scenario_dir / "replay.json", {
                    "study": prereg["study"], "scenario": scenario,
                    "seeds": prereg["seeds"], "rows": rows,
                    "stage3_prediction_cache": "off"})
            _write(output / "analysis.json", {"status": "running",
                    "scenarios": analyses, "all_scenarios_passed": False})
    except Exception as exc:
        manifest.update(status="interrupted", interruption=repr(exc),
                        completed_scenarios=list(analyses))
        _write(manifest_path, manifest)
        raise
    analysis = {"status": "complete", "study": prereg["study"],
                "scenarios": analyses,
                "all_scenarios_passed": all(item["gates"]["scenario_passed"]
                                            for item in analyses.values()),
                "scope_limit": "Synthetic 2D target-0 fixed-three-agent clean and 4x-imbalance "
                               "scenarios; changes Stage 3 objective, not initial-alpha formula. "
                               "Pooled scores only after selection."}
    _write(output / "analysis.json", analysis)
    manifest.update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                    completed_scenarios=list(analyses))
    _write(manifest_path, manifest)
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preregistration")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(assess(args.preregistration, args.output_dir),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

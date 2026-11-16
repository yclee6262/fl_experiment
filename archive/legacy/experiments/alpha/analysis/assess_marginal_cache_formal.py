"""Assess preregistered marginal-alpha plus Stage 3 cache comparison."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from research_analysis import paired_bootstrap_mean_ci


CONTROL = "validation_calibrated_multistart"
TREATMENT = "marginal_information_portfolio"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assess(prereg_path, baseline_root, replay_path):
    prereg_path, baseline_root, replay_path = map(
        Path, (prereg_path, baseline_root, replay_path))
    prereg = json.loads(prereg_path.read_text())
    seeds = prereg["formal_seeds"]
    if (prereg["study"] != "marginal_alpha_with_stage3_cache_clean_formal_v1"
            or prereg["scenario"] != "reference_clean"
            or len(seeds) != 180):
        raise ValueError("Wrong formal preregistration")
    for name, expected in prereg["source_sha256"].items():
        snapshot = Path(prereg["source_snapshot"])/name
        if _sha(name) != expected or _sha(snapshot) != expected:
            raise ValueError(f"Changed registered source: {name}")
    suite = json.loads((baseline_root/"suite_manifest.json").read_text())
    verification = json.loads((baseline_root/"selection_verification.json").read_text())
    pilot = json.loads((Path(prereg["pilot_baseline_root"])/
                        "suite_manifest.json").read_text())
    if (suite["status"] != "complete"
            or suite["scenarios"] != ["reference_clean"]
            or suite["seeds"] != seeds
            or suite["candidate_initializations"] !=
                pilot["candidate_initializations"]
            or suite["methods"] != pilot["methods"]):
        raise ValueError("Baseline condition set differs")
    for key, value in pilot["arguments"].items():
        if key not in ("output_dir", "seeds") and suite["arguments"].get(key) != value:
            raise ValueError(f"Baseline setting differs from pilot: {key}")
    replay = json.loads(replay_path.read_text())
    rows = replay["rows"]
    if (replay["status"] != "complete"
            or replay["scenario"] != "reference_clean"
            or replay["seeds"] != seeds
            or [row["seed"] for row in rows] != seeds):
        raise ValueError("Incomplete or misaligned cache replay")
    valid = [row for row in rows if row["status"] == "valid"]
    failures = [{"seed": row["seed"], "status": row["status"],
                 "missing_fixed_ids": row.get("missing_fixed_ids"),
                 "failed_candidates": row.get("failed_candidates"),
                 "preparation": row.get("preparation")}
                for row in rows if row["status"] != "valid"]
    complete = len(valid) == 180 and verification["passed"]
    pooled = []
    requests = []
    input_samples = []
    control_errors = []
    treatment_errors = []
    control_requests = []
    treatment_requests = []
    max_x = max_score = 0.0
    for row in valid:
        control = row["portfolios"][CONTROL]
        treatment = row["portfolios"][TREATMENT]
        pooled.append(treatment["pooled_error"]-control["pooled_error"])
        requests.append(treatment["cached_method_requests"]
                        -control["raw_method_requests"])
        input_samples.append(treatment["cached_method_input_samples"]
                             -control["raw_method_input_samples"])
        control_errors.append(control["pooled_error"])
        treatment_errors.append(treatment["pooled_error"])
        control_requests.append(control["raw_method_requests"])
        treatment_requests.append(treatment["cached_method_requests"])
        max_x = max(max_x, *(item["max_solution_diff"]
                             for item in row["candidates"].values()))
        max_score = max(max_score, *(item["pooled_score_diff"]
                                     for item in row["candidates"].values()))
    pooled = np.asarray(pooled, dtype=float)
    requests = np.asarray(requests, dtype=float)
    input_samples = np.asarray(input_samples, dtype=float)
    if (not np.isfinite(pooled).all() or not np.isfinite(requests).all()
            or not np.isfinite(input_samples).all()
            or max_x > 1e-7 or max_score > 1e-7):
        raise ValueError("Invalid pooled, cost, or replay score")
    quality_ci = paired_bootstrap_mean_ci(pooled)
    quality_gate = complete and quality_ci[1] is not None and quality_ci[1] < 0
    cost_gate = complete and bool(len(requests)) and requests.mean() < 0
    sample_gate = complete and bool(len(input_samples)) and input_samples.mean() < 0
    return {
        "study": prereg["study"], "status": "complete",
        "preregistration_sha256": _sha(prereg_path),
        "expected_seeds": 180, "valid_pairs": len(valid),
        "failure_status_counts": dict(Counter(row["status"] for row in rows
                                              if row["status"] != "valid")),
        "failure_records": failures,
        "baseline_verification": verification,
        "mean_pooled_error_control": float(np.mean(control_errors)) if valid else None,
        "mean_pooled_error_treatment": float(np.mean(treatment_errors)) if valid else None,
        "mean_paired_pooled_delta_treatment_minus_control":
            float(pooled.mean()) if valid else None,
        "paired_pooled_95pct_ci": quality_ci,
        "pooled_wins_ties_losses": [int((pooled < -1e-7).sum()),
                                     int((abs(pooled) <= 1e-7).sum()),
                                     int((pooled > 1e-7).sum())],
        "mean_model_requests_control": float(np.mean(control_requests)) if valid else None,
        "mean_model_requests_treatment": float(np.mean(treatment_requests)) if valid else None,
        "mean_paired_request_delta": float(requests.mean()) if valid else None,
        "paired_request_95pct_ci": paired_bootstrap_mean_ci(requests),
        "mean_paired_input_sample_delta": (
            float(input_samples.mean()) if valid else None),
        "max_cached_solution_difference": max_x,
        "max_cached_pooled_score_difference": max_score,
        "gates": {
            "complete_valid_pairs": bool(complete),
            "pooled_ci_upper_strictly_below_zero": bool(quality_gate),
            "mean_model_requests_decreased": bool(cost_gate),
            "mean_model_input_samples_decreased": bool(sample_gate),
            "registered_success": bool(quality_gate and cost_gate and sample_gate),
        },
        "scope_limit": "Synthetic clean 2D, target 0, fixed IDs; cache assumes fixed "
                       "deterministic local models; API prices and latency unmeasured",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preregistration")
    parser.add_argument("baseline_root")
    parser.add_argument("cache_replay_json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    result = assess(args.preregistration, args.baseline_root,
                    args.cache_replay_json)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                               allow_nan=False)+"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

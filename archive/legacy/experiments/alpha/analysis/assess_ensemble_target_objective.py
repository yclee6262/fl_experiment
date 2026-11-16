"""Incrementally assess a preregistered pooled-quality Stage 3 objective study."""

import argparse
import hashlib
import json
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


def _write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assess(root, prereg_path, output):
    root, prereg_path, output = Path(root), Path(prereg_path), Path(output)
    prereg = json.loads(prereg_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Assessment output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "assessment_manifest.json"
    manifest = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "preregistration_sha256": _sha(prereg_path),
                "study_root": str(root), "seeds": prereg["seeds"]}
    _write(manifest_path, manifest)
    source_mismatches = [name for name, expected in prereg["source_sha256"].items()
                         if _sha(_source_workspace(__file__) / name) != expected]
    if source_mismatches:
        manifest.update(status="source_mismatch", mismatches=source_mismatches)
        _write(manifest_path, manifest)
        raise ValueError(f"Preregistered source changed: {source_mismatches}")
    source_snapshot = output / "source_snapshot"
    source_snapshot.mkdir()
    for name in prereg["source_sha256"]:
        (source_snapshot / name).write_bytes((_source_workspace(__file__) / name).read_bytes())
    (output / "preregistration.json").write_bytes(prereg_path.read_bytes())
    suite = json.loads((root / "suite_manifest.json").read_text())
    verification = json.loads((root / "selection_verification.json").read_text())
    if (suite["status"] != "complete" or suite["seeds"] != prereg["seeds"]
            or suite["scenarios"] != [prereg["scenario"]]
            or suite["candidate_initializations"] != prereg["candidate_initializations"]
            or suite["methods"] != prereg["baseline_methods"]
            or any(suite["arguments"].get(key) != value
                   for key, value in prereg["baseline_arguments"].items())
            or not verification["passed"]):
        manifest["status"] = "incompatible_or_unverified_baseline"
        _write(manifest_path, manifest)
        raise ValueError("Formal baseline is incomplete or not verified")
    rows = []
    row_path = output / "seed_rows.jsonl"
    try:
        with row_path.open("w") as handle:
            for seed in prereg["seeds"]:
                try:
                    result = replay_one_seed(root, [seed])
                    row = result["rows"][0]
                    row["status"] = ("valid" if row["candidate_failures"] == 0
                                     else "candidate_failure")
                except ValueError as exc:
                    if any(message in str(exc) for message in (
                            "Invalid saved candidate", "Missing coalition",
                            "No feasible candidate could be internally scored")):
                        row = {"seed": seed, "status": "expected_failure",
                               "failure_reason": str(exc)}
                    else:
                        raise
                handle.write(json.dumps(row, ensure_ascii=False,
                                        allow_nan=False) + "\n")
                handle.flush()
                rows.append(row)
    except Exception as exc:
        manifest.update(status="interrupted", completed_seed_rows=len(rows),
                        interruption=repr(exc))
        _write(manifest_path, manifest)
        raise
    valid = [r for r in rows if r["status"] == "valid"]
    pooled = np.asarray([r["new_pooled_error"]-r["old_pooled_error"]
                         for r in valid], dtype=float)
    truth = np.asarray([r["new_truth_error"]-r["old_truth_error"]
                        for r in valid], dtype=float)
    requests = np.asarray([r["new_stage3_requests"]-r["old_stage3_requests"]
                           for r in valid], dtype=float)
    ci = paired_bootstrap_mean_ci(pooled)
    cost_ci = paired_bootstrap_mean_ci(requests)
    complete = len(rows) == len(prereg["seeds"]) and len(valid) == len(rows)
    quality_gate = complete and ci[1] is not None and ci[1] < 0
    cost_gate = complete and len(requests) > 0 and float(requests.mean()) <= 0
    analysis = {
        "status": "complete", "scenario": prereg["scenario"],
        "expected_seeds": len(prereg["seeds"]), "recorded_seeds": len(rows),
        "valid_pairs": len(valid),
        "failure_statuses": {status: sum(r["status"] == status for r in rows)
                             for status in sorted(set(r["status"] for r in rows))
                             if status != "valid"},
        "mean_pooled_delta_new_minus_old": float(pooled.mean()) if len(pooled) else None,
        "paired_bootstrap_95pct_ci_pooled": ci,
        "mean_stage3_request_delta_new_minus_old": (float(requests.mean())
                                                       if len(requests) else None),
        "paired_bootstrap_95pct_ci_stage3_requests": cost_ci,
        "pooled_wins_ties_losses": ([int((pooled < -1e-7).sum()),
                                      int((np.abs(pooled) <= 1e-7).sum()),
                                      int((pooled > 1e-7).sum())] if len(pooled) else None),
        "pooled_q95_regression": float(np.quantile(pooled, .95)) if len(pooled) else None,
        "pooled_worst_regression": float(pooled.max()) if len(pooled) else None,
        "mean_truth_delta_diagnostic": float(truth.mean()) if len(truth) else None,
        "pooled_truth_direction_conflicts": (int((np.sign(pooled)*np.sign(truth)<0).sum())
                                             if len(truth) else None),
        "gates": {"complete_valid_pairs": complete,
                  "pooled_quality_ci_upper_below_zero": quality_gate,
                  "mean_stage3_requests_nonincreasing": cost_gate,
                  "overall_passed": bool(quality_gate and cost_gate)},
        "conclusion": ("preregistered_lowquality_pooled_quality_and_cost_passed"
                       if quality_gate and cost_gate
                       else "preregistered_combined_gate_not_passed"),
        "scope_limit": "Synthetic low-quality 2D target-0 fixed-three-agent scenario only. "
                       "This changes the Stage 3 objective, not the initial-alpha formula. "
                       "Pooled evaluator only scores delivered X after selection.",
    }
    _write(output / "analysis.json", analysis)
    manifest.update(status="complete", completed_seed_rows=len(rows),
                    completed_at_utc=datetime.now(timezone.utc).isoformat())
    _write(manifest_path, manifest)
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("preregistration")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = assess(args.root, args.preregistration, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

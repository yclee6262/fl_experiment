"""Compare guarded no-progress stopping with a same-environment control run."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


REQUEST_FIELDS = ("stage3_requests", "contribution_requests", "payment_requests")


def load_summary(path):
    with (Path(path) / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        if row["seed"] in result:
            raise ValueError(f"Duplicate seed row in {path}: {row['seed']}")
        result[row["seed"]] = row
    return result


def request_total(row):
    values = [row.get(field, "") for field in REQUEST_FIELDS]
    values = [int(value) for value in values if value not in (None, "")]
    return sum(values) if values else None


def optional_number(value):
    try:
        result=float(value)
    except (TypeError,ValueError):
        return None
    return result if np.isfinite(result) else None


def compare(control_dir, treatment_dir, tolerance=1e-12):
    control = load_summary(control_dir)
    treatment = load_summary(treatment_dir)
    if set(control) != set(treatment):
        raise ValueError("Control and treatment must contain identical seeds.")
    pairs = []
    for seed in sorted(control, key=int):
        before, after = control[seed], treatment[seed]
        comparable = all(row.get("delivery_status") == "valid"
                         and row.get("final_solution")
                         and row.get("final_optimization_weights")
                         and row.get("external_global_error") not in (None, "")
                         for row in (before, after))
        if comparable:
            solution_delta = float(np.max(np.abs(
                np.asarray(json.loads(before["final_solution"]), dtype=float)
                - np.asarray(json.loads(after["final_solution"]), dtype=float))))
            alpha_delta = float(np.max(np.abs(
                np.asarray(json.loads(before["final_optimization_weights"]), dtype=float)
                - np.asarray(json.loads(after["final_optimization_weights"]), dtype=float))))
            external_delta = abs(float(before["external_global_error"])-float(after["external_global_error"]))
        else:
            solution_delta = alpha_delta = external_delta = None
        control_requests = request_total(before)
        treatment_requests = request_total(after)
        pairs.append({
            "seed": seed,
            "poison_ratio": before["poison_ratio"],
            "control_rounds": int(before["rounds_executed"]) if before.get("rounds_executed") else None,
            "treatment_rounds": int(after["rounds_executed"]) if after.get("rounds_executed") else None,
            "control_stop_reason": before.get("stop_reason", ""),
            "treatment_stop_reason": after.get("stop_reason", ""),
            "no_progress_triggered": after.get("no_progress_triggered", ""),
            "control_delivery_status": before.get("delivery_status", "unknown"),
            "treatment_delivery_status": after.get("delivery_status", "unknown"),
            "accepted_state_comparable": comparable,
            "solution_max_abs_delta": solution_delta,
            "alpha_max_abs_delta": alpha_delta,
            "external_error_abs_delta": external_delta,
            "accepted_state_matches": (
                (solution_delta <= tolerance
                and alpha_delta <= tolerance
                and external_delta <= tolerance
                and before.get("final_accepted_round") == after.get("final_accepted_round"))
                if comparable else None
            ),
            "control_requests": control_requests,
            "treatment_requests": treatment_requests,
            "control_stage3_input_samples": int(before["stage3_input_samples"]) if before.get("stage3_input_samples") else None,
            "treatment_stage3_input_samples": int(after["stage3_input_samples"]) if after.get("stage3_input_samples") else None,
            "control_contribution_input_samples": int(before["contribution_input_samples"]) if before.get("contribution_input_samples") else None,
            "treatment_contribution_input_samples": int(after["contribution_input_samples"]) if after.get("contribution_input_samples") else None,
            **{f"shared_{field}": optional_number(before.get(f"shared_{field}")) for field in (
                "training_seconds", "qualification_seconds", "proposal_seconds",
                "fixed_proposal_seconds", "evaluator_seconds", "qualification_requests",
                "proposal_requests", "fixed_proposal_requests", "qualification_input_samples",
                "proposal_input_samples", "fixed_proposal_input_samples")},
            "requests_saved": (control_requests - treatment_requests
                if control_requests is not None and treatment_requests is not None else None),
            "request_savings_fraction": (
                (control_requests - treatment_requests) / control_requests
                if control_requests and treatment_requests is not None else None
            ),
        })
    return pairs


def aggregate(pairs):
    valid_cost_pairs = [row for row in pairs if row["control_requests"] is not None
                        and row["treatment_requests"] is not None
                        and row["control_delivery_status"] == "valid"
                        and row["treatment_delivery_status"] == "valid"]
    control_requests = sum(row["control_requests"] for row in valid_cost_pairs)
    treatment_requests = sum(row["treatment_requests"] for row in valid_cost_pairs)
    triggered = [row for row in valid_cost_pairs if row["no_progress_triggered"] == "True"]
    control_rounds = [row["control_rounds"] for row in pairs if row["control_rounds"] is not None]
    treatment_rounds = [row["treatment_rounds"] for row in pairs if row["treatment_rounds"] is not None]
    return {
        "poison_ratio": pairs[0]["poison_ratio"],
        "seeds": len(pairs),
        "triggered_seeds": len(triggered),
        "accepted_state_mismatches": sum(
            row["accepted_state_matches"] is False for row in pairs
        ),
        "uncomparable_accepted_state_pairs": sum(not row["accepted_state_comparable"] for row in pairs),
        "qualification_failure_pairs": sum(row["control_delivery_status"] == "qualification_failure"
            and row["treatment_delivery_status"] == "qualification_failure" for row in pairs),
        "mean_control_rounds": float(np.mean(control_rounds)) if control_rounds else None,
        "mean_treatment_rounds": float(np.mean(treatment_rounds)) if treatment_rounds else None,
        "mean_control_requests": control_requests / len(valid_cost_pairs) if valid_cost_pairs else None,
        "mean_treatment_requests": treatment_requests / len(valid_cost_pairs) if valid_cost_pairs else None,
        "mean_control_api_input_samples": float(np.mean([
            row["control_stage3_input_samples"] + row["control_contribution_input_samples"]
            for row in valid_cost_pairs if row["control_stage3_input_samples"] is not None
            and row["control_contribution_input_samples"] is not None])) if valid_cost_pairs else None,
        "mean_treatment_api_input_samples": float(np.mean([
            row["treatment_stage3_input_samples"] + row["treatment_contribution_input_samples"]
            for row in valid_cost_pairs if row["treatment_stage3_input_samples"] is not None
            and row["treatment_contribution_input_samples"] is not None])) if valid_cost_pairs else None,
        "overall_request_savings_fraction_valid_pairs_only": (
            (control_requests - treatment_requests) / control_requests if control_requests else None
        ),
        "triggered_request_savings_fraction": (
            (
                sum(row["control_requests"] for row in triggered)
                - sum(row["treatment_requests"] for row in triggered)
            ) / sum(row["control_requests"] for row in triggered)
            if triggered and all(row["control_requests"] is not None and row["treatment_requests"] is not None for row in triggered) else None
        ),
        **{f"mean_{field}": float(np.mean([row[field] for row in pairs if row.get(field) is not None]))
           if any(row.get(field) is not None for row in pairs) else None
           for field in ("shared_training_seconds", "shared_qualification_seconds",
                         "shared_proposal_seconds", "shared_fixed_proposal_seconds",
                         "shared_evaluator_seconds", "shared_qualification_requests",
                         "shared_proposal_requests", "shared_fixed_proposal_requests",
                         "shared_qualification_input_samples", "shared_proposal_input_samples",
                         "shared_fixed_proposal_input_samples")},
    }


def write_csv(path, rows):
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", action="append", required=True)
    parser.add_argument("--treatment", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    if len(args.control) != len(args.treatment):
        raise ValueError("Provide the same number of control and treatment directories.")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Analysis output directory must be empty.")
    output.mkdir(parents=True, exist_ok=True)
    all_pairs = []
    aggregates = []
    for control, treatment in zip(args.control, args.treatment):
        pairs = compare(control, treatment, args.tolerance)
        all_pairs.extend(pairs)
        aggregates.append(aggregate(pairs))
    write_csv(output / "paired_results.csv", all_pairs)
    write_csv(output / "aggregate_results.csv", aggregates)
    (output / "analysis_manifest.json").write_text(json.dumps({
        "control_dirs": args.control,
        "treatment_dirs": args.treatment,
        "accepted_state_tolerance": args.tolerance,
        "scope": "five-seed P4 pilot",
    }, indent=2) + "\n")
    print(f"Analyzed {len(all_pairs)} paired seeds")


if __name__ == "__main__":
    main()

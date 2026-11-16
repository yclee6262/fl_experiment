"""Audit the four-method single-solve study; no pooled score enters initialization."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from agent_client import AgentNode
from initial_alpha_methods import initial_alpha
from research_analysis import audit_coverage, finite_float, paired_bootstrap_mean_ci
from verify_alpha_initialization_study import verify as verify_deliveries


NEW_METHODS = ("bootstrap_quality", "exponential_quality")


def read_rows(root, manifest):
    rows = []
    for scenario in manifest["scenarios"]:
        path = root / scenario / "summary.csv"
        if path.exists():
            with path.open(newline="") as handle:
                rows.extend(csv.DictReader(handle))
    return rows


def valid(row):
    if not row or row.get("delivery_status") != "valid":
        return False
    try:
        x = np.asarray(json.loads(row["final_solution"]), float)
        violation = max(0., float(np.max(-1.-x)), float(np.max(x-1.)))
        return (x.ndim == 1 and len(x) == 2 and np.isfinite(x).all()
                and violation <= 1e-8 and row["final_box_feasible"] == "True"
                and finite_float(row["external_global_error"]) is not None)
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def describe(values):
    values = [value for value in values if value is not None]
    if not values:
        return {"n": 0, "mean": None, "median": None, "q1": None, "q3": None}
    return {"n": len(values), "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "q1": float(np.quantile(values, .25)), "q3": float(np.quantile(values, .75))}


def combined_cost(row, suffix):
    values = [finite_float(row.get(f"{stage}_{suffix}"))
              for stage in ("initialization", "stage3")]
    return None if any(v is None for v in values) else sum(values)


def verify_weights(root, manifest, rows):
    errors, max_difference, checked = [], 0., 0
    settings = manifest["initialization_settings"]
    index = {(r["scenario"], r["seed"], r["method"]): r for r in rows}
    for scenario in manifest["scenarios"]:
        shared_file = root / scenario / "shared_costs.jsonl"
        if not shared_file.exists():
            errors.append(f"missing shared data: {scenario}")
            continue
        for item in map(json.loads, shared_file.read_text().splitlines()):
            if item.get("missing_fixed_ids") or not item.get("artifact_path"):
                continue
            payload = torch.load(root / scenario / item["artifact_path"] / "local_models.pt",
                                 map_location="cpu", weights_only=True)
            agents = []
            for ident in item["coalition_ids"]:
                agent = AgentNode(ident, None, n_features=payload["n_features"])
                agent.model.load_state_dict(payload["agents"][ident])
                agent.model.eval()
                agents.append(agent)
            for method in manifest["methods"]:
                row = index.get((scenario, str(item["seed"]), method))
                if not row or not row.get("initial_alpha"):
                    continue
                audit_saved = json.loads(row["alpha_audit"])
                mse = np.asarray(json.loads(row["stage0_mse"]), float)
                alpha, audit = initial_alpha(method, agents, mse,
                    item["blind_test_X"], item["blind_test_y"], item["proposals"], 0.,
                    bootstrap_replicates=settings["bootstrap_replicates"],
                    bootstrap_seed=settings["bootstrap_seed"] + item["seed"],
                    exponential_temperature_ratio=settings["exponential_temperature_ratio"])
                saved = np.asarray(json.loads(row["initial_alpha"]), float)
                difference = float(np.max(np.abs(alpha-saved)))
                max_difference = max(max_difference, difference)
                key = f"{scenario}/{item['seed']}/{method}"
                if difference > 1e-7 or list(audit["agent_ids"]) != item["coalition_ids"]:
                    errors.append(f"alpha reload mismatch: {key}")
                if audit["fallback_reason"] != audit_saved["fallback_reason"]:
                    errors.append(f"fallback mismatch: {key}")
                if method == "bootstrap_quality":
                    if np.max(np.abs(np.asarray(audit["recomputed_blind_mse"])-mse)) > 1e-7:
                        errors.append(f"Stage0 MSE not same blind data: {key}")
                    if not np.allclose(audit["bootstrap_weight_std"],
                                       audit_saved["bootstrap_weight_std"], atol=1e-7, rtol=0):
                        errors.append(f"bootstrap audit mismatch: {key}")
                if row.get("extra_trusted_labels") != "0":
                    errors.append(f"unexpected new trusted labels: {key}")
                expected_requests = len(agents) if method == "bootstrap_quality" else 0
                if finite_float(row["initialization_requests"]) != expected_requests:
                    errors.append(f"initialization cost mismatch: {key}")
                checked += 1
    return {"passed": not errors, "checked_weights": checked,
            "max_alpha_difference": max_difference, "errors": errors}


def analyze(root, preregistration=None, run_verification=True):
    root = Path(root)
    manifest = json.loads((root / "suite_manifest.json").read_text())
    rows = read_rows(root, manifest)
    expected = [(s, str(seed), m) for s in manifest["scenarios"]
                for seed in manifest["seeds"] for m in manifest["methods"]]
    coverage = audit_coverage(rows, expected, ("scenario", "seed", "method"))
    index = {(r["scenario"], r["seed"], r["method"]): r for r in rows}
    summary, comparisons, conflicts, shared_summary = [], [], [], []
    for scenario in manifest["scenarios"]:
        for method in manifest["methods"]:
            group = [index.get((scenario, str(seed), method)) for seed in manifest["seeds"]]
            existing = [r for r in group if r]
            good = [r for r in existing if valid(r)]
            summary.append({"scenario": scenario, "method": method,
                "expected_n": len(group), "valid_n": len(good),
                "failure_counts": dict(Counter("missing" if not r else r["status"]
                    for r in group if not valid(r))),
                "fallback_n": sum(bool(json.loads(r["alpha_audit"])["fallback_reason"])
                                   for r in existing if r.get("alpha_audit")),
                "pooled": describe([finite_float(r["external_global_error"]) for r in good]),
                "truth": describe([finite_float(r["final_target_error"]) for r in good]),
                "method_requests": describe([combined_cost(r, "requests") for r in existing]),
                "method_input_samples": describe([combined_cost(r, "input_samples") for r in existing]),
                "method_seconds": describe([combined_cost(r, "seconds") for r in existing]),
                **{field: describe([finite_float(r.get(field)) for r in existing]) for field in (
                    "initialization_requests", "stage3_requests", "initialization_seconds", "stage3_seconds")}})
        plans = [(m, b) for m in NEW_METHODS for b in ("A_uniform", "B_stage0")]
        plans.append(("B_stage0", "A_uniform"))
        for method, baseline in plans:
            pooled, truth, requests, seconds, excluded = [], [], [], [], Counter()
            paired_seeds = []
            for seed in manifest["seeds"]:
                new, old = index.get((scenario, str(seed), method)), index.get((scenario, str(seed), baseline))
                if not new or not old:
                    excluded["missing_row"] += 1
                    continue
                if not valid(new) or not valid(old):
                    excluded["invalid_or_failed_delivery"] += 1
                    continue
                delta = float(new["external_global_error"])-float(old["external_global_error"])
                pooled.append(delta)
                paired_seeds.append(seed)
                ta, tb = finite_float(new["final_target_error"]), finite_float(old["final_target_error"])
                if ta is not None and tb is not None:
                    truth.append(ta-tb)
                    if delta*(ta-tb) < 0:
                        conflicts.append({"scenario": scenario, "seed": seed, "method": method,
                            "baseline": baseline, "pooled_delta": delta, "truth_delta": ta-tb,
                            "method_X": json.loads(new["final_solution"]),
                            "baseline_X": json.loads(old["final_solution"])})
                for field, target in (("requests", requests), ("seconds", seconds)):
                    a, b = combined_cost(new, field), combined_cost(old, field)
                    if a is not None and b is not None:
                        target.append(a-b)
            primary = method in NEW_METHODS and baseline == "B_stage0"
            simultaneous_confidence = 1-.05/(len(manifest["scenarios"])*len(NEW_METHODS))
            failures_new = len(manifest["seeds"])-sum(valid(index.get((scenario, str(s), method))) for s in manifest["seeds"])
            failures_old = len(manifest["seeds"])-sum(valid(index.get((scenario, str(s), baseline))) for s in manifest["seeds"])
            ci = paired_bootstrap_mean_ci(pooled)
            simultaneous = paired_bootstrap_mean_ci(pooled, confidence=simultaneous_confidence) if primary else (None, None)
            comparisons.append({"scenario": scenario, "method": method, "baseline": baseline,
                "paired_n": len(pooled), "paired_seeds": paired_seeds, "excluded": dict(excluded),
                "pooled_delta": describe(pooled), "ci95": ci,
                "primary": primary, "simultaneous_confidence": simultaneous_confidence if primary else None,
                "simultaneous_ci": simultaneous, "truth_delta": describe(truth),
                "request_delta": describe(requests), "seconds_delta": describe(seconds),
                "wins": sum(d < -1e-12 for d in pooled), "losses": sum(d > 1e-12 for d in pooled),
                "ties": sum(abs(d) <= 1e-12 for d in pooled),
                "candidate_failures": failures_new, "baseline_failures": failures_old})
        path = root/scenario/"shared_costs.jsonl"
        items = list(map(json.loads, path.read_text().splitlines())) if path.exists() else []
        shared_summary.append({"scenario": scenario, "n": len(items),
            "training_seconds": describe([finite_float(r.get("preparation", {}).get("training_seconds")) for r in items]),
            "qualification_seconds": describe([finite_float(r.get("preparation", {}).get("qualification_seconds")) for r in items]),
            "shared_requests": describe([sum(r.get("preparation", {}).get(k, 0) for k in
                ("qualification_requests", "proposal_requests"))+r.get("fixed_proposal_requests", 0) for r in items
                if r.get("preparation")]),
            "shared_input_samples": describe([sum(r.get("preparation", {}).get(k, 0) for k in
                ("qualification_input_samples", "proposal_input_samples"))+r.get("fixed_proposal_input_samples", 0) for r in items
                if r.get("preparation")]),
            "fixed_proposal_seconds": describe([finite_float(r.get("fixed_proposal_seconds")) for r in items]),
            "evaluator_seconds": describe([finite_float(r.get("evaluator_seconds")) for r in items])})
    verification = None
    if run_verification:
        verification = {"deliveries": verify_deliveries(root),
                        "weights": verify_weights(root, manifest, rows)}
    prereg_errors = []
    formal = False
    if preregistration:
        registration = json.loads(Path(preregistration).read_text())
        formal = manifest["seeds"] == registration["formal_seeds"]
        for field in ("scenarios", "methods", "initialization_settings", "stage3_objective"):
            if manifest[field] != registration[field]:
                prereg_errors.append(f"preregistration mismatch: {field}")
        if not formal and manifest["seeds"] != registration["pilot_seeds"]:
            prereg_errors.append("seed set is neither preregistered pilot nor formal")
        for scenario in manifest["scenarios"]:
            run_file = root/scenario/"run_manifest.json"
            if not run_file.exists():
                prereg_errors.append(f"missing run manifest: {scenario}")
                continue
            saved_hashes = json.loads(run_file.read_text())["source_sha256"]
            for name, digest in registration["source_sha256"].items():
                if saved_hashes.get(name) != digest:
                    prereg_errors.append(f"preregistered source mismatch: {scenario}/{name}")
    audited = (manifest["status"] == "complete" and coverage["complete"] and not prereg_errors
               and verification is not None and verification["deliveries"]["passed"]
               and verification["weights"]["passed"])
    for c in comparisons:
        c["confirmed_quality_advantage_over_B"] = bool(formal and audited and c["primary"]
            and c["paired_n"] == len(manifest["seeds"]) and c["simultaneous_ci"][1] is not None
            and c["simultaneous_ci"][1] < 0 and c["candidate_failures"] <= c["baseline_failures"])
    report = {"root": str(root), "suite_status": manifest["status"], "formal": formal,
              "expected_conditions": len(expected), "observed_rows": len(rows),
              "coverage": coverage, "verification": verification,
              "preregistration_errors": prereg_errors, "acceptance_passed": audited,
              "summary": summary, "comparisons": comparisons, "shared_costs": shared_summary,
              "pooled_truth_conflicts": conflicts,
              "interpretation": "Seed-paired pooled error is primary. Six new-vs-B comparisons use Bonferroni family-wise 95% intervals. Quality and costs are separate; intervals against A are descriptive. Shared preparation charged once per seed in the batch, or once per independently executed method. No money utility is assumed."}
    (root/"robust_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--preregistration", required=True)
    args = parser.parse_args()
    report = analyze(args.root, args.preregistration)
    print(json.dumps({k: report[k] for k in ("expected_conditions", "observed_rows", "formal",
        "acceptance_passed", "preregistration_errors")}, indent=2))
    for comparison in report["comparisons"]:
        if comparison["primary"]:
            print(comparison["scenario"], comparison["method"], comparison["pooled_delta"]["mean"],
                  comparison["ci95"], comparison["simultaneous_ci"],
                  comparison["confirmed_quality_advantage_over_B"])
    raise SystemExit(0 if report["acceptance_passed"] else 1)

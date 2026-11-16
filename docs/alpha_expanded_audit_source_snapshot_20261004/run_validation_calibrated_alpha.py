"""Clean paired study of Stage 0 calibrated multistart alpha selection."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from alpha_weight_experiments import (attach_query_counter, build_base_server,
                                      write_run_manifest)
from host_server import HostServer
from p5_scenario_suite import build_scenario_args
from research_evaluation import save_scenario_artifacts
from run_alpha_initialization_study import _append, _json, _one_method
from validation_calibrated_alpha import (CANDIDATE_METHODS, fit_residual_calibrator,
                                         select_candidate, select_candidate_local_bias,
                                         LOCAL_BIAS_BANDWIDTH, diverse_four_methods)


META_METHOD = "validation_calibrated_multistart"
SPECIALIST_METHOD = "specialist_multistart"
ECONOMIC_METHOD = "economic_multistart"
LOCAL_METHOD = "local_calibrated_multistart"
PORTFOLIO_METHOD = "four_candidate_portfolio"
DIVERSE_METHOD = "diversified_four_portfolio"
AUDIT_METHOD = "expanded_audit_multistart"
AUDIT_CANDIDATE = "local_bias_risk_100"
ECONOMIC_EXTRAS = ("best_agent", "local_bias_risk")
SPECIALIST_CANDIDATES = CANDIDATE_METHODS + ("best_agent",)
PORTFOLIO_CANDIDATES = (
    "A_uniform", "B_stage0", "mse_power_1p5", "corr_precision_0p5")
SUMMARY_METHODS = ("A_uniform", "B_stage0", META_METHOD)


def _meta_result(candidates, base, artifact_path, seed,
                 method_order=CANDIDATE_METHODS, method_name=META_METHOD,
                 selector_rule="global", charge_all_initializations=False):
    counter = attach_query_counter(base.trusted_agents)
    counter["stage"] = "calibration"
    baseline = candidates["B_stage0"]
    baseline_alpha = json.loads(baseline["initial_alpha"])
    selected_candidates = {method: candidates[method] for method in method_order}
    solutions = {method: (json.loads(row["final_solution"])
                          if row["delivery_status"] == "valid" else None)
                 for method, row in selected_candidates.items()}
    if selector_rule == "global":
        coefficients, calibration = fit_residual_calibrator(
            base.trusted_agents, base.test_X, base.test_y, baseline_alpha)
        counter["stage"] = "candidate_scoring"
        chosen, internal_scores = select_candidate(
            base.trusted_agents, baseline_alpha, coefficients, solutions, base.target_T,
            base.feasible_lower, base.feasible_upper, base.feasibility_tolerance,
            method_order=method_order)
    elif selector_rule == "local_bias":
        coefficients = None
        calibration = {"calibration_samples": len(base.test_y),
                       "ridge_penalty": [0.01] + [1.0] * base.n_features,
                       "local_bandwidth": LOCAL_BIAS_BANDWIDTH}
        chosen, internal_scores = select_candidate_local_bias(
            base.trusted_agents, baseline_alpha, base.test_X, base.test_y,
            solutions, base.target_T, base.feasible_lower, base.feasible_upper,
            base.feasibility_tolerance, method_order=method_order)
    else:
        raise ValueError(f"Unknown selector rule: {selector_rule}")
    counter["stage"] = "finished"
    selected = candidates[chosen]
    row = {key: selected.get(key, "") for key in selected}
    row["experiment_id"] = ("diversified_four_portfolio_v1" if method_name == DIVERSE_METHOD
                            else "four_candidate_portfolio_v1" if method_name == PORTFOLIO_METHOD
                            else "local_calibrated_multistart_v1" if method_name == LOCAL_METHOD
                            else "economic_multistart_v2" if method_name == ECONOMIC_METHOD
                            else "specialist_multistart_v2" if method_name == SPECIALIST_METHOD
                            else "expanded_audit_multistart_v1" if method_name == AUDIT_METHOD
                            else "validation_calibrated_alpha_v1")
    row["method"] = method_name
    row["trusted_label_count"] = max(int(r.get("trusted_label_count") or 0)
                                     for r in selected_candidates.values())
    row["extra_trusted_labels"] = max(int(r.get("extra_trusted_labels") or 0)
                                     for r in selected_candidates.values())
    row["trusted_label_source"] = ("synthetic_true_function"
                                   if row["extra_trusted_labels"] else "stage0")
    row["status"] = "valid"
    row["delivery_status"] = "valid"
    row["alpha_audit"] = _json({
        "selected_initialization": chosen,
        "selection_rule": selector_rule,
        "internal_scores": internal_scores,
        "candidate_method_order": list(method_order),
        "calibration_coefficients": coefficients.tolist() if coefficients is not None else None,
        "calibration": calibration,
        "candidate_statuses": {m: r["status"] for m, r in selected_candidates.items()},
    })
    initialization_rows = (candidates.values() if charge_all_initializations
                           else selected_candidates.values())
    for metric in ("requests", "input_samples", "seconds"):
        suffix = "input_samples" if metric == "input_samples" else metric
        row[f"initialization_{suffix}"] = sum(float(r[f"initialization_{suffix}"])
                                               for r in initialization_rows)
        row[f"stage3_{suffix}"] = sum(float(r[f"stage3_{suffix}"])
                                       for r in selected_candidates.values())
    row["initialization_requests"] += (counter["requests"]["calibration"]
                                        + counter["requests"]["candidate_scoring"])
    row["initialization_input_samples"] += (counter["input_samples"]["calibration"]
                                             + counter["input_samples"]["candidate_scoring"])
    row["initialization_seconds"] += (counter["seconds"]["calibration"]
                                       + counter["seconds"]["candidate_scoring"])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--external-evaluator-epochs", type=int, default=100)
    parser.add_argument("--economic-extras", action="store_true",
                        help="Compare the five-candidate control with two extra initial alphas")
    parser.add_argument("--local-selector", action="store_true",
                        help="Also choose among the same five alphas using candidate-local Stage 0 bias")
    parser.add_argument("--portfolio-four", action="store_true",
                        help="Also select among A, B, MSE power, and residual-second-moment alpha")
    parser.add_argument("--diverse-four", action="store_true",
                        help="Also select four ex-ante alphas with maximal pairwise L1 spread")
    parser.add_argument("--expanded-audit-extra", action="store_true",
                        help="Add local-bias-risk alpha using 100 trusted blind labels")
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate seed")
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError("Output directory must be empty")
    name, current = next((name, current) for name, current in
                         build_scenario_args(root, seeds, args.external_evaluator_epochs)
                         if name == "reference_clean")
    output = Path(current.output_dir)
    candidate_methods = (CANDIDATE_METHODS
                         + (ECONOMIC_EXTRAS if args.economic_extras else ())
                         + ((AUDIT_CANDIDATE,) if args.expanded_audit_extra else ()))
    summary_methods = (SUMMARY_METHODS
                       + ((SPECIALIST_METHOD, ECONOMIC_METHOD) if args.economic_extras else ())
                       + ((LOCAL_METHOD,) if args.local_selector else ())
                       + ((PORTFOLIO_METHOD,) if args.portfolio_four else ())
                       + ((DIVERSE_METHOD,) if args.diverse_four else ()))
    if args.expanded_audit_extra:
        summary_methods += (AUDIT_METHOD,)
    selection_variants = {META_METHOD: CANDIDATE_METHODS}
    selector_rules = {META_METHOD: "global"}
    if args.economic_extras:
        selection_variants[SPECIALIST_METHOD] = SPECIALIST_CANDIDATES
        selection_variants[ECONOMIC_METHOD] = candidate_methods
        selector_rules[SPECIALIST_METHOD] = "global"
        selector_rules[ECONOMIC_METHOD] = "global"
    if args.local_selector:
        selection_variants[LOCAL_METHOD] = CANDIDATE_METHODS
        selector_rules[LOCAL_METHOD] = "local_bias"
    if args.portfolio_four:
        selection_variants[PORTFOLIO_METHOD] = PORTFOLIO_CANDIDATES
        selector_rules[PORTFOLIO_METHOD] = "global"
    if args.diverse_four:
        selection_variants[DIVERSE_METHOD] = CANDIDATE_METHODS
        selector_rules[DIVERSE_METHOD] = "global"
    if args.expanded_audit_extra:
        selection_variants[AUDIT_METHOD] = CANDIDATE_METHODS + (AUDIT_CANDIDATE,)
        selector_rules[AUDIT_METHOD] = "global"
    manifest = {"schema_version": 1,
                "study": "expanded_audit_multistart_v1" if args.expanded_audit_extra
                         else "diversified_four_portfolio_v1" if args.diverse_four
                         else "four_candidate_portfolio_v1" if args.portfolio_four
                         else "local_calibrated_multistart_v1" if args.local_selector
                         else "economic_multistart_v2" if args.economic_extras
                         else "validation_calibrated_alpha_v1",
                "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "scenarios": [name], "seeds": seeds, "methods": summary_methods,
                "candidate_initializations": candidate_methods,
                "selection_variants": selection_variants,
                "selector_rules": selector_rules,
                "dynamic_portfolio_rule": ("mean_pairwise_L1_keep_A_B_four"
                                           if args.diverse_four else ""),
                "expanded_audit": ({"candidate": AUDIT_CANDIDATE,
                                    "initialization_formula": "local_bias_risk",
                                    "total_trusted_labels": 100,
                                    "extra_trusted_labels": 80,
                                    "source": "synthetic_true_function",
                                    "selector_labels": 20,
                                    "label_acquisition_cost": "unpriced_separate_resource"}
                                   if args.expanded_audit_extra else None),
                "expected_conditions": len(seeds)*len(summary_methods),
                "primary_baseline": META_METHOD if args.economic_extras else "B_stage0",
                "external_evaluator": "posthoc_only",
                "selection_inputs": "Stage 0 blind labels, local APIs, candidate X; no pooled or truth scores",
                "box_tolerance": 1e-8, "reload_score_tolerance": 1e-7,
                "statistical_rule": (f"{len(seeds)} seed-paired comparisons; "
                                     "use the cohort's preregistration for baselines, "
                                     "simultaneous intervals, and decision threshold"),
                "arguments": vars(current)}
    path = root / "suite_manifest.json"
    path.write_text(_json(manifest)+"\n")
    try:
        output.mkdir()
        write_run_manifest(output, current, "running")
        for seed in seeds:
            print(f"=== {name} seed {seed} ===", flush=True)
            base = build_base_server(current, seed, 2, 0.0, 0.0)
            eligible = {int(x["agent_id"]) for x in base.phase1_report
                        if x.get("reason") in {"selected", "not_selected_by_coalition_rule"}}
            missing = sorted(set(current.fixed_participant_ids)-eligible)
            shared = {"scenario": name, "seed": seed, "preparation": base.preparation_costs,
                      "qualification": base.phase1_report,
                      "fixed_ids": current.fixed_participant_ids,
                      "missing_fixed_ids": missing}
            if missing:
                for method in summary_methods:
                    _append(output / "summary.csv", {"experiment_id": manifest["study"],
                        "scenario": name, "seed": seed, "method": method,
                        "status": "qualification_failure", "failure_stage": "qualification",
                        "failure_reason": _json(missing), "delivery_status": "none"})
            else:
                by_id = {a.agent_id:a for a in base.all_agents}
                base.trusted_agents = [by_id[i] for i in current.fixed_participant_ids]
                base.I_list = []
                base.preparation_counter["stage"] = "fixed_participant_proposal"
                started = time.perf_counter()
                for agent in base.trusted_agents:
                    torch.manual_seed(seed*100000+agent.agent_id)
                    base.I_list.append(agent.infer_parameters_D(0.0, steps=current.inverse_steps))
                base.preparation_counter["stage"] = "finished"
                shared["fixed_proposal_seconds"] = time.perf_counter()-started
                shared["fixed_proposal_requests"] = base.preparation_counter["requests"]["fixed_participant_proposal"]
                shared["fixed_proposal_input_samples"] = base.preparation_counter["input_samples"]["fixed_participant_proposal"]
                shared["coalition_ids"] = [a.agent_id for a in base.trusted_agents]
                shared["proposals"] = [np.asarray(x,float).tolist() for x in base.I_list]
                shared["blind_test_X"] = np.asarray(base.test_X,float).tolist()
                shared["blind_test_y"] = np.asarray(base.test_y,float).tolist()
                if args.expanded_audit_extra:
                    numpy_state = np.random.get_state()
                    try:
                        expanded = HostServer(0.0, base.n_features,
                                              test_seed=seed+7919, n_test=100,
                                              constraint_mode="box_constrained")
                    finally:
                        np.random.set_state(numpy_state)
                    if (not np.array_equal(expanded.test_X[:len(base.test_X)],base.test_X)
                            or not np.array_equal(expanded.test_y[:len(base.test_y)],base.test_y)):
                        raise AssertionError("Expanded audit changed the first 20 labels")
                    shared["expanded_blind_test_X"] = expanded.test_X.tolist()
                    shared["expanded_blind_test_y"] = expanded.test_y.tolist()
                    shared["extra_trusted_labels"] = len(expanded.test_y)-len(base.test_y)
                    shared["trusted_label_source"] = "synthetic_true_function"
                mse_by_id = {int(r["agent_id"]):float(r["mse"]) for r in base.phase1_report}
                mse = [mse_by_id[i] for i in current.fixed_participant_ids]
                started = time.perf_counter()
                evaluator, artifact_dir, metrics = save_scenario_artifacts(output,
                    {"seed":seed,"dimension":2,"poison_ratio":0.0,"target":0.0},
                    base.all_agents, base.study_records,
                    evaluator_epochs=current.external_evaluator_epochs,
                    validation_fraction=current.external_validation_fraction,
                    test_fraction=current.external_test_fraction)
                shared["evaluator_seconds"] = time.perf_counter()-started
                shared["evaluator_metrics"] = metrics
                shared["artifact_path"] = str(artifact_dir.relative_to(output))
                del base.study_records
                candidates = {}
                for method in candidate_methods:
                    options = ({"blind_x":expanded.test_X, "blind_y":expanded.test_y,
                                "initialization_method":"local_bias_risk",
                                "trusted_label_source":"synthetic_true_function"}
                               if method == AUDIT_CANDIDATE else {})
                    row = _one_method(base, method, mse, evaluator,
                                      shared["artifact_path"], current, name, seed,
                                      **options)
                    candidates[method] = row
                    _append(output / "candidate_rows.csv", row)
                    if method in ("A_uniform", "B_stage0"):
                        _append(output / "summary.csv", row)
                meta = _meta_result(candidates, base, shared["artifact_path"], seed)
                _append(output / "summary.csv", meta)
                if args.economic_extras:
                    specialist = _meta_result(candidates, base, shared["artifact_path"], seed,
                                              method_order=SPECIALIST_CANDIDATES,
                                              method_name=SPECIALIST_METHOD)
                    _append(output / "summary.csv", specialist)
                    economic = _meta_result(candidates, base, shared["artifact_path"], seed,
                                            method_order=candidate_methods,
                                            method_name=ECONOMIC_METHOD)
                    _append(output / "summary.csv", economic)
                if args.local_selector:
                    local = _meta_result(candidates, base, shared["artifact_path"], seed,
                                         method_name=LOCAL_METHOD,
                                         selector_rule="local_bias")
                    _append(output / "summary.csv", local)
                if args.portfolio_four:
                    portfolio = _meta_result(candidates, base, shared["artifact_path"], seed,
                                             method_order=PORTFOLIO_CANDIDATES,
                                             method_name=PORTFOLIO_METHOD)
                    _append(output / "summary.csv", portfolio)
                if args.diverse_four:
                    all_alphas = {method: json.loads(row["initial_alpha"])
                                  for method, row in candidates.items()}
                    selection_started = time.perf_counter()
                    diverse_methods = diverse_four_methods(all_alphas)
                    portfolio_selection_seconds = time.perf_counter() - selection_started
                    diverse = _meta_result(candidates, base, shared["artifact_path"], seed,
                                           method_order=diverse_methods,
                                           method_name=DIVERSE_METHOD,
                                           charge_all_initializations=True)
                    diverse["initialization_seconds"] += portfolio_selection_seconds
                    diverse_audit = json.loads(diverse["alpha_audit"])
                    diverse_audit["portfolio_rule"] = "mean_pairwise_L1_keep_A_B_four"
                    diverse_audit["portfolio_selection_seconds"] = portfolio_selection_seconds
                    diverse["alpha_audit"] = _json(diverse_audit)
                    _append(output / "summary.csv", diverse)
                if args.expanded_audit_extra:
                    audit_meta = _meta_result(candidates, base, shared["artifact_path"], seed,
                                              method_order=selection_variants[AUDIT_METHOD],
                                              method_name=AUDIT_METHOD)
                    _append(output / "summary.csv", audit_meta)
            with (output / "shared_costs.jsonl").open("a") as handle:
                handle.write(_json(shared)+"\n")
            write_run_manifest(output, current, "running")
        write_run_manifest(output, current, "complete")
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "interrupted"
        manifest["error"] = repr(exc)
        raise
    finally:
        manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        path.write_text(_json(manifest)+"\n")


if __name__ == "__main__":
    main()

"""Paired, static-alpha inverse study. The pooled model scores only delivered X."""

import argparse
import csv
import hashlib
import json
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from alpha_weight_experiments import (attach_query_counter, build_base_server,
                                      run_stage3, write_run_manifest)
from initial_alpha_methods import METHODS, initial_alpha
from p5_scenario_suite import build_scenario_args
from research_evaluation import save_scenario_artifacts, score_solution
from run_experiments import true_function


CONFIRMATORY = ("reference_clean", "sample_imbalance_4x", "low_quality_25pct",
                "dimension_5", "controlled_coalition_5")
EXPLORATORY = ("hard_target_2p5",)
FIELDS = ("experiment_id", "scenario", "seed", "method", "status", "failure_stage",
          "failure_reason", "delivery_status", "final_solution", "initial_alpha",
          "agent_ids", "stage0_mse", "alpha_audit", "actual_solver", "solver_status",
          "final_box_feasible", "max_box_violation", "external_global_prediction",
          "external_global_error", "final_target_error", "initialization_requests",
          "initialization_input_samples", "initialization_seconds", "stage3_requests",
          "stage3_input_samples", "stage3_seconds", "external_scoring_seconds",
          "artifact_path", "trusted_label_count", "extra_trusted_labels",
          "trusted_label_source")


def _append(path, row):
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in FIELDS})
        handle.flush()


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _costs(counter, row):
    for stage in ("initialization", "stage3"):
        row[f"{stage}_requests"] = counter["requests"][stage]
        row[f"{stage}_input_samples"] = counter["input_samples"][stage]
        row[f"{stage}_seconds"] = counter["seconds"][stage]


def _one_method(base, method, mse, evaluator, artifact_path, args, scenario, seed,
                blind_x=None, blind_y=None, initialization_method=None,
                trusted_label_source="stage0"):
    server = deepcopy(base)
    counter = attach_query_counter(server.trusted_agents)
    row = {"experiment_id": "alpha_initial_static_v1", "scenario": scenario,
           "seed": seed, "method": method, "status": "running",
           "delivery_status": "none", "artifact_path": artifact_path,
           "agent_ids": _json([a.agent_id for a in server.trusted_agents]),
           "stage0_mse": _json(mse)}
    x = server.test_X if blind_x is None else np.asarray(blind_x, dtype=float)
    y = server.test_y if blind_y is None else np.asarray(blind_y, dtype=float)
    if len(x) != len(y) or len(x) < len(server.test_y):
        raise ValueError("Blind labels must be paired and include Stage 0")
    row["trusted_label_count"] = len(y)
    row["extra_trusted_labels"] = len(y) - len(server.test_y)
    row["trusted_label_source"] = trusted_label_source
    counter["stage"] = "initialization"
    try:
        alpha, audit = initial_alpha(initialization_method or method,
                                     server.trusted_agents, mse, x, y, server.I_list,
                                     server.target_T, server.feasible_lower,
                                     server.feasible_upper)
        server.alphas = alpha.tolist()
        row["initial_alpha"] = _json(alpha.tolist())
        row["alpha_audit"] = _json(audit)
        counter["stage"] = "stage3"
        try:
            solution, _, _ = run_stage3(server, args.optimizer, args.custom_iterations)
            solver = getattr(server, "last_subspace_optimization", {})
            row["actual_solver"] = solver.get("actual_solver", solver.get("chosen_engine", ""))
            row["solver_status"] = solver.get("message", "")
            violation = server.box_violation(solution)
            row["max_box_violation"] = violation
            row["final_box_feasible"] = violation <= 1e-8
            if not np.isfinite(solution).all() or violation > 1e-8:
                row.update(status="invalid_output", failure_stage="stage3",
                           failure_reason="nonfinite_or_box_violation")
            else:
                row["final_solution"] = _json(solution.tolist())
                row["delivery_status"] = "valid"
                row["status"] = "valid"
        except (ValueError, FloatingPointError) as exc:
            row.update(status="solver_failure", failure_stage="stage3",
                       failure_reason=str(exc), final_box_feasible=False)
        counter["stage"] = "finished"
        if row["delivery_status"] == "valid":
            started = time.perf_counter()
            prediction, error = score_solution(evaluator, solution, server.target_T)
            row["external_global_prediction"] = prediction
            row["external_global_error"] = error
            row["final_target_error"] = abs(true_function(solution) - server.target_T)
            row["external_scoring_seconds"] = time.perf_counter() - started
    except Exception as exc:
        row.update(status="interrupted", failure_stage=counter["stage"],
                   failure_reason=repr(exc), delivery_status="none")
        exc.partial_row = row
        raise
    finally:
        if counter["stage"] != "finished":
            counter["stage"] = "finished"
        _costs(counter, row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--scenarios", default=",".join(CONFIRMATORY))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--primary-baseline", choices=["A_uniform", "B_stage0"],
                        default="A_uniform")
    parser.add_argument("--external-evaluator-epochs", type=int, default=100)
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    names = args.scenarios.split(",")
    selected_methods = args.methods.split(",")
    if (len(set(seeds)) != len(seeds) or len(set(names)) != len(names)
            or len(set(selected_methods)) != len(selected_methods)):
        raise ValueError("Seeds, scenarios, and methods must be unique")
    if set(names) - set(CONFIRMATORY + EXPLORATORY):
        raise ValueError("Unknown scenario")
    if not selected_methods or set(selected_methods) - set(METHODS):
        raise ValueError("Unknown method")
    if args.primary_baseline not in selected_methods:
        raise ValueError("Primary baseline must be included in methods")
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError("Output directory must be empty")
    scenarios = {name: current for name, current in build_scenario_args(root, seeds,
                  args.external_evaluator_epochs) if name in names}
    manifest = {"schema_version": 1, "experiment_id": "alpha_initial_static_v1",
                "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "seeds": seeds, "scenarios": names, "methods": selected_methods,
                "primary_baseline": args.primary_baseline,
                "expected_conditions": len(seeds) * len(names) * len(selected_methods),
                "pooled_role": "posthoc_primary_only", "truth_role": "posthoc_diagnostic_only",
                "alpha_update": False, "pruning": False, "payment": "not_run",
                "box_tolerance": 1e-8, "reload_score_tolerance": 1e-7,
                "statistical_rule": "seed-paired mean D=E_method-E_primary_baseline; 10000 bootstrap; Bonferroni across selected candidate methods",
                "method_definitions": {
                    "A_uniform": "uniform",
                    "B_stage0": "normalized inverse Stage 0 blind MSE",
                    "quality_shrink": "half uniform plus half Stage 0 quality",
                    "ensemble_shrink": "half uniform plus half simplex fit on Stage 0 blind labels",
                    "target_peer_shrink": "half uniform plus half quality times peer target agreement on Stage 2 proposals",
                    "target_local_precision": "inverse target-weighted Stage 0 MSE with five global-MSE pseudo-observations and target-distance bandwidth 0.5",
                },
                "source_sha256": {Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                "scenario_arguments": {name: vars(scenarios[name]) for name in names}}
    manifest_path = root / "suite_manifest.json"
    manifest_path.write_text(_json(manifest) + "\n")
    try:
        for name in names:
            current = scenarios[name]
            output = Path(current.output_dir)
            output.mkdir()
            write_run_manifest(output, current, "running")
            result_file = output / "summary.csv"
            shared_file = output / "shared_costs.jsonl"
            for seed in seeds:
                print(f"=== {name} seed {seed} ===", flush=True)
                dimension = current.dimensions[0]
                poison = current.poison_ratios[0]
                target = current.targets[0]
                try:
                    base = build_base_server(current, seed, dimension, poison, target)
                except ValueError as exc:
                    if not any(phrase in str(exc) for phrase in (
                            "No agents passed", "No agents were selected")):
                        raise
                    for method in selected_methods:
                        _append(result_file, {"experiment_id": "alpha_initial_static_v1",
                            "scenario": name, "seed": seed, "method": method,
                            "status": "qualification_failure", "failure_stage": "qualification",
                            "failure_reason": str(exc), "delivery_status": "none"})
                    with shared_file.open("a") as handle:
                        handle.write(_json({"scenario": name, "seed": seed,
                            "preparation_costs_status": "unknown_due_to_qualification_exception",
                            "failure_reason": str(exc)}) + "\n")
                    continue
                eligible = {int(x["agent_id"]) for x in base.phase1_report
                            if x.get("reason") in {"selected", "not_selected_by_coalition_rule"}}
                missing = sorted(set(current.fixed_participant_ids) - eligible)
                shared = {"scenario": name, "seed": seed, "preparation": base.preparation_costs,
                          "qualification": base.phase1_report, "fixed_ids": current.fixed_participant_ids,
                          "missing_fixed_ids": missing}
                if missing:
                    for method in selected_methods:
                        _append(result_file, {"experiment_id": "alpha_initial_static_v1",
                            "scenario": name, "seed": seed, "method": method,
                            "status": "qualification_failure", "failure_stage": "qualification",
                            "failure_reason": _json(missing), "delivery_status": "none"})
                else:
                    by_id = {a.agent_id: a for a in base.all_agents}
                    base.trusted_agents = [by_id[i] for i in current.fixed_participant_ids]
                    base.I_list = []
                    base.preparation_counter["stage"] = "fixed_participant_proposal"
                    started = time.perf_counter()
                    for agent in base.trusted_agents:
                        torch.manual_seed(seed * 100000 + agent.agent_id)
                        base.I_list.append(agent.infer_parameters_D(target, steps=current.inverse_steps))
                    base.preparation_counter["stage"] = "finished"
                    shared["coalition_ids"] = [agent.agent_id for agent in base.trusted_agents]
                    shared["proposals"] = [np.asarray(x, dtype=float).tolist() for x in base.I_list]
                    shared["blind_test_X"] = np.asarray(base.test_X, dtype=float).tolist()
                    shared["blind_test_y"] = np.asarray(base.test_y, dtype=float).tolist()
                    shared["fixed_proposal_seconds"] = time.perf_counter() - started
                    shared["fixed_proposal_requests"] = base.preparation_counter["requests"]["fixed_participant_proposal"]
                    shared["fixed_proposal_input_samples"] = base.preparation_counter["input_samples"]["fixed_participant_proposal"]
                    mse_by_id = {int(r["agent_id"]): float(r["mse"]) for r in base.phase1_report}
                    mse = [mse_by_id[i] for i in current.fixed_participant_ids]
                    started = time.perf_counter()
                    evaluator, artifact_dir, metrics = save_scenario_artifacts(output,
                        {"seed": seed, "dimension": dimension, "poison_ratio": poison,
                         "target": target}, base.all_agents, base.study_records,
                        evaluator_epochs=current.external_evaluator_epochs,
                        validation_fraction=current.external_validation_fraction,
                        test_fraction=current.external_test_fraction)
                    shared["evaluator_seconds"] = time.perf_counter() - started
                    shared["evaluator_metrics"] = metrics
                    shared["artifact_path"] = str(artifact_dir.relative_to(output))
                    del base.study_records
                    for method in selected_methods:
                        try:
                            row = _one_method(base, method, mse, evaluator,
                                              shared["artifact_path"], current, name, seed)
                        except Exception as exc:
                            _append(result_file, getattr(exc, "partial_row", {"experiment_id": "alpha_initial_static_v1",
                                "scenario": name, "seed": seed, "method": method,
                                "status": "interrupted", "failure_stage": "program",
                                "failure_reason": repr(exc), "delivery_status": "none",
                                "artifact_path": shared["artifact_path"]}))
                            raise
                        _append(result_file, row)
                with shared_file.open("a") as handle:
                    handle.write(_json(shared) + "\n")
                    handle.flush()
                write_run_manifest(output, current, "running")
            write_run_manifest(output, current, "complete")
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "interrupted"
        manifest["error"] = repr(exc)
        raise
    finally:
        manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(_json(manifest) + "\n")


if __name__ == "__main__":
    main()

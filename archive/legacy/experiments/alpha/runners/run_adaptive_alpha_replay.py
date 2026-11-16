"""Sequential B-then-Uniform alpha replay on frozen model artifacts.

This is development work. The pooled evaluator and true function are accessed
only after the adaptive decision has been made. The 0.01 trigger was selected
after inspecting existing data and is not a preregistered quality rule.
"""

import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from agent_client import AgentNode
from alpha_weight_experiments import attach_query_counter
from explore_ensemble_target_objective import _ensemble_objective
from host_server import HostServer
from research_evaluation import load_evaluator_artifact, score_solution
from run_experiments import true_function
from validation_calibrated_alpha import fit_residual_calibrator, select_candidate


B = "B_stage0"
U = "A_uniform"


def should_solve_uniform(b_status, b_internal_error, threshold=0.01):
    """Decide before querying the Uniform Stage 3 candidate."""
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Trigger threshold must be finite and nonnegative")
    if b_status != "valid":
        return True, "B_failed"
    if b_internal_error is None or not np.isfinite(b_internal_error):
        return True, "B_unscorable"
    if b_internal_error > threshold:
        return True, "B_internal_error_above_threshold"
    return False, "B_internal_error_within_threshold"


def _csv_by_key(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = [(int(row["seed"]), row["method"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate candidate key in {path}")
    return dict(zip(keys, rows))


def _solve(server, agents, alpha, optimizer, iterations):
    server.alphas = list(alpha)
    counter = attach_query_counter(agents)
    counter["stage"] = "stage3"
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = server.optimize_candidate_subspace(
                strategy=optimizer, custom_iterations=iterations,
                history_metric=lambda _solution: float("nan"))
        solution = np.asarray(result["solution"], dtype=float)
        violation = float(server.box_violation(solution))
        if not np.isfinite(solution).all() or violation > 1e-8:
            raise ValueError(f"Invalid delivered solution, box violation={violation}")
        return {"status": "valid", "solution": solution.tolist(),
                "solver": result["actual_solver"],
                "max_box_violation": violation,
                "stage3_requests": counter["requests"]["stage3"],
                "stage3_input_samples": counter["input_samples"]["stage3"]}
    except (ValueError, FloatingPointError) as exc:
        return {"status": "solver_failure", "failure_reason": str(exc),
                "stage3_requests": counter["requests"]["stage3"],
                "stage3_input_samples": counter["input_samples"]["stage3"]}
    finally:
        counter["stage"] = "finished"


def replay_seed(root, scenario, seed, shared, candidates, args, threshold):
    directory = root / scenario
    item = shared[seed]
    artifact = directory / item["artifact_path"]
    payload = torch.load(artifact / "local_models.pt", map_location="cpu",
                         weights_only=True)
    agents = []
    for ident in item["coalition_ids"]:
        agent = AgentNode(ident, None, n_features=payload["n_features"])
        agent.model.load_state_dict(payload["agents"][ident])
        agent.model.eval()
        agents.append(agent)
    target = float(args["targets"][0])
    server = HostServer(target, payload["n_features"],
                        constraint_mode=args["constraint_mode"],
                        feasible_lower=args["feasible_lower"],
                        feasible_upper=args["feasible_upper"])
    server.trusted_agents = agents
    server.I_list = [np.asarray(p, dtype=float) for p in item["proposals"]]
    server._consensus_loss = MethodType(_ensemble_objective, server)
    b_alpha = json.loads(candidates[seed, B]["initial_alpha"])
    u_alpha = json.loads(candidates[seed, U]["initial_alpha"])
    methods = {B: _solve(server, agents, b_alpha, args["optimizer"],
                         args["custom_iterations"])}
    initial_selection_counter = attach_query_counter(agents)
    initial_selection_counter["stage"] = "selection"
    coefficients, calibration = fit_residual_calibrator(
        agents, item["blind_test_X"], item["blind_test_y"], b_alpha)
    b_score = None
    scores = {}
    if methods[B]["status"] == "valid":
        _, scores = select_candidate(
            agents, b_alpha, coefficients, {B: methods[B]["solution"]}, target,
            server.feasible_lower, server.feasible_upper,
            server.feasibility_tolerance, method_order=(B,))
        b_score = scores[B]["internal_target_error"]
    trigger, trigger_reason = should_solve_uniform(
        methods[B]["status"], b_score, threshold)
    initial_selection_counter["stage"] = "finished"
    if trigger:
        methods[U] = _solve(server, agents, u_alpha, args["optimizer"],
                            args["custom_iterations"])
    final_selection_counter = attach_query_counter(agents)
    final_selection_counter["stage"] = "selection"
    if U in methods and methods[U]["status"] == "valid":
        _, u_scores = select_candidate(
            agents, b_alpha, coefficients, {U: methods[U]["solution"]},
            target, server.feasible_lower, server.feasible_upper,
            server.feasibility_tolerance, method_order=(U,))
        scores.update(u_scores)
    if scores:
        chosen = min((B, U), key=lambda method: scores.get(
            method, {}).get("internal_target_error", np.inf))
        delivered = methods[chosen]["solution"]
        delivery_status = "valid"
    else:
        chosen, scores, delivered = None, {}, None
        delivery_status = "solver_failure"
    final_selection_counter["stage"] = "finished"

    # Post-decision assessment only. No evaluator or truth data above this line.
    pooled_error = truth_error = None
    if delivered is not None:
        evaluator, _ = load_evaluator_artifact(artifact)
        _, pooled_error = score_solution(evaluator, delivered, target)
        truth_error = abs(true_function(np.asarray(delivered))-target)
    return {
        "scenario": scenario, "seed": seed, "delivery_status": delivery_status,
        "threshold": threshold, "trigger_uniform": trigger,
        "trigger_reason": trigger_reason, "selected_initialization": chosen,
        "b_internal_error_at_trigger": b_score,
        "calibration": calibration, "internal_scores": scores,
        "candidates": methods, "delivered_solution": delivered,
        "pooled_error": pooled_error, "truth_error_diagnostic": truth_error,
        "selection_requests": (initial_selection_counter["requests"]["selection"]
                               + final_selection_counter["requests"]["selection"]),
        "selection_input_samples": (
            initial_selection_counter["input_samples"]["selection"]
            + final_selection_counter["input_samples"]["selection"]),
        "stage3_requests": sum(row["stage3_requests"] for row in methods.values()),
        "stage3_input_samples": sum(row["stage3_input_samples"]
                                    for row in methods.values()),
        "extra_trusted_labels": 0,
    }


def replay(root, seeds, threshold, output):
    root = Path(root)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((root / "suite_manifest.json").read_text())
    if (manifest["status"] != "complete" or len(manifest["scenarios"]) != 1
            or len(manifest["arguments"]["targets"]) != 1
            or not set(seeds) <= set(manifest["seeds"])):
        raise ValueError("Need a complete, matching single-scenario saved run")
    scenario = manifest["scenarios"][0]
    directory = root / scenario
    shared = {int(row["seed"]): row for row in map(
        json.loads, (directory / "shared_costs.jsonl").read_text().splitlines())}
    candidates = _csv_by_key(directory / "candidate_rows.csv")
    rows = []
    for seed in seeds:
        if seed not in shared or any((seed, method) not in candidates
                                     for method in (B, U)):
            raise ValueError(f"Missing saved input: {seed}")
        row = replay_seed(root, scenario, seed, shared, candidates,
                          manifest["arguments"], threshold)
        rows.append(row)
        output.write_text(json.dumps({
            "study": "adaptive_B_then_uniform_development_replay",
            "scope": "reused_model_artifacts_not_independent_formal_evidence",
            "status": "complete" if len(rows) == len(seeds) else "running",
            "source_root": str(root), "scenario": scenario, "seeds": seeds,
            "threshold": threshold, "completed_seeds": len(rows), "rows": rows,
        }, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_root")
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--output", required=True)
    options = parser.parse_args()
    seeds = [int(s) for s in options.seeds.split(",")]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be nonempty and unique")
    rows = replay(options.baseline_root, seeds, options.threshold,
                  options.output)
    print(json.dumps({"completed": len(rows),
                      "valid": sum(r["delivery_status"] == "valid" for r in rows),
                      "triggered": sum(r["trigger_uniform"] for r in rows),
                      "mean_stage3_requests": float(np.mean([
                          r["stage3_requests"] for r in rows]))}, indent=2))


if __name__ == "__main__":
    main()

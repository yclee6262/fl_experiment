"""Exploratory Stage 3 objective replay on saved alpha candidates.

No pooled score or synthetic truth enters the solver or candidate selector.
The comparison changes the Stage 3 objective while preserving alpha, proposal,
models, box constraints, solver, and the existing B20 ridge selector.
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
from alpha_weight_experiments import attach_query_counter, stage3_prediction_cache
from host_server import HostServer
from research_evaluation import load_evaluator_artifact, score_solution
from run_experiments import true_function
from run_validation_calibrated_alpha import META_METHOD
from validation_calibrated_alpha import (CANDIDATE_METHODS, fit_residual_calibrator,
                                         select_candidate)


def _rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _ensemble_objective(server, solution, agents=None, mode="optimization",
                        weights=None, trim_fraction=0.2):
    if mode != "optimization":
        raise ValueError("Exploration only supports optimization mode")
    agents = server.trusted_agents if agents is None else agents
    alpha = np.asarray(server.alphas, dtype=float)
    if len(alpha) != len(agents) or not np.isfinite(alpha).all() or np.any(alpha < 0):
        raise ValueError("Invalid alpha")
    alpha = alpha / alpha.sum()
    predictions = np.asarray([agent.api_predict(solution)[0] for agent in agents],
                             dtype=float)
    return float(abs(np.dot(alpha, predictions) - server.target_T))


def analyze(root, seeds, cache_stage3=False):
    root = Path(root)
    manifest = json.loads((root / "suite_manifest.json").read_text())
    if manifest["status"] != "complete" or len(manifest["scenarios"]) != 1:
        raise ValueError("Expected a complete single-scenario study")
    if not set(seeds) <= set(manifest["seeds"]):
        raise ValueError("Unrecognized seed")
    scenario = manifest["scenarios"][0]
    directory = root / scenario
    args = manifest["arguments"]
    target = float(args["targets"][0])
    settings = {"optimizer": args["optimizer"],
                "custom_iterations": int(args["custom_iterations"]),
                "constraint_mode": args["constraint_mode"],
                "feasible_lower": float(args["feasible_lower"]),
                "feasible_upper": float(args["feasible_upper"])}
    shared = {int(r["seed"]): r for r in
              map(json.loads, (directory / "shared_costs.jsonl").read_text().splitlines())}
    candidates = {}
    for row in _rows(directory / "candidate_rows.csv"):
        key = (int(row["seed"]), row["method"])
        if key in candidates:
            raise ValueError(f"Duplicate candidate: {key}")
        candidates[key] = row
    summaries = {}
    for row in _rows(directory / "summary.csv"):
        key = (int(row["seed"]), row["method"])
        if key in summaries:
            raise ValueError(f"Duplicate summary: {key}")
        summaries[key] = row
    output = []
    for seed in seeds:
        item = shared[seed]
        if item.get("missing_fixed_ids"):
            raise ValueError(f"Missing coalition for seed {seed}")
        artifact = directory / item["artifact_path"]
        payload = torch.load(artifact / "local_models.pt", map_location="cpu",
                             weights_only=True)
        agents = []
        for ident in item["coalition_ids"]:
            agent = AgentNode(ident, None, n_features=payload["n_features"])
            agent.model.load_state_dict(payload["agents"][ident])
            agent.model.eval()
            agents.append(agent)
        evaluator, _ = load_evaluator_artifact(artifact)
        server = HostServer(target, payload["n_features"],
                            constraint_mode=settings["constraint_mode"],
                            feasible_lower=settings["feasible_lower"],
                            feasible_upper=settings["feasible_upper"])
        server.trusted_agents = agents
        server.I_list = [np.asarray(p, dtype=float) for p in item["proposals"]]
        server._consensus_loss = MethodType(_ensemble_objective, server)
        candidate_output = {}
        solutions = {}
        for method in CANDIDATE_METHODS:
            old = candidates[seed, method]
            if old["delivery_status"] != "valid":
                raise ValueError(f"Invalid saved candidate: {seed}/{method}")
            alpha = json.loads(old["initial_alpha"])
            server.alphas = alpha
            counter = attach_query_counter(agents)
            counter["stage"] = "stage3"
            try:
                cache_context = (stage3_prediction_cache(agents) if cache_stage3
                                 else contextlib.nullcontext({"hits": 0, "misses": 0}))
                with cache_context as cache_stats:
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = server.optimize_candidate_subspace(
                            strategy=settings["optimizer"],
                            custom_iterations=settings["custom_iterations"],
                            history_metric=lambda _solution: float("nan"))
                        solution = np.asarray(result["solution"], dtype=float)
                counter["stage"] = "finished"
                violation = server.box_violation(solution)
                if not np.isfinite(solution).all() or violation > 1e-8:
                    raise ValueError(f"Invalid candidate box solution: {violation}")
                prediction, pooled = score_solution(evaluator, solution, target)
                solutions[method] = solution
                candidate_output[method] = {
                    "status": "valid", "alpha": alpha, "solution": solution.tolist(),
                    "pooled_error": pooled,
                    "truth_error": abs(true_function(solution) - target),
                    "ensemble_target_error": server.last_subspace_optimization["loss"],
                    "stage3_requests": counter["requests"]["stage3"],
                    "stage3_input_samples": counter["input_samples"]["stage3"],
                    "stage3_cache_hits": cache_stats["hits"],
                    "stage3_cache_misses": cache_stats["misses"],
                    "solver": server.last_subspace_optimization["actual_solver"],
                    "max_box_violation": violation,
                    "old_pooled_error": float(old["external_global_error"]),
                    "old_stage3_requests": int(float(old["stage3_requests"])),
                }
            except (ValueError, FloatingPointError) as exc:
                counter["stage"] = "finished"
                solutions[method] = None
                candidate_output[method] = {
                    "status": "solver_failure", "reason": str(exc),
                    "stage3_requests": counter["requests"]["stage3"],
                    "stage3_cache_hits": cache_stats["hits"] if cache_stage3 else 0,
                    "stage3_cache_misses": cache_stats["misses"] if cache_stage3 else 0,
                    "old_stage3_requests": int(float(old["stage3_requests"]))}
        baseline_alpha = json.loads(candidates[seed, "B_stage0"]["initial_alpha"])
        selector_counter = attach_query_counter(agents)
        selector_counter["stage"] = "selection"
        coefficients, _ = fit_residual_calibrator(agents, item["blind_test_X"],
                                                   item["blind_test_y"], baseline_alpha)
        selected, internal_scores = select_candidate(
            agents, baseline_alpha, coefficients, solutions, target,
            server.feasible_lower, server.feasible_upper,
            server.feasibility_tolerance)
        selector_counter["stage"] = "finished"
        old_meta = summaries[seed, META_METHOD]
        old_solution = np.asarray(json.loads(old_meta["final_solution"]), dtype=float)
        _, reloaded_old_error = score_solution(evaluator, old_solution, target)
        if abs(reloaded_old_error - float(old_meta["external_global_error"])) > 1e-7:
            raise ValueError(f"Saved pooled score did not reload: {seed}")
        output.append({"scenario": scenario, "seed": seed,
                       "delivery_status": "valid",
                       "candidate_failures": sum(v["status"] != "valid"
                                                 for v in candidate_output.values()),
                       "new_selected_initialization": selected,
                       "old_selected_initialization": json.loads(old_meta["alpha_audit"])[
                           "selected_initialization"],
                       "new_pooled_error": candidate_output[selected]["pooled_error"],
                       "old_pooled_error": float(old_meta["external_global_error"]),
                       "new_truth_error": candidate_output[selected]["truth_error"],
                       "old_truth_error": float(old_meta["final_target_error"]),
                       "new_stage3_requests": sum(v["stage3_requests"]
                                                  for v in candidate_output.values()),
                       "old_stage3_requests": sum(v["old_stage3_requests"]
                                                  for v in candidate_output.values()
                                                  if "old_stage3_requests" in v),
                       "selection_requests": selector_counter["requests"]["selection"],
                       "internal_scores": internal_scores,
                       "candidates": candidate_output})
    deltas = np.asarray([r["new_pooled_error"]-r["old_pooled_error"]
                         for r in output])
    return {"study": "ensemble_target_stage3_objective_development",
            "scope": "exploratory_saved_model_replay_not_formal_quality_evidence",
            "stage3_prediction_cache": ("per_candidate_agentnode_float32"
                                        if cache_stage3 else "off"),
            "objective_old": "sum_i alpha_i * abs(f_i(X)-target)",
            "objective_new": "abs(sum_i alpha_i*f_i(X)-target)",
            "scenario": scenario, "seeds": seeds,
            "mean_paired_pooled_delta_new_minus_old": float(deltas.mean()),
            "wins_ties_losses": [int((deltas < -1e-7).sum()),
                                   int((np.abs(deltas) <= 1e-7).sum()),
                                   int((deltas > 1e-7).sum())],
            "mean_stage3_requests_old": float(np.mean([r["old_stage3_requests"] for r in output])),
            "mean_stage3_requests_new": float(np.mean([r["new_stage3_requests"] for r in output])),
            "rows": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--seeds", default="2400,2401,2402,2403,2404")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-stage3", action="store_true",
                        help="Cache repeated AgentNode predictions within each Stage 3 solve")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be nonempty and unique")
    result = analyze(args.root, seeds, cache_stage3=args.cache_stage3)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    concise = {key: value for key, value in result.items()
               if key not in ("rows", "seeds")}
    concise["seed_count"] = len(seeds)
    concise["seed_range"] = [min(seeds), max(seeds)]
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()

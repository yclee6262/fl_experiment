"""Explore one-solve initial alphas under the ensemble-target Stage 3 loss.

Only Stage 0 blind labels, local APIs, and saved proposals set alpha and X.
The pooled evaluator and synthetic truth score X after each solve.
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
from initial_alpha_methods import initial_alpha
from research_evaluation import load_evaluator_artifact, score_solution
from run_experiments import true_function
from run_validation_calibrated_alpha import META_METHOD


METHODS = ("uniform_stacked_half", "B_stacked_half", "stacked_full")


def explore(root, seeds):
    root = Path(root)
    manifest = json.loads((root / "suite_manifest.json").read_text())
    if manifest["status"] != "complete" or len(manifest["scenarios"]) != 1:
        raise ValueError("Expected complete single-scenario study")
    if not set(seeds) <= set(manifest["seeds"]):
        raise ValueError("Seed not in saved study")
    scenario = manifest["scenarios"][0]
    directory = root / scenario
    args = manifest["arguments"]
    target = float(args["targets"][0])
    shared = {int(row["seed"]): row for row in
              map(json.loads, (directory / "shared_costs.jsonl").read_text().splitlines())}
    with (directory / "candidate_rows.csv").open(newline="") as handle:
        candidates = {(int(row["seed"]), row["method"]): row
                      for row in csv.DictReader(handle)}
    with (directory / "summary.csv").open(newline="") as handle:
        summary = {(int(row["seed"]), row["method"]): row
                   for row in csv.DictReader(handle)}
    output = []
    for seed in seeds:
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
        evaluator, _ = load_evaluator_artifact(artifact)
        b_row = candidates[seed, "B_stage0"]
        mse = json.loads(b_row["stage0_mse"])
        b = np.asarray(json.loads(b_row["initial_alpha"]), dtype=float)
        init_counter = attach_query_counter(agents)
        init_counter["stage"] = "initialization"
        uniform_shrink, audit = initial_alpha(
            "ensemble_shrink", agents, mse, item["blind_test_X"],
            item["blind_test_y"], item["proposals"], target)
        init_counter["stage"] = "finished"
        if audit["fallback_reason"] or not audit["raw_scores"]:
            raise ValueError(f"Stacking fallback at seed {seed}: {audit['fallback_reason']}")
        fitted = np.asarray(audit["raw_scores"], dtype=float)
        alpha_by_method = {
            "uniform_stacked_half": uniform_shrink,
            "B_stacked_half": .5*b + .5*fitted,
            "stacked_full": fitted,
        }
        method_output = {}
        for method in METHODS:
            alpha = np.asarray(alpha_by_method[method], dtype=float)
            server = HostServer(target, payload["n_features"],
                                constraint_mode=args["constraint_mode"],
                                feasible_lower=args["feasible_lower"],
                                feasible_upper=args["feasible_upper"])
            server.trusted_agents = agents
            server.I_list = [np.asarray(p, dtype=float) for p in item["proposals"]]
            server.alphas = alpha.tolist()
            server._consensus_loss = MethodType(_ensemble_objective, server)
            counter = attach_query_counter(agents)
            counter["stage"] = "stage3"
            with contextlib.redirect_stdout(io.StringIO()):
                result = server.optimize_candidate_subspace(
                    strategy=args["optimizer"],
                    custom_iterations=args["custom_iterations"],
                    history_metric=lambda _solution: float("nan"))
            counter["stage"] = "finished"
            solution = np.asarray(result["solution"], dtype=float)
            violation = server.box_violation(solution)
            if violation > 1e-8 or not np.isfinite(solution).all():
                raise ValueError(f"Invalid delivered X: {seed}/{method}")
            _, pooled = score_solution(evaluator, solution, target)
            method_output[method] = {
                "alpha": alpha.tolist(), "solution": solution.tolist(),
                "pooled_error": pooled,
                "truth_error": abs(true_function(solution)-target),
                "stage3_requests": counter["requests"]["stage3"],
                "initialization_requests": init_counter["requests"]["initialization"],
                "initialization_input_samples": init_counter["input_samples"]["initialization"],
                "solver": result["actual_solver"],
                "max_box_violation": violation,
            }
        old = summary[seed, META_METHOD]
        output.append({"scenario": scenario, "seed": seed,
                       "old_five_pooled_error": float(old["external_global_error"]),
                       "old_five_stage3_requests": int(float(old["stage3_requests"])),
                       "fitted_alpha": fitted.tolist(),
                       "B_alpha": b.tolist(),
                       "methods": method_output})
    overview = {}
    for method in METHODS:
        differences = np.asarray([r["methods"][method]["pooled_error"]-
                                  r["old_five_pooled_error"] for r in output])
        overview[method] = {
            "mean_pooled_error": float(np.mean([r["methods"][method]["pooled_error"]
                                                for r in output])),
            "mean_pooled_delta_vs_old_five": float(differences.mean()),
            "mean_stage3_requests": float(np.mean([
                r["methods"][method]["stage3_requests"] for r in output])),
            "mean_initialization_requests": float(np.mean([
                r["methods"][method]["initialization_requests"] for r in output])),
        }
    return {"study": "stacked_one_solve_ensemble_objective_development",
            "scenario": scenario, "seeds": seeds,
            "methods": overview, "rows": output,
            "scope": "exploratory_saved_models_not_formal_quality_evidence"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be nonempty and unique")
    result = explore(args.root, seeds)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False,
                                           indent=2, allow_nan=False) + "\n")
    print(json.dumps({"scenario": result["scenario"], "seed_count": len(seeds),
                      "methods": result["methods"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

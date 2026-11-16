"""Synthetic paid-label pilot for a new proposal-local initial alpha.

Nine independent labels are acquired around Stage 2 proposals before Stage 3.
Only those labels, Stage 0 MSE, and local-model predictions determine alpha.
The pooled evaluator is loaded only after X is solved and never selects alpha.
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
from research_analysis import paired_bootstrap_mean_ci
from research_evaluation import load_evaluator_artifact, score_solution
from run_experiments import true_function


LABELS_PER_PROPOSAL = 3
NEIGHBORHOOD_STD = 0.1
STAGE0_PRIOR_COUNT = 5.0
EPSILON = 1e-5


def proposal_local_points(proposals, seed, lower, upper):
    values = np.asarray(proposals, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all() or not lower < upper:
        raise ValueError("Invalid Stage 2 proposals or box")
    rng = np.random.default_rng(20261005 + int(seed))
    points = values[:, None, :] + rng.normal(
        0.0, NEIGHBORHOOD_STD,
        size=(len(values), LABELS_PER_PROPOSAL, values.shape[1]))
    return np.clip(points.reshape(-1, values.shape[1]), lower, upper)


def proposal_local_alpha(stage0_mse, predictions, labels):
    mse = np.asarray(stage0_mse, dtype=float)
    predicted = np.asarray(predictions, dtype=float)
    y = np.asarray(labels, dtype=float).reshape(-1)
    if (mse.ndim != 1 or predicted.shape != (len(y), len(mse))
            or len(y) < 1 or not np.isfinite(mse).all() or np.any(mse < 0)
            or not np.isfinite(predicted).all() or not np.isfinite(y).all()):
        raise ValueError("Invalid Stage 0 or proposal-local evidence")
    local_mse = np.mean((predicted - y[:, None]) ** 2, axis=0)
    combined = (STAGE0_PRIOR_COUNT*mse + len(y)*local_mse) / (STAGE0_PRIOR_COUNT + len(y))
    scores = 1.0 / (combined + EPSILON)
    alpha = scores / scores.sum()
    return alpha, {"stage0_mse": mse.tolist(), "proposal_local_mse": local_mse.tolist(),
                   "combined_mse": combined.tolist(),
                   "stage0_prior_pseudocount": STAGE0_PRIOR_COUNT,
                   "proposal_local_label_count": len(y),
                   "inverse_mse_epsilon": EPSILON}


def _load_rows(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = [(int(row["seed"]), row["method"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate candidate key: {path}")
    return dict(zip(keys, rows))


def explore(root, benchmark_json, seeds):
    root = Path(root)
    manifest = json.loads((root / "suite_manifest.json").read_text())
    benchmark = json.loads(Path(benchmark_json).read_text())
    if (manifest["status"] != "complete" or len(manifest["scenarios"]) != 1
            or not set(seeds) <= set(manifest["seeds"])
            or not set(seeds) <= set(benchmark["seeds"])
            or manifest["scenarios"][0] != benchmark["scenario"]):
        raise ValueError("Baseline and new-objective benchmark do not align")
    scenario = manifest["scenarios"][0]
    directory = root / scenario
    args = manifest["arguments"]
    target = float(args["targets"][0])
    shared = {int(row["seed"]): row for row in
              map(json.loads, (directory / "shared_costs.jsonl").read_text().splitlines())}
    candidates = _load_rows(directory / "candidate_rows.csv")
    benchmark_rows = {int(row["seed"]): row for row in benchmark["rows"]}
    output = []
    for seed in seeds:
        item = shared[seed]
        old = candidates[seed, "B_stage0"]
        control = benchmark_rows[seed]
        if (old["delivery_status"] != "valid" or
                control["candidates"]["B_stage0"]["status"] != "valid"):
            raise ValueError(f"Invalid saved B control: {seed}")
        artifact = directory / item["artifact_path"]
        model_payload = torch.load(artifact / "local_models.pt", map_location="cpu",
                                   weights_only=True)
        agents = []
        for ident in item["coalition_ids"]:
            agent = AgentNode(ident, None, n_features=model_payload["n_features"])
            agent.model.load_state_dict(model_payload["agents"][ident])
            agent.model.eval()
            agents.append(agent)
        stage0_mse = json.loads(old["stage0_mse"])
        b_alpha = np.asarray(json.loads(old["initial_alpha"]), dtype=float)
        if not np.allclose(b_alpha, control["candidates"]["B_stage0"]["alpha"], atol=1e-12):
            raise ValueError(f"Saved B alpha mismatch: {seed}")
        points = proposal_local_points(item["proposals"], seed,
                                       args["feasible_lower"], args["feasible_upper"])
        # This is a synthetic paid-information assumption, not a free oracle in deployment.
        labels = np.asarray([true_function(point) for point in points], dtype=float)
        init_counter = attach_query_counter(agents)
        init_counter["stage"] = "initialization"
        predictions = np.column_stack([
            np.asarray(agent.api_predict(points), dtype=float).reshape(-1)
            for agent in agents])
        alpha, audit = proposal_local_alpha(stage0_mse, predictions, labels)
        init_counter["stage"] = "finished"
        server = HostServer(target, model_payload["n_features"],
                            constraint_mode=args["constraint_mode"],
                            feasible_lower=args["feasible_lower"],
                            feasible_upper=args["feasible_upper"])
        server.trusted_agents = agents
        server.I_list = [np.asarray(proposal, dtype=float) for proposal in item["proposals"]]
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
        if not np.isfinite(solution).all() or violation > 1e-8:
            raise ValueError(f"Invalid delivered X: {seed}")
        evaluator, _ = load_evaluator_artifact(artifact)
        _, pooled = score_solution(evaluator, solution, target)
        b_control = control["candidates"]["B_stage0"]
        _, b_reloaded = score_solution(evaluator, b_control["solution"], target)
        if abs(b_reloaded - b_control["pooled_error"]) > 1e-7:
            raise ValueError(f"B control pooled score did not reload: {seed}")
        output.append({
            "scenario": scenario, "seed": seed, "status": "valid",
            "agent_ids": item["coalition_ids"],
            "proposal_local_points": points.tolist(),
            "proposal_local_labels": labels.tolist(),
            "proposal_local_predictions": predictions.tolist(),
            "initial_alpha": alpha.tolist(), "alpha_audit": audit,
            "solution": solution.tolist(), "solver": result["actual_solver"],
            "max_box_violation": violation,
            "pooled_error": pooled,
            "truth_error_diagnostic": abs(true_function(solution)-target),
            "trusted_label_count": len(labels),
            "trusted_label_source": "synthetic_true_function_before_stage3",
            "initialization_requests": init_counter["requests"]["initialization"],
            "initialization_input_samples": init_counter["input_samples"]["initialization"],
            "stage3_requests": counter["requests"]["stage3"],
            "stage3_input_samples": counter["input_samples"]["stage3"],
            "B_single_pooled_error": b_control["pooled_error"],
            "B_single_stage3_requests": b_control["stage3_requests"],
            "new_five_pooled_error": control["new_pooled_error"],
            "new_five_stage3_requests": control["new_stage3_requests"],
            "old_five_pooled_error": control["old_pooled_error"],
        })
    delta_b = np.asarray([row["pooled_error"]-row["B_single_pooled_error"]
                          for row in output], dtype=float)
    delta_five = np.asarray([row["pooled_error"]-row["new_five_pooled_error"]
                             for row in output], dtype=float)
    return {
        "study": "proposal_local_paid_label_initial_alpha_development",
        "scope": "synthetic_paid_label_assumption_reused_development_seeds_not_formal_evidence",
        "scenario": scenario, "seeds": seeds,
        "method_definition": {
            "labels_per_proposal": LABELS_PER_PROPOSAL,
            "neighborhood_std": NEIGHBORHOOD_STD,
            "stage0_prior_pseudocount": STAGE0_PRIOR_COUNT,
            "initial_alpha": "inverse of (5*Stage0_MSE + 9*proposal_neighborhood_MSE)/14, normalized",
            "stage3_objective": "abs(sum_i alpha_i*f_i(X)-target)",
            "paid_information": "nine independent synthetic truth labels before Stage 3",
        },
        "mean_paired_pooled_delta_vs_B_single": float(delta_b.mean()),
        "descriptive_95pct_ci_vs_B_single": paired_bootstrap_mean_ci(delta_b),
        "mean_paired_pooled_delta_vs_new_five": float(delta_five.mean()),
        "descriptive_95pct_ci_vs_new_five": paired_bootstrap_mean_ci(delta_five),
        "mean_stage3_requests": float(np.mean([row["stage3_requests"] for row in output])),
        "mean_initialization_requests": float(np.mean([row["initialization_requests"]
                                                         for row in output])),
        "trusted_labels_per_seed": LABELS_PER_PROPOSAL * len(agents),
        "rows": output,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_root")
    parser.add_argument("new_objective_benchmark_json")
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be nonempty and unique")
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(target)
    result = explore(args.baseline_root, args.new_objective_benchmark_json, seeds)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                 allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("rows", "seeds")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

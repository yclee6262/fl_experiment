"""Develop a no-new-label sparse initial alpha with paid-query accounting.

Top two Stage 0 inverse-MSE weights are retained and renormalized. Stage 3
keeps all candidate proposals but does not query a zero-weight local model.
The pooled evaluator and synthetic truth only score X after the solve.
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
from host_server import HostServer
from research_analysis import paired_bootstrap_mean_ci
from research_evaluation import load_evaluator_artifact, score_solution
from run_experiments import true_function


def top_two_stage0_alpha(stage0_mse):
    mse = np.asarray(stage0_mse, dtype=float)
    if mse.ndim != 1 or len(mse) < 2 or not np.isfinite(mse).all() or np.any(mse < 0):
        raise ValueError("Need at least two finite nonnegative Stage 0 MSEs")
    quality = 1.0 / (mse + 1e-5)
    selected = np.argsort(mse, kind="stable")[:2]
    sparse = np.zeros(len(mse), dtype=float)
    sparse[selected] = quality[selected]
    sparse /= sparse.sum()
    return sparse, selected.tolist()


def sparse_ensemble_objective(server, solution, agents=None, mode="optimization",
                              weights=None, trim_fraction=0.2):
    if mode != "optimization":
        raise ValueError("Sparse replay only supports optimization mode")
    agents = server.trusted_agents if agents is None else agents
    alpha = np.asarray(server.alphas, dtype=float)
    if (len(alpha) != len(agents) or not np.isfinite(alpha).all()
            or np.any(alpha < 0) or not np.isclose(alpha.sum(), 1.0)):
        raise ValueError("Invalid sparse alpha")
    weighted_prediction = 0.0
    for agent, share in zip(agents, alpha):
        if share > 0:
            prediction = float(np.asarray(agent.api_predict(solution)).reshape(-1)[0])
            weighted_prediction += share * prediction
            if hasattr(server, "sparse_agent_calls"):
                server.sparse_agent_calls[int(agent.agent_id)] += 1
    return abs(weighted_prediction - server.target_T)


def _candidate_rows(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = [(int(row["seed"]), row["method"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate candidate keys")
    return dict(zip(keys, rows))


def explore(root, benchmark_json, seeds):
    root = Path(root)
    manifest = json.loads((root / "suite_manifest.json").read_text())
    benchmark = json.loads(Path(benchmark_json).read_text())
    if (manifest["status"] != "complete" or len(manifest["scenarios"]) != 1
            or not set(seeds) <= set(manifest["seeds"])
            or not set(seeds) <= set(benchmark["seeds"])
            or manifest["scenarios"][0] != benchmark["scenario"]):
        raise ValueError("Saved baseline and benchmark do not align")
    scenario = manifest["scenarios"][0]
    directory = root / scenario
    args = manifest["arguments"]
    target = float(args["targets"][0])
    shared = {int(row["seed"]): row for row in
              map(json.loads, (directory / "shared_costs.jsonl").read_text().splitlines())}
    candidates = _candidate_rows(directory / "candidate_rows.csv")
    controls = {int(row["seed"]): row for row in benchmark["rows"]}
    output = []
    for seed in seeds:
        item = shared[seed]
        old_b = candidates[seed, "B_stage0"]
        control = controls[seed]
        if (old_b["delivery_status"] != "valid" or
                control["candidates"]["B_stage0"]["status"] != "valid"):
            raise ValueError(f"Invalid B control: {seed}")
        artifact = directory / item["artifact_path"]
        payload = torch.load(artifact / "local_models.pt", map_location="cpu",
                             weights_only=True)
        agents = []
        for ident in item["coalition_ids"]:
            agent = AgentNode(ident, None, n_features=payload["n_features"])
            agent.model.load_state_dict(payload["agents"][ident])
            agent.model.eval()
            agents.append(agent)
        mse = json.loads(old_b["stage0_mse"])
        alpha, selected_indices = top_two_stage0_alpha(mse)
        server = HostServer(target, payload["n_features"],
                            constraint_mode=args["constraint_mode"],
                            feasible_lower=args["feasible_lower"],
                            feasible_upper=args["feasible_upper"])
        server.trusted_agents = agents
        server.I_list = [np.asarray(proposal, dtype=float) for proposal in item["proposals"]]
        server.alphas = alpha.tolist()
        server.sparse_agent_calls = {int(agent.agent_id): 0 for agent in agents}
        server._consensus_loss = MethodType(sparse_ensemble_objective, server)
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
            raise ValueError(f"Invalid sparse delivered X: {seed}")
        calls = server.sparse_agent_calls
        if (sum(calls.values()) != counter["requests"]["stage3"]
                or any(calls[int(agents[i].agent_id)] != 0
                       for i in range(len(agents)) if i not in selected_indices)):
            raise AssertionError(f"Skipped-agent query accounting failed: {seed}")
        evaluator, _ = load_evaluator_artifact(artifact)
        _, pooled = score_solution(evaluator, solution, target)
        b_control = control["candidates"]["B_stage0"]
        _, b_reloaded = score_solution(evaluator, b_control["solution"], target)
        if abs(b_reloaded-b_control["pooled_error"]) > 1e-7:
            raise ValueError(f"B control did not reload: {seed}")
        output.append({
            "scenario": scenario, "seed": seed, "status": "valid",
            "agent_ids": item["coalition_ids"], "stage0_mse": mse,
            "initial_alpha": alpha.tolist(),
            "active_agent_ids": [int(agents[i].agent_id) for i in selected_indices],
            "per_agent_stage3_requests": calls,
            "solution": solution.tolist(), "solver": result["actual_solver"],
            "max_box_violation": violation, "pooled_error": pooled,
            "truth_error_diagnostic": abs(true_function(solution)-target),
            "stage3_requests": counter["requests"]["stage3"],
            "stage3_input_samples": counter["input_samples"]["stage3"],
            "initialization_requests": 0,
            "extra_trusted_labels": 0,
            "B_single_pooled_error": b_control["pooled_error"],
            "B_single_stage3_requests": b_control["stage3_requests"],
            "new_five_pooled_error": control["new_pooled_error"],
            "new_five_stage3_requests": control["new_stage3_requests"],
        })
    versus_b = np.asarray([row["pooled_error"]-row["B_single_pooled_error"]
                           for row in output], dtype=float)
    versus_five = np.asarray([row["pooled_error"]-row["new_five_pooled_error"]
                              for row in output], dtype=float)
    return {
        "study": "sparse_top_two_initial_alpha_development",
        "scope": "reused_development_seeds_not_formal_quality_evidence",
        "scenario": scenario, "seeds": seeds,
        "method_definition": "Retain the two smallest Stage 0 MSE agents, renormalize their B weights, "
                             "keep all Stage 2 proposals, skip zero-weight model API calls",
        "mean_pooled_delta_vs_B_single": float(versus_b.mean()),
        "descriptive_95pct_ci_vs_B_single": paired_bootstrap_mean_ci(versus_b),
        "mean_pooled_delta_vs_new_five": float(versus_five.mean()),
        "descriptive_95pct_ci_vs_new_five": paired_bootstrap_mean_ci(versus_five),
        "mean_stage3_requests": float(np.mean([row["stage3_requests"] for row in output])),
        "mean_B_single_stage3_requests": float(np.mean([
            row["B_single_stage3_requests"] for row in output])),
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

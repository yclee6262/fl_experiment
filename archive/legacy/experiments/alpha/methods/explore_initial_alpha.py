"""Offline method development on archived models; pooled scores never set alpha.

This script may inspect pooled *outcomes* to select a formula for a future,
independent-seed study. It is not confirmatory evidence for that formula.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

from agent_client import AgentNode
from alpha_weight_experiments import run_stage3
from host_server import HostServer
from research_evaluation import load_evaluator_artifact, score_solution


def weights_from_blind(agents, x, y, target, mse_override=None, proposals=None):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    predictions = np.column_stack([a.api_predict(x) for a in agents])
    residual = predictions - y[:, None]
    mse = (np.asarray(mse_override, float) if mse_override is not None
           else np.mean(residual ** 2, axis=0))
    mae = np.mean(np.abs(residual), axis=0)
    def normalize(v):
        v = np.asarray(v, float)
        return v / v.sum()
    methods = {"B_stage0": normalize(1 / (mse + 1e-5))}
    for p in (1.25, 1.5, 2., 3.):
        methods[f"mse_power_{p:g}"] = normalize((mse + 1e-5) ** (-p))
    methods["mae_inverse_sq"] = normalize(1 / (mae + 1e-5) ** 2)
    methods["best_agent"] = np.eye(len(agents))[int(np.argmin(mse))]
    top_two = np.argsort(mse)[:min(2, len(agents))]
    top_two_scores = np.zeros(len(agents))
    top_two_scores[top_two] = 1 / (mse[top_two] + 1e-5)
    methods["stage0_top2"] = normalize(top_two_scores)
    second_moment = residual.T @ residual / len(y)
    base = methods["B_stage0"]
    for correlation_fraction in (.1, .25, .5, .75, 1.):
        covariance = ((1-correlation_fraction) * np.diag(mse+1e-5)
                      + correlation_fraction * second_moment)
        fit = minimize(lambda w: float(w @ covariance @ w), base, method="SLSQP",
                       bounds=[(0., 1.)]*len(agents),
                       constraints={"type":"eq", "fun":lambda w: float(w.sum()-1)},
                       options={"ftol":1e-13, "maxiter":500})
        if fit.success and np.all(np.isfinite(fit.x)):
            methods[f"corr_precision_{correlation_fraction:g}"] = normalize(np.maximum(fit.x,0))
    for label, bias in (("global", residual.mean(axis=0)),
                        ("target", (residual * np.exp(-np.abs(y-target)/.5)[:,None]).sum(axis=0)
                         / np.exp(-np.abs(y-target)/.5).sum())):
        scaled_bias = bias / (np.sqrt(np.mean(mse)) + 1e-5)
        for penalty in (1., 5.):
            fit = minimize(
                lambda w: float(np.sum((w-base)**2/(base+1e-5))
                                + penalty*(w @ scaled_bias)**2),
                base, method="SLSQP", bounds=[(0.,1.)]*len(agents),
                constraints={"type":"eq", "fun":lambda w: float(w.sum()-1)},
                options={"ftol":1e-13, "maxiter":500})
            if fit.success and np.all(np.isfinite(fit.x)):
                methods[f"bias_balance_{label}_{penalty:g}"] = normalize(np.maximum(fit.x,0))
    for bandwidth in (.25, .5, 1.):
        kernel = np.exp(-np.abs(y - target) / bandwidth)
        for pseudo in (5., 20.):
            local_mse = ((residual ** 2 * kernel[:, None]).sum(axis=0) + pseudo * mse) / (kernel.sum() + pseudo)
            methods[f"target_mse_bw{bandwidth:g}_prior{pseudo:g}"] = normalize(1 / (local_mse + 1e-5))
    if proposals is not None:
        proposals = np.asarray(proposals, float)
        center = proposals.mean(axis=0)
        for bandwidth in (.5, 1.):
            center_kernel = np.exp(-np.sum((x-center)**2, axis=1)/(2*bandwidth**2))
            hull_kernel = np.exp(-np.sum((x[:,None,:]-proposals[None,:,:])**2,
                                         axis=2)/(2*bandwidth**2)).mean(axis=1)
            for label, kernel in (("center", center_kernel), ("proposals", hull_kernel)):
                for pseudo in (5., 20.):
                    local_mse = ((residual**2 * kernel[:,None]).sum(axis=0)
                                 + pseudo*mse)/(kernel.sum()+pseudo)
                    methods[f"spatial_{label}_{bandwidth:g}_{pseudo:g}"] = normalize(1/(local_mse+1e-5))
        features = np.column_stack([np.ones(len(x)), x])
        coefficients = np.linalg.solve(features.T@features + np.diag([.01, 1., 1.]),
                                       features.T@residual)
        local_bias = np.r_[1.,center]@coefficients
        residual_variance = np.mean((residual-features@coefficients)**2,axis=0)
        methods["local_bias_risk"] = normalize(1/(local_bias**2 + residual_variance + 1e-5))
    return methods


def evaluate(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    scenario = root / "reference_clean"
    baseline_rows = {int(r["seed"]): r for r in csv.DictReader(
        (scenario / "summary.csv").open()) if r["method"] == "B_stage0"}
    for line in (scenario / "shared_costs.jsonl").read_text().splitlines():
        item = json.loads(line)
        if "artifact_path" not in item:
            continue
        artifact = scenario / item["artifact_path"]
        payload = torch.load(artifact / "local_models.pt", map_location="cpu", weights_only=True)
        agents = []
        for ident in item["coalition_ids"]:
            agent = AgentNode(int(ident), None, n_features=payload["n_features"])
            agent.model.load_state_dict(payload["agents"][ident])
            agent.model.eval()
            agents.append(agent)
        meta = json.loads((artifact / "metadata.json").read_text())
        target = float(meta["scenario"]["target"])
        evaluator = load_evaluator_artifact(artifact)[0]
        baseline = baseline_rows[int(item["seed"])]
        methods = weights_from_blind(agents, item["blind_test_X"], item["blind_test_y"], target,
                                     mse_override=json.loads(baseline["stage0_mse"]),
                                     proposals=item["proposals"])
        methods["B_stage0"] = np.asarray(json.loads(baseline["initial_alpha"]), float)
        for sample_count in (100, 200):
            expanded = HostServer(target, payload["n_features"],
                                  test_seed=int(item["seed"])+7919,
                                  n_test=sample_count, constraint_mode="box_constrained")
            expanded_methods = weights_from_blind(
                agents, expanded.test_X, expanded.test_y, target,
                proposals=item["proposals"])
            methods[f"stage0_mse_{sample_count}"] = expanded_methods["B_stage0"]
            methods[f"target_mse_{sample_count}"] = expanded_methods["target_mse_bw0.5_prior5"]
            methods[f"corr_precision_{sample_count}"] = expanded_methods["corr_precision_0.5"]
            for method in ("bias_balance_global_1", "bias_balance_global_5",
                           "bias_balance_target_1", "bias_balance_target_5"):
                methods[f"{method}_{sample_count}"] = expanded_methods[method]
            for method in ("spatial_center_0.5_5", "spatial_center_1_5",
                           "spatial_proposals_0.5_5", "spatial_proposals_1_5",
                           "local_bias_risk"):
                methods[f"{method}_{sample_count}"] = expanded_methods[method]
        for method, alpha in methods.items():
            server = HostServer(target, payload["n_features"], test_seed=int(item["seed"])+7919,
                                n_test=len(item["blind_test_y"]), constraint_mode="box_constrained",
                                feasible_lower=-1., feasible_upper=1.)
            server.trusted_agents = agents
            server.I_list = [np.asarray(v, float) for v in item["proposals"]]
            server.alphas = alpha.tolist()
            try:
                solution, _, _ = run_stage3(server, "best_of", 30)
                violation = server.box_violation(solution)
                _, error = score_solution(evaluator, solution, target) if violation <= 1e-8 else (None, None)
                lines.append({"seed": item["seed"], "method": method,
                              "alpha": json.dumps(alpha.tolist()),
                              "solution": json.dumps(solution.tolist()),
                              "pooled_error": error, "box_violation": violation,
                              "status": "valid" if error is not None else "invalid_output"})
            except (ValueError, FloatingPointError) as exc:
                lines.append({"seed": item["seed"], "method": method,
                              "alpha": json.dumps(alpha.tolist()), "status": "solver_failure",
                              "reason": str(exc)})
        print("completed seed", item["seed"], flush=True)
    fields = ("seed", "method", "alpha", "solution", "pooled_error", "box_violation", "status", "reason")
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("output")
    args = parser.parse_args()
    evaluate(args.root, args.output)

"""Development-only alpha-option selectors with smoother Stage 0 bias estimates.

Candidate alphas and their Stage 3 solutions are fixed. Every rule scores only
blind labels and local predictions; pooled outcomes are read after selection.
Do not treat selection over these exploratory rules as confirmatory evidence.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from agent_client import AgentNode
from research_analysis import paired_bootstrap_mean_ci
from validation_calibrated_alpha import CANDIDATE_METHODS


def _quadratic(x):
    x = np.asarray(x, float)
    features = [np.ones(len(x)), *[x[:, i] for i in range(x.shape[1])]]
    for i in range(x.shape[1]):
        for j in range(i, x.shape[1]):
            features.append(x[:, i] * x[:, j])
    return np.column_stack(features)


def _smooth_biases(blind_x, residual, candidate_x):
    x = np.asarray(blind_x, float)
    r = np.asarray(residual, float).reshape(-1)
    candidates = np.asarray(candidate_x, float)
    result = {}
    features = _quadratic(x)
    candidate_features = _quadratic(candidates)
    for penalty in (1.0, 5.0, 20.0):
        ridge = np.diag([0.01] + [1.0] * x.shape[1]
                        + [penalty] * (features.shape[1]-x.shape[1]-1))
        coefficients = np.linalg.solve(features.T@features + ridge,
                                       features.T@r)
        result[f"quadratic_ridge_{penalty:g}"] = candidate_features@coefficients
    global_bias = float(np.mean(r))
    for bandwidth in (0.5, 1.0):
        distances = np.sum((x[:, None, :]-candidates[None, :, :])**2, axis=2)
        weights = np.exp(-distances/(2*bandwidth**2))
        for pseudo in (2.0, 5.0, 20.0):
            estimate = (r@weights + pseudo*global_bias)/(weights.sum(axis=0)+pseudo)
            result[f"local_constant_bw{bandwidth:g}_prior{pseudo:g}"] = estimate
    pair_distance = np.sum((x[:, None, :]-x[None, :, :])**2, axis=2)
    candidate_distance = np.sum((candidates[:, None, :]-x[None, :, :])**2, axis=2)
    centered = r-global_bias
    for bandwidth in (0.5, 1.0, 2.0):
        kernel = np.exp(-pair_distance/(2*bandwidth**2))
        candidate_kernel = np.exp(-candidate_distance/(2*bandwidth**2))
        for regularization in (0.1, 1.0):
            coefficients = np.linalg.solve(
                kernel + regularization*np.eye(len(x)), centered)
            result[f"rbf_ridge_bw{bandwidth:g}_lambda{regularization:g}"] = (
                global_bias + candidate_kernel@coefficients)
    return result


def evaluate(roots, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for root_arg in roots:
        root = Path(root_arg)/"reference_clean"
        cohort = Path(root_arg).name
        candidate_rows = defaultdict(dict)
        for row in csv.DictReader((root/"candidate_rows.csv").open()):
            candidate_rows[int(row["seed"])][row["method"]] = row
        baseline_rows = {int(row["seed"]): row for row in
                         csv.DictReader((root/"summary.csv").open())
                         if row["method"] == "validation_calibrated_multistart"}
        for item in map(json.loads, (root/"shared_costs.jsonl").read_text().splitlines()):
            seed = int(item["seed"])
            if seed not in baseline_rows:
                continue
            payload = torch.load(root/item["artifact_path"]/"local_models.pt",
                                 map_location="cpu", weights_only=True)
            agents = []
            for agent_id in item["coalition_ids"]:
                agent = AgentNode(agent_id, None, n_features=payload["n_features"])
                agent.model.load_state_dict(payload["agents"][agent_id])
                agent.model.eval()
                agents.append(agent)
            x = np.asarray(item["blind_test_X"], float)
            y = np.asarray(item["blind_test_y"], float)
            b = np.asarray(json.loads(candidate_rows[seed]["B_stage0"]["initial_alpha"]), float)
            blind_predictions = np.column_stack([a.api_predict(x) for a in agents])
            residual = blind_predictions@b-y
            solutions = np.asarray([json.loads(candidate_rows[seed][m]["final_solution"])
                                    for m in CANDIDATE_METHODS], float)
            option_predictions = np.column_stack([a.api_predict(solutions) for a in agents])
            b_prediction = option_predictions@b
            old = baseline_rows[seed]
            old_error = float(old["external_global_error"])
            old_selected = json.loads(old["alpha_audit"])["selected_initialization"]
            methods = _smooth_biases(x, residual, solutions)
            for method, estimated_bias in methods.items():
                internal = np.abs(b_prediction-estimated_bias)
                winner_index = int(np.argmin(internal))
                winner = CANDIDATE_METHODS[winner_index]
                outcome = float(candidate_rows[seed][winner]["external_global_error"])
                rows.append({"cohort":cohort,"seed":seed,"rule":method,
                             "chosen":winner,"original_chosen":old_selected,
                             "pooled_error":outcome,"original_pooled":old_error,
                             "pooled_delta":outcome-old_error,
                             "internal_score":float(internal[winner_index])})
        print("completed",cohort,flush=True)
    fields = list(rows[0])
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["cohort"],row["rule"])].append(row["pooled_delta"])
    for (cohort,rule),values in sorted(grouped.items()):
        ci = paired_bootstrap_mean_ci(values)
        print(cohort,rule,len(values),float(np.mean(values)),ci,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    evaluate(args.roots, args.output)

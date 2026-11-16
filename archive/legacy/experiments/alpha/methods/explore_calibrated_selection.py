"""Development-only validation calibrators for selecting an initial alpha.

The archived pooled evaluator is used only to score the selected delivery,
never as a feature or selection criterion for an individual seed.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from agent_client import AgentNode


METHODS = ("A_uniform", "B_stage0", "mse_power_1.5",
           "corr_precision_0.5", "target_mse_bw0.5_prior5")


def score_functions(agents, blind_x, blind_y, b_alpha, solution):
    x = np.asarray(blind_x, float)
    y = np.asarray(blind_y, float)
    pred_blind = np.column_stack([a.api_predict(x) for a in agents])
    pred_solution = np.asarray([a.api_predict(solution)[0] for a in agents])
    aggregate_blind = pred_blind @ b_alpha
    aggregate_solution = float(pred_solution @ b_alpha)
    bias = aggregate_blind - y
    corrected = aggregate_solution - float(bias.mean())
    # Ridge regression of the ensemble's signed error on position.
    features = np.column_stack([np.ones(len(x)), x])
    coefficients = np.linalg.solve(features.T@features + np.diag([.01,1.,1.]),
                                   features.T@bias)
    spatial_corrected = aggregate_solution - float(np.r_[1.,solution]@coefficients)
    # Affine calibration shrunk toward the original aggregate prediction.
    design = np.column_stack([np.ones(len(y)), aggregate_blind])
    prior = np.array([0., 1.])
    affine = np.linalg.solve(design.T@design + np.diag([1.,1.]),
                             design.T@y + prior)
    affine_corrected = float(np.array([1.,aggregate_solution])@affine)
    each_bias = (pred_blind-y[:,None]).mean(axis=0)
    median_corrected = float(np.median(pred_solution-each_bias))
    return {"B_bias":abs(corrected), "B_spatial":abs(spatial_corrected),
            "B_affine":abs(affine_corrected), "median_bias":abs(median_corrected),
            "B_raw":abs(aggregate_solution)}


def evaluate(root, development_csv, output):
    root = Path(root) / "reference_clean"
    development = defaultdict(dict)
    for row in csv.DictReader(Path(development_csv).open()):
        if row["status"] == "valid":
            development[int(row["seed"])][row["method"]] = row
    original = defaultdict(dict)
    for row in csv.DictReader((root / "summary.csv").open()):
        original[int(row["seed"])][row["method"]] = row
    results = []
    for item in map(json.loads, (root / "shared_costs.jsonl").read_text().splitlines()):
        seed = item["seed"]
        artifact = root / item["artifact_path"]
        payload = torch.load(artifact / "local_models.pt", map_location="cpu", weights_only=True)
        agents = []
        for ident in item["coalition_ids"]:
            agent = AgentNode(ident, None, n_features=payload["n_features"])
            agent.model.load_state_dict(payload["agents"][ident])
            agent.model.eval()
            agents.append(agent)
        b_alpha = np.asarray(json.loads(original[seed]["B_stage0"]["initial_alpha"]), float)
        candidates = dict(development[seed])
        candidates["A_uniform"] = original[seed]["A_uniform"]
        b_error = float(original[seed]["B_stage0"]["external_global_error"])
        for method in METHODS:
            row = candidates[method]
            solution = np.asarray(json.loads(row["solution"] if "solution" in row
                                             else row["final_solution"]), float)
            scores = score_functions(agents, item["blind_test_X"],
                                     item["blind_test_y"], b_alpha, solution)
            pooled = float(row["pooled_error"] if "pooled_error" in row
                           else row["external_global_error"])
            for criterion, estimated_error in scores.items():
                results.append({"seed":seed, "method":method, "criterion":criterion,
                                "estimated_error":estimated_error,
                                "pooled_error":pooled, "B_error":b_error})
    with Path(output).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("development_csv")
    parser.add_argument("output")
    args = parser.parse_args()
    evaluate(args.root,args.development_csv,args.output)

"""Development-only risk-aware selection on archived five-candidate solutions.

All selection scores use only Stage 0 labels and local-model predictions.
Saved pooled scores are read only after a candidate has been selected to
evaluate potential rules before a new-seed confirmation experiment.
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


def _fit_bias(blind_x, blind_y, blind_pred, alpha, slope_penalty=1., kernel=None):
    x = np.asarray(blind_x, float)
    y = np.asarray(blind_y, float)
    design = np.column_stack([np.ones(len(x)), x])
    residual = blind_pred @ alpha - y
    weight = np.ones(len(y)) if kernel is None else np.asarray(kernel, float)
    penalty = np.diag([.01] + [slope_penalty] * x.shape[1])
    coef = np.linalg.solve(design.T @ (weight[:, None] * design) + penalty,
                           design.T @ (weight * residual))
    fitted = design @ coef
    estimated_noise = float(np.mean((residual - fitted)**2))
    return coef, estimated_noise


def _rules(blind_x, blind_y, blind_pred, b_alpha, candidate_x, candidate_pred,
           candidate_alpha, target):
    x = np.asarray(blind_x, float)
    y = np.asarray(blind_y, float)
    b = np.asarray(b_alpha, float)
    u = np.full(len(b), 1/len(b))
    rules = {}
    for weight_name, alpha in (("B", b), ("half_uniform", .5*b+.5*u),
                               ("uniform", u), ("own", candidate_alpha)):
        for slope in (.1, 1., 5.):
            coef, noise = _fit_bias(x, y, blind_pred, alpha, slope)
            predicted = float(candidate_pred @ alpha - np.r_[1.,candidate_x] @ coef)
            basic = abs(predicted-target)
            name = f"{weight_name}_ridge_{slope:g}"
            rules[name] = basic
            if weight_name == "B" and slope == 1.:
                corrected_agents = candidate_pred - np.mean(blind_pred-y[:,None], axis=0)
                disagreement = float(np.std(corrected_agents))
                distance = float(np.min(np.linalg.norm(x-candidate_x, axis=1)))
                for aversion in (.1, .25, .5, 1.):
                    rules[f"B_disagreement_{aversion:g}"] = basic + aversion*disagreement
                    rules[f"B_distance_{aversion:g}"] = basic + aversion*np.sqrt(noise)*distance
                    rules[f"B_risk_l2_{aversion:g}"] = float(np.hypot(basic,
                                                                        aversion*disagreement))
    for bandwidth in (.25, .5, 1.):
        local = np.exp(-np.sum((x-candidate_x)**2, axis=1)/(2*bandwidth**2))
        coef, _ = _fit_bias(x, y, blind_pred, b, 1., local)
        predicted = float(candidate_pred @ b - np.r_[1.,candidate_x] @ coef)
        rules[f"B_local_bias_{bandwidth:g}"] = abs(predicted-target)
    for bandwidth in (.25, .5, 1.):
        kernel = np.exp(-np.abs(y-target)/bandwidth)
        coef, _ = _fit_bias(x, y, blind_pred, b, 1., kernel)
        predicted = float(candidate_pred @ b - np.r_[1.,candidate_x] @ coef)
        rules[f"B_target_bias_{bandwidth:g}"] = abs(predicted-target)
    return rules


def evaluate(cohorts, output):
    results = []
    for label, root in cohorts:
        root = Path(root)/"reference_clean"
        candidates = defaultdict(dict)
        for row in csv.DictReader((root/"candidate_rows.csv").open()):
            candidates[int(row["seed"])][row["method"]] = row
        meta = {int(row["seed"]): row for row in
                csv.DictReader((root/"summary.csv").open())
                if row["method"] == "validation_calibrated_multistart"}
        for shared in map(json.loads, (root/"shared_costs.jsonl").read_text().splitlines()):
            seed = int(shared["seed"])
            if seed not in meta or not set(CANDIDATE_METHODS).issubset(candidates[seed]):
                continue
            payload = torch.load(root/shared["artifact_path"]/"local_models.pt",
                                 map_location="cpu", weights_only=True)
            agents = []
            for ident in shared["coalition_ids"]:
                agent = AgentNode(ident, None, n_features=payload["n_features"])
                agent.model.load_state_dict(payload["agents"][ident])
                agent.model.eval()
                agents.append(agent)
            blind_x = np.asarray(shared["blind_test_X"], float)
            blind_y = np.asarray(shared["blind_test_y"], float)
            blind_pred = np.column_stack([a.api_predict(blind_x) for a in agents])
            b_alpha = np.asarray(json.loads(candidates[seed]["B_stage0"]["initial_alpha"]), float)
            scores = defaultdict(dict)
            for method in CANDIDATE_METHODS:
                row = candidates[seed][method]
                if row["delivery_status"] != "valid":
                    continue
                solution = np.asarray(json.loads(row["final_solution"]), float)
                predictions = np.asarray([a.api_predict(solution)[0] for a in agents], float)
                own_alpha = np.asarray(json.loads(row["initial_alpha"]), float)
                each = _rules(blind_x, blind_y, blind_pred, b_alpha,
                              solution, predictions, own_alpha, 0.)
                for rule, value in each.items():
                    scores[rule][method] = value
            actual = float(meta[seed]["external_global_error"])
            for rule, by_method in scores.items():
                chosen = min(CANDIDATE_METHODS, key=lambda m: by_method.get(m, np.inf))
                selected = candidates[seed][chosen]
                results.append({"cohort":label,"seed":seed,"rule":rule,
                                "chosen":chosen,"internal_score":by_method[chosen],
                                "pooled_error":float(selected["external_global_error"]),
                                "current_error":actual})
    path=Path(output)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=results[0].keys())
        writer.writeheader(); writer.writerows(results)
    grouped=defaultdict(list)
    for row in results:
        grouped[(row["cohort"],row["rule"])].append(row["pooled_error"]-row["current_error"])
    for (cohort,rule),values in sorted(grouped.items()):
        ci=paired_bootstrap_mean_ci(values)
        print(cohort,rule,len(values),round(float(np.mean(values)),6),
              tuple(round(float(x),6) for x in ci))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--cohort",action="append",nargs=2,metavar=("NAME","ROOT"),required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    evaluate(args.cohort,args.output)

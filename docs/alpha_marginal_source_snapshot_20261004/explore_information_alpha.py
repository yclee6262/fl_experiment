"""Development-only test of economically interpretable initial-alpha shares.

The formulas and candidate selection use Stage 0 labels/local models only.
Frozen pooled evaluators are loaded solely after each delivered X is chosen.
All roots passed to this script are development cohorts, not confirmation data.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from agent_client import AgentNode
from alpha_weight_experiments import run_stage3
from host_server import HostServer
from initial_alpha_methods import initial_alpha
from research_evaluation import load_evaluator_artifact, score_solution
from validation_calibrated_alpha import (CANDIDATE_METHODS,
                                         fit_residual_calibrator, select_candidate)


BOOTSTRAPS = 512
BLEND = 0.5


def information_weights(predictions, labels, baseline_alpha, seed):
    """Return ex-ante shares; none of these inputs are external-evaluator scores."""
    predictions = np.asarray(predictions, dtype=float)
    labels = np.asarray(labels, dtype=float).reshape(-1)
    baseline = np.asarray(baseline_alpha, dtype=float).reshape(-1)
    if (predictions.ndim != 2 or predictions.shape != (len(labels), len(baseline))
            or len(labels) < 2 or not np.isfinite(predictions).all()
            or not np.isfinite(labels).all() or not np.isfinite(baseline).all()
            or np.any(baseline < 0) or not np.isclose(baseline.sum(), 1)):
        raise ValueError("Invalid Stage 0 predictions, labels, or baseline alpha")
    square_errors = (predictions - labels[:, None]) ** 2
    rng = np.random.default_rng(20261004 + int(seed))
    bootstrap_rows = rng.integers(0, len(labels), (BOOTSTRAPS, len(labels)))
    winners = np.argmin(square_errors[bootstrap_rows].mean(axis=1), axis=1)
    expertise = np.bincount(winners, minlength=len(baseline)) / BOOTSTRAPS
    bootstrap_share = BLEND * baseline + (1 - BLEND) * expertise

    full_loss = np.mean((predictions @ baseline - labels) ** 2)
    gains = np.zeros(len(baseline))
    for index in range(len(baseline)):
        remaining = baseline.copy()
        remaining[index] = 0
        if remaining.sum() <= 0:
            continue
        remaining /= remaining.sum()
        without_loss = np.mean((predictions @ remaining - labels) ** 2)
        gains[index] = without_loss - full_loss
    positive = np.maximum(gains, 0)
    if positive.sum() > 0:
        marginal_share = BLEND * baseline + (1 - BLEND) * positive / positive.sum()
    else:
        marginal_share = baseline.copy()
    top_two = np.argsort(square_errors.mean(axis=0))[:min(2, len(baseline))]
    top_two_share = np.zeros(len(baseline))
    top_two_share[top_two] = baseline[top_two]
    top_two_share /= top_two_share.sum()
    return {
        "bootstrap_expertise_50": (bootstrap_share, {"winner_frequency": expertise.tolist()}),
        "marginal_information_50": (marginal_share, {"blind_leave_one_out_gain": gains.tolist()}),
        "top_two_expertise": (top_two_share, {"top_two_indices": top_two.tolist()}),
    }


def _rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def evaluate(roots, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for root_arg in roots:
        root = Path(root_arg) / "reference_clean"
        cohort = Path(root_arg).name
        candidates = {(int(row["seed"]), row["method"]): row
                      for row in _rows(root / "candidate_rows.csv")}
        baselines = {int(row["seed"]): row for row in _rows(root / "summary.csv")
                     if row["method"] == "validation_calibrated_multistart"}
        for item in map(json.loads, (root / "shared_costs.jsonl").read_text().splitlines()):
            seed = int(item["seed"])
            if seed not in baselines or "artifact_path" not in item:
                continue
            artifact = root / item["artifact_path"]
            payload = torch.load(artifact / "local_models.pt", map_location="cpu",
                                 weights_only=True)
            agents = []
            for agent_id in item["coalition_ids"]:
                agent = AgentNode(agent_id, None, n_features=payload["n_features"])
                agent.model.load_state_dict(payload["agents"][agent_id])
                agent.model.eval()
                agents.append(agent)
            x = np.asarray(item["blind_test_X"], float)
            y = np.asarray(item["blind_test_y"], float)
            predictions = np.column_stack([a.api_predict(x) for a in agents])
            baseline_alpha = np.asarray(json.loads(
                candidates[(seed, "B_stage0")]["initial_alpha"]), float)
            weights = information_weights(predictions, y, baseline_alpha, seed)
            stage0_mse = json.loads(candidates[(seed, "B_stage0")]["stage0_mse"])
            for method in ("quality_shrink", "ensemble_shrink", "target_peer_shrink"):
                weights[method] = initial_alpha(
                    method, agents, stage0_mse, x, y, item["proposals"], 0.0)
            coefficients, _ = fit_residual_calibrator(agents, x, y, baseline_alpha)
            saved_solutions = {method: json.loads(candidates[(seed, method)]["final_solution"])
                               for method in CANDIDATE_METHODS}
            baseline_choice, _ = select_candidate(
                agents, baseline_alpha, coefficients, saved_solutions, 0.0)
            baseline_pooled = float(candidates[(seed, baseline_choice)]["external_global_error"])
            if abs(baseline_pooled-float(baselines[seed]["external_global_error"])) > 1e-7:
                raise AssertionError(f"Selector replay differs: {cohort}/{seed}")
            evaluator = load_evaluator_artifact(artifact)[0]
            for method, (alpha, audit) in weights.items():
                server = HostServer(0.0, payload["n_features"],
                                    test_seed=seed + 7919, n_test=len(y),
                                    constraint_mode="box_constrained",
                                    feasible_lower=-1.0, feasible_upper=1.0)
                server.trusted_agents = agents
                server.I_list = [np.asarray(v, float) for v in item["proposals"]]
                server.alphas = alpha.tolist()
                record = {"cohort": cohort, "seed": seed, "method": method,
                          "alpha": json.dumps(alpha.tolist()),
                          "audit": json.dumps(audit),
                          "baseline_pooled": baseline_pooled}
                try:
                    solution, _, _ = run_stage3(server, "best_of", 30)
                    violation = float(server.box_violation(solution))
                    if violation > 1e-8:
                        raise ValueError(f"box violation {violation}")
                    record["solution"] = json.dumps(np.asarray(solution, float).tolist())
                    record["candidate_pooled"] = float(score_solution(
                        evaluator, solution, 0.0)[1])
                    chosen, scores = select_candidate(
                        agents, baseline_alpha, coefficients,
                        {**saved_solutions, method: solution}, 0.0,
                        method_order=CANDIDATE_METHODS + (method,))
                    record["selected_extra"] = int(chosen == method)
                    record["new_pooled"] = (record["candidate_pooled"]
                                             if chosen == method else baseline_pooled)
                    record["pooled_delta"] = record["new_pooled"] - baseline_pooled
                    record["candidate_internal_score"] = scores[method]["internal_target_error"]
                    record["status"] = "valid"
                except (ValueError, FloatingPointError) as exc:
                    record["status"] = "solver_failure"
                    record["reason"] = str(exc)
                records.append(record)
            print(f"completed {cohort} seed {seed}", flush=True)
    fields = sorted({key for row in records for key in row})
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    for cohort in sorted({r["cohort"] for r in records}):
        for method in sorted({r["method"] for r in records}):
            rows = [r for r in records if r["cohort"] == cohort and r["method"] == method]
            valid = [r for r in rows if r["status"] == "valid"]
            print(cohort, method, len(valid), "/", len(rows),
                  "mean_delta", float(np.mean([r["pooled_delta"] for r in valid]))
                  if valid else None,
                  "selected", sum(r.get("selected_extra", 0) for r in valid), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    evaluate(args.roots, args.output)

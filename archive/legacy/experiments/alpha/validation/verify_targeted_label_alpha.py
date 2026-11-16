"""Independently reload proposal-local labels, alpha, X, and pooled scores."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from agent_client import AgentNode
from explore_targeted_label_alpha import (LABELS_PER_PROPOSAL,
                                          proposal_local_alpha,
                                          proposal_local_points)
from research_evaluation import load_evaluator_artifact, score_solution
from run_experiments import true_function


def verify(root, result_path):
    root, result_path = Path(root), Path(result_path)
    saved = json.loads(result_path.read_text())
    scenario = saved["scenario"]
    manifest = json.loads((root / "suite_manifest.json").read_text())
    if (manifest["status"] != "complete" or manifest["scenarios"] != [scenario]
            or saved["seeds"] != [row["seed"] for row in saved["rows"]]
            or len(saved["seeds"]) != len(set(saved["seeds"]))
            or not set(saved["seeds"]) <= set(manifest["seeds"])):
        raise ValueError("Saved-study identity or seed coverage mismatch")
    directory = root / scenario
    shared = {int(row["seed"]): row for row in map(
        json.loads, (directory / "shared_costs.jsonl").read_text().splitlines())}
    with (directory / "candidate_rows.csv").open(newline="") as handle:
        stage0_mse = {int(candidate["seed"]): json.loads(candidate["stage0_mse"])
                      for candidate in csv.DictReader(handle)
                      if candidate["method"] == "B_stage0"}
    target = float(manifest["arguments"]["targets"][0])
    lower = float(manifest["arguments"]["feasible_lower"])
    upper = float(manifest["arguments"]["feasible_upper"])
    max_points = max_labels = max_predictions = max_alpha = max_pooled = 0.0
    problems = []
    for row in saved["rows"]:
        seed = row["seed"]
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
        points = proposal_local_points(item["proposals"], seed, lower, upper)
        labels = np.asarray([true_function(point) for point in points], dtype=float)
        predictions = np.column_stack([
            np.asarray(agent.api_predict(points), dtype=float).reshape(-1)
            for agent in agents])
        alpha, audit = proposal_local_alpha(
            stage0_mse[seed], predictions, labels)
        x = np.asarray(row["solution"], dtype=float)
        if (row["status"] != "valid" or row["agent_ids"] != item["coalition_ids"]
                or x.shape != (payload["n_features"],) or not np.isfinite(x).all()
                or np.any(x < lower-1e-8) or np.any(x > upper+1e-8)
                or row["trusted_label_count"] != LABELS_PER_PROPOSAL*len(agents)
                or row["initialization_requests"] != len(agents)
                or row["initialization_input_samples"] != len(points)*len(agents)
                or row["stage3_requests"] <= 0):
            problems.append(f"status_box_or_cost_mismatch:{seed}")
            continue
        max_points = max(max_points, float(np.max(np.abs(points-row["proposal_local_points"]))))
        max_labels = max(max_labels, float(np.max(np.abs(labels-row["proposal_local_labels"]))))
        max_predictions = max(max_predictions, float(np.max(np.abs(
            predictions-row["proposal_local_predictions"]))))
        max_alpha = max(max_alpha, float(np.max(np.abs(alpha-row["initial_alpha"]))))
        if any(not np.allclose(audit[key], row["alpha_audit"][key], atol=1e-12)
               for key in ("stage0_mse", "proposal_local_mse", "combined_mse")):
            problems.append(f"alpha_audit_mismatch:{seed}")
        evaluator, _ = load_evaluator_artifact(artifact)
        _, pooled = score_solution(evaluator, x, target)
        max_pooled = max(max_pooled, abs(pooled-row["pooled_error"]))
    if (max_points > 1e-12 or max_labels > 1e-12 or max_predictions > 1e-12
            or max_alpha > 1e-12 or max_pooled > 1e-7):
        problems.append("numeric_reload_tolerance_exceeded")
    return {"passed": not problems, "scenario": scenario, "seeds": len(saved["rows"]),
            "max_points_delta": max_points, "max_labels_delta": max_labels,
            "max_predictions_delta": max_predictions, "max_alpha_delta": max_alpha,
            "max_pooled_score_delta": max_pooled, "problems": problems}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_root")
    parser.add_argument("result_json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = verify(args.baseline_root, args.result_json)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

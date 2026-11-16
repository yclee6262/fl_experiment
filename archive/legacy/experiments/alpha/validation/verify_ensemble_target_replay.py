"""Reload a saved-model ensemble-target pilot and independently verify its X scores."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from agent_client import AgentNode
from research_evaluation import load_evaluator_artifact, score_solution
from run_experiments import true_function
from validation_calibrated_alpha import (CANDIDATE_METHODS, fit_residual_calibrator,
                                         select_candidate)


def verify(root, replay):
    root, replay = Path(root), Path(replay)
    manifest = json.loads((root / "suite_manifest.json").read_text())
    payload = json.loads(replay.read_text())
    scenario = payload["scenario"]
    directory = root / scenario
    target = float(manifest["arguments"]["targets"][0])
    lower = float(manifest["arguments"]["feasible_lower"])
    upper = float(manifest["arguments"]["feasible_upper"])
    shared = {int(item["seed"]): item for item in map(
        json.loads, (directory / "shared_costs.jsonl").read_text().splitlines())}
    with (directory / "candidate_rows.csv").open(newline="") as handle:
        candidates = {(int(row["seed"]), row["method"]): row
                      for row in csv.DictReader(handle)}
    with (directory / "summary.csv").open(newline="") as handle:
        summaries = {(int(row["seed"]), row["method"]): row
                     for row in csv.DictReader(handle)}
    rows = payload["rows"]
    problems = []
    if (manifest["status"] != "complete" or manifest["scenarios"] != [scenario]
            or not rows or len({row["seed"] for row in rows}) != len(rows)
            or set(row["seed"] for row in rows) != set(payload["seeds"])
            or not set(payload["seeds"]) <= set(manifest["seeds"])):
        problems.append("manifest_or_seed_coverage_mismatch")
    maximum_pooled_difference = maximum_alpha_difference = 0.0
    maximum_internal_difference = 0.0
    verified_candidates = verified_selections = 0
    for row in rows:
        seed = row["seed"]
        if row.get("delivery_status") != "valid" or row.get("candidate_failures") != 0:
            problems.append(f"invalid_delivery:{seed}")
            continue
        item = shared[seed]
        artifact = directory / item["artifact_path"]
        model_payload = torch.load(artifact / "local_models.pt", map_location="cpu",
                                   weights_only=True)
        evaluator, _ = load_evaluator_artifact(artifact)
        agents = []
        for ident in item["coalition_ids"]:
            agent = AgentNode(ident, None, n_features=model_payload["n_features"])
            agent.model.load_state_dict(model_payload["agents"][ident])
            agent.model.eval()
            agents.append(agent)
        solutions = {}
        if set(row["candidates"]) != set(CANDIDATE_METHODS):
            problems.append(f"candidate_set_mismatch:{seed}")
            continue
        for method in CANDIDATE_METHODS:
            candidate = row["candidates"][method]
            if candidate["status"] != "valid":
                problems.append(f"candidate_failure:{seed}/{method}")
                continue
            solution = np.asarray(candidate["solution"], dtype=float)
            alpha = np.asarray(candidate["alpha"], dtype=float)
            saved_alpha = np.asarray(json.loads(
                candidates[seed, method]["initial_alpha"]), dtype=float)
            if alpha.shape != saved_alpha.shape or solution.shape != (model_payload["n_features"],):
                problems.append(f"candidate_shape_mismatch:{seed}/{method}")
                continue
            if (not np.isfinite(alpha).all() or np.any(alpha < 0)
                    or not np.isclose(alpha.sum(), 1.0)
                    or not np.isfinite(solution).all()):
                problems.append(f"nonfinite_or_invalid_candidate:{seed}/{method}")
                continue
            maximum_alpha_difference = max(maximum_alpha_difference,
                                           float(np.max(np.abs(alpha - saved_alpha))))
            if not (np.isfinite(solution).all() and
                    np.all(solution >= lower - 1e-8) and
                    np.all(solution <= upper + 1e-8)):
                problems.append(f"infeasible_candidate:{seed}/{method}")
            _, pooled = score_solution(evaluator, solution, target)
            if (not np.isfinite(candidate["pooled_error"])
                    or not np.isfinite(candidate["truth_error"])):
                problems.append(f"nonfinite_saved_score:{seed}/{method}")
                continue
            maximum_pooled_difference = max(maximum_pooled_difference,
                                            abs(pooled - candidate["pooled_error"]))
            if abs(abs(true_function(solution) - target) - candidate["truth_error"]) > 1e-7:
                problems.append(f"truth_score_mismatch:{seed}/{method}")
            solutions[method] = solution
            verified_candidates += 1
        if len(solutions) != len(CANDIDATE_METHODS):
            continue
        baseline_alpha = json.loads(candidates[seed, "B_stage0"]["initial_alpha"])
        coefficients, _ = fit_residual_calibrator(
            agents, item["blind_test_X"], item["blind_test_y"], baseline_alpha)
        selected, scores = select_candidate(
            agents, baseline_alpha, coefficients, solutions, target,
            lower, upper, method_order=CANDIDATE_METHODS)
        if selected != row["new_selected_initialization"]:
            problems.append(f"selection_mismatch:{seed}")
        if (not np.isfinite(row["new_pooled_error"]) or
                abs(row["new_pooled_error"] - row["candidates"][selected]["pooled_error"]) > 1e-7):
            problems.append(f"delivered_score_mismatch:{seed}")
        for method in CANDIDATE_METHODS:
            maximum_internal_difference = max(maximum_internal_difference,
                abs(scores[method]["internal_target_error"] -
                    row["internal_scores"][method]["internal_target_error"]))
        old = summaries[seed, "validation_calibrated_multistart"]
        old_solution = json.loads(old["final_solution"])
        _, old_pooled = score_solution(evaluator, old_solution, target)
        if abs(old_pooled - row["old_pooled_error"]) > 1e-7:
            problems.append(f"old_pooled_score_mismatch:{seed}")
        verified_selections += 1
    if (maximum_pooled_difference > 1e-7 or maximum_alpha_difference > 1e-12
            or maximum_internal_difference > 1e-12):
        problems.append("numeric_reload_tolerance_exceeded")
    return {"passed": not problems, "scenario": scenario, "seeds": len(rows),
            "verified_candidates": verified_candidates,
            "verified_selections": verified_selections,
            "max_pooled_score_reload_delta": maximum_pooled_difference,
            "max_initial_alpha_delta": maximum_alpha_difference,
            "max_internal_score_replay_delta": maximum_internal_difference,
            "problems": problems}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_root")
    parser.add_argument("replay_json")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = verify(args.baseline_root, args.replay_json)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()

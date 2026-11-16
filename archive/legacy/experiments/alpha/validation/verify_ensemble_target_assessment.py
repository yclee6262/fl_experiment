"""Independently reload and verify delivered-X scores and selector replay."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from agent_client import AgentNode
from research_analysis import paired_bootstrap_mean_ci
from research_evaluation import load_evaluator_artifact, score_solution
from run_experiments import true_function
from validation_calibrated_alpha import (CANDIDATE_METHODS, fit_residual_calibrator,
                                         select_candidate)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root, assessment, preregistration):
    root, assessment, preregistration = map(Path, (root, assessment, preregistration))
    prereg = json.loads(preregistration.read_text())
    manifest = json.loads((assessment / "assessment_manifest.json").read_text())
    analysis = json.loads((assessment / "analysis.json").read_text())
    problems = []
    if manifest["status"] != "complete" or analysis["status"] != "complete":
        problems.append("assessment_not_complete")
    if _sha(preregistration) != manifest["preregistration_sha256"]:
        problems.append("preregistration_hash_mismatch")
    for name, expected in prereg["source_sha256"].items():
        snapshot = assessment / "source_snapshot" / name
        if not snapshot.exists() or _sha(snapshot) != expected:
            problems.append(f"source_snapshot_mismatch:{name}")
    directory = root / prereg["scenario"]
    shared = {int(r["seed"]): r for r in
              map(json.loads, (directory / "shared_costs.jsonl").read_text().splitlines())}
    with (directory / "candidate_rows.csv").open(newline="") as handle:
        old_candidates = {(int(r["seed"]), r["method"]): r
                          for r in csv.DictReader(handle)}
    rows = [json.loads(line) for line in
            (assessment / "seed_rows.jsonl").read_text().splitlines()]
    seeds = [int(r["seed"]) for r in rows]
    if seeds != prereg["seeds"] or len(set(seeds)) != len(seeds):
        problems.append("seed_coverage_or_order_mismatch")
    max_score_delta = max_alpha_delta = max_internal_delta = 0.0
    verified_candidates = verified_selections = 0
    for row in rows:
        seed = int(row["seed"])
        if row.get("status") != "valid":
            continue
        item = shared[seed]
        artifact = directory / item["artifact_path"]
        evaluator, _ = load_evaluator_artifact(artifact)
        payload = torch.load(artifact / "local_models.pt", map_location="cpu",
                             weights_only=True)
        agents = []
        for ident in item["coalition_ids"]:
            agent = AgentNode(ident, None, n_features=payload["n_features"])
            agent.model.load_state_dict(payload["agents"][ident])
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
                old_candidates[seed, method]["initial_alpha"]), dtype=float)
            max_alpha_delta = max(max_alpha_delta,
                                  float(np.max(np.abs(alpha-saved_alpha))))
            if not (np.isfinite(solution).all() and
                    np.all(solution >= -1.0-1e-8) and
                    np.all(solution <= 1.0+1e-8)):
                problems.append(f"infeasible_solution:{seed}/{method}")
            _, error = score_solution(evaluator, solution, 0.0)
            max_score_delta = max(max_score_delta,
                                  abs(error-candidate["pooled_error"]))
            if abs(abs(true_function(solution))-candidate["truth_error"]) > 1e-7:
                problems.append(f"truth_mismatch:{seed}/{method}")
            solutions[method] = solution
            verified_candidates += 1
        if len(solutions) != len(CANDIDATE_METHODS):
            continue
        baseline_alpha = json.loads(old_candidates[seed, "B_stage0"]["initial_alpha"])
        coefficients, _ = fit_residual_calibrator(
            agents, item["blind_test_X"], item["blind_test_y"], baseline_alpha)
        selected, scores = select_candidate(
            agents, baseline_alpha, coefficients, solutions, 0.0,
            method_order=CANDIDATE_METHODS)
        if selected != row["new_selected_initialization"]:
            problems.append(f"selection_mismatch:{seed}")
        for method in CANDIDATE_METHODS:
            max_internal_delta = max(max_internal_delta,
                abs(scores[method]["internal_target_error"]-
                    row["internal_scores"][method]["internal_target_error"]))
        if abs(row["new_pooled_error"]-
               row["candidates"][selected]["pooled_error"]) > 1e-7:
            problems.append(f"delivered_score_mismatch:{seed}")
        verified_selections += 1
    valid = [r for r in rows if r.get("status") == "valid"]
    delta = np.asarray([r["new_pooled_error"]-r["old_pooled_error"]
                        for r in valid])
    cost = np.asarray([r["new_stage3_requests"]-r["old_stage3_requests"]
                       for r in valid])
    if valid:
        if abs(float(delta.mean())-analysis["mean_pooled_delta_new_minus_old"]) > 1e-12:
            problems.append("pooled_mean_mismatch")
        if not np.allclose(paired_bootstrap_mean_ci(delta),
                           analysis["paired_bootstrap_95pct_ci_pooled"], atol=1e-12):
            problems.append("pooled_ci_mismatch")
        if abs(float(cost.mean())-
               analysis["mean_stage3_request_delta_new_minus_old"]) > 1e-12:
            problems.append("cost_mean_mismatch")
    if max_score_delta > 1e-7 or max_alpha_delta > 1e-12 or max_internal_delta > 1e-12:
        problems.append("numeric_reload_tolerance_exceeded")
    report = {"passed": not problems, "expected_seeds": len(prereg["seeds"]),
              "observed_seeds": len(rows),
              "verified_candidates": verified_candidates,
              "verified_selections": verified_selections,
              "max_pooled_score_reload_delta": max_score_delta,
              "max_initial_alpha_delta": max_alpha_delta,
              "max_internal_score_replay_delta": max_internal_delta,
              "problems": problems}
    (assessment / "independent_verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("assessment")
    parser.add_argument("preregistration")
    args = parser.parse_args()
    report = verify(args.root, args.assessment, args.preregistration)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()

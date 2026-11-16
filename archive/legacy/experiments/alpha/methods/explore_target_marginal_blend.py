"""Exploratory five-solve portfolio with a blended Target-local/marginal alpha.

Uses saved local models, proposals, and pooled evaluators. The blended alpha
and delivered X are chosen using Stage 0 information only; pooled scores are
read after delivery. All input cohorts were previously used and are not an
independent confirmation set.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from agent_client import AgentNode
from alpha_weight_experiments import attach_query_counter, run_stage3
from host_server import HostServer
from initial_alpha_methods import initial_alpha
from research_evaluation import load_evaluator_artifact, score_solution
from validation_calibrated_alpha import (
    CANDIDATE_METHODS, fit_residual_calibrator, select_candidate,
)


HYBRID = "target_marginal_blend_50"
HYBRID_ORDER = tuple(m for m in CANDIDATE_METHODS
                     if m != "target_local_precision") + (HYBRID,)


def _rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def explore(root_arg, seeds):
    root = Path(root_arg)
    manifest = json.loads((root / "suite_manifest.json").read_text())
    scenario, = manifest["scenarios"]
    if manifest["status"] != "complete" or not set(seeds).issubset(manifest["seeds"]):
        raise ValueError(f"Incomplete root or unknown seeds: {root}")
    folder = root / scenario
    candidate_rows = _rows(folder / "candidate_rows.csv")
    candidates = {(int(r["seed"]), r["method"]): r for r in candidate_rows}
    summary_rows = _rows(folder / "summary.csv")
    baselines = {int(r["seed"]): r for r in summary_rows
                 if r["method"] == "validation_calibrated_multistart"}
    shared = {r["seed"]: r for r in map(
        json.loads, (folder / "shared_costs.jsonl").read_text().splitlines())}
    records = []
    for seed in seeds:
        item = shared[seed]
        baseline = baselines[seed]
        artifact = folder / item["artifact_path"]
        payload = torch.load(artifact / "local_models.pt", map_location="cpu",
                             weights_only=True)
        agents = []
        for agent_id in item["coalition_ids"]:
            agent = AgentNode(agent_id, None, n_features=payload["n_features"])
            agent.model.load_state_dict(payload["agents"][agent_id])
            agent.model.eval()
            agents.append(agent)
        x = np.asarray(item["blind_test_X"], dtype=float)
        y = np.asarray(item["blind_test_y"], dtype=float)
        b = np.asarray(json.loads(candidates[seed, "B_stage0"]["initial_alpha"]),
                       dtype=float)
        mse = json.loads(candidates[seed, "B_stage0"]["stage0_mse"])
        counter = attach_query_counter(agents)
        counter["stage"] = "initialization"
        hybrid, audit = initial_alpha(HYBRID, agents, mse, x, y,
                                      item["proposals"],
                                      float(manifest["arguments"]["inverse_target"]))
        target = np.asarray(json.loads(candidates[seed, "target_local_precision"][
            "initial_alpha"]), dtype=float)
        marginal = np.asarray(json.loads(candidates[seed, "marginal_information_50"][
            "initial_alpha"]), dtype=float)
        if not np.allclose(hybrid, .5*target+.5*marginal, atol=1e-10, rtol=0):
            raise AssertionError(f"Alpha components differ: {scenario}/{seed}")
        server = HostServer(float(manifest["arguments"]["inverse_target"]),
                            payload["n_features"], test_seed=seed+7919,
                            n_test=len(y), constraint_mode="box_constrained")
        if not np.array_equal(server.test_X, x) or not np.array_equal(server.test_y, y):
            raise AssertionError(f"Stage 0 replay differs: {scenario}/{seed}")
        server.trusted_agents = agents
        server.I_list = [np.asarray(v, dtype=float) for v in item["proposals"]]
        server.alphas = hybrid.tolist()
        row = {"cohort": root.name, "scenario": scenario, "seed": seed,
               "method": HYBRID, "alpha": json.dumps(hybrid.tolist()),
               "alpha_audit": json.dumps(audit), "status": "running"}
        try:
            counter["stage"] = "stage3"
            solution, _, _ = run_stage3(server, "best_of", 30)
            violation = float(server.box_violation(solution))
            if not np.isfinite(solution).all() or violation > 1e-8:
                row.update(status="invalid_output", violation=violation)
                records.append(row)
                continue
            row["solution"] = json.dumps(solution.tolist())
            row["violation"] = violation
            row["actual_solver"] = server.last_subspace_optimization.get(
                "actual_solver", server.last_subspace_optimization.get("chosen_engine", ""))
            counter["stage"] = "selector"
            coefficients, _ = fit_residual_calibrator(agents, x, y, b)
            solutions = {method: json.loads(candidates[seed, method]["final_solution"])
                         for method in HYBRID_ORDER if method != HYBRID}
            solutions[HYBRID] = solution
            chosen, scores = select_candidate(
                agents, b, coefficients, solutions,
                float(manifest["arguments"]["inverse_target"]),
                method_order=HYBRID_ORDER)
            old_audit = json.loads(baseline["alpha_audit"])
            old_order = tuple(old_audit["candidate_method_order"])
            old_choice, old_scores = select_candidate(
                agents, b, coefficients,
                {method: json.loads(candidates[seed, method]["final_solution"])
                 for method in old_order},
                float(manifest["arguments"]["inverse_target"]),
                method_order=old_order)
            if old_choice != old_audit["selected_initialization"]:
                raise AssertionError(f"Old selector differs: {scenario}/{seed}")
            for method in old_order:
                old_value = old_audit["internal_scores"][method]["internal_target_error"]
                if abs(old_scores[method]["internal_target_error"]-old_value) > 1e-7:
                    raise AssertionError(f"Internal score differs: {scenario}/{seed}/{method}")
            evaluator = load_evaluator_artifact(artifact)[0]
            hybrid_pooled = float(score_solution(evaluator, solution,
                                float(manifest["arguments"]["inverse_target"]))[1])
            delivered_pooled = (hybrid_pooled if chosen == HYBRID else
                                float(candidates[seed, chosen]["external_global_error"]))
            old_pooled = float(baseline["external_global_error"])
            row.update(status="valid", chosen=chosen,
                       internal_scores=json.dumps(scores),
                       hybrid_pooled=hybrid_pooled,
                       delivered_pooled=delivered_pooled,
                       old_pooled=old_pooled,
                       pooled_delta=delivered_pooled-old_pooled,
                       hybrid_initialization_requests=counter["requests"]["initialization"],
                       hybrid_stage3_requests=counter["requests"]["stage3"],
                       selector_requests=counter["requests"]["selector"])
        except (ValueError, FloatingPointError) as exc:
            row.update(status="solver_failure", reason=str(exc))
        finally:
            counter["stage"] = "finished"
        records.append(row)
        print("completed", scenario, seed, row["status"], flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    seeds = [int(v) for v in args.seeds.split(",")]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate seeds")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for root in args.roots:
        rows.extend(explore(root, seeds))
    fields = sorted({key for row in rows for key in row})
    with (output / "seed_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for cohort in sorted({r["cohort"] for r in rows}):
        group = [r for r in rows if r["cohort"] == cohort]
        valid = [r for r in group if r["status"] == "valid"]
        deltas = [r["pooled_delta"] for r in valid]
        summaries.append({
            "cohort": cohort, "scenario": group[0]["scenario"],
            "evidence_status": "exploratory_reused_seeds",
            "valid": len(valid), "total": len(group),
            "mean_pooled_delta": float(np.mean(deltas)) if valid else None,
            "wins_ties_losses": [sum(d < 0 for d in deltas),
                                 sum(d == 0 for d in deltas),
                                 sum(d > 0 for d in deltas)],
            "hybrid_selected": sum(r["chosen"] == HYBRID for r in valid),
            "mean_extra_candidate_requests": float(np.mean([
                r["hybrid_initialization_requests"]+r["hybrid_stage3_requests"]
                for r in valid])) if valid else None,
        })
    (output / "summary.json").write_text(json.dumps(
        summaries, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

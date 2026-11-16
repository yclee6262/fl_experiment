"""Exhaustively measure the P6 recruitment proxy on every 1-3 person coalition."""

import argparse
import csv
import hashlib
import itertools
import json
import time
from pathlib import Path


def _source_workspace(source_file):
    """Find the flat workspace or a standalone historical source snapshot."""
    start = Path(source_file).resolve().parent
    for candidate in (start, *start.parents):
        if all((candidate / name).is_file() for name in
               ("host_server.py", "run_experiments.py", "dataset.py")):
            return candidate
        if (candidate / "project_support/legacy.py").is_file():
            from project_support.legacy import prepare
            return prepare()
    raise FileNotFoundError(f"Cannot locate source workspace from {source_file}")


import numpy as np
from scipy.stats import spearmanr
import torch

from alpha_weight_experiments import attach_query_counter, build_base_server, build_parser as alpha_parser, box_feasible, run_stage3
from p6_recruitment_experiment import provenance, qualified_candidates
from recruitment_selection import coalition_value, feasible, selection_report
from research_evaluation import save_scenario_artifacts, score_solution
from run_experiments import true_function


METHOD = "objective_exact"
METHODS = ("exact", "objective_greedy", "legacy_greedy")
BUDGETS = (2.5, 3.5, 8.0)
ETAS = (0.0, 0.5)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(output_dir, seeds=(0, 1, 2, 3, 4)):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError(f"Enumeration directory must be empty: {root}")
    snapshot = root / "source_snapshot"
    snapshot.mkdir()
    for source in _source_workspace(__file__).glob("*.py"):
        (snapshot / source.name).write_bytes(source.read_bytes())
    manifest = {**provenance(),
        "schema_version": 1, "status": "running", "seeds": list(seeds),
        "budgets": list(BUDGETS), "etas": list(ETAS), "max_coalition_size": 3,
        "candidate_pool_expected": 10, "max_coalitions_per_seed": 175,
        "selection_objective": "sum(Stage0 quality_share) + eta * normalized pairwise residual cosine dissimilarity",
        "quality_reference": "frozen pooled-data global evaluator; never fed back into selection",
        "truth_score_role": "diagnostic only", "payout_status": "not_run_by_configuration",
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in _source_workspace(__file__).glob("*.py")},
    }
    manifest_path = root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    coalition_rows, condition_rows, candidate_rows, preparation_rows = [], [], [], []
    try:
        for seed in seeds:
            args = alpha_parser().parse_args([])
            args.num_agents = 10
            args.constraint_mode = "box_constrained"
            args.settle_payment = False
            started = time.perf_counter()
            server = build_base_server(args, seed, 2, 0.0, 0.5)
            counter = server.preparation_counter
            counter["stage"] = "selection_precompute"
            precompute_started=time.perf_counter()
            candidates = qualified_candidates(server)
            residual_precompute_seconds=time.perf_counter()-precompute_started
            counter["stage"] = "proposal"
            if len(candidates) != 10:
                manifest.setdefault("candidate_pool_warnings", []).append(
                    {"seed": seed, "observed": len(candidates), "expected": 10})
            evaluator_started=time.perf_counter()
            evaluator, artifact, evaluator_metrics = save_scenario_artifacts(
                root, {"seed": seed, "dimension": 2, "poison_ratio": 0.0, "target": 0.5},
                server.all_agents, server.study_records, evaluator_epochs=args.external_evaluator_epochs)
            evaluator_seconds=time.perf_counter()-evaluator_started
            proposals = {}
            local_proposal_seconds=0.0
            for candidate in candidates:
                proposal_started=time.perf_counter()
                torch.manual_seed(seed * 100000 + candidate["agent_id"])
                proposals[candidate["agent_id"]] = candidate["agent"].infer_parameters_D(
                    0.5, steps=args.inverse_steps)
                elapsed=time.perf_counter()-proposal_started
                local_proposal_seconds+=elapsed
                candidate_rows.append({"seed":seed,"agent_id":candidate["agent_id"],
                    "mse":candidate["mse"],"bid":candidate["bid"],
                    "quality_share":candidate["quality_share"],
                    "legacy_selected":candidate["legacy_selected"],
                    "residual":json.dumps(candidate["residual"].tolist()),
                    "formal_target":0.5,"qualification_target":0.0,
                    "proposal":json.dumps(proposals[candidate["agent_id"]].tolist()),
                    "local_proposal_seconds":elapsed})
            write_csv(root/"candidate_pool.csv",candidate_rows)
            preparation_rows.append({"seed":seed,**server.preparation_costs,
                "residual_precompute_seconds":residual_precompute_seconds,
                "local_formal_proposal_seconds":local_proposal_seconds,
                "external_evaluator_seconds":evaluator_seconds})
            write_csv(root/"preparation_costs.csv",preparation_rows)
            coalition_started = time.perf_counter()
            for size in range(1, min(3, len(candidates)) + 1):
                for indices in itertools.combinations(range(len(candidates)), size):
                    chosen = [candidates[index] for index in indices]
                    optimizer = type(server)(
                        target_T=0.5, n_features=2, total_budget=args.total_budget,
                        test_seed=seed + 7919, n_test=args.n_test,
                        constraint_mode="box_constrained", feasible_lower=-1.0,
                        feasible_upper=1.0, feasibility_tolerance=1e-8)
                    optimizer.trusted_agents = [item["agent"] for item in chosen]
                    optimizer.I_list = [proposals[item["agent_id"]] for item in chosen]
                    optimizer.alphas = (np.ones(size) / size).tolist()
                    solve_counter = attach_query_counter(optimizer.trusted_agents)
                    solve_counter["stage"] = "stage3"
                    solve_start = time.perf_counter()
                    try:
                        solution, _, _ = run_stage3(optimizer, args.optimizer, args.custom_iterations)
                        prediction, pooled_error = score_solution(evaluator, solution, 0.5)
                        truth_error = abs(true_function(solution) - 0.5)
                        violation = optimizer.box_violation(solution)
                        delivered = box_feasible(solution, {"feasible_lower": -1.0, "feasible_upper": 1.0})
                        status, reason = ("valid" if delivered else "box_infeasible"), ""
                        serialized = json.dumps(np.asarray(solution).tolist())
                        solver_name = optimizer.last_subspace_optimization.get("actual_solver")
                    except (ValueError, RuntimeError, FloatingPointError) as exc:
                        status, reason, serialized = "solver_failure", str(exc), ""
                        prediction = pooled_error = truth_error = violation = None
                        solver_name = getattr(optimizer, "last_subspace_optimization", {}).get("actual_solver") if getattr(optimizer, "last_subspace_optimization", None) else None
                    except Exception as exc:
                        status, reason, serialized = "interrupted", f"{type(exc).__name__}: {exc}", ""
                        prediction = pooled_error = truth_error = violation = None
                        solver_name = None
                        coalition_rows.append({
                            "seed": seed, "coalition_indices": json.dumps(indices),
                            "agent_ids": json.dumps([int(item["agent_id"]) for item in chosen]),
                            "coalition_size": size, "delivery_status": status,
                            "failure_stage": "unexpected_exception", "failure_reason": reason,
                            "final_solution": "", "artifact_dir": str(artifact.relative_to(root)),
                            "stage3_requests": solve_counter["requests"]["stage3"],
                            "stage3_input_samples": solve_counter["input_samples"]["stage3"],
                            "stage3_seconds": time.perf_counter()-solve_start,
                        })
                        write_csv(root / "coalitions.csv", coalition_rows)
                        manifest.update(status="interrupted",error=reason)
                        manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
                        raise
                    row = {
                        "seed": seed, "coalition_indices": json.dumps(indices),
                        "agent_ids": json.dumps([int(item["agent_id"]) for item in chosen]),
                        "coalition_size": size,
                        "bid_sum": float(sum(item["bid"] for item in chosen)),
                        "quality_share_sum": float(sum(item["quality_share"] for item in chosen)),
                        "final_solution": serialized, "delivery_status": status,
                        "failure_stage": "stage3" if status == "solver_failure" else "",
                        "failure_reason": reason, "pooled_prediction": prediction,
                        "pooled_error": pooled_error, "truth_error": truth_error,
                        "max_box_violation": violation, "solver": solver_name,
                        "stage3_requests": solve_counter["requests"]["stage3"],
                        "stage3_input_samples": solve_counter["input_samples"]["stage3"],
                        "stage3_seconds": time.perf_counter() - solve_start,
                        "artifact_dir": str(artifact.relative_to(root)),
                    }
                    coalition_rows.append(row)
                    # Persist the single expensive solve before continuing.
                    write_csv(root / "coalitions.csv", coalition_rows)
            valid = [row for row in coalition_rows if int(row["seed"]) == seed
                     and row["delivery_status"] == "valid"]
            for budget in BUDGETS:
                for eta in ETAS:
                    feasible_rows = [row for row in valid if float(row["bid_sum"]) <= budget + 1e-12]
                    ranked = []
                    for row in feasible_rows:
                        ids = json.loads(row["agent_ids"])
                        indices = tuple(i for i, candidate in enumerate(candidates)
                                        if candidate["agent_id"] in ids)
                        objective = coalition_value(indices, candidates, eta, 3)
                        ranked.append((objective, row))
                    rank_values = [item[0] for item in ranked]
                    error_values = [-float(item[1]["pooled_error"]) for item in ranked]
                    rho = (float(spearmanr(rank_values, error_values).statistic)
                           if len(ranked) >= 2 and np.ptp(rank_values) > 0
                           and np.ptp(error_values) > 0 else None)
                    top_objective = {tuple(json.loads(row["agent_ids"])) for _, row in
                                     sorted(ranked, key=lambda item: (-item[0], float(item[1]["bid_sum"])))[:5]}
                    top_quality = {tuple(json.loads(row["agent_ids"])) for row in
                                   sorted(feasible_rows, key=lambda row: float(row["pooled_error"]))[:5]}
                    chosen = max(ranked, key=lambda item: (item[0], -float(item[1]["bid_sum"]))) if ranked else None
                    best_quality = min(feasible_rows, key=lambda row: float(row["pooled_error"])) if feasible_rows else None
                    overlap_denominator = min(5, len(feasible_rows))
                    condition_rows.append({
                        "seed": seed, "budget": budget, "eta": eta,
                        "valid_feasible_coalitions": len(feasible_rows),
                        "failed_coalitions": sum(row["delivery_status"] != "valid" for row in coalition_rows if int(row["seed"]) == seed),
                        "spearman_objective_vs_negative_pooled_error": rho,
                        "spearman_status": "undefined_constant_or_lt2" if rho is None else "defined",
                        "top5_overlap_count": len(top_objective & top_quality),
                        "top5_overlap_fraction": (len(top_objective & top_quality) / overlap_denominator
                            if overlap_denominator else None),
                        "top5_denominator": overlap_denominator,
                        "selected_agent_ids": "" if chosen is None else chosen[1]["agent_ids"],
                        "selected_pooled_error": "" if chosen is None else chosen[1]["pooled_error"],
                        "best_valid_feasible_agent_ids": "" if best_quality is None else best_quality["agent_ids"],
                        "best_valid_feasible_pooled_error": "" if best_quality is None else best_quality["pooled_error"],
                        "selected_pooled_error_regret": "" if chosen is None or best_quality is None else
                            float(chosen[1]["pooled_error"]) - float(best_quality["pooled_error"]),
                        "shared_preparation_seconds": sum(server.preparation_costs[key] for key in
                            ("training_seconds", "qualification_seconds", "proposal_seconds")),
                        "external_evaluator_seconds":evaluator_seconds,
                        "residual_precompute_seconds":residual_precompute_seconds,
                        "local_formal_proposal_seconds":local_proposal_seconds,
                        "enumeration_seconds": time.perf_counter() - coalition_started,
                        "seed_wall_seconds": time.perf_counter() - started,
                        "artifact_dir": str(artifact.relative_to(root)),
                    })
                    write_csv(root / "objective_validity.csv", condition_rows)
            manifest.setdefault("seed_summary", []).append({
                "seed": seed, "candidate_count": len(candidates),
                "enumerated_coalitions": sum(int(row["seed"]) == seed for row in coalition_rows),
                "pooled_test_mse": evaluator_metrics["test_mse"],
            })
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        manifest["status"] = "complete"
    except Exception as exc:
        if manifest.get("status") != "interrupted":
            manifest.update(status="failed", error=str(exc))
        raise
    finally:
        write_csv(root / "coalitions.csv", coalition_rows)
        write_csv(root / "objective_validity.csv", condition_rows)
        write_csv(root/"candidate_pool.csv",candidate_rows)
        write_csv(root/"preparation_costs.csv",preparation_rows)
        manifest["updated_at_utc"] = time.time()
        manifest["artifact_sha256"] = {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file() and path.name != "run_manifest.json"
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    args = parser.parse_args()
    run(args.output_dir, [int(value) for value in args.seeds.split(",") if value.strip()])


if __name__ == "__main__":
    main()

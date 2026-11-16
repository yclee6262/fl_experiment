"""P6 exact/greedy recruitment benchmark with delivered-X evaluation."""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _source_workspace(source_file):
    """Find the flat workspace or a standalone historical source snapshot."""
    start = Path(source_file).resolve().parent
    for candidate in (start, *start.parents):
        if all((candidate / name).is_file() for name in
               ("host_server.py", "run_experiments.py", "dataset.py")):
            return candidate
    raise FileNotFoundError(f"Cannot locate source workspace from {source_file}")


import numpy as np
import torch

from alpha_weight_experiments import (
    attach_query_counter,
    box_feasible,
    build_base_server,
    build_parser as build_alpha_parser,
    run_stage3,
)
from host_server import HostServer
from recruitment_selection import (
    coalition_value,
    exact_select,
    greedy_select,
    selection_report,
)
from research_evaluation import save_scenario_artifacts, score_solution
from run_experiments import true_function


def provenance():
    """Capture the exact local source/environment identity without secrets."""
    root = _source_workspace(__file__)

    def git(*arguments):
        result = subprocess.run(
            ["git", "-C", str(root), *arguments], capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else None

    packages = {}
    for name in ("numpy", "scipy", "torch"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.glob("*.py")
        },
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def qualified_candidates(server):
    by_id = {agent.agent_id: agent for agent in server.all_agents}
    candidates = []
    for row in server.phase1_report:
        if row.get("reason") not in {"selected", "not_selected_by_coalition_rule"}:
            continue
        agent = by_id[row["agent_id"]]
        residual = agent.api_predict(server.test_X) - server.test_y
        candidates.append({
            "agent_id": int(row["agent_id"]),
            "agent": agent,
            "mse": float(row["mse"]),
            "bid": float(row["bid"]),
            "quality_share": float(row["alpha_pre"]),
            "residual": np.asarray(residual, dtype=float),
            "legacy_selected": bool(row["selected"]),
        })
    candidates.sort(key=lambda row: row["agent_id"])
    return candidates


def evaluate_selection(name, report, candidates, proposals, evaluator, args, seed):
    selected = [candidates[index] for index in report["indices"]]
    server = HostServer(
        target_T=args.formal_target,
        n_features=args.dimension,
        total_budget=args.total_budget,
        test_seed=seed + 7919,
        n_test=args.n_test,
        constraint_mode=getattr(args, "constraint_mode", "box_constrained"),
        feasible_lower=args.feasible_lower, feasible_upper=args.feasible_upper,
    )
    server.trusted_agents = [row["agent"] for row in selected]
    server.I_list = [proposals[row["agent_id"]] for row in selected]
    server.alphas = (np.ones(len(selected)) / len(selected)).tolist()
    counter = attach_query_counter(server.trusted_agents)
    counter["stage"] = "stage3"
    solution, _, _ = run_stage3(
        server, args.optimizer, args.custom_iterations
    )
    prediction, external_error = score_solution(
        evaluator, solution, args.formal_target
    )
    bid_sum = report["bid_sum"]
    return {
        "selection_method": name,
        "selected_agent_ids": json.dumps(report["agent_ids"]),
        "coalition_size": len(selected),
        "selection_objective": report["objective"],
        "selection_budget": report["budget"],
        "recruitment_bid_commitment": bid_sum,
        "budget_feasible": report["budget_feasible"],
        "base_reward_liability": bid_sum,
        "accounting_total_committed_cost": bid_sum,
        "accounting_relation": "base reward is the realized recruitment bid commitment",
        "unallocated_surplus_pool": max(0.0, report["budget"] - bid_sum),
        "contribution_settlement_status": "not_run_in_recruitment_benchmark",
        "final_solution": json.dumps(solution.tolist()),
        "external_global_prediction": prediction,
        "external_global_error": external_error,
        "true_error": abs(true_function(solution) - args.formal_target),
        "box_feasible": box_feasible(solution, {
            "feasible_lower": args.feasible_lower,
            "feasible_upper": args.feasible_upper,
        }),
        "stage3_requests": counter["requests"]["stage3"],
        "stage3_input_samples": counter["input_samples"]["stage3"],
        "solver_success": server.last_subspace_optimization["success"],
        "actual_solver": server.last_subspace_optimization.get("actual_solver"),
        "max_box_violation": server.box_violation(solution),
    }


def run(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("P6 output directory must be empty.")
    manifest = {
        "schema_version": 1,
        "status": "running",
        "arguments": vars(args),
        "qualification_inverse_target": args.qualification_inverse_target,
        "formal_inversion_target": args.formal_target,
        "selection_objective": (
            "sum(Stage0 normalized quality share) + diversity_eta * "
            "normalized sum(pairwise residual cosine dissimilarity)"
        ),
        "constraints": "nonempty, bid sum <= selection budget, size <= k_max",
        "accounting": {
            "recruitment_bid_commitment": "sum of selected reservation bids",
            "base_reward_liability": "same bid commitment if recruited",
            "unallocated_surplus_pool": "total budget minus base rewards",
            "contribution_settlement": "out of scope; no fairness claim",
        },
    }
    manifest.update(provenance())
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    selection_rows = []
    candidate_rows = []
    try:
        base_args = build_alpha_parser().parse_args([])
        base_args.num_agents = args.num_agents
        base_args.samples_per_agent = args.samples_per_agent
        base_args.epochs = args.epochs
        base_args.n_test = args.n_test
        base_args.total_budget = args.total_budget
        base_args.budget_fraction = args.budget_fraction
        base_args.diversity_eta = args.diversity_eta
        base_args.inverse_target = args.qualification_inverse_target
        base_args.inverse_steps = args.inverse_steps
        for seed in args.seeds:
            base_server = build_base_server(
                base_args, seed, args.dimension, args.poison_ratio,
                args.formal_target,
            )
            evaluator, artifact_dir, evaluator_metrics = save_scenario_artifacts(
                output,
                {"seed": seed, "dimension": args.dimension,
                 "poison_ratio": args.poison_ratio, "target": args.formal_target},
                base_server.all_agents,
                base_server.study_records,
                evaluator_epochs=args.external_evaluator_epochs,
            )
            del base_server.study_records
            candidates = qualified_candidates(base_server)
            if len(candidates) > args.exact_max_candidates:
                candidates = sorted(
                    candidates,
                    key=lambda row: (-row["quality_share"], row["agent_id"]),
                )[:args.exact_max_candidates]
                candidates.sort(key=lambda row: row["agent_id"])
            budget = args.budget_fraction * args.total_budget
            k_max = min(args.max_coalition_size, len(candidates))
            exact = exact_select(candidates, budget, k_max, args.diversity_eta)
            greedy = greedy_select(candidates, budget, k_max, args.diversity_eta)
            legacy_indices = tuple(
                index for index, row in enumerate(candidates)
                if row["legacy_selected"]
            )
            if not legacy_indices:
                raise ValueError("Legacy Stage 1 selected no candidate in exact pool.")
            legacy = selection_report(
                legacy_indices, candidates, budget, k_max, args.diversity_eta
            )
            proposals = {}
            for row in candidates:
                torch.manual_seed(seed * 100000 + row["agent_id"])
                proposals[row["agent_id"]] = row["agent"].infer_parameters_D(
                    args.formal_target, steps=args.inverse_steps
                )
                candidate_rows.append({
                    "seed": seed,
                    "agent_id": row["agent_id"],
                    "qualification_target": args.qualification_inverse_target,
                    "formal_target": args.formal_target,
                    "mse": row["mse"],
                    "quality_share": row["quality_share"],
                    "bid": row["bid"],
                    "legacy_selected": row["legacy_selected"],
                    "proposal": json.dumps(proposals[row["agent_id"]].tolist()),
                })
            evaluated = []
            for name, report in (
                ("exact", exact), ("objective_greedy", greedy),
                ("legacy_greedy", legacy),
            ):
                row = evaluate_selection(
                    name, report, candidates, proposals, evaluator, args, seed
                )
                row.update({
                    "seed": seed,
                    "dimension": args.dimension,
                    "poison_ratio": args.poison_ratio,
                    "qualification_target": args.qualification_inverse_target,
                    "formal_target": args.formal_target,
                    "qualified_candidates": len(candidates),
                    "max_coalition_size": k_max,
                    "objective_gap_from_exact": exact["objective"] - report["objective"],
                    "external_evaluator_test_truth_mse": evaluator_metrics[
                        "test_truth_mse"
                    ],
                    "scenario_artifact_dir": str(artifact_dir.relative_to(output)),
                })
                evaluated.append(row)
            selection_rows.extend(evaluated)
            write_csv(output / "candidates.csv", candidate_rows)
            write_csv(output / "selection_results.csv", selection_rows)
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", type=lambda x: [int(v) for v in x.split(",")],
                        default=list(range(5)))
    parser.add_argument("--num-agents", type=int, default=10)
    parser.add_argument("--exact-max-candidates", type=int, default=12)
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--poison-ratio", type=float, default=0.0)
    parser.add_argument("--samples-per-agent", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--inverse-steps", type=int, default=500)
    parser.add_argument("--qualification-inverse-target", type=float, default=0.0)
    parser.add_argument("--formal-target", type=float, default=0.5)
    parser.add_argument("--total-budget", type=float, default=10.0)
    parser.add_argument("--budget-fraction", type=float, default=0.8)
    parser.add_argument("--max-coalition-size", type=int, default=3)
    parser.add_argument("--diversity-eta", type=float, default=0.5)
    parser.add_argument("--optimizer", choices=("best_of", "custom", "bfgs"),
                        default="best_of")
    parser.add_argument("--custom-iterations", type=int, default=30)
    parser.add_argument("--external-evaluator-epochs", type=int, default=100)
    parser.add_argument("--feasible-lower", type=float, default=-1.0)
    parser.add_argument("--feasible-upper", type=float, default=1.0)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

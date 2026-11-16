"""P3 pilot: compare candidate exclusion with full participant removal.

The operational contribution removes one candidate point while retaining every
participant model and the current alpha vector in the Stage 3 objective.  The
full-removal counterfactual removes the participant's candidate, model, and
alpha, renormalizes the remaining alpha, and reruns Stage 3 from one fixed
feature-space starting point.  Both delivered solutions are judged only after
optimization by the same frozen pooled evaluator.
"""

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from alpha_weight_experiments import (
    attach_query_counter,
    box_feasible,
    build_base_server,
    build_parser as build_alpha_parser,
    configure_controlled_coalition,
    normalize,
    raw_contribution_distribution,
    run_stage3,
    write_csv,
)
from research_evaluation import save_scenario_artifacts, score_solution
from run_experiments import true_function


def full_participant_removal(
    server,
    excluded_index,
    initial_solution,
    optimizer="best_of",
    custom_iterations=30,
):
    """Rerun Stage 3 after removing candidate, engine, and alpha.

    The caller's coalition state is restored even if optimization fails.
    """
    if len(server.trusted_agents) <= 1:
        raise ValueError("Full participant removal requires at least two agents.")
    if excluded_index < 0 or excluded_index >= len(server.trusted_agents):
        raise IndexError("excluded_index is outside the active coalition.")
    original_agents = server.trusted_agents
    original_candidates = server.I_list
    original_alphas = server.alphas
    remaining = [index for index in range(len(original_agents)) if index != excluded_index]
    try:
        server.trusted_agents = [original_agents[index] for index in remaining]
        server.I_list = [original_candidates[index] for index in remaining]
        server.alphas = normalize([original_alphas[index] for index in remaining]).tolist()
        solution, _, _ = run_stage3(
            server,
            optimizer,
            custom_iterations,
            initial_solution=np.asarray(initial_solution, dtype=float),
        )
        return {
            "solution": np.asarray(solution, dtype=float),
            "remaining_agent_ids": [agent.agent_id for agent in server.trusted_agents],
            "remaining_alphas": np.asarray(server.alphas, dtype=float),
            "optimization_loss": float(server.last_subspace_optimization["loss"]),
            "chosen_engine": server.last_subspace_optimization["chosen_engine"],
        }
    finally:
        server.trusted_agents = original_agents
        server.I_list = original_candidates
        server.alphas = original_alphas


def _manifest(output_dir, args, status, error=None):
    root = Path(__file__).resolve().parent

    def git(*arguments):
        result = subprocess.run(
            ["git", "-C", str(root), *arguments], capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else None

    path = Path(output_dir) / "run_manifest.json"
    if path.exists():
        payload = json.loads(path.read_text())
    else:
        sources = {}
        for name in (
            "p3_contribution_validation.py", "alpha_weight_experiments.py",
            "host_server.py", "research_evaluation.py", "run_experiments.py",
            "agent_client.py", "models.py", "dataset.py",
        ):
            sources[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
        packages = {}
        for name in ("numpy", "scipy", "torch"):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
        payload = {
            "schema_version": 2,
            "experiment_id": args.experiment_id,
            "study": "p3_candidate_vs_full_participant_removal",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git("rev-parse", "HEAD"),
            "git_status": git("status", "--short"),
            "source_sha256": sources,
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "packages": packages,
            "arguments": vars(args),
            "semantics": {
                "operational": "remove candidate point only; retain all engines and alpha",
                "full_removal": "remove candidate, engine, and alpha; renormalize alpha",
                "retraining": "local models are not retrained",
                "quality": "frozen pooled evaluator scored after optimization only",
                "pooled_evaluator_feedback": "forbidden",
            },
            "acceptance": {
                "feasibility_tolerance": 1e-8,
                "reload_score_tolerance": 1e-7,
                "primary_dead_zone": 1e-7,
            },
        }
    payload["status"] = status
    payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    if status == "complete":
        artifact_root = Path(output_dir) / "artifacts"
        payload["artifact_sha256"] = {
            str(path.relative_to(output_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in artifact_root.rglob("*") if path.is_file()
        } if artifact_root.exists() else {}
    if error is not None:
        payload["error"] = str(error)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _max_box_violation(solution, lower, upper):
    values = np.asarray(solution, dtype=float)
    if not np.all(np.isfinite(values)):
        return float("inf")
    return float(max(0.0, np.max(lower - values), np.max(values - upper)))


def _snapshot_sources(output_dir):
    root = Path(__file__).resolve().parent
    snapshot = Path(output_dir) / "source_snapshot"
    snapshot.mkdir(exist_ok=False)
    hashes = {}
    for name in (
        "p3_contribution_validation.py", "alpha_weight_experiments.py",
        "host_server.py", "research_evaluation.py", "run_experiments.py",
        "agent_client.py", "models.py", "dataset.py",
    ):
        target = snapshot / name
        shutil.copy2(root / name, target)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    return hashes


def _base_args(args):
    base = build_alpha_parser().parse_args([])
    for name in (
        "num_agents", "samples_per_agent", "epochs", "n_test", "total_budget",
        "mse_threshold", "inverse_target", "inverse_loss_threshold", "inverse_steps",
        "feasible_lower", "feasible_upper", "budget_fraction", "diversity_eta",
        "min_selection_score", "k_api", "k_red", "custom_iterations",
        "constraint_mode",
    ):
        setattr(base, name, getattr(args, name))
    base.disable_inverse_check = args.disable_inverse_check
    return base


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("Use an empty output directory to preserve previous experiments.")
    if args.formal and args.constraint_mode != "box_constrained":
        raise ValueError("Formal P3 requires --constraint-mode box_constrained.")
    if args.formal and args.scenario == "clean" and not (
        args.poison_ratio == 0.0 and args.coalition_mode == "stage1"
        and args.evaluator == "uniform"
    ):
        raise ValueError("Formal clean scenario requires stage1, poison=0, uniform evaluator.")
    if args.formal and args.scenario == "controlled_poison" and not (
        args.poison_ratio > 0.0 and args.coalition_mode == "controlled"
        and args.controlled_coalition_size == 5
        and args.controlled_poison_count == 1 and args.evaluator == "trimmed"
    ):
        raise ValueError("Formal controlled_poison scenario configuration is inconsistent.")
    if args.coalition_mode == "controlled" and args.controlled_coalition_size > 5:
        raise ValueError("P3 pilot coalitions are limited to at most five participants.")
    source_hashes = _snapshot_sources(output_dir)
    _manifest(output_dir, args, "running")
    manifest_path = output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_snapshot_sha256"] = source_hashes
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    effect_rows = []
    scenario_rows = []
    try:
        for seed in args.seeds:
            base_args = _base_args(args)
            server = build_base_server(
                base_args, seed, args.dimension, args.poison_ratio, args.target
            )
            poisoned_count = int(args.num_agents * args.poison_ratio)
            poisoned_ids = set(
                range(args.num_agents - poisoned_count + 1, args.num_agents + 1)
            ) if poisoned_count else set()
            if args.coalition_mode == "controlled":
                configure_controlled_coalition(
                    server, poisoned_ids, args.controlled_coalition_size,
                    args.controlled_poison_count,
                )
            if len(server.trusted_agents) > 5:
                raise ValueError(
                    f"Seed {seed} selected {len(server.trusted_agents)} agents; "
                    "P3 pilot allows at most five."
                )
            server.alphas = (
                np.ones(len(server.trusted_agents), dtype=float)
                / len(server.trusted_agents)
            ).tolist()
            scenario = {
                "experiment_id": args.experiment_id,
                "scenario": args.scenario,
                "seed": seed,
                "dimension": args.dimension,
                "poison_ratio": args.poison_ratio,
                "target": args.target,
            }
            evaluator, artifact_dir, evaluator_metrics = save_scenario_artifacts(
                output_dir,
                {key: scenario[key] for key in ("seed", "dimension", "poison_ratio", "target")},
                server.all_agents, server.study_records,
                evaluator_epochs=args.external_evaluator_epochs,
            )
            del server.study_records
            fixed_start = np.mean(np.asarray(server.I_list, dtype=float), axis=0)
            feasibility = {
                "feasible_lower": args.feasible_lower,
                "feasible_upper": args.feasible_upper,
            }
            counter = attach_query_counter(server.trusted_agents)
            counter["stage"] = "baseline_stage3"
            baseline_started = time.perf_counter()
            baseline_solution, _, _ = run_stage3(
                server, args.optimizer, args.custom_iterations,
                initial_solution=fixed_start,
            )
            baseline_seconds = time.perf_counter() - baseline_started
            baseline_solver = dict(server.last_subspace_optimization)
            baseline_violation = _max_box_violation(
                baseline_solution, args.feasible_lower, args.feasible_upper
            )
            pooled_started = time.perf_counter()
            baseline_external_prediction, baseline_external_error = score_solution(
                evaluator, baseline_solution, args.target
            )
            baseline_pooled_seconds = time.perf_counter() - pooled_started
            baseline_true_error = abs(true_function(baseline_solution) - args.target)
            evaluation_weights = None
            counter["stage"] = "candidate_exclusion"
            candidate_started = time.perf_counter()
            base_internal_loss, reports, exclusion_summary = server._compute_exclusion_reports(
                baseline_solution,
                verbose=False,
                evaluation_mode=args.evaluator,
                evaluation_weights=evaluation_weights,
                trim_fraction=args.trim_fraction,
                optimizer=args.optimizer,
                custom_iterations=args.custom_iterations,
                return_summary=True,
            )
            candidate_seconds = time.perf_counter() - candidate_started
            operational_requests = counter["requests"]["candidate_exclusion"]
            raw_q = raw_contribution_distribution(reports)
            original_agents = list(server.trusted_agents)
            original_candidates = [np.asarray(value, dtype=float) for value in server.I_list]
            original_alphas = list(server.alphas)
            full_total_seconds = 0.0
            pooled_total_seconds = baseline_pooled_seconds

            for index, report in enumerate(reports):
                candidate_solution = np.asarray(report["restricted_solution"], dtype=float)
                pooled_started = time.perf_counter()
                candidate_prediction, candidate_external_error = score_solution(
                    evaluator, candidate_solution, args.target
                )
                candidate_pooled_seconds = time.perf_counter() - pooled_started
                before = counter["requests"]["full_participant_removal"]
                counter["stage"] = "full_participant_removal"
                full_started = time.perf_counter()
                removal = full_participant_removal(
                    server, index, fixed_start, args.optimizer, args.custom_iterations
                )
                full_seconds = time.perf_counter() - full_started
                full_total_seconds += full_seconds
                full_requests = (
                    counter["requests"]["full_participant_removal"] - before
                )
                pooled_started = time.perf_counter()
                full_prediction, full_external_error = score_solution(
                    evaluator, removal["solution"], args.target
                )
                full_pooled_seconds = time.perf_counter() - pooled_started
                pooled_total_seconds += candidate_pooled_seconds + full_pooled_seconds
                full_true_error = abs(true_function(removal["solution"]) - args.target)
                effect_rows.append({
                    **scenario,
                    "coalition_mode": args.coalition_mode,
                    "evaluator": args.evaluator,
                    "agent_id": report["agent_id"],
                    "execution_status": "complete",
                    "delivery_status": "valid_delivery",
                    "failure_stage": "",
                    "failure_reason": "",
                    "is_poisoned": report["agent_id"] in poisoned_ids,
                    "alpha": original_alphas[index],
                    "candidate_point": json.dumps(original_candidates[index].tolist()),
                    "fixed_initial_solution": json.dumps(fixed_start.tolist()),
                    "baseline_solution": json.dumps(baseline_solution.tolist()),
                    "baseline_external_prediction": baseline_external_prediction,
                    "baseline_external_error": baseline_external_error,
                    "baseline_true_error": baseline_true_error,
                    "baseline_box_feasible": box_feasible(
                        baseline_solution, feasibility
                    ),
                    "baseline_max_box_violation": baseline_violation,
                    "baseline_solver": baseline_solver.get("chosen_engine"),
                    "baseline_actual_solver": baseline_solver.get("actual_solver"),
                    "baseline_seconds": baseline_seconds,
                    "baseline_pooled_scoring_seconds": baseline_pooled_seconds,
                    "operational_internal_contribution": report["marginal_contribution"],
                    "operational_internal_loss_without": report["loss_without_agent"],
                    "operational_restricted_solution": json.dumps(candidate_solution.tolist()),
                    "operational_external_prediction_without": candidate_prediction,
                    "operational_external_error_without": candidate_external_error,
                    "operational_external_contribution": (
                        candidate_external_error - baseline_external_error
                    ),
                    "operational_box_feasible": box_feasible(
                        candidate_solution, feasibility
                    ),
                    "operational_max_box_violation": _max_box_violation(
                        candidate_solution, args.feasible_lower, args.feasible_upper
                    ),
                    "operational_solver": report["restricted_engine"],
                    "operational_pooled_scoring_seconds": candidate_pooled_seconds,
                    "full_removal_solution": json.dumps(removal["solution"].tolist()),
                    "full_removal_agent_ids": json.dumps(removal["remaining_agent_ids"]),
                    "full_removal_alphas": json.dumps(removal["remaining_alphas"].tolist()),
                    "full_removal_engine": removal["chosen_engine"],
                    "full_removal_optimization_loss": removal["optimization_loss"],
                    "full_removal_external_prediction_without": full_prediction,
                    "full_removal_external_error_without": full_external_error,
                    "full_removal_external_contribution": (
                        full_external_error - baseline_external_error
                    ),
                    "full_removal_true_error_without": full_true_error,
                    "full_removal_true_contribution": full_true_error - baseline_true_error,
                    "full_removal_box_feasible": box_feasible(
                        removal["solution"], feasibility
                    ),
                    "full_removal_max_box_violation": _max_box_violation(
                        removal["solution"], args.feasible_lower, args.feasible_upper
                    ),
                    "full_removal_requests": full_requests,
                    "full_removal_seconds": full_seconds,
                    "full_removal_pooled_scoring_seconds": full_pooled_seconds,
                    "scenario_artifact_dir": str(artifact_dir.relative_to(output_dir)),
                })
                write_csv(output_dir / "participant_effects.csv", effect_rows)

            server.trusted_agents = original_agents
            server.I_list = original_candidates
            server.alphas = original_alphas
            scenario_rows.append({
                **scenario,
                "coalition_mode": args.coalition_mode,
                "evaluator": args.evaluator,
                "coalition_size": len(original_agents),
                "execution_status": "complete",
                "delivery_status": "valid_delivery",
                "failure_stage": "",
                "failure_reason": "",
                "selected_agent_ids": json.dumps([a.agent_id for a in original_agents]),
                "selected_poisoned_ids": json.dumps(
                    [a.agent_id for a in original_agents if a.agent_id in poisoned_ids]
                ),
                "fixed_initial_solution": json.dumps(fixed_start.tolist()),
                "baseline_solution": json.dumps(baseline_solution.tolist()),
                "baseline_internal_loss": base_internal_loss,
                "baseline_external_error": baseline_external_error,
                "baseline_true_error": baseline_true_error,
                "baseline_box_feasible": box_feasible(
                    baseline_solution, feasibility
                ),
                "baseline_max_box_violation": baseline_violation,
                "baseline_solver": baseline_solver.get("chosen_engine"),
                "baseline_actual_solver": baseline_solver.get("actual_solver"),
                "operational_q_valid": raw_q is not None,
                "operational_q_raw": json.dumps(
                    [] if raw_q is None else raw_q.tolist()
                ),
                "baseline_stage3_requests": counter["requests"]["baseline_stage3"],
                "candidate_exclusion_requests": operational_requests,
                "full_removal_requests": counter["requests"]["full_participant_removal"],
                "baseline_stage3_seconds": baseline_seconds,
                "candidate_exclusion_seconds": candidate_seconds,
                "full_removal_seconds": full_total_seconds,
                "pooled_scoring_seconds": pooled_total_seconds,
                "constraint_mode": args.constraint_mode,
                "feasibility_tolerance": 1e-8,
                "external_evaluator_test_truth_mse": evaluator_metrics["test_truth_mse"],
                "external_evaluator_hard_test_truth_mse": evaluator_metrics[
                    "hard_test_truth_mse"
                ],
                "exclusion_semantics": "candidate_only",
                "full_removal_semantics": "candidate_engine_alpha_no_retraining",
                "loo_search_objective": exclusion_summary["loo_search_objective"],
                "scenario_artifact_dir": str(artifact_dir.relative_to(output_dir)),
            })
            write_csv(output_dir / "scenario_summary.csv", scenario_rows)
        _manifest(output_dir, args, "complete")
    except Exception as exc:
        _manifest(output_dir, args, "failed", error=exc)
        raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-id", default="p3_external_validity")
    parser.add_argument("--scenario", choices=("clean", "controlled_poison"),
                        default="clean")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--seeds", type=lambda x: [int(v) for v in x.split(",")],
                        default=list(range(5)))
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--poison-ratio", type=float, default=0.0)
    parser.add_argument("--target", type=float, default=0.0)
    parser.add_argument("--evaluator", choices=("uniform", "median", "trimmed"),
                        default="uniform")
    parser.add_argument("--coalition-mode", choices=("stage1", "controlled"),
                        default="stage1")
    parser.add_argument("--controlled-coalition-size", type=int, default=5)
    parser.add_argument("--controlled-poison-count", type=int, default=1)
    parser.add_argument("--optimizer", choices=("best_of", "custom", "bfgs"),
                        default="best_of")
    parser.add_argument("--custom-iterations", type=int, default=30)
    parser.add_argument("--trim-fraction", type=float, default=0.2)
    parser.add_argument("--external-evaluator-epochs", type=int, default=100)
    parser.add_argument("--num-agents", type=int, default=20)
    parser.add_argument("--samples-per-agent", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--total-budget", type=float, default=10.0)
    parser.add_argument("--mse-threshold", type=float, default=0.1)
    parser.add_argument("--inverse-target", type=float, default=0.0)
    parser.add_argument("--inverse-loss-threshold", type=float, default=0.1)
    parser.add_argument("--inverse-steps", type=int, default=500)
    parser.add_argument("--feasible-lower", type=float, default=-1.0)
    parser.add_argument("--feasible-upper", type=float, default=1.0)
    parser.add_argument("--constraint-mode",
                        choices=("legacy_unconstrained", "box_constrained"),
                        default="box_constrained")
    parser.add_argument("--disable-inverse-check", action="store_true")
    parser.add_argument("--budget-fraction", type=float, default=0.8)
    parser.add_argument("--diversity-eta", type=float, default=0.5)
    parser.add_argument("--min-selection-score", type=float, default=0.0)
    parser.add_argument("--k-api", type=int, default=None)
    parser.add_argument("--k-red", type=int, default=None)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

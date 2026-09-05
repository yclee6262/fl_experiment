"""Experiments for separated reputation, optimization, and contribution weights.

The experiment keeps the Stage 1 coalition and Stage 2 candidates fixed across
conditions.  Stage 1 weights are retained as ex-ante reputation ``r``; Stage 3
uses an independently initialized and iteratively calibrated ``alpha``; and
Subspace Exclusion produces the ex-post contribution distribution ``q``.
"""

import argparse
import copy
import csv
import json
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np

from host_server import HostServer
from run_experiments import build_agents, true_function


def parse_csv(value, cast=str):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def normalize(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("Weights must be a non-empty one-dimensional vector.")
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("Weights must be finite and non-negative.")
    total = float(np.sum(values))
    if total <= 0:
        raise ValueError("Weights must have a positive sum.")
    return values / total


def initialize_optimization_weights(kind, reputation, rng):
    reputation = normalize(reputation)
    n_agents = len(reputation)
    if kind == "stage0":
        return reputation.copy()
    if kind == "uniform":
        return np.ones(n_agents, dtype=float) / n_agents
    if kind == "dirichlet":
        return rng.dirichlet(np.ones(n_agents, dtype=float))
    if kind == "reversed":
        return normalize(reputation[::-1])
    raise ValueError(f"Unknown initialization: {kind}")


def contribution_distribution(exclusion_reports, exploration_mass=0.0):
    positives = np.asarray(
        [
            row["positive_contribution"]
            if np.isfinite(row["positive_contribution"])
            else 0.0
            for row in exclusion_reports
        ],
        dtype=float,
    )
    total = float(np.sum(positives))
    if total <= 0:
        return None
    distribution = positives / total
    if not 0.0 <= exploration_mass < 1.0:
        raise ValueError("exploration_mass must be in [0, 1).")
    if exploration_mass:
        distribution = (
            (1.0 - exploration_mass) * distribution
            + exploration_mass / len(distribution)
        )
    return normalize(distribution)


def attach_query_counter(agents):
    counter = {"stage": "unassigned", "requests": defaultdict(int)}
    for agent in agents:
        original = agent.api_predict

        def counted(X_array, _original=original):
            counter["requests"][counter["stage"]] += 1
            return _original(X_array)

        agent.api_predict = counted
    return counter


def run_stage3(server, optimizer, custom_iterations):
    if optimizer == "bfgs":
        final_S, history = server.phase3_global_optimization()
        states = ["bfgs"] * len(history)
    else:
        final_S, history, states = server.phase3_custom_secant_optimization(
            num_iterations=custom_iterations,
            use_annealing=True,
            allow_tangent=True,
        )
    return np.asarray(final_S, dtype=float), history, states


def evaluator_arguments(evaluator, reputation):
    if evaluator == "reputation":
        return "fixed", reputation
    return evaluator, None


def settle_with_separated_weights(
    server,
    reputation,
    contribution,
    positive_contribution,
    reputation_mix,
):
    """Settle the surplus from reputation and contribution, never optimization alpha."""
    if contribution is None:
        return "rejected_uninformative_contribution", [], None

    positive_indices = [
        idx for idx, value in enumerate(positive_contribution) if value > 0
    ]
    if not positive_indices:
        return "rejected_no_positive_contributors", [], None

    bids = np.asarray(
        [server._agent_bid(server.trusted_agents[idx]) for idx in positive_indices],
        dtype=float,
    )
    bid_sum = float(np.sum(bids))
    if bid_sum > server.total_budget:
        return "infeasible_minimum_bids_exceed_budget", [], None

    reputation_active = normalize(np.asarray(reputation)[positive_indices])
    contribution_active = normalize(np.asarray(contribution)[positive_indices])
    surplus = server.total_budget - bid_sum
    rows = []
    for local_idx, agent_idx in enumerate(positive_indices):
        surplus_share = (
            reputation_mix * reputation_active[local_idx]
            + (1.0 - reputation_mix) * contribution_active[local_idx]
        )
        payment = bids[local_idx] + surplus * surplus_share
        rows.append(
            {
                "agent_id": server.trusted_agents[agent_idx].agent_id,
                "reputation_share": float(reputation_active[local_idx]),
                "optimization_weight": float(server.alphas[agent_idx]),
                "contribution_share": float(contribution_active[local_idx]),
                "surplus_share": float(surplus_share),
                "bid": float(bids[local_idx]),
                "payment": float(payment),
            }
        )
    return "ok", rows, float(sum(row["payment"] for row in rows))


def calibrate_one_condition(server, config, reputation, poisoned_ids):
    rng = np.random.default_rng(config["condition_seed"])
    alpha = initialize_optimization_weights(
        config["initialization"], reputation, rng
    )
    server.alphas = [float(value) for value in alpha]
    counter = attach_query_counter(server.trusted_agents)
    round_rows = []
    agent_rows = []
    final_S = None
    final_q = None
    final_positive_contribution = None
    final_eval_loss = None
    converged = False
    stop_reason = "max_rounds"

    evaluation_mode, evaluation_weights = evaluator_arguments(
        config["evaluator"], reputation
    )

    for round_idx in range(config["max_rounds"]):
        alpha_before = np.asarray(server.alphas, dtype=float)
        counter["stage"] = "stage3"
        final_S, _, _ = run_stage3(
            server, config["optimizer"], config["custom_iterations"]
        )
        target_error = abs(true_function(final_S) - server.target_T)

        counter["stage"] = "contribution"
        final_eval_loss, exclusion_reports = server._compute_exclusion_reports(
            final_S,
            verbose=False,
            evaluation_mode=evaluation_mode,
            evaluation_weights=evaluation_weights,
            trim_fraction=config["trim_fraction"],
        )
        positive_contribution = np.asarray(
            [report["positive_contribution"] for report in exclusion_reports],
            dtype=float,
        )
        q = contribution_distribution(
            exclusion_reports,
            exploration_mass=config["exploration_mass"],
        )

        informative = q is not None
        if informative:
            consistency_l1 = float(np.sum(np.abs(alpha_before - q)))
            if config["update_rate"] == 0:
                alpha_after = alpha_before.copy()
                stop_reason = "no_update_baseline"
            else:
                alpha_after = normalize(
                    (1.0 - config["update_rate"]) * alpha_before
                    + config["update_rate"] * q
                )
            update_l1 = float(np.sum(np.abs(alpha_after - alpha_before)))
        else:
            q_for_update = alpha_before.copy()
            consistency_l1 = None
            update_l1 = 0.0
            alpha_after = alpha_before.copy()
            stop_reason = "uninformative_contribution"

        if informative:
            q_for_update = q

        round_rows.append(
            {
                **config["identity"],
                "round": round_idx,
                "target_error": float(target_error),
                "evaluation_loss": float(final_eval_loss),
                "consistency_l1": consistency_l1,
                "update_l1": update_l1,
                "informative": informative,
                "stage3_requests_cumulative": counter["requests"]["stage3"],
                "contribution_requests_cumulative": counter["requests"]["contribution"],
            }
        )
        for idx, report in enumerate(exclusion_reports):
            agent_rows.append(
                {
                    **config["identity"],
                    "round": round_idx,
                    "agent_id": report["agent_id"],
                    "is_poisoned": report["agent_id"] in poisoned_ids,
                    "reputation_share": float(reputation[idx]),
                    "optimization_before": float(alpha_before[idx]),
                    "marginal_contribution": float(report["marginal_contribution"]),
                    "positive_contribution": float(report["positive_contribution"]),
                    "contribution_share": (
                        float(q_for_update[idx]) if informative else None
                    ),
                    "optimization_after": float(alpha_after[idx]),
                }
            )

        final_q = q if informative else None
        final_positive_contribution = positive_contribution
        server.alphas = [float(value) for value in alpha_after]
        if not informative or config["update_rate"] == 0:
            break
        if consistency_l1 < config["tolerance"]:
            converged = True
            stop_reason = "fixed_point_tolerance"
            break
        if round_idx == config["max_rounds"] - 1:
            # Keep alpha aligned with the solution evaluated in this final round.
            server.alphas = [float(value) for value in alpha_before]

    payment_status, payment_rows, paid_total = settle_with_separated_weights(
        server,
        reputation,
        final_q if stop_reason != "uninformative_contribution" else None,
        final_positive_contribution,
        config["payment_reputation_mix"],
    )
    for row in payment_rows:
        row.update(config["identity"])

    final_alpha = np.asarray(server.alphas, dtype=float)
    selected_poison_weight = float(
        sum(
            final_alpha[idx]
            for idx, agent in enumerate(server.trusted_agents)
            if agent.agent_id in poisoned_ids
        )
    )
    summary = {
        **config["identity"],
        "selected_agent_ids": json.dumps(
            [agent.agent_id for agent in server.trusted_agents]
        ),
        "reputation_weights": json.dumps([float(x) for x in reputation]),
        "final_optimization_weights": json.dumps([float(x) for x in final_alpha]),
        "final_contribution_weights": json.dumps(
            [float(x) for x in final_q] if final_q is not None else []
        ),
        "rounds_executed": len(round_rows),
        "converged": converged,
        "stop_reason": stop_reason,
        "final_target_error": float(abs(true_function(final_S) - server.target_T)),
        "final_evaluation_loss": float(final_eval_loss),
        "final_consistency_l1": round_rows[-1]["consistency_l1"],
        "selected_poison_weight": selected_poison_weight,
        "stage3_requests": counter["requests"]["stage3"],
        "contribution_requests": counter["requests"]["contribution"],
        "payment_status": payment_status,
        "paid_total": paid_total,
    }
    return summary, round_rows, agent_rows, payment_rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_base_server(args, seed, dimension, poison_ratio, target):
    agents = build_agents(
        num_agents=args.num_agents,
        n_features=dimension,
        poison_ratio=poison_ratio,
        samples_per_agent=args.samples_per_agent,
        epochs=args.epochs,
        seed=seed,
    )
    server = HostServer(
        target_T=target,
        n_features=dimension,
        total_budget=args.total_budget,
        test_seed=seed + 7919,
        n_test=args.n_test,
    )
    server.phase1_filter_agents(
        agents,
        mse_threshold=args.mse_threshold,
        budget_fraction=args.budget_fraction,
        diversity_eta=args.diversity_eta,
        min_selection_score=args.min_selection_score,
        k_api=args.k_api,
        k_red=args.k_red,
        enable_inverse_check=not args.disable_inverse_check,
        inverse_target=args.inverse_target,
        inverse_loss_threshold=args.inverse_loss_threshold,
        inverse_steps=args.inverse_steps,
        feasible_lower=args.feasible_lower,
        feasible_upper=args.feasible_upper,
    )
    server.phase2_collect_proposals()
    return server


def run(args):
    output_dir = Path(args.output_dir)
    summaries = []
    rounds = []
    agents = []
    payments = []
    condition_index = 0

    scenarios = product(args.seeds, args.dimensions, args.poison_ratios, args.targets)
    conditions = list(product(args.initializations, args.evaluators, args.update_rates))
    for seed, dimension, poison_ratio, target in scenarios:
        base_server = build_base_server(args, seed, dimension, poison_ratio, target)
        reputation = normalize(base_server.alphas)
        num_poisoned = int(args.num_agents * poison_ratio)
        poisoned_ids = set(
            range(args.num_agents - num_poisoned + 1, args.num_agents + 1)
        ) if num_poisoned else set()

        for initialization, evaluator, update_rate in conditions:
            identity = {
                "seed": seed,
                "dimension": dimension,
                "poison_ratio": poison_ratio,
                "target": target,
                "initialization": initialization,
                "evaluator": evaluator,
                "update_rate": update_rate,
            }
            config = {
                "identity": identity,
                "condition_seed": args.condition_seed + condition_index,
                "initialization": initialization,
                "evaluator": evaluator,
                "update_rate": update_rate,
                "max_rounds": args.max_calibration_rounds,
                "tolerance": args.alpha_tolerance,
                "exploration_mass": args.exploration_mass,
                "trim_fraction": args.trim_fraction,
                "optimizer": args.optimizer,
                "custom_iterations": args.custom_iterations,
                "payment_reputation_mix": args.payment_reputation_mix,
            }
            condition_index += 1
            server = copy.deepcopy(base_server)
            result = calibrate_one_condition(
                server, config, reputation.copy(), poisoned_ids
            )
            summary, round_rows, agent_rows, payment_rows = result
            summaries.append(summary)
            rounds.extend(round_rows)
            agents.extend(agent_rows)
            payments.extend(payment_rows)

    write_csv(output_dir / "summary.csv", summaries)
    write_csv(output_dir / "rounds.csv", rounds)
    write_csv(output_dir / "agents.csv", agents)
    write_csv(output_dir / "payments.csv", payments)
    print(f"Alpha-weight experiments complete: {len(summaries)} conditions")
    print(f"Outputs written to: {output_dir}")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare ex-ante reputation, iterative Stage 3 optimization weights, "
            "and ex-post Subspace Exclusion contribution shares."
        )
    )
    parser.add_argument("--output-dir", default="alpha_weight_outputs")
    parser.add_argument("--seeds", type=lambda x: parse_csv(x, int), default=[0])
    parser.add_argument("--dimensions", type=lambda x: parse_csv(x, int), default=[2])
    parser.add_argument(
        "--poison-ratios", type=lambda x: parse_csv(x, float), default=[0.0, 0.3]
    )
    parser.add_argument("--targets", type=lambda x: parse_csv(x, float), default=[0.0])
    parser.add_argument(
        "--initializations",
        type=lambda x: parse_csv(x, str),
        choices=None,
        default=["stage0", "uniform"],
        help="Comma-separated: stage0,uniform,dirichlet,reversed",
    )
    parser.add_argument(
        "--evaluators",
        type=lambda x: parse_csv(x, str),
        default=["optimization", "uniform"],
        help="Comma-separated: optimization,reputation,uniform,median,trimmed",
    )
    parser.add_argument(
        "--update-rates", type=lambda x: parse_csv(x, float), default=[0.0, 0.3]
    )
    parser.add_argument("--max-calibration-rounds", type=int, default=6)
    parser.add_argument("--alpha-tolerance", type=float, default=1e-3)
    parser.add_argument("--exploration-mass", type=float, default=0.02)
    parser.add_argument("--trim-fraction", type=float, default=0.2)
    parser.add_argument("--payment-reputation-mix", type=float, default=0.4)
    parser.add_argument("--condition-seed", type=int, default=20260904)
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
    parser.add_argument("--disable-inverse-check", action="store_true")
    parser.add_argument("--budget-fraction", type=float, default=0.8)
    parser.add_argument("--diversity-eta", type=float, default=0.5)
    parser.add_argument("--min-selection-score", type=float, default=0.0)
    parser.add_argument("--k-api", type=int, default=None)
    parser.add_argument("--k-red", type=int, default=None)
    parser.add_argument("--optimizer", choices=["custom", "bfgs"], default="custom")
    parser.add_argument("--custom-iterations", type=int, default=30)
    return parser


def validate_args(args):
    valid_initializations = {"stage0", "uniform", "dirichlet", "reversed"}
    valid_evaluators = {"optimization", "reputation", "uniform", "median", "trimmed"}
    unknown_initializations = set(args.initializations) - valid_initializations
    unknown_evaluators = set(args.evaluators) - valid_evaluators
    if unknown_initializations:
        raise ValueError(f"Unknown initializations: {sorted(unknown_initializations)}")
    if unknown_evaluators:
        raise ValueError(f"Unknown evaluators: {sorted(unknown_evaluators)}")
    if any(rate < 0 or rate > 1 for rate in args.update_rates):
        raise ValueError("Every update rate must be in [0, 1].")
    if args.max_calibration_rounds < 1:
        raise ValueError("max-calibration-rounds must be positive.")
    if not 0 <= args.payment_reputation_mix <= 1:
        raise ValueError("payment-reputation-mix must be in [0, 1].")
    if not 0 <= args.exploration_mass < 1:
        raise ValueError("exploration-mass must be in [0, 1).")


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    validate_args(parsed_args)
    run(parsed_args)

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from agent_client import AgentNode
from dataset import generate_agent_dataloaders
from host_server import HostServer


def true_function(S):
    S = np.asarray(S, dtype=float)
    y = np.sum(S)
    if len(S) > 1:
        y += np.sum(S[:-1] * S[1:])
    return float(y)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_agents(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    loaders = generate_agent_dataloaders(
        num_agents=args.num_agents,
        n_samples_per_agent=args.samples_per_agent,
        poison_ratio=args.poison_ratio,
        n_features=args.n_features,
        seed=args.seed,
    )

    agents = []
    for idx, loader in enumerate(loaders):
        torch.manual_seed(args.seed * 1000 + idx)
        agent = AgentNode(agent_id=idx + 1, dataloader=loader, n_features=args.n_features)
        print(f"Training Agent {idx + 1}...")
        agent.train_local_model(epochs=args.epochs)
        agents.append(agent)
    return agents


def normalize(values):
    values = np.asarray(values, dtype=float)
    total = float(np.sum(values))
    if total <= 0 or not np.isfinite(total):
        return np.ones_like(values) / len(values)
    return values / total


def contribution_distribution(exclusion_reports):
    positives = np.asarray(
        [
            row["positive_contribution"]
            if np.isfinite(row["positive_contribution"]) else 0.0
            for row in exclusion_reports
        ],
        dtype=float,
    )
    return normalize(positives)


def run_stage3(server, args):
    if args.optimizer == "bfgs":
        final_S, history = server.phase3_global_optimization()
        states = ["bfgs"] * len(history)
    else:
        final_S, history, states = server.phase3_custom_secant_optimization(
            num_iterations=args.custom_iterations,
            use_annealing=True,
            allow_tangent=True,
        )
    return final_S, history, states


def calibrate_alphas(server, args):
    alpha_history_rows = []
    contribution_rows = []

    for round_idx in range(args.max_calibration_rounds + 1):
        print(f"\n=== Alpha calibration round {round_idx} ===")
        old_alpha = np.asarray(server.alphas, dtype=float)

        final_S, history, states = run_stage3(server, args)
        base_loss, exclusion_reports = server._compute_exclusion_reports(final_S)
        contrib_share = contribution_distribution(exclusion_reports)

        for idx, (agent, alpha, pi) in enumerate(
            zip(server.trusted_agents, old_alpha, contrib_share)
        ):
            alpha_history_rows.append({
                "round": round_idx,
                "agent_id": agent.agent_id,
                "alpha_before": float(alpha),
                "contribution_share": float(pi),
                "alpha_after": "",
                "delta_alpha_l1": "",
            })

        for row in exclusion_reports:
            contribution_rows.append({
                "round": round_idx,
                "agent_id": row["agent_id"],
                "alpha": row["alpha"],
                "base_loss": base_loss,
                "loss_without_agent": row["loss_without_agent"],
                "marginal_contribution": row["marginal_contribution"],
                "positive_contribution": row["positive_contribution"],
                "contribution_share": float(contrib_share[row["index"]]),
            })

        if round_idx == args.max_calibration_rounds:
            return final_S, history, states, alpha_history_rows, contribution_rows

        new_alpha = (1.0 - args.alpha_update_rate) * old_alpha + (
            args.alpha_update_rate * contrib_share
        )
        new_alpha = normalize(new_alpha)
        delta = float(np.sum(np.abs(new_alpha - old_alpha)))

        for row in alpha_history_rows[-len(server.trusted_agents):]:
            agent_index = [agent.agent_id for agent in server.trusted_agents].index(row["agent_id"])
            row["alpha_after"] = float(new_alpha[agent_index])
            row["delta_alpha_l1"] = delta

        server.alphas = [float(value) for value in new_alpha]
        print(f"alpha L1 change = {delta:.6f}")

        if delta < args.alpha_tolerance:
            print("Alpha calibration converged.")
            return final_S, history, states, alpha_history_rows, contribution_rows

    return final_S, history, states, alpha_history_rows, contribution_rows


def run():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    agents = build_agents(args)
    server = HostServer(
        target_T=args.target,
        n_features=args.n_features,
        total_budget=args.total_budget,
        test_seed=args.seed + 7919,
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
    initial_coalition_ids = [agent.agent_id for agent in server.trusted_agents]
    initial_alphas = list(server.alphas)

    server.phase2_collect_proposals()
    final_S, history, states, alpha_rows, contribution_rows = calibrate_alphas(server, args)

    pruning_report = None
    if not args.disable_pruning:
        pruning_report = server.prune_negative_contributors(
            final_S,
            epsilon=args.pruning_epsilon,
            optimizer=args.optimizer,
            custom_iterations=args.custom_iterations,
        )
        final_S = pruning_report["final_solution"]
        if pruning_report["final_history"]:
            history = pruning_report["final_history"]
            states = pruning_report["final_states"]

    try:
        profit_report = server.phase4_profit_sharing(final_S)
    except ValueError as exc:
        profit_report = {"status": "error", "error": str(exc), "payments": []}

    final_error = abs(true_function(final_S) - args.target)
    final_coalition_ids = [agent.agent_id for agent in server.trusted_agents]

    summary_rows = [{
        "seed": args.seed,
        "D": args.n_features,
        "rho": args.poison_ratio,
        "target_T": args.target,
        "initial_coalition_ids": json.dumps(initial_coalition_ids),
        "final_coalition_ids": json.dumps(final_coalition_ids),
        "initial_alphas": json.dumps(initial_alphas),
        "final_alphas": json.dumps(server.alphas),
        "final_error": final_error,
        "payment_status": profit_report.get("status", ""),
        "paid_total": profit_report.get("paid_total", ""),
    }]

    payment_rows = []
    for row in profit_report.get("payments", []):
        payment_rows.append({
            "agent_id": row["agent_id"],
            "alpha": row["alpha"],
            "bid": row["bid"],
            "marginal_contribution": row["marginal_contribution"],
            "positive_contribution": row["positive_contribution"],
            "alpha_share": row["alpha_share"],
            "contribution_share": row["contribution_share"],
            "profit_share": row["profit_share"],
            "payment": row["payment"],
        })

    convergence_rows = []
    for iteration, error in enumerate(history):
        convergence_rows.append({
            "iteration": iteration,
            "target_error": error,
            "state": states[iteration] if iteration < len(states) else "",
        })

    write_csv(
        output_dir / "summary.csv",
        summary_rows,
        [
            "seed", "D", "rho", "target_T", "initial_coalition_ids",
            "final_coalition_ids", "initial_alphas", "final_alphas",
            "final_error", "payment_status", "paid_total",
        ],
    )
    write_csv(
        output_dir / "alpha_history.csv",
        alpha_rows,
        [
            "round", "agent_id", "alpha_before", "contribution_share",
            "alpha_after", "delta_alpha_l1",
        ],
    )
    write_csv(
        output_dir / "contributions.csv",
        contribution_rows,
        [
            "round", "agent_id", "alpha", "base_loss", "loss_without_agent",
            "marginal_contribution", "positive_contribution", "contribution_share",
        ],
    )
    write_csv(
        output_dir / "payments.csv",
        payment_rows,
        [
            "agent_id", "alpha", "bid", "marginal_contribution",
            "positive_contribution", "alpha_share", "contribution_share",
            "profit_share", "payment",
        ],
    )
    write_csv(
        output_dir / "convergence.csv",
        convergence_rows,
        ["iteration", "target_error", "state"],
    )

    if pruning_report is not None:
        pruning_rows = [
            {
                "round": row["round"],
                "coalition_ids": json.dumps(row["coalition_ids"]),
                "base_loss": row["base_loss"],
                "removed_agent_id": row["removed_agent_id"],
                "status": row["status"],
            }
            for row in pruning_report["pruning_log"]
        ]
        write_csv(
            output_dir / "pruning.csv",
            pruning_rows,
            ["round", "coalition_ids", "base_loss", "removed_agent_id", "status"],
        )

    print("\n=== Alpha calibration experiment complete ===")
    print(f"Initial coalition: {initial_coalition_ids}")
    print(f"Final coalition: {final_coalition_ids}")
    print(f"Initial alpha: {initial_alphas}")
    print(f"Final alpha: {server.alphas}")
    print(f"Final target error: {final_error:.6f}")
    print(f"Outputs written to: {output_dir}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Try contribution-calibrated alpha updates without modifying the original main.py."
    )
    parser.add_argument("--output-dir", default="alpha_calibration_outputs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", type=float, default=0.0)
    parser.add_argument("--num-agents", type=int, default=20)
    parser.add_argument("--poison-ratio", type=float, default=0.3)
    parser.add_argument("--n-features", type=int, default=5)
    parser.add_argument("--samples-per-agent", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-test", type=int, default=5)
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
    parser.add_argument("--max-calibration-rounds", type=int, default=3)
    parser.add_argument("--alpha-update-rate", type=float, default=0.5)
    parser.add_argument("--alpha-tolerance", type=float, default=1e-3)
    parser.add_argument("--disable-pruning", action="store_true")
    parser.add_argument("--pruning-epsilon", type=float, default=1e-6)
    return parser


if __name__ == "__main__":
    run()

import argparse
import csv
import json
import os
import random
from collections import defaultdict
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


def attach_query_counter(agents):
    counter = {
        "stage": "unassigned",
        "requests": defaultdict(int),
        "samples": defaultdict(int),
    }

    for agent in agents:
        original_api_predict = agent.api_predict

        def counted_api_predict(X_array, _original=original_api_predict):
            stage = counter["stage"]
            X_np = np.asarray(X_array)
            batch_size = 1 if X_np.ndim == 1 else len(X_np)
            counter["requests"][stage] += 1
            counter["samples"][stage] += int(batch_size)
            return _original(X_array)

        agent.api_predict = counted_api_predict

    return counter


def set_counter_stage(counter, stage):
    counter["stage"] = stage


def safe_float(value):
    if value is None:
        return ""
    try:
        if np.isfinite(value):
            return float(value)
    except TypeError:
        pass
    return ""


def write_rows(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_agents(num_agents, n_features, poison_ratio, samples_per_agent, epochs, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    loaders = generate_agent_dataloaders(
        num_agents=num_agents,
        n_samples_per_agent=samples_per_agent,
        poison_ratio=poison_ratio,
        n_features=n_features,
        seed=seed,
    )

    agents = []
    for idx, loader in enumerate(loaders):
        torch.manual_seed(seed * 1000 + idx)
        agent = AgentNode(agent_id=idx + 1, dataloader=loader, n_features=n_features)
        agent.train_local_model(epochs=epochs)
        agents.append(agent)
    return agents


def filtering_metrics(num_agents, poison_ratio, phase1_report):
    num_poisoned = int(num_agents * poison_ratio)
    poisoned_ids = set(range(num_agents - num_poisoned + 1, num_agents + 1))
    selected_ids = {row["agent_id"] for row in phase1_report if row.get("selected")}
    filtered_ids = {
        row["agent_id"]
        for row in phase1_report
        if row.get("reason") in {"failed_mse_threshold", "failed_inverse_feasibility"}
    }

    true_poisoned_filtered = len(filtered_ids & poisoned_ids)
    false_benign_filtered = len(filtered_ids - poisoned_ids)
    poisoned_selected = len(selected_ids & poisoned_ids)

    precision = (
        true_poisoned_filtered / len(filtered_ids)
        if filtered_ids else None
    )
    recall = (
        true_poisoned_filtered / len(poisoned_ids)
        if poisoned_ids else None
    )
    selected_poison_rate = (
        poisoned_selected / len(selected_ids)
        if selected_ids else None
    )

    return {
        "num_poisoned": num_poisoned,
        "poisoned_ids": sorted(poisoned_ids),
        "selected_ids": sorted(selected_ids),
        "filtered_ids": sorted(filtered_ids),
        "true_poisoned_filtered": true_poisoned_filtered,
        "false_benign_filtered": false_benign_filtered,
        "poisoned_selected": poisoned_selected,
        "filtering_precision": precision,
        "filtering_recall": recall,
        "selected_poison_rate": selected_poison_rate,
    }


def run_one_setting(args, seed, n_features, poison_ratio, target):
    agents = build_agents(
        num_agents=args.num_agents,
        n_features=n_features,
        poison_ratio=poison_ratio,
        samples_per_agent=args.samples_per_agent,
        epochs=args.epochs,
        seed=seed,
    )
    counter = attach_query_counter(agents)

    server = HostServer(
        target_T=target,
        n_features=n_features,
        total_budget=args.total_budget,
        test_seed=seed + 7919,
        n_test=args.current_blind_test_size,
    )

    set_counter_stage(counter, "stage1")
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

    set_counter_stage(counter, "stage2")
    server.phase2_collect_proposals()

    set_counter_stage(counter, "stage3")
    final_S, hist_custom, states_custom = server.phase3_custom_secant_optimization(
        num_iterations=args.custom_iterations,
        use_annealing=True,
        allow_tangent=True,
    )

    set_counter_stage(counter, "stage3_5")
    pruning_report = server.prune_negative_contributors(
        final_S,
        epsilon=args.pruning_epsilon,
        optimizer="custom",
        custom_iterations=args.custom_iterations,
    )
    final_S = pruning_report["final_solution"]
    if pruning_report["final_history"]:
        hist_custom = pruning_report["final_history"]
        states_custom = pruning_report["final_states"]

    set_counter_stage(counter, "stage4")
    try:
        profit_report = server.phase4_profit_sharing(final_S)
    except ValueError as exc:
        profit_report = {
            "status": "error",
            "error": str(exc),
            "payments": [],
            "exclusion_reports": [],
        }

    final_error = abs(true_function(final_S) - target)
    metrics = filtering_metrics(args.num_agents, poison_ratio, server.phase1_report)

    stage3_requests = counter["requests"].get("stage3", 0)
    contribution_requests = (
        counter["requests"].get("stage3_5", 0)
        + counter["requests"].get("stage4", 0)
    )
    additional_query_ratio = (
        contribution_requests / stage3_requests
        if stage3_requests else None
    )

    summary_row = {
        "seed": seed,
        "D": n_features,
        "n_test": args.current_blind_test_size,
        "rho": poison_ratio,
        "target_T": target,
        "num_agents": args.num_agents,
        "num_poisoned": metrics["num_poisoned"],
        "passed_agents": sum(
            1 for row in server.phase1_report
            if row.get("reason") in {"selected", "not_selected_by_coalition_rule"}
        ),
        "selected_agents": len(metrics["selected_ids"]),
        "final_coalition_ids": json.dumps(pruning_report["final_coalition_ids"]),
        "final_error": final_error,
        "stage1_requests": counter["requests"].get("stage1", 0),
        "stage2_requests": counter["requests"].get("stage2", 0),
        "stage3_requests": stage3_requests,
        "stage3_5_requests": counter["requests"].get("stage3_5", 0),
        "stage4_requests": counter["requests"].get("stage4", 0),
        "additional_contribution_requests": contribution_requests,
        "additional_query_ratio": safe_float(additional_query_ratio),
        "filtering_precision": safe_float(metrics["filtering_precision"]),
        "filtering_recall": safe_float(metrics["filtering_recall"]),
        "selected_poison_rate": safe_float(metrics["selected_poison_rate"]),
        "payment_status": profit_report.get("status", ""),
        "total_budget": args.total_budget,
        "minimum_bid_sum": profit_report.get("minimum_bid_sum", ""),
        "paid_total": profit_report.get("paid_total", ""),
    }

    context = {
        "seed": seed,
        "D": n_features,
        "n_test": args.current_blind_test_size,
        "rho": poison_ratio,
        "target_T": target,
    }

    stage1_rows = []
    poisoned_ids = set(metrics["poisoned_ids"])
    for row in server.phase1_report:
        stage1_rows.append({
            **context,
            "agent_id": row["agent_id"],
            "is_poisoned": row["agent_id"] in poisoned_ids,
            "mse": row.get("mse", ""),
            "bid": row.get("bid", ""),
            "inverse_loss": row.get("inverse_loss", ""),
            "inverse_feasible": row.get("inverse_feasible", ""),
            "alpha_pre": row.get("alpha_pre", ""),
            "cost_performance": row.get("cost_performance", ""),
            "diversity": row.get("diversity", ""),
            "selection_score": row.get("selection_score", ""),
            "selected": row.get("selected", False),
            "reason": row.get("reason", ""),
        })

    pruning_rows = []
    for round_log in pruning_report["pruning_log"]:
        pruning_rows.append({
            **context,
            "round": round_log["round"],
            "coalition_ids": json.dumps(round_log["coalition_ids"]),
            "base_loss": round_log["base_loss"],
            "removed_agent_id": round_log["removed_agent_id"],
            "status": round_log["status"],
        })

    payment_rows = []
    for row in profit_report.get("payments", []):
        payment_rows.append({
            **context,
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
    for iteration, error in enumerate(hist_custom):
        convergence_rows.append({
            **context,
            "method": "custom_secant_dynamic",
            "iteration": iteration,
            "target_error": error,
            "state": states_custom[iteration] if iteration < len(states_custom) else "",
        })

    return summary_row, stage1_rows, pruning_rows, payment_rows, convergence_rows


def parse_number_list(text, value_type):
    return [value_type(item.strip()) for item in text.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Run thesis experiment batches and export CSV tables."
    )
    parser.add_argument("--output-dir", default="experiment_outputs")
    parser.add_argument("--seeds", default="0", help="Comma-separated seeds, e.g. 0,1,2")
    parser.add_argument("--dimensions", default="2", help="Comma-separated D values")
    parser.add_argument("--rhos", default="0.0,0.4", help="Comma-separated poison ratios")
    parser.add_argument("--targets", default="0.0", help="Comma-separated target T values")
    parser.add_argument("--blind-test-sizes", default="5", help="Comma-separated n_test values")
    parser.add_argument("--num-agents", type=int, default=20)
    parser.add_argument("--samples-per-agent", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--custom-iterations", type=int, default=30)
    parser.add_argument("--total-budget", type=float, default=10.0)
    parser.add_argument("--mse-threshold", type=float, default=0.1)
    parser.add_argument("--inverse-target", type=float, default=0.0)
    parser.add_argument("--inverse-loss-threshold", type=float, default=0.1)
    parser.add_argument("--inverse-steps", type=int, default=500)
    parser.add_argument("--feasible-lower", type=float, default=-1.0)
    parser.add_argument("--feasible-upper", type=float, default=1.0)
    parser.add_argument(
        "--disable-inverse-check",
        action="store_true",
        help="Disable Stage 0 inverse-feasibility checking for ablation.",
    )
    parser.add_argument("--budget-fraction", type=float, default=0.8)
    parser.add_argument("--diversity-eta", type=float, default=0.5)
    parser.add_argument("--min-selection-score", type=float, default=0.0)
    parser.add_argument("--k-api", type=int, default=None)
    parser.add_argument("--k-red", type=int, default=None)
    parser.add_argument("--pruning-epsilon", type=float, default=1e-6)
    args = parser.parse_args()

    seeds = parse_number_list(args.seeds, int)
    dimensions = parse_number_list(args.dimensions, int)
    rhos = parse_number_list(args.rhos, float)
    targets = parse_number_list(args.targets, float)
    blind_test_sizes = parse_number_list(args.blind_test_sizes, int)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_fields = [
        "seed", "D", "n_test", "rho", "target_T", "num_agents", "num_poisoned",
        "passed_agents", "selected_agents", "final_coalition_ids",
        "final_error", "stage1_requests", "stage2_requests",
        "stage3_requests", "stage3_5_requests", "stage4_requests",
        "additional_contribution_requests", "additional_query_ratio",
        "filtering_precision", "filtering_recall", "selected_poison_rate",
        "payment_status", "total_budget", "minimum_bid_sum", "paid_total",
    ]
    stage1_fields = [
        "seed", "D", "n_test", "rho", "target_T", "agent_id", "is_poisoned",
        "mse", "bid", "inverse_loss", "inverse_feasible", "alpha_pre",
        "cost_performance", "diversity", "selection_score", "selected", "reason",
    ]
    pruning_fields = [
        "seed", "D", "n_test", "rho", "target_T", "round", "coalition_ids",
        "base_loss", "removed_agent_id", "status",
    ]
    payment_fields = [
        "seed", "D", "n_test", "rho", "target_T", "agent_id", "alpha", "bid",
        "marginal_contribution", "positive_contribution", "alpha_share",
        "contribution_share", "profit_share", "payment",
    ]
    convergence_fields = [
        "seed", "D", "n_test", "rho", "target_T", "method", "iteration",
        "target_error", "state",
    ]

    for seed in seeds:
        for n_features in dimensions:
            for n_test in blind_test_sizes:
                args.current_blind_test_size = n_test
                for rho in rhos:
                    for target in targets:
                        print(
                            f"\n=== Experiment seed={seed}, D={n_features}, "
                            f"n_test={n_test}, rho={rho}, T={target} ==="
                        )
                        rows = run_one_setting(args, seed, n_features, rho, target)
                        summary_row, stage1_rows, pruning_rows, payment_rows, convergence_rows = rows

                        write_rows(output_dir / "summary.csv", [summary_row], summary_fields)
                        write_rows(output_dir / "stage1_agents.csv", stage1_rows, stage1_fields)
                        write_rows(output_dir / "pruning.csv", pruning_rows, pruning_fields)
                        write_rows(output_dir / "payments.csv", payment_rows, payment_fields)
                        write_rows(output_dir / "convergence.csv", convergence_rows, convergence_fields)

    print(f"\nDone. CSV outputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

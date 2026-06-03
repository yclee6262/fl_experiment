import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent_client import AgentNode
from dataset import generate_agent_dataloaders
from host_server import HostServer


def true_function(S):
    S = np.asarray(S, dtype=float)
    y = np.sum(S)
    if len(S) > 1:
        y += np.sum(S[:-1] * S[1:])
    return float(y)


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


METHOD_LABELS = {
    "candidate_mean": "Candidate mean",
    "best_single_candidate": "Best single candidate",
    "bfgs": "ZO-BFGS",
    "secant_only": "Secant only",
    "dynamic_only": "Dynamic/Tangent only",
    "secant_with_annealing": "Secant + annealing",
    "dynamic_with_annealing": "Dynamic/Tangent + annealing",
    "adaptive_secant_dynamic_annealing": "Adaptive secant/dynamic",
    "current_dual_engine_bfgs_plus_adaptive": "Current dual engine",
}


STATE_STYLES = {
    "start": ("Start", "#6b7280", "o"),
    "bfgs": ("ZO-BFGS", "#1f77b4", "s"),
    "secant": ("Secant", "#ff7f0e", "o"),
    "secant_annealed": ("Secant + annealing", "#d62728", "^"),
    "switch": ("Switch to dynamic", "#9467bd", "D"),
    "dynamic": ("Dynamic/Tangent", "#2ca02c", "*"),
    "dynamic_annealed": ("Dynamic/Tangent + annealing", "#16a34a", "P"),
    "stopped": ("Stopped", "#111827", "x"),
}


def weighted_consensus_loss(server, S_array):
    total_loss = 0.0
    for agent, alpha in zip(server.trusted_agents, server.alphas):
        pred = agent.api_predict(S_array)[0]
        total_loss += alpha * abs(pred - server.target_T)
    return float(total_loss)


def attach_query_counter(agents):
    counter = {"requests": 0, "samples": 0}
    originals = []

    for agent in agents:
        original_api_predict = agent.api_predict
        originals.append((agent, original_api_predict))

        def counted_api_predict(X_array, _original=original_api_predict):
            X_np = np.asarray(X_array)
            batch_size = 1 if X_np.ndim == 1 else len(X_np)
            counter["requests"] += 1
            counter["samples"] += int(batch_size)
            return _original(X_array)

        agent.api_predict = counted_api_predict

    return counter, originals


def restore_api_predict(originals):
    for agent, original_api_predict in originals:
        agent.api_predict = original_api_predict


def run_with_query_count(server, runner):
    counter, originals = attach_query_counter(server.trusted_agents)
    try:
        result = runner()
    finally:
        restore_api_predict(originals)
    return result, counter


def run_candidate_mean(server):
    I_matrix = np.asarray(server.I_list)
    final_S = np.mean(I_matrix, axis=0)
    return {
        "solution": final_S,
        "history": [abs(true_function(final_S) - server.target_T)],
        "states": ["mean"],
    }


def run_best_single_candidate(server):
    best_S = None
    best_loss = float("inf")
    best_idx = None

    for idx, candidate in enumerate(server.I_list):
        loss = weighted_consensus_loss(server, candidate)
        if loss < best_loss:
            best_loss = loss
            best_S = np.asarray(candidate).copy()
            best_idx = idx

    return {
        "solution": best_S,
        "history": [abs(true_function(best_S) - server.target_T)],
        "states": [f"agent_{server.trusted_agents[best_idx].agent_id}"],
    }


def run_bfgs(server):
    final_S, history = server.phase3_global_optimization()
    if not history:
        history = [abs(true_function(final_S) - server.target_T)]
    return {
        "solution": final_S,
        "history": history,
        "states": ["bfgs"] * len(history),
    }


def run_directional_optimizer(
    server,
    *,
    num_iterations,
    initial_method,
    use_annealing,
    allow_switch,
    label,
):
    I_matrix = np.asarray(server.I_list)
    S_current = np.mean(I_matrix, axis=0)
    n_agents = len(server.trusted_agents)

    def evaluate_S(S_array):
        return weighted_consensus_loss(server, S_array)

    loss_anchors = [evaluate_S(I_i) for I_i in server.I_list]
    best_loss = evaluate_S(S_current)
    global_best_S = S_current.copy()
    global_best_loss = best_loss

    eta = 0.1
    delta = 1e-4
    current_method = initial_method

    history = [abs(true_function(S_current) - server.target_T)]
    states = ["start"]

    for iteration in range(num_iterations):
        grad_S = np.zeros_like(S_current)

        for i in range(n_agents):
            direction = np.asarray(server.I_list[i]) - S_current
            dist = np.linalg.norm(direction)
            if dist < 1e-8:
                continue

            unit_dir = direction / dist
            if current_method == "secant":
                deriv = (loss_anchors[i] - best_loss) / dist
            elif current_method == "dynamic":
                S_perturb = S_current + delta * unit_dir
                loss_p = evaluate_S(S_perturb)
                deriv = (loss_p - best_loss) / delta
            else:
                raise ValueError(f"Unsupported directional method: {current_method}")

            grad_S += server.alphas[i] * deriv * unit_dir

        grad_norm = np.linalg.norm(grad_S)
        if grad_norm <= 1e-8:
            states.append(f"{current_method}_zero_gradient")
            history.append(abs(true_function(S_current) - server.target_T))
            break

        grad_S = grad_S / grad_norm
        current_eta = eta
        success = False
        state_this_iter = current_method

        if use_annealing:
            for attempt in range(10):
                S_try = S_current - current_eta * grad_S
                try_loss = evaluate_S(S_try)
                if try_loss < best_loss:
                    S_current = S_try
                    best_loss = try_loss
                    success = True
                    if attempt > 0:
                        state_this_iter = f"{current_method}_annealed"
                    else:
                        state_this_iter = current_method
                    eta = min(0.5, current_eta * 1.5)
                    break
                current_eta *= 0.5
        else:
            S_try = S_current - current_eta * grad_S
            try_loss = evaluate_S(S_try)
            if try_loss < best_loss:
                S_current = S_try
                best_loss = try_loss
                success = True

        if best_loss < global_best_loss:
            global_best_loss = best_loss
            global_best_S = S_current.copy()

        if not success:
            if allow_switch and current_method == "secant":
                current_method = "dynamic"
                eta = 0.5
                state_this_iter = "switch_to_dynamic"
            else:
                states.append(f"{current_method}_stopped")
                history.append(abs(true_function(S_current) - server.target_T))
                break

        history.append(abs(true_function(S_current) - server.target_T))
        states.append(state_this_iter)

    return {
        "solution": global_best_S,
        "history": history,
        "states": states,
        "label": label,
    }


def run_adaptive_dual_engine(server, num_iterations):
    bfgs_result, bfgs_counter = run_with_query_count(server, lambda: run_bfgs(server))
    adaptive_result, adaptive_counter = run_with_query_count(
        server,
        lambda: run_directional_optimizer(
            server,
            num_iterations=num_iterations,
            initial_method="secant",
            use_annealing=True,
            allow_switch=True,
            label="adaptive_secant_dynamic_annealing",
        ),
    )

    bfgs_loss = weighted_consensus_loss(server, bfgs_result["solution"])
    adaptive_loss = weighted_consensus_loss(server, adaptive_result["solution"])

    if bfgs_loss <= adaptive_loss:
        chosen = bfgs_result
        chosen["states"] = [f"dual_bfgs_{state}" for state in chosen["states"]]
        chosen_engine = "bfgs"
        consensus_loss = bfgs_loss
    else:
        chosen = adaptive_result
        chosen["states"] = [f"dual_adaptive_{state}" for state in chosen["states"]]
        chosen_engine = "adaptive"
        consensus_loss = adaptive_loss

    chosen["chosen_engine"] = chosen_engine
    chosen["consensus_loss_override"] = consensus_loss
    return chosen, {
        "requests": bfgs_counter["requests"] + adaptive_counter["requests"] + 2 * len(server.trusted_agents),
        "samples": bfgs_counter["samples"] + adaptive_counter["samples"] + 2 * len(server.trusted_agents),
    }


def build_server(args, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    loaders = generate_agent_dataloaders(
        num_agents=args.num_agents,
        n_samples_per_agent=args.samples_per_agent,
        poison_ratio=args.poison_ratio,
        n_features=args.n_features,
        seed=seed,
    )

    all_agents = []
    for idx, loader in enumerate(loaders):
        torch.manual_seed(seed * 1000 + idx)
        agent = AgentNode(agent_id=idx + 1, dataloader=loader, n_features=args.n_features)
        agent.train_local_model(epochs=args.epochs)
        all_agents.append(agent)

    server = HostServer(
        target_T=args.target,
        n_features=args.n_features,
        total_budget=args.total_budget,
        test_seed=seed,
        n_test=args.n_test,
    )
    server.phase1_filter_agents(
        all_agents,
        mse_threshold=args.mse_threshold,
        budget_fraction=args.budget_fraction,
        diversity_eta=args.diversity_eta,
        k_api=args.k_api,
        k_red=args.k_red,
    )
    server.phase2_collect_proposals()
    return server


def summarize_result(args, seed, server, method_name, result, counter, consensus_loss=None):
    final_S = np.asarray(result["solution"], dtype=float)
    if consensus_loss is None:
        consensus_loss = weighted_consensus_loss(server, final_S)
        counter["requests"] += len(server.trusted_agents)
        counter["samples"] += len(server.trusted_agents)

    state_counts = Counter(result["states"])
    return {
        "seed": seed,
        "D": args.n_features,
        "rho": args.poison_ratio,
        "target_T": args.target,
        "method": method_name,
        "selected_agents": len(server.trusted_agents),
        "selected_agent_ids": json.dumps([agent.agent_id for agent in server.trusted_agents]),
        "final_error": abs(true_function(final_S) - args.target),
        "consensus_loss": consensus_loss,
        "query_requests": counter["requests"],
        "query_samples": counter["samples"],
        "iterations": max(0, len(result["history"]) - 1),
        "state_counts": json.dumps(dict(state_counts), sort_keys=True),
        "chosen_engine": result.get("chosen_engine", ""),
        "final_solution": json.dumps([float(x) for x in final_S]),
    }


def convergence_rows(args, seed, method_name, result):
    rows = []
    for iteration, error in enumerate(result["history"]):
        rows.append({
            "seed": seed,
            "D": args.n_features,
            "rho": args.poison_ratio,
            "target_T": args.target,
            "method": method_name,
            "iteration": iteration,
            "target_error": error,
            "state": result["states"][iteration] if iteration < len(result["states"]) else "",
        })
    return rows


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_convergence(convergence, plotted_methods):
    by_method_seed = {}
    for row in convergence:
        if row["method"] not in plotted_methods:
            continue
        key = (row["method"], row["seed"])
        by_method_seed.setdefault(key, []).append(row)

    histories = {}
    for (method, seed), rows in by_method_seed.items():
        rows = sorted(rows, key=lambda item: item["iteration"])
        histories.setdefault(method, {})[seed] = [row["target_error"] for row in rows]

    aggregated = {}
    for method, seed_histories in histories.items():
        max_len = max(len(history) for history in seed_histories.values())
        mean_values = []
        for index in range(max_len):
            values = []
            for history in seed_histories.values():
                values.append(history[index] if index < len(history) else history[-1])
            mean_values.append(float(np.mean(values)))
        aggregated[method] = mean_values
    return aggregated


def plot_convergence(output_dir, convergence, plotted_methods):
    aggregated = aggregate_convergence(convergence, plotted_methods)

    plt.figure(figsize=(11, 6))
    for method in plotted_methods:
        if method not in aggregated:
            continue
        values = aggregated[method]
        plt.plot(
            range(len(values)),
            values,
            marker="o",
            linewidth=1.5,
            label=METHOD_LABELS.get(method, method),
        )

    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Mean target error |g(S)-T|")
    plt.title("Stage 3 Optimizer Ablation (mean across seeds)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "optimizer_ablation_convergence.png", dpi=300)
    plt.close()


def state_category(raw_state):
    state = str(raw_state).lower()
    if "start" in state:
        return "start"
    if "bfgs" in state:
        return "bfgs"
    if "switch" in state:
        return "switch"
    if "dynamic" in state and "anneal" in state:
        return "dynamic_annealed"
    if "secant" in state and "anneal" in state:
        return "secant_annealed"
    if "dynamic" in state:
        return "dynamic"
    if "secant" in state:
        return "secant"
    if "stop" in state or "zero_gradient" in state:
        return "stopped"
    return "stopped"


def plot_adaptive_engine_state_transition(output_dir, convergence, seed):
    method = "adaptive_secant_dynamic_annealing"
    rows = [
        row for row in convergence
        if row["method"] == method and row["seed"] == seed
    ]
    if not rows:
        rows = [row for row in convergence if row["method"] == method]
    if not rows:
        return

    rows = sorted(rows, key=lambda item: (item["seed"], item["iteration"]))
    selected_seed = rows[0]["seed"]
    rows = [row for row in rows if row["seed"] == selected_seed]

    iterations = [row["iteration"] for row in rows]
    errors = [row["target_error"] for row in rows]

    plt.figure(figsize=(10, 5.8))
    plt.plot(iterations, errors, color="#4b5563", linewidth=1.4, alpha=0.75)

    used_labels = set()
    for row in rows:
        category = state_category(row["state"])
        label, color, marker = STATE_STYLES[category]
        legend_label = label if label not in used_labels else None
        used_labels.add(label)
        plt.scatter(
            row["iteration"],
            row["target_error"],
            color=color,
            marker=marker,
            s=90 if marker != "*" else 140,
            edgecolors="white" if marker not in {"x", "*"} else color,
            linewidths=0.8,
            label=legend_label,
            zorder=3,
        )

    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Target error |g(S)-T|")
    plt.title(f"Adaptive Secant/Dynamic Engine State Transition (seed={selected_seed})")
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "adaptive_engine_state_transition.png", dpi=300)
    plt.close()


def run_ablation_study(args):
    output_dir = Path(args.output_dir)
    summary_rows = []
    convergence = []

    for seed in parse_int_list(args.seeds):
        print(f"\n=== Ablation seed={seed}, D={args.n_features}, rho={args.poison_ratio}, T={args.target} ===")
        server = build_server(args, seed)

        method_specs = [
            ("candidate_mean", lambda: run_candidate_mean(server)),
            ("best_single_candidate", lambda: run_best_single_candidate(server)),
            ("bfgs", lambda: run_bfgs(server)),
            (
                "secant_only",
                lambda: run_directional_optimizer(
                    server,
                    num_iterations=args.iterations,
                    initial_method="secant",
                    use_annealing=False,
                    allow_switch=False,
                    label="secant_only",
                ),
            ),
            (
                "dynamic_only",
                lambda: run_directional_optimizer(
                    server,
                    num_iterations=args.iterations,
                    initial_method="dynamic",
                    use_annealing=False,
                    allow_switch=False,
                    label="dynamic_only",
                ),
            ),
            (
                "secant_with_annealing",
                lambda: run_directional_optimizer(
                    server,
                    num_iterations=args.iterations,
                    initial_method="secant",
                    use_annealing=True,
                    allow_switch=False,
                    label="secant_with_annealing",
                ),
            ),
            (
                "dynamic_with_annealing",
                lambda: run_directional_optimizer(
                    server,
                    num_iterations=args.iterations,
                    initial_method="dynamic",
                    use_annealing=True,
                    allow_switch=False,
                    label="dynamic_with_annealing",
                ),
            ),
            (
                "adaptive_secant_dynamic_annealing",
                lambda: run_directional_optimizer(
                    server,
                    num_iterations=args.iterations,
                    initial_method="secant",
                    use_annealing=True,
                    allow_switch=True,
                    label="adaptive_secant_dynamic_annealing",
                ),
            ),
        ]

        for method_name, runner in method_specs:
            result, counter = run_with_query_count(server, runner)
            summary_rows.append(summarize_result(args, seed, server, method_name, result, counter))
            convergence.extend(convergence_rows(args, seed, method_name, result))

        dual_result, dual_counter = run_adaptive_dual_engine(server, args.iterations)
        summary_rows.append(
            summarize_result(
                args,
                seed,
                server,
                "current_dual_engine_bfgs_plus_adaptive",
                dual_result,
                dual_counter,
                consensus_loss=dual_result["consensus_loss_override"],
            )
        )
        convergence.extend(
            convergence_rows(args, seed, "current_dual_engine_bfgs_plus_adaptive", dual_result)
        )

    summary_fields = [
        "seed", "D", "rho", "target_T", "method", "selected_agents", "selected_agent_ids",
        "final_error", "consensus_loss", "query_requests", "query_samples", "iterations",
        "state_counts", "chosen_engine", "final_solution",
    ]
    convergence_fields = [
        "seed", "D", "rho", "target_T", "method", "iteration", "target_error", "state",
    ]

    write_csv(output_dir / "summary.csv", summary_rows, summary_fields)
    write_csv(output_dir / "convergence.csv", convergence, convergence_fields)
    plotted_methods = parse_str_list(args.plot_methods)
    plot_convergence(output_dir, convergence, plotted_methods)
    state_plot_seed = parse_int_list(args.seeds)[0] if args.state_plot_seed is None else args.state_plot_seed
    plot_adaptive_engine_state_transition(output_dir, convergence, state_plot_seed)

    print(f"\nAblation outputs written to: {output_dir}")
    print(f"- {output_dir / 'summary.csv'}")
    print(f"- {output_dir / 'convergence.csv'}")
    print(f"- {output_dir / 'optimizer_ablation_convergence.png'}")
    print(f"- {output_dir / 'adaptive_engine_state_transition.png'}")
    print(f"Plotted methods: {', '.join(plotted_methods)}")


def build_parser():
    parser = argparse.ArgumentParser(description="Stage 3 dual-engine optimizer ablation.")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--target", type=float, default=0)
    parser.add_argument("--num-agents", type=int, default=20)
    parser.add_argument("--poison-ratio", type=float, default=0.4)
    parser.add_argument("--n-features", type=int, default=5)
    parser.add_argument("--samples-per-agent", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-test", type=int, default=5)
    parser.add_argument("--total-budget", type=float, default=10.0)
    parser.add_argument("--mse-threshold", type=float, default=0.1)
    parser.add_argument("--budget-fraction", type=float, default=0.8)
    parser.add_argument("--diversity-eta", type=float, default=0.5)
    parser.add_argument("--k-api", type=int, default=None)
    parser.add_argument("--k-red", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output-dir", default="ablation_outputs/dual_engine")
    parser.add_argument(
        "--state-plot-seed",
        type=int,
        default=None,
        help="Seed to visualize for the adaptive secant/dynamic engine state transition plot.",
    )
    parser.add_argument(
        "--plot-methods",
        default=(
            "bfgs,"
            "secant_only,"
            "dynamic_only,"
            "secant_with_annealing,"
            "current_dual_engine_bfgs_plus_adaptive"
        ),
        help="Comma-separated methods to include in the convergence plot.",
    )
    return parser


if __name__ == "__main__":
    run_ablation_study(build_parser().parse_args())

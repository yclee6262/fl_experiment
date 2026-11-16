"""Small configurable numerical FedAvg and Method 5 execution comparison."""
import argparse
import contextlib
import copy
import json
from pathlib import Path
import numpy as np
import torch
from run_experiments import build_agents, true_function
from host_server import HostServer
from core.pipeline import solve_decision
import fl_comparison_numerical as baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dimensions", type=int, default=2)
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--poison-ratio", type=float, default=0.)
    parser.add_argument("--mse-threshold", type=float, default=.1)
    parser.add_argument("--disable-inverse-check", action="store_true")
    args = parser.parse_args()
    if min(args.rounds, args.samples, args.agents, args.dimensions) < 1:
        parser.error("rounds, samples, agents and dimensions must be positive")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(json.dumps({"status": "running", **vars(args)}))
    with (output / "execution.log").open("w") as log, contextlib.redirect_stdout(log):
        # Preserve the same dataset and one local training epoch per FL round.
        agents = build_agents(num_agents=args.agents, n_features=args.dimensions,
                              poison_ratio=args.poison_ratio, samples_per_agent=args.samples,
                              epochs=args.rounds, seed=args.seed)
        loaders = [copy.deepcopy(a.dataloader) for a in agents]
        baseline.N_FEATURES = args.dimensions
        baseline.N_AGENTS = args.agents
        baseline.NUM_ROUNDS = args.rounds
        baseline.NUM_POISONED = int(args.agents * args.poison_ratio)
        baseline.TARGET_T = 0.
        torch.manual_seed(args.seed)
        errors = baseline.run_traditional_fl_baseline(loaders, "fedavg")
        server = HostServer(0., args.dimensions, constraint_mode="box_constrained")
        result = solve_decision(server, agents, qualification_options={
            "mse_threshold": args.mse_threshold,
            "enable_inverse_check": not args.disable_inverse_check})
        solution = np.asarray(result["solution"])
        if not np.isfinite(solution).all() or server.box_violation(solution) > 1e-8:
            raise ValueError("Invalid Method 5 decision")
    report = {"status": "complete", "arguments": vars(args), "fedavg_target_error_by_round": errors,
              "method5_solution": solution.tolist(), "method5_target_error": float(abs(true_function(solution))),
              "limitations": "Execution comparison using synthetic truth, not a validated fairness or communication-cost benchmark. Model initialization, admission and solver budgets differ; rounds are not API iterations."}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps({"status": "complete", **vars(args)}))


if __name__ == "__main__":
    main()

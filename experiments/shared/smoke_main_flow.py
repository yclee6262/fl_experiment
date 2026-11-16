"""Small synthetic integration check, not a research-quality experiment."""
import argparse
import json
from pathlib import Path

import numpy as np

from core.pipeline import solve_decision
from host_server import HostServer
from run_experiments import build_agents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    agents = build_agents(num_agents=3, n_features=2, poison_ratio=0.0,
                          samples_per_agent=32, epochs=1, seed=42)
    server = HostServer(target_T=0.0, n_features=2, n_test=5,
                        constraint_mode="box_constrained")
    result = solve_decision(server, agents, custom_iterations=3,
                            qualification_options={"mse_threshold": 1e6,
                                                   "enable_inverse_check": False})
    solution = np.asarray(result["solution"], dtype=float)
    if solution.shape != (2,) or not np.all(np.isfinite(solution)):
        raise AssertionError("Non-finite or malformed decision")
    if np.any(solution < -1.0 - 1e-8) or np.any(solution > 1.0 + 1e-8):
        raise AssertionError("Decision outside configured bounds")
    report = {"status": "passed", "seed": 42, "solution": solution.tolist(),
              "scope": "synthetic integration only; relaxed qualification; no pruning or payment"}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

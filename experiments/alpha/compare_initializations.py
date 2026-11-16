"""Replay the eight 1112 initializations on saved, qualified models."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from four_alpha_initialization import initializer
from pure_es_initialization import initializer as pure_initializer
import oob_shrinkage_research as shared
from research_evaluation import load_evaluator_artifact, score_solution

METHODS = {"U": "A_uniform", "Q": "B_stage0", "E": "stacked_full",
           "S": "pure_support", "EU": "ensemble_shrink", "EQ": "B_stacked_half",
           "SU": "peer_support_shrink", "SQ": "support_quality_mix50"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--scenarios", default=",".join(shared.SCENARIOS))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    shared.dump(output / "manifest.json", {"status": "running", **vars(args), "methods": METHODS})
    original = shared.initial_alpha
    try:
        for scenario in args.scenarios.split(","):
            for seed in map(int, args.seeds.split(",")):
                server, item, artifact = shared.load_base(args.source, scenario, seed)
                predictions = np.column_stack([a.api_predict(server.test_X) for a in server.trusted_agents])
                mse = ((predictions - server.test_y[:, None]) ** 2).mean(axis=0)
                peer = np.full((len(server.I_list), len(server.I_list)), np.nan)
                for j, agent in enumerate(server.trusted_agents):
                    for i, point in enumerate(server.I_list):
                        if i != j:
                            peer[j, i] = float(np.asarray(agent.api_predict(point)).reshape(-1)[0])
                # Keep each method's solve independent; external scoring starts only after all decisions are locked.
                decisions = []
                for label, method in METHODS.items():
                    shared.initial_alpha = pure_initializer(peer) if label in ("E", "S") else initializer(peer)
                    row = shared.solve(copy.deepcopy(server), method, mse, server.test_X, server.test_y, predictions, seed, 1000)
                    row.update(method=method, method_label=label, seed=seed, scenario=scenario)
                    shared.append(output / "decisions.jsonl", row)
                    decisions.append(row)
                evaluator, _ = load_evaluator_artifact(artifact)
                for row in decisions:
                    if row["status"] == "valid":
                        _, row["pooled_error"] = score_solution(evaluator, np.asarray(row["X"]), 0.)
                    shared.append(output / "results.jsonl", row)
        shared.dump(output / "manifest.json", {"status": "complete", **vars(args), "methods": METHODS,
                    "limitations": "Historical model replay, not new independent confirmation. Shared blind/support preparation excluded from per-method solver costs."})
    finally:
        shared.initial_alpha = original


if __name__ == "__main__":
    main()

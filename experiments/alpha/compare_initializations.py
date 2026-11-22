"""Replay all ten fixed alpha initializations on saved, qualified models."""
import argparse
import copy
import json
import hashlib
import subprocess
from pathlib import Path
import numpy as np
from four_alpha_initialization import initializer
from pure_es_initialization import initializer as pure_initializer
from se_qu_initialization import initializer as pair_initializer
import oob_shrinkage_research as shared
from research_evaluation import load_evaluator_artifact, score_solution
from experiments.shared.solver_policy import apply_solver_policy, POLICIES, POLICY_PATH

METHODS = {"U": "A_uniform", "Q": "B_stage0", "E": "stacked_full",
           "S": "pure_support", "EU": "ensemble_shrink", "EQ": "B_stacked_half",
           "SU": "peer_support_shrink", "SQ": "support_quality_mix50",
           "SE": "support_ensemble_mix50", "QU": "quality_shrink"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--scenarios", default=",".join(shared.SCENARIOS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--solver-policy", choices=POLICIES, default="historical")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    provenance = {"solver_policy_definition": json.loads(POLICY_PATH.read_text()),
                  "solver_policy_sha256": hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest(),
                  "git_head": subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    shared.dump(output / "manifest.json", {"status": "running", **vars(args), "methods": METHODS, **provenance})
    original = shared.initial_alpha
    try:
        for scenario in args.scenarios.split(","):
            for seed in map(int, args.seeds.split(",")):
                server, item, artifact = shared.load_base(args.source, scenario, seed)
                policy = apply_solver_policy(server, args.solver_policy)
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
                    shared.initial_alpha = (pure_initializer(peer) if label in ("E", "S")
                                            else pair_initializer(peer) if label in ("SE", "QU")
                                            else initializer(peer))
                    row = shared.solve(copy.deepcopy(server), method, mse, server.test_X, server.test_y, predictions, seed, 1000)
                    row.update(method=method, method_label=label, seed=seed, scenario=scenario, solver_policy=policy)
                    shared.append(output / "decisions.jsonl", row)
                    decisions.append(row)
                evaluator, _ = load_evaluator_artifact(artifact)
                for row in decisions:
                    if row["status"] == "valid":
                        _, row["pooled_error"] = score_solution(evaluator, np.asarray(row["X"]), 0.)
                    shared.append(output / "results.jsonl", row)
        shared.dump(output / "manifest.json", {"status": "complete", **vars(args), "methods": METHODS, **provenance,
                    "limitations": "Historical model replay, not new independent confirmation. Shared blind/support preparation excluded from per-method solver costs."})
    finally:
        shared.initial_alpha = original


if __name__ == "__main__":
    main()

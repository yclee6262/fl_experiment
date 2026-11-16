"""Replay marginal-alpha candidates with exact per-solve API caching.

Pooled scores are used only after each cached Stage 3 solution is produced.
The source run, preregistration, and seed history determine whether a replay
is retrospective development or part of an independent formal comparison.
"""

import argparse
import contextlib
import csv
import io
import json
from pathlib import Path

import numpy as np
import torch

from agent_client import AgentNode
from alpha_weight_experiments import attach_query_counter, run_stage3
from host_server import HostServer
from research_evaluation import load_evaluator_artifact, score_solution


def _rows(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = [(int(row["seed"]), row["method"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate seed/method in {path}")
    return dict(zip(keys, rows))


def replay(root, seeds, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((root/"suite_manifest.json").read_text())
    if (manifest["status"] != "complete"
            or len(manifest["scenarios"]) != 1
            or not set(seeds) <= set(manifest["seeds"])
            or not manifest.get("marginal_replacement")):
        raise ValueError("Need complete marginal-replacement study")
    scenario = manifest["scenarios"][0]
    directory = root/scenario
    settings = manifest["arguments"]
    shared = {int(row["seed"]): row for row in map(
        json.loads, (directory/"shared_costs.jsonl").read_text().splitlines())}
    candidates = _rows(directory/"candidate_rows.csv")
    summaries = _rows(directory/"summary.csv")
    methods = manifest["candidate_initializations"]
    target = float(settings["targets"][0])
    output_rows = []
    def persist():
        output.write_text(json.dumps({
            "study": "marginal_alpha_exact_stage3_cache_replay",
            "scope": "saved_model_per_candidate_exact_prediction_cache",
            "status": "complete" if len(output_rows) == len(seeds) else "running",
            "source_root": str(root), "scenario": scenario,
            "seeds": seeds, "completed_seeds": len(output_rows),
            "rows": output_rows,
        }, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    for seed in seeds:
        item = shared[seed]
        if item.get("missing_fixed_ids"):
            output_rows.append({
                "scenario": scenario, "seed": seed,
                "status": "qualification_failure",
                "missing_fixed_ids": item["missing_fixed_ids"],
                "preparation": item["preparation"],
                "candidates": {}, "portfolios": {},
            })
            persist()
            continue
        bad = []
        for method in methods:
            candidate = candidates.get((seed, method))
            if candidate is None or candidate["delivery_status"] != "valid":
                bad.append({"method": method,
                            "status": candidate["status"] if candidate else "missing",
                            "failure_stage": (candidate["failure_stage"]
                                              if candidate else "missing"),
                            "failure_reason": (candidate["failure_reason"]
                                               if candidate else "missing")})
        if bad:
            output_rows.append({"scenario": scenario, "seed": seed,
                                "status": "candidate_failure",
                                "failed_candidates": bad,
                                "preparation": item["preparation"],
                                "candidates": {}, "portfolios": {}})
            persist()
            continue
        artifact = directory/item["artifact_path"]
        payload = torch.load(artifact/"local_models.pt", map_location="cpu",
                             weights_only=True)
        agents = []
        for ident in item["coalition_ids"]:
            agent = AgentNode(ident, None, n_features=payload["n_features"])
            agent.model.load_state_dict(payload["agents"][ident])
            agent.model.eval()
            agents.append(agent)
        evaluator, _ = load_evaluator_artifact(artifact)
        server = HostServer(target, payload["n_features"],
                            constraint_mode=settings["constraint_mode"],
                            feasible_lower=settings["feasible_lower"],
                            feasible_upper=settings["feasible_upper"])
        server.trusted_agents = agents
        server.I_list = [np.asarray(p, dtype=float) for p in item["proposals"]]
        replayed = {}
        for method in methods:
            old = candidates[seed, method]
            if old["delivery_status"] != "valid":
                raise ValueError(f"Invalid original candidate: {seed}/{method}")
            server.alphas = json.loads(old["initial_alpha"])
            counter = attach_query_counter(agents)
            counter["stage"] = "stage3"
            with contextlib.redirect_stdout(io.StringIO()):
                solution, _, _ = run_stage3(
                    server, settings["optimizer"],
                    settings["custom_iterations"], cache_predictions=True)
            counter["stage"] = "finished"
            saved_x = np.asarray(json.loads(old["final_solution"]), dtype=float)
            max_x_diff = float(np.max(np.abs(solution-saved_x)))
            violation = float(server.box_violation(solution))
            _, pooled_error = score_solution(evaluator, solution, target)
            pooled_diff = abs(pooled_error-float(old["external_global_error"]))
            actual_solver = server.last_subspace_optimization["actual_solver"]
            if (max_x_diff > 1e-7 or pooled_diff > 1e-7 or violation > 1e-8
                    or actual_solver != old["actual_solver"]):
                raise ValueError(f"Cached solve changed delivery: {seed}/{method}")
            replayed[method] = {
                "raw_stage3_requests": int(float(old["stage3_requests"])),
                "cached_stage3_requests": counter["requests"]["stage3"],
                "raw_stage3_input_samples": int(float(old["stage3_input_samples"])),
                "cached_stage3_input_samples": counter["input_samples"]["stage3"],
                "cache_hits": server.last_stage3_cache["hits"],
                "max_solution_diff": max_x_diff,
                "pooled_score_diff": pooled_diff,
                "max_box_violation": violation,
                "actual_solver": actual_solver,
            }
        portfolios = {}
        for portfolio in ("validation_calibrated_multistart",
                          "marginal_information_portfolio"):
            saved = summaries[seed, portfolio]
            order = json.loads(saved["alpha_audit"])["candidate_method_order"]
            cached_stage3 = sum(replayed[name]["cached_stage3_requests"]
                                for name in order)
            raw_stage3 = sum(replayed[name]["raw_stage3_requests"]
                             for name in order)
            cached_samples = sum(replayed[name]["cached_stage3_input_samples"]
                                 for name in order)
            raw_samples = sum(replayed[name]["raw_stage3_input_samples"]
                              for name in order)
            if abs(raw_stage3-float(saved["stage3_requests"])) > 0:
                raise ValueError(f"Portfolio request attribution differs: {seed}")
            if abs(raw_samples-float(saved["stage3_input_samples"])) > 0:
                raise ValueError(f"Portfolio sample attribution differs: {seed}")
            initialization = float(saved["initialization_requests"])
            initialization_samples = float(saved["initialization_input_samples"])
            portfolios[portfolio] = {
                "selected_initialization": json.loads(saved["alpha_audit"])[
                    "selected_initialization"],
                "pooled_error": float(saved["external_global_error"]),
                "initialization_and_selection_requests": initialization,
                "raw_stage3_requests": raw_stage3,
                "cached_stage3_requests": cached_stage3,
                "raw_method_requests": initialization+raw_stage3,
                "cached_method_requests": initialization+cached_stage3,
                "initialization_and_selection_input_samples": initialization_samples,
                "raw_stage3_input_samples": raw_samples,
                "cached_stage3_input_samples": cached_samples,
                "raw_method_input_samples": initialization_samples+raw_samples,
                "cached_method_input_samples": initialization_samples+cached_samples,
            }
        output_rows.append({"scenario": scenario, "seed": seed,
                            "status": "valid",
                            "candidates": replayed, "portfolios": portfolios})
        persist()
    return output_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--output", required=True)
    options = parser.parse_args()
    seeds = [int(x) for x in options.seeds.split(",")]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be nonempty and unique")
    rows = replay(options.root, seeds, options.output)
    print(json.dumps({"completed_seeds": len(rows),
                      "candidate_runs": sum(len(r["candidates"]) for r in rows),
                      "failures": sum(r["status"] != "valid" for r in rows)},
                     indent=2))


if __name__ == "__main__":
    main()

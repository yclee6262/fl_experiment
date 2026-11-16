"""Replay saved Stage 3 candidates and measure exact local-model query reuse.

This diagnostic does not modify the production runner or count pooled scoring.
Cache keys use the exact input shape and float32 bytes seen by AgentNode.
"""

import argparse
import contextlib
import csv
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from agent_client import AgentNode
from alpha_weight_experiments import run_stage3
from host_server import HostServer


def _key(values, key_mode):
    x = np.asarray(values, dtype=np.float32 if key_mode == "model_float32" else None)
    return x.dtype.str, x.shape, x.tobytes()


def _agents(payload, identifiers, mode, seen, counters, key_mode):
    agents = []
    for ident in identifiers:
        agent = AgentNode(ident, None, n_features=payload["n_features"])
        agent.model.load_state_dict(payload["agents"][ident])
        agent.model.eval()
        original = agent.api_predict

        def queried(x, _ident=ident, _original=original):
            key = (_ident, _key(x, key_mode))
            counters["requests"] += 1
            if key in seen:
                counters["duplicate_requests"] += 1
                if mode == "cache":
                    return seen[key].copy()
            output = _original(x)
            if key not in seen:
                seen[key] = output.copy()
                counters["unique_requests"] += 1
            return output

        agent.api_predict = queried
        agents.append(agent)
    return agents


def _run(payload, shared, row, mode, seen, key_mode):
    counters = defaultdict(int)
    server = HostServer(0.0, payload["n_features"], n_test=20,
                        constraint_mode="box_constrained")
    server.trusted_agents = _agents(payload, shared["coalition_ids"],
                                    mode, seen, counters, key_mode)
    server.I_list = [np.asarray(p, dtype=float) for p in shared["proposals"]]
    server.alphas = json.loads(row["initial_alpha"])
    with contextlib.redirect_stdout(io.StringIO()):
        solution, _, _ = run_stage3(server, "best_of", 30)
    return solution, dict(counters), server.last_subspace_optimization["chosen_engine"]


def analyze(root, seeds, key_mode="model_float32"):
    root = Path(root)
    manifest = json.loads((root / "suite_manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["scenarios"] != ["low_quality_25pct"]:
        raise ValueError("Expected complete low-quality candidate study")
    if not set(seeds) <= set(manifest["seeds"]):
        raise ValueError("Requested seed was not in the study")
    directory = root / "low_quality_25pct"
    shared = {int(item["seed"]): item for item in
              map(json.loads, (directory / "shared_costs.jsonl").read_text().splitlines())}
    with (directory / "candidate_rows.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    index = {}
    for row in rows:
        key = (int(row["seed"]), row["method"])
        if key in index:
            raise ValueError(f"Duplicate candidate: {key}")
        index[key] = row
    methods = manifest["candidate_initializations"]
    output = []
    for seed in seeds:
        item = shared[seed]
        if item.get("missing_fixed_ids"):
            raise ValueError(f"No fixed coalition for seed {seed}")
        artifact = directory / item["artifact_path"] / "local_models.pt"
        payload = torch.load(artifact, map_location="cpu", weights_only=True)
        seen_raw, seen_cache = {}, {}
        seed_rows = []
        for method in methods:
            row = index[seed, method]
            if row["delivery_status"] != "valid":
                raise ValueError(f"Invalid saved candidate: {seed}/{method}")
            original, raw_count, original_solver = _run(payload, item, row, "trace", seen_raw, key_mode)
            _, local_count, _ = _run(payload, item, row, "trace", {}, key_mode)
            cached, cache_count, cached_solver = _run(payload, item, row, "cache", seen_cache, key_mode)
            saved = np.asarray(json.loads(row["final_solution"]), dtype=float)
            saved_delta = float(np.max(np.abs(original - saved)))
            cache_delta = float(np.max(np.abs(cached - original)))
            if saved_delta > 1e-7 or cache_delta > 1e-7 or original_solver != cached_solver:
                raise ValueError(f"Stage 3 replay changed: {seed}/{method}: "
                                 f"saved={saved_delta}, cache={cache_delta}")
            seed_rows.append({"seed": seed, "method": method,
                              "saved_x_max_diff": saved_delta,
                              "cached_x_max_diff": cache_delta,
                              "actual_solver": original_solver,
                              "raw": raw_count, "within_candidate": local_count,
                              "cached": cache_count})
        output.extend(seed_rows)
    raw = sum(row["raw"]["requests"] for row in output)
    unique = sum(row["raw"].get("unique_requests", 0) for row in output)
    cached = sum(row["cached"].get("unique_requests", 0) for row in output)
    local_unique = sum(row["within_candidate"].get("unique_requests", 0)
                       for row in output)
    return {"study": "exact_stage3_query_reuse_diagnostic", "seeds": seeds,
            "methods": methods, "candidate_runs": len(output), "key_mode": key_mode,
            "raw_requests": raw, "unique_per_seed_requests": unique,
            "unique_per_candidate_requests": local_unique,
            "cached_model_calls": cached,
            "potential_model_call_saving_fraction": (raw-cached)/raw,
            "within_candidate_saving_fraction": (raw-local_unique)/raw,
            "cross_candidate_additional_saving_fraction": (local_unique-cached)/raw,
            "interpretation": "Identical model-visible inputs per agent can reuse a prediction "
                              "only while that agent model is unchanged. This is a saved-model "
                              "replay diagnostic, not a new quality comparison.",
            "rows": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--seeds", default="2400,2401,2402,2403,2404")
    parser.add_argument("--key-mode", choices=("model_float32", "raw"),
                        default="model_float32")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be nonempty and unique")
    result = analyze(args.root, seeds, args.key_mode)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()

"""Frozen-model bootstrap mechanism study; evaluator is post-delivery only.

Uses saved qualification/model artifacts. Calibration points are independently
generated uniform points with the same exact synthetic labels as Stage 0;
they are not evaluator train/validation/test data. Outer seed is the analysis
unit; blind draws nested within seed are never independent replications.
"""
import argparse
import contextlib
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from agent_client import AgentNode
from alpha_weight_experiments import attach_query_counter, run_stage3
from host_server import HostServer
from initial_alpha_methods import initial_alpha
from research_evaluation import load_evaluator_artifact, score_solution, synthetic_truth

METHODS = ["A_uniform", "B_stage0", "quality_shrink", "bootstrap_quality"]
SCENARIOS = ["reference_clean", "low_quality_25pct", "sample_imbalance_4x"]
KEYS = ["scenario", "seed", "draw", "blind_n", "target", "method"]


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False, default=json_numpy) + "\n")


def json_numpy(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value {type(value)}")


def append(path, value):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                default=json_numpy) + "\n")
        handle.flush()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibration(seed, draw, count):
    rng = np.random.default_rng(np.random.SeedSequence([20261009, seed, draw]))
    x = rng.uniform(-1., 1., (count, 2))
    return x, synthetic_truth(x)


def load_base(source, scenario, seed):
    directory = Path(source) / scenario
    items = [json.loads(line) for line in (directory / "shared_costs.jsonl").read_text().splitlines()]
    item = next(row for row in items if row["seed"] == seed)
    if item.get("missing_fixed_ids") or not item.get("artifact_path"):
        raise ValueError("Missing qualified fixed coalition")
    artifact = directory / item["artifact_path"]
    payload = torch.load(artifact / "local_models.pt", map_location="cpu", weights_only=True)
    server = HostServer(0., payload["n_features"], n_test=20,
                        constraint_mode="box_constrained", feasibility_tolerance=1e-8)
    server.trusted_agents = []
    for ident in item["coalition_ids"]:
        agent = AgentNode(ident, None, n_features=payload["n_features"])
        agent.model.load_state_dict(payload["agents"][ident])
        agent.model.eval()
        server.trusted_agents.append(agent)
    server.I_list = [np.asarray(x, float) for x in item["proposals"]]
    server.test_X = np.asarray(item["blind_test_X"], float)
    server.test_y = np.asarray(item["blind_test_y"], float)
    return server, item, artifact


def solve(server, method, mse, x, y, predictions, rng_seed, replicates, reuse=True):
    counter = attach_query_counter(server.trusted_agents)
    counter["stage"] = "initialization"
    start = time.perf_counter()
    try:
        alpha, audit = initial_alpha(method, server.trusted_agents, mse, x, y,
            server.I_list, server.target_T, bootstrap_replicates=replicates,
            bootstrap_seed=rng_seed,
            cached_blind_predictions=predictions if reuse else None)
    except Exception as exc:
        exc.partial_row = {"status":"interrupted", "failure_stage":"initialization",
            "reason":repr(exc), "initialization_seconds":time.perf_counter()-start,
            "initialization_requests":counter["requests"]["initialization"],
            "initialization_input_samples":counter["input_samples"]["initialization"],
            "stage3_requests":0,"stage3_input_samples":0,"stage3_seconds":0,
            "X":None,"pooled_error":None,"truth_error":None}
        raise
    initial_seconds = time.perf_counter() - start
    server.alphas = alpha.tolist()
    row = {"alpha": alpha.tolist(), "alpha_audit": audit, "mse": mse.tolist(),
           "initialization_seconds": initial_seconds,
           "initialization_requests": counter["requests"]["initialization"],
           "initialization_input_samples": counter["input_samples"]["initialization"],
           "status": "running", "X": None, "pooled_error": None, "truth_error": None}
    counter["stage"] = "stage3"
    start = time.perf_counter()
    try:
        solution, _, _ = run_stage3(server, "best_of", 30, cache_predictions=False)
        violation = server.box_violation(solution)
        solver = dict(getattr(server, "last_subspace_optimization", {}))
        losses = solver.get("engine_losses",{})
        solver["nonfinite_engine_losses"] = [k for k,v in losses.items() if not np.isfinite(v)]
        solver["engine_losses"] = {k:v if np.isfinite(v) else None for k,v in losses.items()}
        row.update(max_box_violation=violation if np.isfinite(violation) else None, solver=solver)
        if not np.isfinite(solution).all() or violation > 1e-8:
            row.update(status="invalid_output", reason="nonfinite_or_infeasible")
        else:
            row.update(status="valid", X=solution.tolist())
    except (ValueError, FloatingPointError) as exc:
        row.update(status="solver_failure", reason=str(exc))
    except Exception as exc:
        row.update(status="interrupted", failure_stage="stage3", reason=repr(exc))
        exc.partial_row = row
        raise
    finally:
        row.update(stage3_seconds=time.perf_counter()-start,
                   stage3_requests=counter["requests"]["stage3"],
                   stage3_input_samples=counter["input_samples"]["stage3"])
        counter["stage"] = "finished"
    return row


def run(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError("Output must be new/empty")
    seeds = [int(s) for s in args.seeds.split(",")]
    sizes = [int(n) for n in args.sizes.split(",")]
    targets = [float(t) for t in args.targets.split(",")]
    source_manifest = json.loads((Path(args.source)/"suite_manifest.json").read_text())
    if source_manifest["status"] != "complete" or not set(seeds) <= set(source_manifest["seeds"]):
        raise ValueError("Source is incomplete or does not contain seeds")
    if args.original_blind and sizes != [20]:
        raise ValueError("Original blind path must use n=20")
    draws = [-1] if args.original_blind else list(range(args.draws))
    manifest = {"status": "running", "source": str(Path(args.source).resolve()),
        "seeds": seeds, "scenarios": SCENARIOS, "draws": draws, "sizes": sizes,
        "targets": targets, "methods": METHODS, "bootstrap_replicates": 1000,
        "bootstrap_seed_base": 20261010, "expected": len(seeds)*3*len(draws)*len(sizes)*len(targets)*4,
        "keys": KEYS, "original_blind": args.original_blind,
        "calibration_distribution": "independent Uniform([-1,1]^2), exact synthetic Stage0 labels",
        "study_design": "fixed trained models/coalition; calibration does NOT requalify agents",
        "pooled_role": "posthoc only; no input to solve/alpha", "truth_role": "diagnostic",
        "source_sha256": {p.name: digest(p) for p in Path('.').glob('*.py')},
        "box_tolerance": 1e-8, "reload_tolerance": 1e-7,
        "objective": "sum_i alpha_i abs(local_prediction_i(X)-target)",
        "settlement": "not_run", "alpha_update": False, "pruning": False,
        "bootstrap_sensitivity": {"replicates": [100,500,1000,10000], "rng_repeats": 5}}
    snapshot = root / "source_snapshot"
    snapshot.mkdir()
    for name in manifest["source_sha256"]:
        (snapshot/name).write_bytes(Path(name).read_bytes())
    dump(root/"manifest.json", manifest)
    with (root/"solver.log").open("w") as solver_log:
        for scenario in SCENARIOS:
            for seed in seeds:
                print(f"{scenario} seed={seed}", flush=True)
                server, item, artifact = load_base(args.source, scenario, seed)
                artifact_hash = {p.name: digest(p) for p in artifact.iterdir() if p.is_file()}
                dump(root/f"artifact_reference_{scenario}_{seed}.json", {
                    "artifact": str(artifact.resolve()), "sha256": artifact_hash,
                    "original_shared": item})
                # The frozen evaluator is loaded for scoring only, never passed
                # to solve() or initial_alpha().
                evaluator, payload = load_evaluator_artifact(artifact)
                for target in targets:
                    server.target_T = target
                    if target == 0:
                        server.I_list = [np.asarray(p, float) for p in item["proposals"]]
                        proposal_cost = {"seconds": item["fixed_proposal_seconds"],
                            "requests": item["fixed_proposal_requests"],
                            "input_samples": item["fixed_proposal_input_samples"], "source": "saved"}
                    else:
                        counter = attach_query_counter(server.trusted_agents)
                        counter["stage"] = "proposal"
                        start = time.perf_counter()
                        server.I_list = []
                        for agent in server.trusted_agents:
                            torch.manual_seed(seed*100000+agent.agent_id)
                            server.I_list.append(agent.infer_parameters_D(target, steps=500))
                        proposal_cost = {"seconds": time.perf_counter()-start,
                            "requests": counter["requests"]["proposal"],
                            "input_samples": counter["input_samples"]["proposal"], "source": "recomputed"}
                    # Coverage for the additional target uses evaluator's test
                    # split only and is never used to generate calibration.
                    pooled = np.load(artifact/"pooled_data.npz")
                    tx = pooled["X"][pooled["test_indices"]]
                    ty = synthetic_truth(tx)
                    with torch.no_grad():
                        pred = evaluator(torch.tensor(tx, dtype=torch.float32)).numpy().reshape(-1)
                    distance = np.abs(ty-target)
                    coverage = []
                    for lo, hi in [(0,.1),(.1,.25),(.25,.5),(.5,1),(1,float('inf'))]:
                        mask = (distance >= lo if lo == 0 else distance > lo) & (distance <= hi)
                        coverage.append({"lower": lo, "upper": hi if np.isfinite(hi) else None,
                            "n": int(mask.sum()), "sparse_lt30": int(mask.sum())<30,
                            "truth_mse": float(np.mean((pred[mask]-ty[mask])**2)) if mask.any() else None})
                    append(root/"shared.jsonl", {"scenario": scenario, "seed": seed, "target": target,
                        "proposal_cost": proposal_cost, "proposals": [p.tolist() for p in server.I_list],
                        "evaluator_coverage": coverage, "global_metrics": payload["metrics"]})
                    for draw in draws:
                        if args.original_blind:
                            full_x, full_y = server.test_X, server.test_y
                        else:
                            full_x, full_y = calibration(seed, draw, max(sizes))
                        counter = attach_query_counter(server.trusted_agents)
                        counter["stage"] = "calibration"
                        start = time.perf_counter()
                        prediction = np.column_stack([a.api_predict(full_x) for a in server.trusted_agents])
                        batches = {len(full_y):prediction}
                        for n in sizes:
                            if n not in batches:
                                batches[n] = np.column_stack([a.api_predict(full_x[:n])
                                                              for a in server.trusted_agents])
                        seconds = time.perf_counter()-start
                        # Batch GEMM kernels may round differently at n=20 vs
                        # n=100: qualification reuse must preserve batch shape,
                        # not merely row identity. Each n has its own cached
                        # qualification batch, shared only across methods.
                        append(root/"calibration.jsonl", {"scenario": scenario, "seed": seed,
                            "target": target, "draw": draw, "X": full_x.tolist(), "y": full_y.tolist(),
                            "prediction": prediction.tolist(),
                            "prediction_by_n": {str(n):batches[n].tolist() for n in sizes},
                            "requests": counter["requests"]["calibration"],
                            "input_samples": counter["input_samples"]["calibration"], "seconds": seconds})
                        for n in sizes:
                            x, y, cache = full_x[:n], full_y[:n], batches[n]
                            mse = ((cache-y[:,None])**2).mean(axis=0)
                            bseed = 20261010 + seed*100 + max(draw,0)
                            for method in METHODS:
                                identity = dict(zip(KEYS, [scenario,seed,draw,n,target,method]))
                                try:
                                    with contextlib.redirect_stdout(solver_log):
                                        row = solve(server, method, mse, x, y, cache, bseed, 1000)
                                    row.update(identity, bootstrap_seed=bseed,
                                        calibration_requests_independent=0 if method=="A_uniform" else 3,
                                        calibration_inputs_independent=0 if method=="A_uniform" else 3*n,
                                        additional_labels=0 if args.original_blind or method=="A_uniform" else n)
                                    if row["status"] == "valid":
                                        start = time.perf_counter()
                                        _, error = score_solution(evaluator, np.asarray(row["X"]), target)
                                        row.update(pooled_error=error, truth_error=abs(float(synthetic_truth(
                                            np.asarray(row["X"])[None,:])[0])-target),
                                            scoring_seconds=time.perf_counter()-start)
                                    append(root/"results.jsonl", row)
                                except Exception as exc:
                                    append(root/"results.jsonl", {**identity, **getattr(exc,"partial_row",{}),
                                        "status": "interrupted", "reason": repr(exc)})
                                    manifest["status"] = "interrupted"
                                    dump(root/"manifest.json", manifest)
                                    raise
                            if target == 0 and n == 20:
                                # Finite Monte Carlo error is distinct from
                                # cross-blind-dataset uncertainty.
                                for b in [100,500,1000,10000]:
                                    for repeat in range(5):
                                        start = time.perf_counter()
                                        alpha, _ = initial_alpha("bootstrap_quality", server.trusted_agents,
                                            mse, x, y, server.I_list, target, bootstrap_replicates=b,
                                            bootstrap_seed=bseed+repeat*1000000,
                                            cached_blind_predictions=cache)
                                        append(root/"bootstrap_sensitivity.jsonl", {"scenario": scenario,
                                            "seed":seed, "draw":draw, "B":b, "repeat":repeat,
                                            "alpha":alpha.tolist(), "seconds":time.perf_counter()-start})
                                # Verify actual delivered X/query counts survive
                                # caching, not just a formula-level assertion.
                                with contextlib.redirect_stdout(solver_log):
                                    fresh = solve(server, "bootstrap_quality", mse, x, y, cache, bseed,1000,False)
                                    reused = solve(server, "bootstrap_quality", mse, x, y, cache,bseed,1000,True)
                                append(root/"reuse.jsonl", {"scenario":scenario,"seed":seed,"draw":draw,
                                    "fresh":fresh, "reused":reused})
                del server, evaluator
    manifest["status"] = "complete"
    manifest["output_sha256"] = {p.name:digest(p) for p in root.iterdir()
                                  if p.is_file() and p.name != "manifest.json"}
    dump(root/"manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--sizes", default="10,20,50,100")
    parser.add_argument("--targets", default="-0.5,0,0.5")
    parser.add_argument("--draws", type=int, default=10)
    parser.add_argument("--original-blind", action="store_true")
    run(parser.parse_args())

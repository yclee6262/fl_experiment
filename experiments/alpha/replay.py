"""Shared preparation and execution helpers used by the retained experiments."""
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
SCENARIOS = ['reference_clean', 'low_quality_25pct', 'sample_imbalance_4x']

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

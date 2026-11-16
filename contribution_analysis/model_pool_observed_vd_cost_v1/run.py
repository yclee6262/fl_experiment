"""Frozen paired cost replay: complete anchor C^A plus read-only model V/D.

The V/D observer accesses only cache entries created by the original full search.
It never runs a model-excluded search branch.
"""
import copy
import hashlib
import json
import math
import os
import resource
import shutil
import signal
import sys
import time
from pathlib import Path

os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import numpy as np

W = Path(__file__).resolve().parents[2]
CA = W / "contribution_analysis"
sys.path.insert(0, str(W))
FROZEN = CA / "model_step_decision_experiment_v1_20261021/source_snapshot"
sys.path.insert(0, str(FROZEN))

from contribution_analysis.custom_realized_gain_v1.runtime import load_server
from contribution_analysis.custom_realized_gain_v1.schema import plain
from contribution_analysis.method5_cost_v1.run import Replies
from contribution_analysis.model_step_decision_v1.engine import run
from contribution_analysis.three_pool_t2_anchor_v1.core import decide
from host_server import HostServer

OUT = CA / "model_pool_observed_vd_cost_20261103_attempt_07"
EPS = 1e-10


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(plain(value), indent=2, ensure_ascii=False, allow_nan=False))


def endpoint(trace, t):
    pre = trace.pre[t]
    post = next((item for item in trace.post if item["next_iteration"] == pre["next_iteration"] + 1), trace.final)
    return {"pre": pre, "post": post, "g": trace.gradients[t]["g"]}


def construct_directions(server, reply, weights, feature):
    """Reconstruct distinct anchor and model decompositions of the full gradient."""
    n = len(weights)
    x = np.asarray(feature["pre"]["S_current"])
    current_losses = reply.losses(x)
    h_by_anchor = np.zeros((n, len(x)))
    g_by_model = np.zeros((n, len(x)))
    directions = []
    seen = set()
    for k, anchor in enumerate(server.I_list):
        distance = float(np.linalg.norm(anchor - x))
        if distance < 1e-8:
            continue
        u = (anchor - x) / distance
        h = distance if feature["pre"]["current_method"] == "secant" else server._bounded_step(x, u, feature["pre"]["delta"])
        if h <= 1e-12:
            continue
        point = anchor if feature["pre"]["current_method"] == "secant" else x + h * u
        losses = reply.losses(point)
        raw_slope = (losses - current_losses) / h
        # H_k belongs to anchor k.  G_i belongs to loss provider/model i.
        h_by_anchor[k] += weights[k] * float(weights @ raw_slope) * u
        g_by_model += np.outer(weights * raw_slope, weights[k] * u)
        key = (point.tobytes(), np.float64(h).tobytes(), u.tobytes())
        if key not in seen:
            directions.append({"k": k, "point": point, "h": h, "u": u, "losses": losses})
            seen.add(key)
    g = h_by_anchor.sum(axis=0)
    assert np.allclose(g, feature["g"], atol=EPS, rtol=EPS), "gradient reconstruction failed"
    assert np.allclose(g_by_model.sum(axis=0), g, atol=EPS, rtol=EPS), "model decomposition failed"
    return current_losses, directions, h_by_anchor, g_by_model


def observe_vd(reply, trace, feature, weights, current_losses, g_by_model):
    """Read V and D from the baseline cache. New physical samples are forbidden."""
    n = len(weights)
    t = feature["pre"]["next_iteration"]
    trials = [item for item in trace.trials if item["iteration"] == t]
    accepted = [item for item in trials if item["accepted"]]
    assert len(accepted) <= 1
    full_x = np.asarray(feature["post"]["S_current"])
    expected = np.asarray(accepted[0]["x"]) if accepted else np.asarray(feature["pre"]["S_current"])
    assert np.allclose(full_x, expected, atol=EPS), "trace endpoint mismatch"
    full_losses = reply.losses(full_x)
    original = np.asarray(feature["pre"]["S_current"])
    observed = []
    for i in range(n):
        others = [j for j in range(n) if j != i]
        remaining_weights = weights[others] / weights[others].sum()
        baseline = float(remaining_weights @ current_losses[others])
        choice = None
        choice_losses = current_losses
        for order, trial in enumerate(trials):
            losses = reply.losses(np.asarray(trial["x"]))
            if float(remaining_weights @ losses[others]) < baseline:
                choice, choice_losses = order, losses
                break
        alt_x = original if choice is None else np.asarray(trials[choice]["x"])
        gi = g_by_model[i]
        g = g_by_model.sum(axis=0)
        rest = g - gi
        d = None
        if np.linalg.norm(g) > EPS and np.linalg.norm(rest) > EPS:
            d = float(np.linalg.norm(g / np.linalg.norm(g) - rest / np.linalg.norm(rest)) / 2)
        observed.append({
            "i": i,
            "V": float(weights @ (choice_losses - full_losses)),
            "D": d,
            "choice": choice,
            "original_choice": None if not accepted else len(trials) - 1,
            "full_loss": float(weights @ full_losses),
            "alternative_loss": float(weights @ choice_losses),
            "n_trials": len(trials),
        })
    return observed


def execute_observed(case, server, path):
    start = time.perf_counter()
    reply = Replies(server, sha(case["models"]) + ":target0:abs:float64:cpu")
    n = len(server.alphas)
    ids = list(range(n))
    weights = np.asarray(server.alphas, dtype=float)
    weights /= weights.sum()
    owners = [str(item) for item in case["ids"]]
    reply.stage = "baseline"
    base, trace = run(server, reply.objective(ids), 30, initial=np.asarray(case["fixed_start"]))
    full = [endpoint(trace, t) for t in range(len(trace.pre))]
    baseline_samples = sum(not event["hit"] for event in reply.events)
    rows = []
    all_vd_hits = True
    for t, feature in enumerate(full):
        reply.stage = "features"
        current_losses, directions, h_by_anchor, g_by_model = construct_directions(server, reply, weights, feature)
        if not directions:
            raise ValueError("undefined_directions")
        reply.stage = "anchor"
        full_losses = reply.losses(np.asarray(feature["post"]["S_current"]))
        for i in ids:
            alternative = decide(server, reply.objective(ids), np.asarray(feature["pre"]["S_current"]),
                                 np.asarray(feature["g"]) - h_by_anchor[i], feature["pre"]["best_loss"],
                                 feature["pre"]["eta"], feature["pre"]["current_method"])
            reply.stage = "common"
            alternative_losses = reply.losses(alternative["x"])
            ca = float(weights @ alternative_losses - weights @ full_losses)
            rows.append({"t": t, "i": i, "owner": owners[i], "C_A": ca, "rho": max(1e-12, 32*np.finfo(float).eps*(1 + abs(float(weights @ alternative_losses)) + abs(float(weights @ full_losses))) )})
        before = len(reply.events)
        reply.stage = "observe_vd"
        vd = observe_vd(reply, trace, feature, weights, current_losses, g_by_model)
        new_events = reply.events[before:]
        all_vd_hits &= all(event["hit"] for event in new_events)
        for item in vd:
            item.update(t=t, owner=owners[item["i"]])
        feature.update(directions=directions, H=h_by_anchor, G=g_by_model, current_losses=current_losses, vd=vd)
    assert all_vd_hits, "V/D read created a physical model sample"
    data = {
        "arm": "anchor_CA_plus_observed_VD", "key": case["key"], "scenario": case["scenario"],
        "base": base, "full": full, "rows": rows, "events": reply.events, "weights": weights,
        "owners": owners, "baseline_samples": baseline_samples,
        "samples": sum(not event["hit"] for event in reply.events), "requests": len(reply.events),
        "hits": sum(event["hit"] for event in reply.events), "vd_new_physical_samples": 0,
        "model_seconds": reply.seconds, "peak_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "core_seconds": time.perf_counter() - start,
    }
    save(path, data)
    data["wall_seconds"] = time.perf_counter() - start
    return data


def execute_loo(case, server, path):
    start = time.perf_counter()
    reply = Replies(server, sha(case["models"]) + ":target0:abs:float64:cpu")
    n = len(server.alphas)
    ids = list(range(n))
    weights = np.asarray(server.alphas, dtype=float)
    weights /= weights.sum()
    owners = [str(item) for item in case["ids"]]
    reply.stage = "baseline"
    base = HostServer._solve_subspace_custom(server, server.I_list, reply.objective(ids), num_iterations=30,
                                               direction_weights=server.alphas, initial_solution=np.asarray(case["fixed_start"]))
    rows = []
    for i in ids:
        panel = [j for j in ids if j != i]
        branch = copy.copy(server)
        branch.I_list = np.asarray(server.I_list)[panel]
        branch.alphas = weights[panel] / weights[panel].sum()
        reply.stage = "full_removal"
        removed = HostServer._solve_subspace_custom(branch, branch.I_list, reply.objective(panel), num_iterations=30,
                                                    direction_weights=branch.alphas, initial_solution=np.asarray(case["fixed_start"]))
        reply.stage = "common"
        full_losses, removed_losses = reply.losses(base["solution"]), reply.losses(removed["solution"])
        rows.append({"i": i, "owner": owners[i], "D_LOO": float(weights @ (removed_losses - full_losses)),
                     "full": base["solution"], "minus": removed["solution"], "result": removed})
    data = {
        "arm": "full_LOO", "key": case["key"], "scenario": case["scenario"], "base": base, "rows": rows,
        "samples": sum(not event["hit"] for event in reply.events), "requests": len(reply.events),
        "hits": sum(event["hit"] for event in reply.events), "model_seconds": reply.seconds,
        "core_seconds": time.perf_counter() - start,
    }
    save(path, data)
    data["wall_seconds"] = time.perf_counter() - start
    return data


def validate_against_frozen(case, observed, loo):
    root = CA / "custom_direction_experiment_v2_20261020"
    historic = json.loads((root / "baselines" / (case["key"] + ".json")).read_text())["result"]
    # The AST observer returns the solver-core fields.  The archival baseline
    # adds delivery metadata afterwards, so compare the exact shared core.
    assert plain(observed["base"]) == {key: historic[key] for key in observed["base"]}
    assert plain(loo["base"]) == {key: historic[key] for key in loo["base"]}
    for row in loo["rows"]:
        reference = json.loads((root / "removals" / (case["key"] + "_" + str(case["ids"][row["i"]]) + ".json")).read_text())["result"]
        assert plain(row["result"]) == {key: reference[key] for key in row["result"]}
    # C_A is not redefined here: it must be exactly the already accepted
    # complete anchor-removal calculation from the frozen paired cost replay.
    previous = json.loads((CA / "method5_cost_experiment_20261030_attempt_04" / "records" / (case["key"] + "_New.json")).read_text())
    indexed = {(item["t"], item["i"]): item for item in previous["rows"]}
    for row in observed["rows"]:
        assert abs(row["C_A"] - indexed[(row["t"], row["i"])]["C_A"]) < EPS
    prior = json.loads((CA / "observed_e0_e1_followup_20261103_attempt_01" / "results.json").read_text())
    prior = {(item["case"], item["t"], item["owner"]): item for item in prior}
    for feature in observed["full"]:
        for item in feature["vd"]:
            reference = prior[(case["key"], item["t"], item["owner"])]
            assert abs(item["V"] - reference["V"]) < EPS
            assert (item["D"] is None) == (reference["D"] is None)
            if item["D"] is not None:
                assert abs(item["D"] - reference["D"]) < EPS


def summary(records):
    observed = [record["observed"] for record in records]
    loo = [record["loo"] for record in records]
    vd = [entry for record in observed for feature in record["full"] for entry in feature["vd"]]
    scores = {}
    for record in observed:
        per_owner = {owner: 0.0 for owner in record["owners"]}
        for feature in record["full"]:
            for item in feature["vd"]:
                per_owner[item["owner"]] += item["V"]
        scores[record["key"]] = per_owner
    return {
        "complete": True, "teams": len(records), "rounds": sum(len(item["full"]) for item in observed),
        "baseline_solver_core_replay_exact": True, "full_LOO_solver_core_replay_exact": True,
        "complete_anchor_CA_replay_exact": True, "prior_observed_VD_replay_exact": True,
        "owner_rounds": len(vd), "V_positive": sum(item["V"] > EPS for item in vd),
        "V_zero": sum(abs(item["V"]) <= EPS for item in vd), "V_negative": sum(item["V"] < -EPS for item in vd),
        "D_defined": sum(item["D"] is not None for item in vd), "D_undefined": sum(item["D"] is None for item in vd),
        "choice_changed": sum(item["choice"] != item["original_choice"] for item in vd),
        "vd_new_physical_samples": sum(item["vd_new_physical_samples"] for item in observed),
        "observed_samples_total": sum(item["samples"] for item in observed),
        "loo_samples_total": sum(item["samples"] for item in loo),
        "observed_minus_loo": sum(item["samples"] for item in observed) - sum(item["samples"] for item in loo),
        "observed_over_loo_pct": 100 * (sum(item["samples"] for item in observed) / sum(item["samples"] for item in loo) - 1),
        "all_teams_observed_lt_loo": all(a["samples"] < b["samples"] for a,b in zip(observed, loo)),
        "V_cumulative_by_team": scores,
    }


def main():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    OUT.mkdir(exist_ok=False)
    (OUT / "records").mkdir()
    (OUT / "source_snapshot").mkdir()
    cases = json.loads((CA / "combined_contribution_layer2_experiment_v1_20261021/cases.json").read_text())["development"]
    source = [Path(__file__), W / "docs/model_pool_observed_vd_cost_design_20261103.md",
              CA / "method5_cost_v1" / "run.py", CA / "model_step_decision_v1" / "engine.py",
              CA / "three_pool_t2_anchor_v1" / "core.py", CA / "custom_realized_gain_v1" / "runtime.py",
              CA / "custom_realized_gain_v1" / "schema.py"]
    source += list(FROZEN.glob("*.py"))
    source += [Path(case["models"]) for case in cases]
    source = list(dict.fromkeys(source))
    lock = {str(item): sha(item) for item in source}
    save(OUT / "source_lock.json", lock)
    for item in source:
        if item.suffix == ".py":
            shutil.copy2(item, OUT / "source_snapshot" / "__".join(item.relative_to(W).parts))
    save(OUT / "registration.json", {"cases": cases, "iterations": 30, "start": "fixed historical", "new": "complete C_A + cache-only V/D", "control": "full participant LOO", "new_model_calls_for_VD": 0, "training": 0, "pooled": 0, "truth": 0, "payment": 0})
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("wall cap")))
    signal.alarm(900)
    began = time.perf_counter()
    complete = []
    try:
        for index, case in enumerate(cases):
            print(case["key"], flush=True)
            server = load_server(case)
            observed = execute_observed(case, server, OUT / "records" / (case["key"] + "_observed.json"))
            loo = execute_loo(case, server, OUT / "records" / (case["key"] + "_loo.json"))
            validate_against_frozen(case, observed, loo)
            complete.append({"key": case["key"], "scenario": case["scenario"], "observed": observed, "loo": loo})
            save(OUT / "progress.json", [{"key": item["key"], "observed_samples": item["observed"]["samples"], "loo_samples": item["loo"]["samples"]} for item in complete])
        for item, digest in lock.items():
            assert sha(item) == digest, "source changed during execution"
        output = summary(complete)
        output["wall_seconds"] = time.perf_counter() - began
        save(OUT / "summary.json", output)
        print(json.dumps({k:v for k,v in output.items() if k != "V_cumulative_by_team"}, ensure_ascii=False), flush=True)
    except Exception as error:
        save(OUT / "failure.json", {"error": repr(error), "completed": len(complete)})
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()

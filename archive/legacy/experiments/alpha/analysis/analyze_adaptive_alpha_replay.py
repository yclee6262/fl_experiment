"""Verify and summarize sequential alpha replays against frozen candidates."""

import argparse
import json
from pathlib import Path

import numpy as np

from research_analysis import paired_bootstrap_mean_ci
from run_adaptive_alpha_replay import should_solve_uniform


def summarize(name, replay_paths, benchmark_paths):
    replays, benchmarks = [], []
    for path in replay_paths:
        payload = json.loads(Path(path).read_text())
        if payload["status"] != "complete":
            raise ValueError(f"Incomplete replay: {path}")
        replays.extend(payload["rows"])
    for path in benchmark_paths:
        benchmarks.extend(json.loads(Path(path).read_text())["rows"])
    by_seed = {}
    for row in benchmarks:
        key = (row["scenario"], row["seed"])
        if key in by_seed:
            raise ValueError(f"Duplicate benchmark: {key}")
        by_seed[key] = row
    keys = [(row["scenario"], row["seed"]) for row in replays]
    if len(keys) != len(set(keys)) or set(keys) != set(by_seed):
        raise ValueError("Duplicate or missing scenario/seed condition")
    vs_b, vs_five, replay_cost, b_cost, five_cost = [], [], [], [], []
    trigger_count = uniform_selected = 0
    max_x_diff = max_pooled_diff = max_stage3_diff = 0.0
    failures = 0
    for row in replays:
        control = by_seed[row["scenario"], row["seed"]]
        expected_trigger, expected_reason = should_solve_uniform(
            row["candidates"]["B_stage0"]["status"],
            row["b_internal_error_at_trigger"], row["threshold"])
        if (row["trigger_uniform"] != expected_trigger
                or row["trigger_reason"] != expected_reason):
            raise ValueError(f"Trigger decision did not replay: {row['seed']}")
        if row["delivery_status"] != "valid":
            failures += 1
            continue
        if (row["selected_initialization"] not in ("B_stage0", "A_uniform")
                or set(row["candidates"]) != ({"B_stage0", "A_uniform"}
                                              if row["trigger_uniform"]
                                              else {"B_stage0"})):
            raise ValueError(f"Invalid adaptive decision: {row['seed']}")
        expected_selection = min(
            ("B_stage0", "A_uniform"),
            key=lambda method: row["internal_scores"].get(method, {}).get(
                "internal_target_error", np.inf))
        if row["selected_initialization"] != expected_selection:
            raise ValueError(f"Internal selection did not replay: {row['seed']}")
        selected = control["candidates"][row["selected_initialization"]]
        max_x_diff = max(max_x_diff, float(np.max(np.abs(
            np.asarray(row["delivered_solution"])-selected["solution"]))))
        max_pooled_diff = max(max_pooled_diff,
                              abs(row["pooled_error"]-selected["pooled_error"]))
        expected_stage3 = sum(control["candidates"][method]["stage3_requests"]
                              for method in row["candidates"])
        max_stage3_diff = max(max_stage3_diff,
                              abs(row["stage3_requests"]-expected_stage3))
        if (not np.isfinite(row["pooled_error"])
                or any(candidate["max_box_violation"] > 1e-8
                       for candidate in row["candidates"].values()
                       if candidate["status"] == "valid")):
            raise ValueError(f"Invalid delivered quality/box: {row['seed']}")
        vs_b.append(row["pooled_error"]
                    - control["candidates"]["B_stage0"]["pooled_error"])
        vs_five.append(row["pooled_error"]-control["new_pooled_error"])
        replay_cost.append(row["stage3_requests"]+row["selection_requests"])
        b_cost.append(control["candidates"]["B_stage0"]["stage3_requests"])
        five_cost.append(control["new_stage3_requests"]
                         + control["selection_requests"])
        trigger_count += row["trigger_uniform"]
        uniform_selected += row["selected_initialization"] == "A_uniform"
    if max(max_x_diff, max_pooled_diff) > 1e-7 or max_stage3_diff != 0:
        raise ValueError("Replay differs from frozen candidate solutions")
    if not vs_b:
        raise ValueError("No valid delivery")
    b_delta = np.asarray(vs_b)
    five_delta = np.asarray(vs_five)
    return {
        "name": name, "conditions": len(replays), "valid": len(vs_b),
        "failures": failures, "trigger_count": trigger_count,
        "uniform_selected_count": uniform_selected,
        "pooled_delta_adaptive_minus_B_mean": float(b_delta.mean()),
        "pooled_delta_adaptive_minus_B_descriptive_95pct_ci":
            paired_bootstrap_mean_ci(b_delta),
        "pooled_delta_adaptive_minus_five_mean": float(five_delta.mean()),
        "pooled_delta_adaptive_minus_five_descriptive_95pct_ci":
            paired_bootstrap_mean_ci(five_delta),
        "vs_B_wins_ties_losses": [int((b_delta < -1e-7).sum()),
                                  int((abs(b_delta) <= 1e-7).sum()),
                                  int((b_delta > 1e-7).sum())],
        "mean_stage3_plus_selection_requests": float(np.mean(replay_cost)),
        "mean_B_stage3_requests": float(np.mean(b_cost)),
        "mean_five_stage3_plus_selection_requests": float(np.mean(five_cost)),
        "max_solution_replay_difference": max_x_diff,
        "max_pooled_replay_difference": max_pooled_diff,
        "max_stage3_request_difference": max_stage3_diff,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs=3, action="append", required=True,
                        metavar=("NAME", "REPLAY_JSON", "BENCHMARK_JSON"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    names = [case[0] for case in args.case]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate case name")
    result = [summarize(name, [replay], [benchmark])
              for name, replay, benchmark in args.case]
    output.write_text(json.dumps(result, ensure_ascii=False,
                                 indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

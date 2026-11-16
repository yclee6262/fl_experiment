"""Exploratory Stage-0 information-gap audit for saved five-candidate runs.

The saved pooled scores are outcomes for diagnosis only. This script never
changes the selected candidate, alpha, or delivered X. All input cohorts have
already been used for method development, so associations are not validation.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from agent_client import AgentNode
from run_validation_calibrated_alpha import CANDIDATE_METHODS, META_METHOD


BANDWIDTH = 0.25


def _csv_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _indexed(rows, expected, name):
    index = {}
    for row in rows:
        key = (int(row["seed"]), row["method"])
        if key in index:
            raise ValueError(f"Duplicate {name} key: {key}")
        index[key] = row
    if set(index) != expected:
        raise ValueError(f"{name} coverage mismatch: missing={expected-set(index)}, "
                         f"unexpected={set(index)-expected}")
    return index


def _load_agents(folder, shared):
    path = folder / shared["artifact_path"] / "local_models.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    agents = []
    for ident in shared["coalition_ids"]:
        agent = AgentNode(ident, None, n_features=payload["n_features"])
        agent.model.load_state_dict(payload["agents"][ident])
        agent.model.eval()
        agents.append(agent)
    return agents


def _safe_spearman(x, y):
    value = float(spearmanr(x, y).statistic)
    return value if np.isfinite(value) else None


def _quartile_contrast(records, feature):
    values = np.asarray([r[feature] for r in records], float)
    regret = np.asarray([r["selector_regret_posthoc"] for r in records], float)
    lo, hi = np.quantile(values, [.25, .75])
    low, high = values <= lo, values >= hi
    return {
        "q1": float(lo), "q3": float(hi),
        "low_count": int(low.sum()), "high_count": int(high.sum()),
        "low_mean_regret": float(regret[low].mean()),
        "high_mean_regret": float(regret[high].mean()),
        "low_regret_gt_1e_6": int((regret[low] > 1e-6).sum()),
        "high_regret_gt_1e_6": int((regret[high] > 1e-6).sum()),
    }


def analyze(root):
    root = Path(root)
    manifest = json.loads((root / "suite_manifest.json").read_text())
    verification = json.loads((root / "selection_verification.json").read_text())
    if (manifest["status"] != "complete" or not verification["passed"]
            or len(manifest["scenarios"]) != 1):
        raise ValueError(f"Incomplete or unverified root: {root}")
    scenario, = manifest["scenarios"]
    folder = root / scenario
    seeds = [int(seed) for seed in manifest["seeds"]]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate manifest seeds: {root}")
    candidate_rows = _csv_rows(folder / "candidate_rows.csv")
    expected_candidates = {(seed, method) for seed in seeds
                           for method in manifest["candidate_initializations"]}
    candidate = _indexed(candidate_rows, expected_candidates, "candidate")
    summary_rows = _csv_rows(folder / "summary.csv")
    expected_summary = {(seed, method) for seed in seeds
                        for method in manifest["methods"]}
    summary = _indexed(summary_rows, expected_summary, "summary")
    shared_rows = [json.loads(line) for line in
                   (folder / "shared_costs.jsonl").read_text().splitlines()]
    shared = {int(row["seed"]): row for row in shared_rows}
    if len(shared) != len(shared_rows) or set(shared) != set(seeds):
        raise ValueError(f"Shared-state coverage mismatch: {root}")
    records = []
    for seed in seeds:
        row = summary[seed, META_METHOD]
        audit = json.loads(row["alpha_audit"])
        methods = tuple(audit["candidate_method_order"])
        if methods != CANDIDATE_METHODS:
            raise ValueError(f"Unexpected baseline candidate order: {scenario}/{seed}")
        if row["delivery_status"] != "valid" or row["final_box_feasible"] != "True":
            raise ValueError(f"Invalid baseline delivery: {scenario}/{seed}")
        chosen = audit["selected_initialization"]
        if any(candidate[seed, method]["delivery_status"] != "valid" or
               candidate[seed, method]["final_box_feasible"] != "True"
               for method in methods):
            raise ValueError(f"Invalid baseline candidate: {scenario}/{seed}")
        scores = np.asarray([float(audit["internal_scores"][m][
            "internal_target_error"]) for m in methods])
        if chosen != methods[int(np.argmin(scores))]:
            raise ValueError(f"Saved choice does not replay: {scenario}/{seed}")
        pooled = np.asarray([float(candidate[seed, m]["external_global_error"])
                             for m in methods])
        if not (np.isfinite(scores).all() and np.isfinite(pooled).all()):
            raise ValueError(f"Nonfinite score: {scenario}/{seed}")
        chosen_index = methods.index(chosen)
        if abs(float(row["external_global_error"])-pooled[chosen_index]) > 1e-7:
            raise ValueError(f"Delivered pooled mismatch: {scenario}/{seed}")
        solution = np.asarray(json.loads(row["final_solution"]), float)
        blind_x = np.asarray(shared[seed]["blind_test_X"], float)
        if (solution.ndim != 1 or blind_x.ndim != 2 or
                solution.size != blind_x.shape[1] or
                not np.isfinite(solution).all() or not np.isfinite(blind_x).all()):
            raise ValueError(f"Invalid Stage-0 geometry: {scenario}/{seed}")
        distance = np.linalg.norm(blind_x-solution, axis=1)
        kernel = np.exp(-distance**2/(2*BANDWIDTH**2))
        effective_n = float(kernel.sum()**2/np.dot(kernel, kernel))
        agents = _load_agents(folder, shared[seed])
        predictions = np.asarray([float(np.asarray(agent.api_predict(
            solution.reshape(1, -1)), float).reshape(-1)[0])
                                  for agent in agents])
        if not np.isfinite(predictions).all():
            raise ValueError(f"Nonfinite local prediction: {scenario}/{seed}")
        sorted_scores = np.sort(scores)
        records.append({
            "scenario": scenario, "seed": seed, "chosen": chosen,
            "selector_regret_posthoc": float(pooled[chosen_index]-pooled.min()),
            "chosen_pooled_error": float(pooled[chosen_index]),
            "pooled_best_error_posthoc": float(pooled.min()),
            "chosen_is_pooled_best_posthoc": bool(pooled[chosen_index] == pooled.min()),
            "nearest_blind_distance": float(distance.min()),
            "blind_points_within_0p25": int((distance <= BANDWIDTH).sum()),
            "kernel_effective_n_0p25": effective_n,
            "kernel_mass_0p25": float(kernel.sum()),
            "local_prediction_std": float(predictions.std()),
            "internal_score_margin": float(sorted_scores[1]-sorted_scores[0]),
            "internal_score_best": float(sorted_scores[0]),
        })
    regret = [row["selector_regret_posthoc"] for row in records]
    features = ("nearest_blind_distance", "blind_points_within_0p25",
                "kernel_effective_n_0p25", "kernel_mass_0p25",
                "local_prediction_std", "internal_score_margin",
                "internal_score_best")
    result = {
        "input_root": str(root), "scenario": scenario,
        "seed_count": len(records),
        "evidence_status": "posthoc_exploratory_no_method_selection",
        "bandwidth": BANDWIDTH,
        "mean_selector_regret_posthoc": float(np.mean(regret)),
        "regret_gt_1e_6_count": sum(value > 1e-6 for value in regret),
        "feature_diagnostics": {feature: {
            "spearman_with_regret": _safe_spearman(
                [row[feature] for row in records], regret),
            "quartile_contrast": _quartile_contrast(records, feature),
        } for feature in features},
    }
    return result, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    summaries, records = [], []
    for root in args.roots:
        result, rows = analyze(root)
        summaries.append(result)
        records.extend(rows)
    (output / "summary.json").write_text(json.dumps(
        summaries, ensure_ascii=False, indent=2)+"\n")
    with (output / "seed_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Development-only jackknife selection on saved five-candidate solutions.

The selected X uses only Stage 0 data and local APIs. Pooled scores are read
after selection for exploratory method screening, never for a delivered X.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from agent_client import AgentNode
from research_analysis import paired_bootstrap_mean_ci
from validation_calibrated_alpha import (
    CANDIDATE_METHODS, select_candidate_jackknife_q90,
)


def _read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def explore(roots):
    records = []
    for root_arg in roots:
        root = Path(root_arg)
        manifest = json.loads((root / "suite_manifest.json").read_text())
        scenario, = manifest["scenarios"]
        folder = root / scenario
        candidates = {(int(r["seed"]), r["method"]): r
                      for r in _read(folder / "candidate_rows.csv")}
        baseline = {int(r["seed"]): r for r in _read(folder / "summary.csv")
                    if r["method"] == "validation_calibrated_multistart"}
        shared = {r["seed"]: r for r in map(
            json.loads, (folder / "shared_costs.jsonl").read_text().splitlines())}
        if set(shared) != set(manifest["seeds"]) or set(baseline) != set(manifest["seeds"]):
            raise ValueError(f"Missing seed: {root}")
        for seed in manifest["seeds"]:
            item = shared[seed]
            payload = torch.load(folder / item["artifact_path"] / "local_models.pt",
                                 map_location="cpu", weights_only=True)
            agents = []
            for agent_id in item["coalition_ids"]:
                agent = AgentNode(agent_id, None, n_features=payload["n_features"])
                agent.model.load_state_dict(payload["agents"][agent_id])
                agent.model.eval()
                agents.append(agent)
            old = baseline[seed]
            alpha = np.asarray(json.loads(candidates[seed, "B_stage0"]["initial_alpha"]),
                               dtype=float)
            solutions = {method: (json.loads(candidates[seed, method]["final_solution"])
                                  if candidates[seed, method]["delivery_status"] == "valid"
                                  else None)
                         for method in CANDIDATE_METHODS}
            started = time.perf_counter()
            chosen, scores = select_candidate_jackknife_q90(
                agents, alpha, item["blind_test_X"], item["blind_test_y"],
                solutions, float(manifest["arguments"]["inverse_target"]))
            elapsed = time.perf_counter()-started
            selected = candidates[seed, chosen]
            old_chosen = json.loads(old["alpha_audit"])["selected_initialization"]
            if abs(float(candidates[seed, old_chosen]["external_global_error"])-
                   float(old["external_global_error"])) > 1e-7:
                raise AssertionError(f"Old selector replay mismatch: {scenario}/{seed}")
            records.append({
                "cohort": root.name, "scenario": scenario, "seed": seed,
                "old_chosen": old_chosen, "jackknife_chosen": chosen,
                "old_pooled": float(old["external_global_error"]),
                "jackknife_pooled": float(selected["external_global_error"]),
                "pooled_delta": (float(selected["external_global_error"])-
                                 float(old["external_global_error"])),
                "old_truth": float(old["final_target_error"]),
                "jackknife_truth": float(selected["final_target_error"]),
                "chosen_jackknife_q90": scores[chosen]["jackknife_q90_target_error"],
                "selector_seconds_exploratory": elapsed,
                "candidate_scores": json.dumps(scores),
            })
        print("completed", root, flush=True)
    summaries = []
    for cohort in sorted({r["cohort"] for r in records}):
        group = [r for r in records if r["cohort"] == cohort]
        deltas = [r["pooled_delta"] for r in group]
        summaries.append({
            "cohort": cohort, "scenario": group[0]["scenario"],
            "evidence_status": "exploratory_saved_formal_seeds_reused",
            "seeds": len(group), "changed_selection": sum(
                r["old_chosen"] != r["jackknife_chosen"] for r in group),
            "mean_delta": float(np.mean(deltas)),
            "paired_95pct_ci_descriptive": paired_bootstrap_mean_ci(deltas),
            "wins_ties_losses": [sum(v < 0 for v in deltas),
                                 sum(v == 0 for v in deltas),
                                 sum(v > 0 for v in deltas)],
            "worst_regression": max(deltas),
            "mean_selector_seconds": float(np.mean([
                r["selector_seconds_exploratory"] for r in group])),
            "extra_stage3_solves": 0,
            "extra_model_requests_vs_global_selector": 0,
        })
    return records, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    records, summaries = explore(args.roots)
    with (output / "seed_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output / "summary.json").write_text(json.dumps(
        summaries, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

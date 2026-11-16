"""Exploratory Stage 0 gate audit on already-used alpha experiment seeds.

All pooled outcomes here are read after computing the Stage 0-only gate. The
output cannot support a new independent advantage claim; it only helps fix a
rule before a new paired-seed cohort is run.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from agent_client import AgentNode
from validation_calibrated_alpha import choose_marginal_or_target_fifth
from run_validation_calibrated_alpha import META_METHOD, MARGINAL_METHOD, MARGINAL_CANDIDATE


THRESHOLDS = (0.7, 0.8, 0.9, 0.95)


def _rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _row_gains(shared, artifact, baseline):
    payload = torch.load(artifact / "local_models.pt", map_location="cpu",
                         weights_only=True)
    agents = []
    for agent_id in shared["coalition_ids"]:
        agent = AgentNode(agent_id, None, n_features=payload["n_features"])
        agent.model.load_state_dict(payload["agents"][agent_id])
        agent.model.eval()
        agents.append(agent)
    x = np.asarray(shared["blind_test_X"], dtype=float)
    y = np.asarray(shared["blind_test_y"], dtype=float)
    predictions = np.column_stack([agent.api_predict(x) for agent in agents])
    baseline = np.asarray(baseline, dtype=float)
    full = (predictions @ baseline - y) ** 2
    gains = np.zeros((len(y), len(agents)))
    for i in range(len(agents)):
        remaining = baseline.copy()
        remaining[i] = 0
        remaining /= remaining.sum()
        gains[:, i] = (predictions @ remaining - y) ** 2 - full
    return gains


def explore(roots):
    records = []
    for path in roots:
        root = Path(path)
        manifest = json.loads((root / "suite_manifest.json").read_text())
        scenario = manifest["scenarios"][0]
        folder = root / scenario
        summary = {(int(r["seed"]), r["method"]): r for r in _rows(folder / "summary.csv")}
        candidate = {(int(r["seed"]), r["method"]): r
                     for r in _rows(folder / "candidate_rows.csv")}
        shared = {int(r["seed"]): r for r in map(
            json.loads, (folder / "shared_costs.jsonl").read_text().splitlines())}
        if set(shared) != set(manifest["seeds"]):
            raise ValueError(f"Shared seed mismatch: {root}")
        for seed in manifest["seeds"]:
            source = shared[seed]
            baseline = json.loads(candidate[seed, "B_stage0"]["initial_alpha"])
            rows = _row_gains(source, folder / source["artifact_path"], baseline)
            saved = np.asarray(json.loads(candidate[seed, MARGINAL_CANDIDATE][
                "alpha_audit"])["blind_leave_one_out_gain"])
            if not np.allclose(rows.mean(axis=0), saved, atol=1e-7, rtol=0):
                raise AssertionError(f"Stage 0 marginal gain replay differs: {scenario}/{seed}")
            choice, audit = choose_marginal_or_target_fifth(rows, seed, threshold=.9)
            old = summary[seed, META_METHOD]
            new = summary[seed, MARGINAL_METHOD]
            if old["delivery_status"] != "valid" or new["delivery_status"] != "valid":
                raise ValueError(f"No paired delivery: {scenario}/{seed}")
            records.append({
                "scenario": scenario, "seed": seed,
                "weighted_sign_support": audit["weighted_sign_support"],
                "per_agent_sign_support": json.dumps(audit["per_agent_sign_support"]),
                "mean_gain": json.dumps(rows.mean(axis=0).tolist()),
                "gate_choice_at_0p9": choice,
                "old_pooled": float(old["external_global_error"]),
                "new_pooled": float(new["external_global_error"]),
                "old_selected": json.loads(old["alpha_audit"])["selected_initialization"],
                "new_selected": json.loads(new["alpha_audit"])["selected_initialization"],
            })
            print(f"Stage 0 replay {scenario}/{seed}", flush=True)
    summaries = []
    for scenario in sorted({r["scenario"] for r in records}):
        subset = [r for r in records if r["scenario"] == scenario]
        old = np.asarray([r["old_pooled"] for r in subset])
        new = np.asarray([r["new_pooled"] for r in subset])
        for threshold in THRESHOLDS:
            chosen = np.asarray([new[i] if r["weighted_sign_support"] >= threshold
                                 else old[i] for i, r in enumerate(subset)])
            summaries.append({
                "scenario": scenario, "threshold": threshold, "count": len(subset),
                "marginal_portfolio_count": sum(
                    r["weighted_sign_support"] >= threshold for r in subset),
                "mean_old": float(old.mean()), "mean_replacement": float(new.mean()),
                "mean_gate": float(chosen.mean()),
                "mean_gate_minus_old": float((chosen-old).mean()),
                "gate_vs_old_wins": int(np.sum(chosen < old)),
                "gate_vs_old_ties": int(np.sum(chosen == old)),
                "gate_vs_old_losses": int(np.sum(chosen > old)),
                "worst_gate_minus_old": float((chosen-old).max()),
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
        {"evidence_status": "exploratory_reuse_of_formal_seeds",
         "thresholds": summaries}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

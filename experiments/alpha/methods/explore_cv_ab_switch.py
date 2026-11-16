"""Exploratory one-solve A/B switch using Stage-0 leave-one-label-out risk.

The switch uses only Stage-0 labels and local-model predictions. Saved pooled
scores are read after the A/B decision for diagnosis. Reused cohorts here are
development data, never independent evidence of quality advantage.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from agent_client import AgentNode
from run_validation_calibrated_alpha import META_METHOD
from research_analysis import paired_bootstrap_mean_ci


METHODS = ("A_uniform", "B_stage0", META_METHOD)
EPS = 1e-5


def _rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _load_agents(folder, shared):
    payload = torch.load(folder / shared["artifact_path"] / "local_models.pt",
                         map_location="cpu", weights_only=True)
    agents = []
    for ident in shared["coalition_ids"]:
        agent = AgentNode(ident, None, n_features=payload["n_features"])
        agent.model.load_state_dict(payload["agents"][ident])
        agent.model.eval()
        agents.append(agent)
    return agents


def cv_choice(predictions, labels, saved_mse):
    prediction = np.asarray(predictions, float)
    y = np.asarray(labels, float).reshape(-1)
    mse = np.asarray(saved_mse, float)
    if (prediction.ndim != 2 or prediction.shape[0] != len(y)
            or prediction.shape[1] != len(mse) or len(y) < 3
            or not np.isfinite(prediction).all() or not np.isfinite(y).all()
            or not np.isfinite(mse).all()):
        raise ValueError("Invalid Stage-0 prediction matrix")
    squared = (prediction-y[:, None])**2
    if not np.allclose(squared.mean(axis=0), mse, rtol=0, atol=1e-7):
        raise ValueError("Saved Stage-0 MSE differs from blind predictions")
    loo_mse = (squared.sum(axis=0)[None, :]-squared)/(len(y)-1)
    inverse = 1.0/(loo_mse+EPS)
    loo_b = inverse/inverse.sum(axis=1, keepdims=True)
    pred_b = np.sum(prediction*loo_b, axis=1)
    pred_a = prediction.mean(axis=1)
    loss_a = float(np.mean((pred_a-y)**2))
    loss_b = float(np.mean((pred_b-y)**2))
    return ("B_stage0" if loss_b < loss_a else "A_uniform"), {
        "cv_loss_A": loss_a, "cv_loss_B": loss_b,
        "cv_loss_gap_B_minus_A": loss_b-loss_a,
        "sample_count": len(y),
    }


def analyze(root):
    root = Path(root)
    manifest = json.loads((root / "suite_manifest.json").read_text())
    verified = json.loads((root / "selection_verification.json").read_text())
    if (manifest["status"] != "complete" or not verified["passed"]
            or len(manifest["scenarios"]) != 1):
        raise ValueError(f"Unverified experiment root: {root}")
    scenario, = manifest["scenarios"]
    seeds = [int(x) for x in manifest["seeds"]]
    folder = root / scenario
    summary_rows = _rows(folder / "summary.csv")
    keys = [(int(r["seed"]),r["method"]) for r in summary_rows]
    expected = {(seed,method) for seed in seeds for method in manifest["methods"]}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(f"Summary coverage mismatch: {root}")
    summary = {key:row for key,row in zip(keys,summary_rows)}
    shared_rows = [json.loads(line) for line in
                   (folder / "shared_costs.jsonl").read_text().splitlines()]
    shared = {int(r["seed"]):r for r in shared_rows}
    if len(shared) != len(shared_rows) or set(shared) != set(seeds):
        raise ValueError(f"Shared-state coverage mismatch: {root}")
    records = []
    for seed in seeds:
        row = shared[seed]
        x = np.asarray(row["blind_test_X"],float)
        y = np.asarray(row["blind_test_y"],float)
        agents = _load_agents(folder,row)
        predictions = np.column_stack([np.asarray(a.api_predict(x),float)
                                       for a in agents])
        saved_mse = json.loads(summary[seed,"B_stage0"]["stage0_mse"])
        expected_b = 1.0/(np.asarray(saved_mse,float)+EPS)
        expected_b /= expected_b.sum()
        if not np.allclose(expected_b, json.loads(summary[seed,"B_stage0"]["initial_alpha"]),
                           rtol=0, atol=1e-10):
            raise ValueError(f"Saved B alpha mismatch: {scenario}/{seed}")
        choice,audit = cv_choice(predictions,y,saved_mse)
        if choice not in METHODS[:2]:
            raise AssertionError("Unexpected switch choice")
        selected = summary[seed,choice]
        if any(summary[seed,method]["delivery_status"] != "valid" or
               summary[seed,method]["final_box_feasible"] != "True"
               for method in METHODS):
            raise ValueError(f"Invalid saved delivery: {scenario}/{seed}")
        pooled = {method:float(summary[seed,method]["external_global_error"])
                  for method in METHODS}
        if not np.isfinite(list(pooled.values())).all():
            raise ValueError(f"Nonfinite pooled score: {scenario}/{seed}")
        requests = float(selected["initialization_requests"])
        requests += float(selected["stage3_requests"]) + len(agents)
        records.append({
            "scenario":scenario, "seed":seed, "choice":choice,
            **audit,
            "pooled_switch":pooled[choice],
            "pooled_A":pooled["A_uniform"],
            "pooled_B":pooled["B_stage0"],
            "pooled_five":pooled[META_METHOD],
            "switch_minus_A":pooled[choice]-pooled["A_uniform"],
            "switch_minus_B":pooled[choice]-pooled["B_stage0"],
            "switch_minus_five":pooled[choice]-pooled[META_METHOD],
            "switch_method_requests_conservative":requests,
            "extra_blind_prediction_requests":len(agents),
            "extra_blind_prediction_input_samples":len(agents)*len(y),
        })
    report={
        "scenario":scenario, "seed_count":len(records),
        "evidence_status":"reused_cohort_exploratory_no_formal_claim",
        "rule":"Choose B iff leave-one-label-out Stage-0 ensemble MSE_B < MSE_A; "
               "use full-sample B weights for one Stage-3 solve; pooled posthoc only.",
        "selected_B":sum(r["choice"]=="B_stage0" for r in records),
        "mean_pooled":{name:float(np.mean([r["pooled_"+name] for r in records]))
                       for name in ("switch","A","B","five")},
        "mean_switch_requests":float(np.mean([
            r["switch_method_requests_conservative"] for r in records])),
        "comparisons":{},
    }
    for method in ("A","B","five"):
        delta=np.asarray([r["switch_minus_"+method] for r in records])
        report["comparisons"]["switch_minus_"+method]={
            "mean":float(delta.mean()),
            "descriptive_95pct_ci":paired_bootstrap_mean_ci(delta),
            "wins_ties_losses":[int((delta<0).sum()),int((delta==0).sum()),
                                 int((delta>0).sum())],
            "q95":float(np.quantile(delta,.95)),
            "worst":float(delta.max()),
        }
    return report,records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots",nargs="+",required=True)
    parser.add_argument("--output-dir",required=True)
    args=parser.parse_args()
    output=Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=False)
    reports,records=[],[]
    for root in args.roots:
        report,rows=analyze(root)
        reports.append(report)
        records.extend(rows)
    (output/"summary.json").write_text(json.dumps(reports,ensure_ascii=False,
                                                  indent=2)+"\n")
    with (output/"seed_rows.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    print(json.dumps(reports,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()

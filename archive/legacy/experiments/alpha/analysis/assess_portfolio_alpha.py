"""Preregistered paired cost-quality assessment of the four-alpha portfolio."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from research_analysis import paired_bootstrap_mean_ci
from run_validation_calibrated_alpha import META_METHOD, PORTFOLIO_METHOD, DIVERSE_METHOD


QUALITY_MARGIN = 0.0005
MIN_REQUEST_REDUCTION = 0.15


def assess(root, candidate_method=PORTFOLIO_METHOD, expected_seed_count=120):
    if candidate_method not in (PORTFOLIO_METHOD, DIVERSE_METHOD):
        raise ValueError("Unknown portfolio method")
    root = Path(root)
    manifest = json.loads((root/"suite_manifest.json").read_text())
    verification = json.loads((root/"selection_verification.json").read_text())
    if manifest["scenarios"] != ["reference_clean"]:
        raise ValueError("Formal portfolio assessment expects reference_clean")
    with (root/"reference_clean"/"summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = Counter((row["scenario"],row["seed"],row["method"]) for row in rows)
    expected = {("reference_clean",str(seed),method)
                for seed in manifest["seeds"] for method in manifest["methods"]}
    complete = (manifest["status"] == "complete" and verification["passed"]
                and set(keys) == expected and all(keys[key] == 1 for key in expected))
    index = {(row["seed"],row["method"]):row for row in rows}
    method_stats = {}
    for method in (META_METHOD,candidate_method):
        group = [index.get((str(seed),method)) for seed in manifest["seeds"]]
        valid = [row for row in group if row is not None
                 and row["delivery_status"] == "valid"
                 and row["final_box_feasible"] == "True"
                 and np.isfinite(float(row["external_global_error"]))]
        method_stats[method] = {
            "valid":len(valid),"failures":len(group)-len(valid),
            "mean_pooled_error":float(np.mean([float(row["external_global_error"])
                                          for row in valid])) if valid else None,
            "mean_truth_error":float(np.mean([float(row["final_target_error"])
                                         for row in valid])) if valid else None,
            "mean_requests":float(np.mean([float(row["initialization_requests"])
                                          +float(row["stage3_requests"])
                                          for row in valid])) if valid else None,
            "mean_method_seconds":float(np.mean([float(row["initialization_seconds"])
                                                +float(row["stage3_seconds"])
                                                for row in valid])) if valid else None,
        }
    differences = []
    request_savings = []
    conflicts = 0
    selected = Counter()
    for seed in manifest["seeds"]:
        old = index.get((str(seed),META_METHOD))
        new = index.get((str(seed),candidate_method))
        if not old or not new or old["delivery_status"] != "valid" or new["delivery_status"] != "valid":
            continue
        delta = float(new["external_global_error"])-float(old["external_global_error"])
        truth_delta = float(new["final_target_error"])-float(old["final_target_error"])
        if not np.isfinite(delta) or not np.isfinite(truth_delta):
            continue
        differences.append(delta)
        request_savings.append((float(old["initialization_requests"])
                                +float(old["stage3_requests"])
                                -float(new["initialization_requests"])
                                -float(new["stage3_requests"])))
        conflicts += delta*truth_delta < 0
        selected[json.loads(new["alpha_audit"])["selected_initialization"]] += 1
    interval = paired_bootstrap_mean_ci(differences)
    old_requests = method_stats[META_METHOD]["mean_requests"]
    new_requests = method_stats[candidate_method]["mean_requests"]
    reduction = (1-new_requests/old_requests
                 if old_requests is not None and new_requests is not None and old_requests > 0
                 else None)
    noninferior = bool(complete and len(manifest["seeds"]) == expected_seed_count
                       and len(differences) == len(manifest["seeds"])
                       and interval[1] is not None and interval[1] < QUALITY_MARGIN
                       and method_stats[candidate_method]["failures"]
                       <= method_stats[META_METHOD]["failures"])
    report = {
        "complete_and_verified":complete,"seed_count":len(manifest["seeds"]),
        "paired_n":len(differences),"methods":method_stats,
        "mean_pooled_difference":float(np.mean(differences)) if differences else None,
        "paired_95pct_ci":interval,
        "wins":sum(d<0 for d in differences),"ties":sum(d==0 for d in differences),
        "losses":sum(d>0 for d in differences),
        "pooled_truth_conflicts":conflicts,
        "selected_initializations":dict(selected),
        "mean_request_savings":float(np.mean(request_savings)) if request_savings else None,
        "request_reduction_fraction":reduction,
        "quality_improvement_gate":bool(noninferior and interval[1] < 0),
        "bounded_quality_cost_gate":bool(noninferior and reduction is not None
                                          and reduction >= MIN_REQUEST_REDUCTION),
        "quality_margin":QUALITY_MARGIN,
        "margin_note":"Research tolerance, about 3% of the development baseline mean; not a validated application threshold or monetary valuation",
        "break_even_cost_to_error_value":(
            max(float(np.mean(differences)),0)/float(np.mean(request_savings))
            if differences and request_savings and np.mean(request_savings)>0 else None),
        "rule":f"{expected_seed_count} valid seed pairs; 10000 seed-bootstrap two-sided 95% upper CI for E_candidate-E_five below 0.0005, at least 15% lower mean model requests, no extra failures; report strict quality improvement separately",
    }
    filename = ("diverse_portfolio_decision.json" if candidate_method == DIVERSE_METHOD
                else "portfolio_decision.json")
    (root/filename).write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--method", choices=(PORTFOLIO_METHOD,DIVERSE_METHOD),
                        default=PORTFOLIO_METHOD)
    parser.add_argument("--expected-seeds",type=int,default=120)
    args=parser.parse_args()
    print(json.dumps(assess(args.root,args.method,args.expected_seeds),indent=2,ensure_ascii=False))

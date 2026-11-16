"""Assess six- and seven-candidate alpha initialization against five candidates."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from research_analysis import paired_bootstrap_mean_ci
from run_validation_calibrated_alpha import ECONOMIC_METHOD, META_METHOD, SPECIALIST_METHOD


def assess(root):
    root = Path(root)
    manifest = json.loads((root / "suite_manifest.json").read_text())
    verification = json.loads((root / "selection_verification.json").read_text())
    rows=[]
    for scenario in manifest["scenarios"]:
        with (root/scenario/"summary.csv").open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    keys=Counter((r["scenario"],r["seed"],r["method"]) for r in rows)
    expected={(s,str(seed),method) for s in manifest["scenarios"]
              for seed in manifest["seeds"] for method in manifest["methods"]}
    complete=(manifest["status"]=="complete" and verification["passed"]
              and set(keys)==expected and all(keys[k]==1 for k in expected))
    index={(r["scenario"],r["seed"],r["method"]):r for r in rows}
    results={}
    for scenario in manifest["scenarios"]:
        method_stats={}
        for method in (META_METHOD,SPECIALIST_METHOD,ECONOMIC_METHOD):
            group=[index.get((scenario,str(seed),method)) for seed in manifest["seeds"]]
            valid=[r for r in group if r and r["delivery_status"]=="valid"
                   and r["final_box_feasible"]=="True"
                   and np.isfinite(float(r["external_global_error"]))]
            method_stats[method]={"valid":len(valid),"failures":len(group)-len(valid),
                "mean_pooled_error":float(np.mean([float(r["external_global_error"]) for r in valid])) if valid else None,
                "mean_requests":float(np.mean([float(r["initialization_requests"])+float(r["stage3_requests"]) for r in valid])) if valid else None,
                "mean_method_seconds":float(np.mean([float(r["initialization_seconds"])+float(r["stage3_seconds"]) for r in valid])) if valid else None}
        comparisons={}
        for method in (SPECIALIST_METHOD,ECONOMIC_METHOD):
            differences=[];conflicts=0;choices=Counter()
            for seed in manifest["seeds"]:
                new=index.get((scenario,str(seed),method))
                old=index.get((scenario,str(seed),META_METHOD))
                if not new or not old or new["delivery_status"]!="valid" or old["delivery_status"]!="valid":
                    continue
                delta=float(new["external_global_error"])-float(old["external_global_error"])
                truth=float(new["final_target_error"])-float(old["final_target_error"])
                if not (np.isfinite(delta) and np.isfinite(truth)):
                    continue
                differences.append(delta)
                conflicts+=delta*truth<0
                choices[json.loads(new["alpha_audit"])["selected_initialization"]]+=1
            interval=paired_bootstrap_mean_ci(differences)
            comparisons[method]={"paired_n":len(differences),
                "mean_pooled_difference":float(np.mean(differences)) if differences else None,
                "paired_95pct_ci":interval,"wins":sum(d<0 for d in differences),
                "ties":sum(d==0 for d in differences),"losses":sum(d>0 for d in differences),
                "pooled_truth_conflicts":conflicts,"selected_initializations":dict(choices),
                "quality_gate":bool(complete and len(differences)==len(manifest["seeds"])
                    and interval[1] is not None and interval[1]<0
                    and method_stats[method]["failures"]<=method_stats[META_METHOD]["failures"])}
        results[scenario]={"methods":method_stats,"comparisons":comparisons,
                           "primary_specialist_passes":comparisons[SPECIALIST_METHOD]["quality_gate"]}
    report={"complete_and_verified":complete,"seeds":manifest["seeds"],
            "scenario_results":results,
            "rule":"primary six-candidate vs five-candidate seed-paired 10000-bootstrap 95% CI wholly below 0, full coverage and no extra failures; seven-candidate comparison exploratory; cost separate"}
    (root/"economic_decision.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("root")
    args=parser.parse_args();print(json.dumps(assess(args.root),indent=2,ensure_ascii=False))

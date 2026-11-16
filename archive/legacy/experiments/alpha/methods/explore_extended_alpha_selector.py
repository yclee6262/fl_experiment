"""Development-only value of adding one ex-ante alpha to the five-candidate selector.

The saved pooled error is used to evaluate a choice, never to make it.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from agent_client import AgentNode
from research_analysis import paired_bootstrap_mean_ci
from validation_calibrated_alpha import CANDIDATE_METHODS, fit_residual_calibrator


def evaluate(root, extras_path, output):
    root = Path(root)/"reference_clean"
    extras = defaultdict(dict)
    for row in csv.DictReader(Path(extras_path).open()):
        extras[int(row["seed"])][row["method"]] = row
    candidates = defaultdict(dict)
    for row in csv.DictReader((root/"candidate_rows.csv").open()):
        candidates[int(row["seed"])][row["method"]] = row
    meta = {int(row["seed"]): row for row in
            csv.DictReader((root/"summary.csv").open())
            if row["method"] == "validation_calibrated_multistart"}
    results=[]
    for shared in map(json.loads, (root/"shared_costs.jsonl").read_text().splitlines()):
        seed=int(shared["seed"])
        if seed not in extras or seed not in meta:
            continue
        payload=torch.load(root/shared["artifact_path"]/"local_models.pt",
                           map_location="cpu",weights_only=True)
        agents=[]
        for ident in shared["coalition_ids"]:
            a=AgentNode(ident,None,n_features=payload["n_features"])
            a.model.load_state_dict(payload["agents"][ident]); a.model.eval()
            agents.append(a)
        b=np.asarray(json.loads(candidates[seed]["B_stage0"]["initial_alpha"]),float)
        coef,_=fit_residual_calibrator(agents,shared["blind_test_X"],
                                       shared["blind_test_y"],b)
        def score(solution):
            s=np.asarray(solution,float)
            pred=np.asarray([a.api_predict(s)[0] for a in agents],float)
            return abs(float(pred@b-np.r_[1.,s]@coef))
        baseline={}
        for method in CANDIDATE_METHODS:
            row=candidates[seed][method]
            if row["delivery_status"]=="valid":
                baseline[method]=(score(json.loads(row["final_solution"])),
                                  float(row["external_global_error"]))
        original=min(baseline,key=lambda m:baseline[m][0])
        original_pooled=baseline[original][1]
        if abs(original_pooled-float(meta[seed]["external_global_error"]))>1e-7:
            raise AssertionError(f"Baseline selector replay mismatch: {seed}")
        for method,row in extras[seed].items():
            if method in baseline or row["status"]!="valid":
                continue
            extra_score=score(json.loads(row["solution"]))
            is_chosen=extra_score<baseline[original][0]
            new_pooled=float(row["pooled_error"]) if is_chosen else original_pooled
            results.append({"seed":seed,"extra_method":method,
                            "selected_extra":int(is_chosen),
                            "current_pooled":original_pooled,
                            "new_pooled":new_pooled,
                            "pooled_delta":new_pooled-original_pooled,
                            "extra_internal_score":extra_score,
                            "current_internal_score":baseline[original][0],
                            "extra_pooled":float(row["pooled_error"])})
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    with output.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=results[0].keys());writer.writeheader();writer.writerows(results)
    grouped=defaultdict(list)
    for row in results:grouped[row["extra_method"]].append(row)
    for method,group in sorted(grouped.items()):
        values=[r["pooled_delta"] for r in group]
        ci=paired_bootstrap_mean_ci(values)
        print(method,len(group),round(float(np.mean(values)),6),
              tuple(round(float(v),6) for v in ci),
              "selected",sum(r["selected_extra"] for r in group),
              "oracle_gain",round(float(np.mean([min(0.,r["extra_pooled"]-r["current_pooled"])
                                               for r in group])),6))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("root");parser.add_argument("extras");parser.add_argument("output")
    args=parser.parse_args();evaluate(args.root,args.extras,args.output)

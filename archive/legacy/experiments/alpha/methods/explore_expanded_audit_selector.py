"""Development-only selection among five saved alphas using 100 blind labels.

Pooled results are read after each rule chooses an existing candidate. Extra
labels are synthetic trusted measurements and their acquisition cost is not
represented by the local-model request counter.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from agent_client import AgentNode
from host_server import HostServer
from validation_calibrated_alpha import (CANDIDATE_METHODS,
                                         fit_residual_calibrator, select_candidate)


def evaluate(roots, output, extra_paths=()):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    extra_candidates={}
    for extra_path in extra_paths:
        for row in csv.DictReader(Path(extra_path).open()):
            if row.get("method") != "local_bias_risk_100" or row.get("status") != "valid":
                continue
            seed=int(row["seed"])
            if seed in extra_candidates:
                raise ValueError(f"Duplicate extra alpha candidate seed: {seed}")
            extra_candidates[seed]={"solution":json.loads(row["solution"]),
                                    "pooled_error":float(row.get("candidate_pooled")
                                                          or row["pooled_error"])}
    records=[]
    for root_arg in roots:
        root=Path(root_arg)/"reference_clean"
        cohort=Path(root_arg).name
        candidates={}
        for row in csv.DictReader((root/"candidate_rows.csv").open()):
            candidates[(int(row["seed"]),row["method"])]=row
        baseline={int(row["seed"]):row for row in csv.DictReader((root/"summary.csv").open())
                  if row["method"]=="validation_calibrated_multistart"}
        for item in map(json.loads,(root/"shared_costs.jsonl").read_text().splitlines()):
            seed=int(item["seed"])
            if seed not in baseline or "artifact_path" not in item:continue
            payload=torch.load(root/item["artifact_path"]/"local_models.pt",
                               map_location="cpu",weights_only=True)
            agents=[]
            for ident in item["coalition_ids"]:
                agent=AgentNode(ident,None,n_features=payload["n_features"])
                agent.model.load_state_dict(payload["agents"][ident]);agent.model.eval()
                agents.append(agent)
            expanded=HostServer(0.,payload["n_features"],test_seed=seed+7919,
                                n_test=100,constraint_mode="box_constrained")
            old_x=np.asarray(item["blind_test_X"],float)
            old_y=np.asarray(item["blind_test_y"],float)
            if not (np.array_equal(expanded.test_X[:len(old_x)],old_x)
                    and np.array_equal(expanded.test_y[:len(old_y)],old_y)):
                raise AssertionError(f"Expanded audit changed original blind labels: {cohort}/{seed}")
            predictions=np.column_stack([a.api_predict(expanded.test_X) for a in agents])
            mse100=np.mean((predictions-expanded.test_y[:,None])**2,axis=0)
            b20=np.asarray(json.loads(candidates[(seed,"B_stage0")]["initial_alpha"]),float)
            b100=(mse100+1e-5)**-1
            b100/=b100.sum()
            solutions={m:json.loads(candidates[(seed,m)]["final_solution"])
                       for m in CANDIDATE_METHODS}
            old_pooled=float(baseline[seed]["external_global_error"])
            for rule,alpha in (("B20_calibrate100",b20),("B100_calibrate100",b100)):
                coeff,_=fit_residual_calibrator(
                    agents,expanded.test_X,expanded.test_y,alpha)
                options=[(rule,solutions,CANDIDATE_METHODS)]
                if seed in extra_candidates:
                    augmented={**solutions,"local_bias_risk_100":extra_candidates[seed]["solution"]}
                    options.append((rule+"_six",augmented,
                                    CANDIDATE_METHODS+("local_bias_risk_100",)))
                for name,available,order in options:
                    chosen,_=select_candidate(
                        agents,alpha,coeff,available,0.,method_order=order)
                    new_pooled=(extra_candidates[seed]["pooled_error"]
                                if chosen=="local_bias_risk_100"
                                else float(candidates[(seed,chosen)]["external_global_error"]))
                    records.append({"cohort":cohort,"seed":seed,"rule":name,
                                    "selected_initialization":chosen,
                                    "old_pooled":old_pooled,"new_pooled":new_pooled,
                                    "pooled_delta":new_pooled-old_pooled,
                                    "extra_trusted_labels":100-len(old_y),
                                    "b100":json.dumps(b100.tolist())})
        print("completed",cohort,flush=True)
    with output.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    from collections import defaultdict
    grouped=defaultdict(list)
    for record in records:grouped[(record["cohort"],record["rule"])].append(record["pooled_delta"])
    for (cohort,rule),values in sorted(grouped.items()):
        print(cohort,rule,len(values),float(np.mean(values)),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--roots",nargs="+",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--extra-paths",nargs="*",default=[])
    args=parser.parse_args();evaluate(args.roots,args.output,args.extra_paths)

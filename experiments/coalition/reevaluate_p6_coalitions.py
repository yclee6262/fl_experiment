"""Re-score every persisted P6 coalition solve; failures remain not applicable."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from research_evaluation import load_evaluator_artifact, score_solution
from run_experiments import true_function


def main():
    parser=argparse.ArgumentParser();parser.add_argument("results_dir");parser.add_argument("--output",required=True)
    parser.add_argument("--tolerance",type=float,default=1e-7);parser.add_argument("--box-tolerance",type=float,default=1e-8)
    args=parser.parse_args();root=Path(args.results_dir);output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    with (root/"coalitions.csv").open(newline="",encoding="utf-8") as handle:rows=list(csv.DictReader(handle))
    cache={};results=[];seen=set()
    for row in rows:
        key=(row["seed"],row["coalition_indices"])
        if key in seen:raise ValueError(f"duplicate coalition identity: {key}")
        seen.add(key);artifact=row["artifact_dir"]
        base={"seed":row["seed"],"coalition_indices":row["coalition_indices"],"agent_ids":row["agent_ids"]}
        if not row.get("final_solution"):
            results.append({**base,"reevaluation_status":"not_applicable","reason":row.get("failure_reason","no delivered X"),
                "external_score_matches":"not_applicable","truth_score_matches":"not_applicable",
                "recomputed_box_feasible":"not_applicable","artifact_dir":artifact});continue
        if artifact not in cache:cache[artifact],_=load_evaluator_artifact(root/artifact)
        solution=np.asarray(json.loads(row["final_solution"]),dtype=float);target=0.5
        prediction,error=score_solution(cache[artifact],solution,target);truth=abs(true_function(solution)-target)
        saved_error=float(row["pooled_error"]);saved_truth=float(row["truth_error"])
        violation=max(0.0,float(np.max(solution-1.0)),float(np.max(-1.0-solution)))
        results.append({**base,"reevaluation_status":"scored","reason":"",
            "recomputed_pooled_error":error,"saved_pooled_error":saved_error,
            "pooled_score_delta":abs(error-saved_error),"external_score_matches":abs(error-saved_error)<=args.tolerance,
            "recomputed_truth_error":truth,"saved_truth_error":saved_truth,
            "truth_score_delta":abs(truth-saved_truth),"truth_score_matches":abs(truth-saved_truth)<=args.tolerance,
            "recomputed_box_violation":violation,"recomputed_box_feasible":bool(np.all(np.isfinite(solution)) and violation<=args.box_tolerance),
            "artifact_dir":artifact})
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",newline="",encoding="utf-8") as handle:
        fields=list(dict.fromkeys(key for row in results for key in row));writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(results)
    mismatches=sum(row.get("external_score_matches")=="False" or row.get("truth_score_matches")=="False"
                   or row.get("recomputed_box_feasible")=="False" for row in results)
    print(f"Re-evaluated {len(results)} coalitions; mismatches={mismatches}")


if __name__=="__main__":main()

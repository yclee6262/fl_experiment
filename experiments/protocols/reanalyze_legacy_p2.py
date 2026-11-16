"""Failure-aware re-analysis of archived P2 runs without changing old outputs."""

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from research_analysis import audit_coverage, finite_float, paired_bootstrap_mean_ci, valid_delivery


BASELINE = "A_uniform_no_update"
METHODS = (BASELINE, "B_stage0_no_update", "C_uniform_fixed", "D_uniform_guarded")
METRICS = ("external_global_error", "final_target_error", "final_evaluation_loss", "final_consistency_l1")


def read_rows(dirs):
    result=[]
    for directory in map(Path, dirs):
        with (directory/"summary.csv").open(newline="",encoding="utf-8") as handle:
            result.extend({**row,"_result_dir":str(directory)} for row in csv.DictReader(handle))
    return result


def key(row):
    return (str(int(row["seed"])), str(int(row["dimension"])),
            str(float(row["poison_ratio"])), str(float(row["target"])))


def analyze(dirs):
    rows=read_rows(dirs)
    grouped={}
    for row in rows:
        identity=key(row); method=row["method_label"]
        if (identity,method) in grouped: raise ValueError(f"Duplicate archived P2 row: {(identity,method)}")
        grouped[(identity,method)]=row
    expected_conditions=set()
    for directory in map(Path,dirs):
        manifest_path=directory/"run_manifest.json"
        if not manifest_path.exists(): continue
        args=json.loads(manifest_path.read_text()).get("arguments",{})
        for seed,dim,poison,target in itertools.product(args.get("seeds",[]),args.get("dimensions",[]),args.get("poison_ratios",[]),args.get("targets",[])):
            expected_conditions.add((str(seed),str(dim),str(float(poison)),str(float(target))))
    if not expected_conditions: expected_conditions={key(row) for row in rows}
    expected=[(*condition,method) for condition in sorted(expected_conditions) for method in METHODS]
    observed=[(*key(row),row["method_label"]) for row in rows]
    coverage=audit_coverage([dict(zip(("seed","dimension","poison_ratio","target","method"),item)) for item in observed],expected,
        ("seed","dimension","poison_ratio","target","method"))
    results=[]; pairs=[]
    for condition in sorted(expected_conditions):
        method_rows={method:grouped.get((condition,method)) for method in METHODS}
        for method,row in method_rows.items():
            if row is None:
                results.append({"seed":condition[0],"dimension":condition[1],"poison_ratio":condition[2],"target":condition[3],
                    "method":method,"execution_status":"missing","delivery_status":"missing"});continue
            ext=finite_float(row.get("external_global_error"))
            results.append({"seed":condition[0],"dimension":condition[1],"poison_ratio":condition[2],"target":condition[3],
                "method":method,"execution_status":row.get("execution_status","completed"),
                "delivery_status":row.get("delivery_status","unknown"),
                "feasible":row.get("final_box_feasible",""),
                "external_error_valid_delivery":ext if valid_delivery(row) else None,
                "external_error_all_runs_diagnostic":ext,
                "failure_stage":row.get("failure_stage",""),"failure_reason":row.get("failure_reason",row.get("stop_reason","")),
                "stage3_requests":row.get("stage3_requests",""),"contribution_requests":row.get("contribution_requests",""),
                "payment_status":row.get("payment_status","unknown"),
                "payment_requests":row.get("payment_requests",""),"condition_seconds":row.get("condition_seconds","")})
        for method in METHODS[1:]:
            left,right=method_rows[BASELINE],method_rows[method]
            for metric in METRICS:
                ok=left is not None and right is not None
                if metric in ("external_global_error","final_target_error"):
                    ok=ok and valid_delivery(left,metric) and valid_delivery(right,metric)
                else:
                    ok=ok and left.get("final_solution") not in (None,"","[]") and right.get("final_solution") not in (None,"","[]")
                a=finite_float(left.get(metric)) if ok else None
                b=finite_float(right.get(metric)) if ok else None
                delta=b-a if a is not None and b is not None else None
                pairs.append({"seed":condition[0],"dimension":condition[1],"poison_ratio":condition[2],"target":condition[3],
                    "baseline":BASELINE,"method":method,"metric":metric,"valid_quality_pair":delta is not None,
                    "delta_method_minus_A":delta,"missing_reason":"" if delta is not None else "missing, invalid, or non-finite"})
    statistics=[]
    groups={}
    for row in pairs:
        group=(row["dimension"],row["poison_ratio"],row["target"],row["method"],row["metric"])
        groups.setdefault(group,[]).append(row)
    for group,items in sorted(groups.items()):
        values=[row["delta_method_minus_A"] for row in items if row["delta_method_minus_A"] is not None]
        low,high=paired_bootstrap_mean_ci(values) if values else (None,None)
        statistics.append({"dimension":group[0],"poison_ratio":group[1],"target":group[2],"method":group[3],"metric":group[4],
            "planned_seed_rows":len(items),"valid_paired_seeds":len(values),"missing_pairs":len(items)-len(values),
            "mean_delta_method_minus_A":float(np.mean(values)) if values else None,
            "median_delta_method_minus_A":float(np.median(values)) if values else None,"ci95_low":low,"ci95_high":high,
            "interpretation":"diagnostic_only_for_invalid_runs; seed is sampling unit"})
    return results,pairs,statistics,{**coverage,"duplicate_keys":json.dumps(coverage["duplicate_keys"]),
        "missing_keys":json.dumps(coverage["missing_keys"]),"unexpected_keys":json.dumps(coverage["unexpected_keys"])}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("result_dirs",nargs="+");parser.add_argument("--output-dir",required=True)
    args=parser.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise FileExistsError(out)
    results,pairs,statistics,coverage=analyze(args.result_dirs)
    for name,rows in (("method_results.csv",results),("paired_results.csv",pairs),
                      ("paired_statistics.csv",statistics),("coverage_audit.csv",[coverage])):
        if rows:
            with (out/name).open("x",newline="",encoding="utf-8") as h:
                writer=csv.DictWriter(h,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));writer.writeheader();writer.writerows(rows)
    (out/"analysis_manifest.json").write_text(json.dumps({"source_dirs":args.result_dirs,"coverage_complete":coverage["complete"],
        "bootstrap_replicates":10000,"bootstrap_seed":20260924,"legacy_output_preserved":True},ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":main()

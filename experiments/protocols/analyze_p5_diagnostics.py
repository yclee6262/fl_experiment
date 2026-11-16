"""Create post-hoc P5 alpha-quality, query-quality, and representative-case data."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from research_analysis import finite_float, valid_delivery


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    if not rows:
        Path(path).write_text("\n", encoding="utf-8")
        return
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("results_root");parser.add_argument("--output-dir",required=True)
    args=parser.parse_args();root=Path(args.results_root);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise FileExistsError(out)
    suite=json.loads((root/"suite_manifest.json").read_text())
    q_scatter=[];round_cost=[];representatives=[];correlations=[]
    for spec in suite["scenarios"]:
        scenario=spec["name"];directory=root/scenario
        summaries=read(directory/"summary.csv")
        pairs={}
        for row in summaries:
            valid=valid_delivery(row)
            gap=finite_float(row.get("final_consistency_l1"));error=finite_float(row.get("external_global_error"))
            q_scatter.append({"scenario":scenario,"seed":row.get("seed"),"method":row.get("method_label"),
                "valid_delivery":valid,"alpha_q_l1":gap if valid else None,
                "pooled_error":error if valid else None,
                "interpretation":"descriptive mechanism diagnostic; not a selection criterion"})
            pairs.setdefault(int(row["seed"]),{})[row["method_label"]]=row
        values=[r for r in q_scatter if r["scenario"]==scenario and r["valid_delivery"]
                and r["alpha_q_l1"] is not None and r["pooled_error"] is not None]
        rho=(float(spearmanr([r["alpha_q_l1"] for r in values],[r["pooled_error"] for r in values]).statistic)
             if len(values)>=3 and np.ptp([r["alpha_q_l1"] for r in values])>0
             and np.ptp([r["pooled_error"] for r in values])>0 else None)
        correlations.append({"scenario":scenario,"valid_method_seed_rows":len(values),
            "spearman_alpha_q_l1_vs_pooled_error":rho,
            "interpretation":"post-hoc association only; correlated method-seed rows are not independent evidence"})
        round_path=directory/"rounds.csv"
        if round_path.exists():
            for row in read(round_path):
                stage3=finite_float(row.get("stage3_requests_cumulative"));contrib=finite_float(row.get("contribution_requests_cumulative"))
                stage_samples=finite_float(row.get("stage3_input_samples_cumulative"));contrib_samples=finite_float(row.get("contribution_input_samples_cumulative"))
                round_cost.append({"scenario":scenario,"seed":row.get("seed"),"method":row.get("method_label"),
                    "round":row.get("round"),"current_state_accepted":row.get("current_state_accepted"),
                    "proposal_generated":row.get("proposal_generated"),
                    "cumulative_stage3_and_contribution_requests":(stage3 or 0)+(contrib or 0)
                        if stage3 is not None or contrib is not None else None,
                    "cumulative_stage3_and_contribution_input_samples":(stage_samples or 0)+(contrib_samples or 0)
                        if stage_samples is not None or contrib_samples is not None else None,
                    "round_external_global_error":finite_float(row.get("external_global_error")),
                    "round_target_error":finite_float(row.get("target_error")),
                    "note":"post-hoc score only; never used to choose delivered round"})
        for seed,methods in pairs.items():
            a,d=methods.get("A_uniform_no_update"),methods.get("D_uniform_guarded")
            if a is None or d is None:continue
            aok,dok=valid_delivery(a),valid_delivery(d)
            delta=(float(d["external_global_error"])-float(a["external_global_error"]) if aok and dok else None)
            representatives.append({"scenario":scenario,"seed":seed,
                "case_type":"paired_valid" if delta is not None else "paired_failure_or_missing_quality",
                "external_error_delta_D_minus_A":delta,
                "A_status":a.get("delivery_status","unknown"),"D_status":d.get("delivery_status","unknown"),
                "A_error":a.get("external_global_error",""),"D_error":d.get("external_global_error",""),
                "A_stop_reason":a.get("stop_reason",""),"D_stop_reason":d.get("stop_reason",""),
                "seed_selection_rule":"case table only; external scores do not affect delivery"})
    # Label the per-scenario median, best, and worst valid paired outcomes explicitly.
    for scenario in [spec["name"] for spec in suite["scenarios"]]:
        selected=[r for r in representatives if r["scenario"]==scenario and r["external_error_delta_D_minus_A"] is not None]
        for label,selector in (("largest_D_improvement",min),("median_delta",None),("largest_D_worsening",max)):
            if not selected:continue
            ordered=sorted(selected,key=lambda row:float(row["external_error_delta_D_minus_A"]))
            chosen=(ordered[0] if label=="largest_D_improvement" else ordered[-1] if label=="largest_D_worsening" else
                    min(ordered,key=lambda row:abs(float(row["external_error_delta_D_minus_A"])-float(np.median([float(x["external_error_delta_D_minus_A"]) for x in ordered])))))
            representatives.append({**chosen,"case_type":label})
    write(out/"alpha_q_quality.csv",q_scatter);write(out/"alpha_q_correlations.csv",correlations)
    write(out/"round_quality_cost.csv",round_cost);write(out/"representative_cases.csv",representatives)


if __name__=="__main__":main()

"""Aggregate P6 proxy-validity evidence using seed as the sampling unit."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from research_analysis import audit_coverage, finite_float


def main():
    parser=argparse.ArgumentParser();parser.add_argument("results_dir");parser.add_argument("--output-dir",required=True)
    args=parser.parse_args();root=Path(args.results_dir);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise FileExistsError(out)
    manifest=json.loads((root/"run_manifest.json").read_text())
    with (root/"objective_validity.csv").open(newline="",encoding="utf-8") as handle:rows=list(csv.DictReader(handle))
    seeds=[str(s) for s in manifest["seeds"]];budgets=[float(b) for b in manifest["budgets"]];etas=[float(e) for e in manifest["etas"]]
    normalized=[{**r,"seed":str(int(r["seed"])),"budget":str(float(r["budget"])),"eta":str(float(r["eta"]))} for r in rows]
    expected=[(s,str(b),str(e)) for s in seeds for b in budgets for e in etas]
    coverage=audit_coverage(normalized,expected,("seed","budget","eta"))
    groups={}
    for row in normalized:groups.setdefault((row["budget"],row["eta"]),[]).append(row)
    summaries=[]
    for (budget,eta),items in sorted(groups.items()):
        rhos=[finite_float(row.get("spearman_objective_vs_negative_pooled_error")) for row in items]
        rhos=[v for v in rhos if v is not None]
        regrets=[finite_float(row.get("selected_pooled_error_regret")) for row in items]
        regrets=[v for v in regrets if v is not None]
        summaries.append({"budget":budget,"eta":eta,"planned_seeds":len(seeds),"observed_seed_rows":len(items),
            "defined_spearman_seeds":len(rhos),"undefined_spearman_seeds":len(items)-len(rhos),
            "mean_spearman_across_seeds":float(np.mean(rhos)) if rhos else None,
            "median_spearman_across_seeds":float(np.median(rhos)) if rhos else None,
            "mean_top5_overlap_fraction_across_seeds":float(np.mean([float(r["top5_overlap_fraction"]) for r in items if r.get("top5_overlap_fraction") not in (None,"")] )) if any(r.get("top5_overlap_fraction") not in (None,"") for r in items) else None,
            "mean_selected_set_pooled_error_regret":float(np.mean(regrets)) if regrets else None,
            "median_selected_set_pooled_error_regret":float(np.median(regrets)) if regrets else None,
            "valid_feasible_coalitions_mean":float(np.mean([int(r["valid_feasible_coalitions"]) for r in items])) if items else None,
            "failed_coalitions_mean":float(np.mean([int(r["failed_coalitions"]) for r in items])) if items else None,
            "statistical_unit":"seed; coalitions are within-seed enumeration items, not independent replicates"})
    for name,data in (("seed_budget_eta.csv",rows),("budget_eta_summary.csv",summaries)):
        path=out/name
        if data:
            with path.open("x",newline="",encoding="utf-8") as handle:
                fields=list(dict.fromkeys(k for row in data for k in row));writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(data)
    coverage_row={**coverage,"duplicate_keys":json.dumps(coverage["duplicate_keys"]),
        "missing_keys":json.dumps(coverage["missing_keys"]),"unexpected_keys":json.dumps(coverage["unexpected_keys"])}
    with (out/"coverage_audit.csv").open("x",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(coverage_row));writer.writeheader();writer.writerow(coverage_row)
    (out/"analysis_manifest.json").write_text(json.dumps({"results_dir":str(root.resolve()),
        "sampling_unit":"seed","coalitions_not_independent":True,"complete_seed_budget_eta_coverage":coverage["complete"]},indent=2)+"\n")


if __name__=="__main__":main()

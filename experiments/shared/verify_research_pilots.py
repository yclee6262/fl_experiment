"""Manifest-driven pilot acceptance checks for P4, P5, and P6."""

import argparse
import csv
import json
import hashlib
from pathlib import Path

from research_analysis import audit_coverage, finite_float


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def check_valid_deliveries(rows, feasibility_field):
    errors=[]
    for row in rows:
        if row.get("delivery_status") != "valid":
            if row.get("execution_status") == "interrupted":
                errors.append(f"unexpected interruption: {row.get('failure_reason','')}")
            elif not row.get("failure_reason", row.get("stop_reason", row.get("error", ""))):
                errors.append("recorded failure lacks reason")
            continue
        if str(row.get(feasibility_field, "")).lower() != "true":
            errors.append("successful delivery flagged infeasible")
        violation=finite_float(row.get("max_box_violation"))
        if violation is not None and violation > 1e-8:
            errors.append(f"box violation {violation} exceeds 1e-8")
    return errors


def verify_hashes(root, source_hashes, artifact_hashes):
    root=Path(root);errors=[]
    for name,expected in (source_hashes or {}).items():
        snapshot=root/"source_snapshot"/name
        if not snapshot.is_file():errors.append(f"missing source snapshot: {snapshot}");continue
        actual=hashlib.sha256(snapshot.read_bytes()).hexdigest()
        if actual!=expected:errors.append(f"source hash mismatch: {snapshot}")
    for name,expected in (artifact_hashes or {}).items():
        artifact=root/name
        if not artifact.is_file():errors.append(f"missing artifact: {artifact}");continue
        actual=hashlib.sha256(artifact.read_bytes()).hexdigest()
        if actual!=expected:errors.append(f"artifact hash mismatch: {artifact}")
    return errors


def verify_p4(root):
    root=Path(root); suite=json.loads((root/"suite_manifest.json").read_text())
    errors=[]; counts=0
    for spec in suite["comparisons"]:
        directory=Path(spec["arguments"]["output_dir"])
        if not directory.is_absolute(): directory=Path.cwd()/directory
        manifest=json.loads((directory/"run_manifest.json").read_text())
        errors.extend(verify_hashes(directory,manifest.get("source_sha256"),manifest.get("artifact_sha256")))
        seeds=[int(s) for s in manifest["arguments"]["seeds"]]
        rows=read_rows(directory/"summary.csv")
        run_manifest=json.loads((directory/"run_manifest.json").read_text())
        errors.extend(verify_hashes(directory,run_manifest.get("source_sha256"),run_manifest.get("artifact_sha256")))
        coverage=audit_coverage(rows,[(str(seed),) for seed in seeds],("seed",))
        if not coverage["complete"]: errors.append(f"P4 incomplete {directory}: {coverage}")
        errors.extend(check_valid_deliveries(rows,"final_box_feasible"))
        if any(row.get("payment_status")!="not_run_by_configuration" for row in rows):
            errors.append(f"P4 settlement unexpectedly ran: {directory}")
        reevaluation=read_rows(directory/"reevaluation.csv")
        if len(reevaluation)!=len(rows): errors.append(f"P4 re-evaluation row mismatch: {directory}")
        for row in reevaluation:
            if row.get("reevaluation_status")=="scored" and (
                row.get("external_score_matches")!="True" or row.get("true_score_matches")!="True"
                or row.get("recomputed_box_feasible")!="True"):
                errors.append(f"P4 re-evaluation mismatch: {directory}/{row.get('seed')}")
            elif row.get("reevaluation_status")=="not_applicable" and not row.get("reason"):
                errors.append(f"P4 non-delivery lacks a re-evaluation reason: {directory}")
        counts+=len(rows)
    analysis=read_rows(root/"analysis/aggregate_results.csv")
    if len(analysis)!=2 or any(int(row["accepted_state_mismatches"]) for row in analysis):
        errors.append("P4 accepted-state comparison did not pass")
    return {"rows":counts,"errors":errors,"accepted":not errors}


def verify_p5(root):
    root=Path(root); suite=json.loads((root/"suite_manifest.json").read_text())
    methods=suite["comparison"]; seeds=[str(s) for s in suite["seeds"]]
    errors=[]; counts=0
    for spec in suite["scenarios"]:
        scenario=spec["name"]; directory=root/scenario
        run_manifest=json.loads((directory/"run_manifest.json").read_text())
        errors.extend(f"{scenario}: {error}" for error in verify_hashes(
            directory,run_manifest.get("source_sha256"),run_manifest.get("artifact_sha256")))
        rows=read_rows(directory/"summary.csv")
        coverage=audit_coverage(rows,[(method,seed) for method in methods for seed in seeds],("method_label","seed"))
        if not coverage["complete"]: errors.append(f"P5 incomplete {scenario}: {coverage}")
        errors.extend(f"{scenario}: {error}" for error in check_valid_deliveries(rows,"final_box_feasible"))
        for row in rows:
            if row.get("payment_status")!="not_run_by_configuration": errors.append(f"P5 payment ran: {scenario}/{row.get('seed')}")
            artifact=directory/row.get("scenario_artifact_dir","")
            if not artifact.is_dir(): errors.append(f"P5 missing artifact {artifact}")
        reevaluation=read_rows(directory/"reevaluation.csv")
        if len(reevaluation)!=len(rows): errors.append(f"P5 re-evaluation row mismatch: {scenario}")
        for row in reevaluation:
            if row.get("reevaluation_status")=="scored" and (
                row.get("external_score_matches")!="True" or row.get("true_score_matches")!="True"
                or row.get("recomputed_box_feasible")!="True"):
                errors.append(f"P5 re-evaluation mismatch: {scenario}/{row.get('seed')}")
            elif row.get("reevaluation_status")=="not_applicable" and not row.get("reason"):
                errors.append(f"P5 non-delivery lacks a re-evaluation reason: {scenario}/{row.get('seed')}")
        counts+=len(rows)
    return {"rows":counts,"errors":errors,"accepted":not errors}


def verify_p6(root):
    root=Path(root); manifest=json.loads((root/"run_manifest.json").read_text())
    errors=verify_hashes(root,manifest.get("source_sha256"),manifest.get("artifact_sha256"))
    seeds=[str(s) for s in manifest["seeds"]]
    budgets=[str(float(b)) for b in manifest["budgets"]]
    etas=[str(float(e)) for e in manifest["etas"]]
    methods=("exact","objective_greedy","legacy_greedy")
    rows=read_rows(root/"selection_results.csv")
    norm=[{**r,"seed":str(int(r["seed"])),"budget":str(float(r.get("budget",r.get("selection_budget")))),
           "diversity_eta":str(float(r["diversity_eta"]))} for r in rows]
    expected=[(s,m,b,e) for s in seeds for b in budgets for e in etas for m in methods]
    coverage=audit_coverage(norm,expected,("seed","selection_method","budget","diversity_eta"))
    if not coverage["complete"]: errors.append(f"P6 incomplete: {coverage}")
    errors.extend(check_valid_deliveries(rows,"box_feasible"))
    for row in rows:
        if row.get("payment_status")!="not_run_by_configuration": errors.append("P6 settlement ran")
        if row.get("delivery_status")=="valid" and row.get("budget_feasible")!="True": errors.append("P6 valid selection violated budget")
        if row.get("delivery_status")=="valid" and int(row.get("coalition_size") or 0)>3: errors.append("P6 coalition over size cap")
    grouped={}
    for row in rows: grouped.setdefault((row["seed"],row["budget"],row["diversity_eta"]),{})[row["selection_method"]]=row
    for key,methods_at in grouped.items():
        a=finite_float(methods_at.get("exact",{}).get("selection_objective"))
        b=finite_float(methods_at.get("objective_greedy",{}).get("selection_objective"))
        if a is not None and b is not None and a+1e-12<b: errors.append(f"exact objective below greedy {key}")
    reevaluation=read_rows(root/"reevaluation.csv")
    if len(reevaluation)!=len(rows): errors.append("P6 re-evaluation row mismatch")
    if any(r.get("reevaluation_status")=="scored" and (r.get("external_score_matches")!="True" or r.get("true_score_matches")!="True") for r in reevaluation):
        errors.append("P6 re-evaluation mismatch")
    return {"rows":len(rows),"errors":errors,"accepted":not errors}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--p4",required=True);parser.add_argument("--p5",required=True);parser.add_argument("--p6",required=True)
    args=parser.parse_args()
    result={"p4":verify_p4(args.p4),"p5":verify_p5(args.p5),"p6":verify_p6(args.p6)}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not all(item["accepted"] for item in result.values()): raise SystemExit(1)


if __name__=="__main__":main()

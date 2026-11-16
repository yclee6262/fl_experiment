"""Seed-level mechanism analysis and artifact reload acceptance for bootstrap alpha."""
import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm

from bootstrap_quality_research import KEYS, METHODS, append, calibration, digest, dump, load_base
from initial_alpha_methods import initial_alpha
from research_analysis import audit_coverage, paired_bootstrap_mean_ci
from research_evaluation import load_evaluator_artifact, score_solution, synthetic_truth


def read(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def valid(row):
    if row.get("status") != "valid":
        return False
    x = np.asarray(row.get("X"), float)
    return (x.shape == (2,) and np.isfinite(x).all() and np.max(np.abs(x)) <= 1+1e-8
            and row.get("pooled_error") is not None and np.isfinite(row["pooled_error"]))


def summary(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n":0,"mean":None,"median":None,"Q1":None,"Q3":None}
    return {"n":len(a),"mean":float(a.mean()),"median":float(np.median(a)),
            "Q1":float(np.quantile(a,.25)),"Q3":float(np.quantile(a,.75))}


def paired_stats(by_seed, formal=False):
    values = [float(np.mean(v)) for v in by_seed.values() if len(v)]
    ordinary = paired_bootstrap_mean_ci(values)
    simultaneous = paired_bootstrap_mean_ci(values, confidence=1-.05/9)
    return {"n_seeds":len(values),"mean":float(np.mean(values)) if values else None,
            "seed_SD":float(np.std(values,ddof=1)) if len(values)>1 else None,
            "CI95":ordinary, "CI_simultaneous_9":simultaneous if formal else None,
            "better_seeds":sum(v<0 for v in values),"worse_seeds":sum(v>0 for v in values)}


def verify(root, manifest, rows):
    errors = []
    expected = list(itertools.product(manifest["scenarios"],manifest["seeds"],manifest["draws"],
                                     manifest["sizes"],manifest["targets"],manifest["methods"]))
    coverage = audit_coverage(rows, expected, KEYS)
    if not coverage["complete"]:
        errors.append("condition coverage incomplete")
    for name, sha in manifest["source_sha256"].items():
        if not (root/"source_snapshot"/name).exists() or digest(root/"source_snapshot"/name) != sha:
            errors.append(f"source snapshot mismatch {name}")
    for name, sha in manifest.get("output_sha256",{}).items():
        if not (root/name).exists() or digest(root/name) != sha:
            errors.append(f"output hash mismatch {name}")
    local = {}
    evaluators = {}
    for scenario in manifest["scenarios"]:
        for seed in manifest["seeds"]:
            reference = json.loads((root/f"artifact_reference_{scenario}_{seed}.json").read_text())
            artifact = Path(reference["artifact"])
            for name, sha in reference["sha256"].items():
                if not (artifact/name).exists() or digest(artifact/name) != sha:
                    errors.append(f"artifact hash mismatch {scenario}/{seed}/{name}")
            local[(scenario,seed)] = load_base(manifest["source"],scenario,seed)[0]
            evaluators[(scenario,seed)] = load_evaluator_artifact(artifact)[0]
    calibration_rows = read(root/"calibration.jsonl")
    cal_index = {}
    max_prediction_diff = 0.
    for row in calibration_rows:
        key = (row["scenario"],row["seed"],row["target"],row["draw"])
        if key in cal_index:
            errors.append(f"duplicate calibration {key}")
        server = local[key[:2]]
        x, y = np.asarray(row["X"]), np.asarray(row["y"])
        expected_x, expected_y = ((server.test_X, server.test_y) if manifest["original_blind"]
                                  else calibration(row["seed"],row["draw"],max(manifest["sizes"])))
        if not np.array_equal(x,expected_x) or not np.array_equal(y,expected_y):
            errors.append(f"calibration regeneration mismatch {key}")
        predicted = np.column_stack([a.api_predict(x) for a in server.trusted_agents])
        delta = float(np.max(np.abs(predicted-np.asarray(row["prediction"]))))
        max_prediction_diff = max(delta,max_prediction_diff)
        if delta > 1e-7:
            errors.append(f"calibration model reload mismatch {key}")
        batches = {}
        for n, saved in row.get("prediction_by_n",{}).items():
            batch = np.column_stack([a.api_predict(x[:int(n)]) for a in server.trusted_agents])
            if not np.array_equal(batch,np.asarray(saved)):
                errors.append(f"qualification batch reload mismatch {key}/{n}")
            batches[int(n)] = batch
        cal_index[key] = (x,y,predicted,batches)
    max_alpha_diff, max_score_diff, successes = 0.,0.,0
    for row in rows:
        key = (row["scenario"],row["seed"],row["target"],row["draw"])
        if key not in cal_index:
            errors.append(f"missing calibration {key}")
            continue
        if row["status"] == "interrupted":
            errors.append(f"program interrupted {key}/{row['method']}")
            continue
        if "alpha" not in row:
            errors.append(f"missing alpha {key}/{row['method']}")
            continue
        x,y,prediction,batches = cal_index[key]
        n = row["blind_n"]
        x,y,prediction = x[:n],y[:n],batches.get(n,prediction[:n])
        mse = ((prediction-y[:,None])**2).mean(axis=0)
        if not np.allclose(mse,np.asarray(row["mse"]),atol=1e-12,rtol=0):
            errors.append(f"saved MSE mismatch {key}/{row['method']}")
        server = local[key[:2]]
        alpha,_ = initial_alpha(row["method"],server.trusted_agents,mse,x,y,
            server.I_list,row["target"],bootstrap_replicates=1000,
            bootstrap_seed=row["bootstrap_seed"],cached_blind_predictions=prediction)
        delta = float(np.max(np.abs(alpha-np.asarray(row["alpha"]))))
        max_alpha_diff = max(delta,max_alpha_diff)
        if delta>1e-7:
            errors.append(f"alpha reload mismatch {key}/{row['method']}")
        if row["initialization_requests"] != 0 or row["initialization_input_samples"] != 0:
            errors.append(f"reuse queried API {key}/{row['method']}")
        if row["status"] == "valid":
            if not valid(row):
                errors.append(f"invalid delivered X/score {key}/{row['method']}")
                continue
            successes += 1
            x_final = np.asarray(row["X"])
            violation = max(0.,float(np.max(np.abs(x_final)))-1)
            if abs(violation-row["max_box_violation"])>1e-8:
                errors.append(f"violation mismatch {key}/{row['method']}")
            _,score = score_solution(evaluators[key[:2]],x_final,row["target"])
            delta = abs(score-row["pooled_error"])
            max_score_diff = max(delta,max_score_diff)
            if delta>1e-7:
                errors.append(f"pooled reload mismatch {key}/{row['method']}")
            truth = abs(float(synthetic_truth(x_final[None,:])[0])-row["target"])
            if abs(truth-row["truth_error"])>1e-7:
                errors.append(f"truth mismatch {key}/{row['method']}")
    for pair in read(root/"reuse.jsonl"):
        fresh,reused = pair["fresh"],pair["reused"]
        if fresh["status"] != reused["status"]:
            errors.append("reuse delivery status changed")
        for field in ["alpha","X"]:
            if fresh[field] is not None and not np.array_equal(fresh[field],reused[field]):
                errors.append(f"reuse changed {field}")
        if (fresh["stage3_requests"] != reused["stage3_requests"]
            or fresh["initialization_requests"] != 3 or reused["initialization_requests"] != 0):
            errors.append("reuse query accounting mismatch")
    result = {"passed":not errors and manifest["status"]=="complete", "errors":errors,
        "coverage":coverage,"expected":len(expected),"observed":len(rows),"valid":successes,
        "max_alpha_reload_diff":max_alpha_diff,"max_pooled_reload_diff":max_score_diff,
        "max_blind_prediction_reload_diff":max_prediction_diff,
        "reuse_pairs_checked":len(read(root/"reuse.jsonl"))}
    dump(root/"verification.json",result)
    return result


def analyze(root, formal=False, verification=True):
    root = Path(root)
    manifest = json.loads((root/"manifest.json").read_text())
    rows = read(root/"results.jsonl")
    checked = verify(root,manifest,rows) if verification else None
    if formal:
        registration = json.loads(Path(
            "docs/bootstrap_quality_independent_formal_preregistration_v2_20261008.json").read_text())
        formal_errors = []
        if (manifest["seeds"] != registration["seeds"] or manifest["sizes"] != [20]
            or not manifest["original_blind"] or manifest["bootstrap_replicates"] != 1000
            or manifest["targets"] != [-.5,0.,.5]
            or manifest["methods"] != registration["methods"]):
            formal_errors.append("formal conditions do not match registration")
        if manifest["source_sha256"] != registration["source_sha256"]:
            formal_errors.append("formal source snapshot does not match registration")
        source_verification = Path(manifest["source"])/"verification.json"
        if not source_verification.exists() or not json.loads(source_verification.read_text())["passed"]:
            formal_errors.append("source preparation not verified")
        if checked:
            checked["errors"].extend(formal_errors)
            checked["passed"] = checked["passed"] and not formal_errors
            dump(root/"verification.json",checked)
    index = {tuple(r[k] for k in KEYS):r for r in rows}
    groups, comparisons, stability, conflicts, cases = [],[],[],[],[]
    for scenario,n,target in itertools.product(manifest["scenarios"],manifest["sizes"],manifest["targets"]):
        for method in METHODS:
            chosen = [r for r in rows if r["scenario"]==scenario and r["blind_n"]==n
                      and r["target"]==target and r["method"]==method]
            values = [r["pooled_error"] for r in chosen if valid(r)]
            per_seed = defaultdict(list)
            for r in chosen:
                if valid(r):per_seed[r["seed"]].append(r)
            noise_stats = []
            for seed,rs in per_seed.items():
                if len(rs) < 2:continue
                alpha = np.asarray([r["alpha"] for r in rs])
                xs = np.asarray([r["X"] for r in rs])
                noise_stats.append({"seed":seed,
                    "alpha_variance":float(np.var(alpha,axis=0,ddof=1).sum()),
                    "X_variance":float(np.var(xs,axis=0,ddof=1).sum()),
                    "pooled_variance":float(np.var([r["pooled_error"] for r in rs],ddof=1)),
                    "pooled_p90":float(np.quantile([r["pooled_error"] for r in rs],.9))})
            if noise_stats:
                stability.append({"scenario":scenario,"n":n,"target":target,"method":method,
                    "n_seeds":len(noise_stats),"per_seed":noise_stats,
                    **{k:float(np.mean([r[k] for r in noise_stats])) for k in
                       ["alpha_variance","X_variance","pooled_variance","pooled_p90"]}})
            def average_cost(field):
                vs = [r.get(field) for r in chosen]
                return float(np.mean(vs)) if vs and all(v is not None for v in vs) else None
            group = {"scenario":scenario,"n":n,"target":target,"method":method,
                     "runs":len(chosen),"failed":sum(not valid(r) for r in chosen),
                     "pooled":summary(values),"truth":summary([r["truth_error"] for r in chosen if valid(r)]),
                     "n_complete_seeds":sum(len(rs)==len(manifest["draws"]) for rs in per_seed.values()),
                     "seed_mean_pooled":summary([np.mean([r["pooled_error"] for r in rs])
                                                 for rs in per_seed.values()])}
            for field in ["initialization_seconds","stage3_seconds","stage3_requests","stage3_input_samples",
                          "calibration_requests_independent","calibration_inputs_independent","additional_labels"]:
                group[field] = average_cost(field)
            groups.append(group)
        for baseline in ["A_uniform","B_stage0","quality_shrink"]:
            by_seed, pair_failures, both_valid = defaultdict(list),defaultdict(int),0
            method_failures,base_failures = 0,0
            for seed,draw in itertools.product(manifest["seeds"],manifest["draws"]):
                a = index.get((scenario,seed,draw,n,target,"bootstrap_quality"))
                b = index.get((scenario,seed,draw,n,target,baseline))
                method_failures += not valid(a) if a else True
                base_failures += not valid(b) if b else True
                if not a or not b or not valid(a) or not valid(b):
                    pair_failures["missing_or_invalid"] += 1
                    continue
                delta = a["pooled_error"]-b["pooled_error"]
                by_seed[seed].append(delta)
                both_valid += 1
                truth_delta = a["truth_error"]-b["truth_error"]
                if delta*truth_delta < 0:
                    conflicts.append({"scenario":scenario,"seed":seed,"draw":draw,"n":n,
                        "target":target,"baseline":baseline,"pooled_delta":delta,"truth_delta":truth_delta})
            complete = {seed:vals for seed,vals in by_seed.items() if len(vals)==len(manifest["draws"])}
            stats = paired_stats(complete,formal and target==0 and n==20)
            comparison = {"scenario":scenario,"n":n,"target":target,"baseline":baseline,
                "valid_run_pairs":both_valid,"incomplete_seed_count":len(manifest["seeds"])-len(complete),
                "excluded":dict(pair_failures),"bootstrap_failures":method_failures,
                "baseline_failures":base_failures,**stats}
            comparison["confirmed_advantage"] = bool(formal and n==20 and target==0
                and checked and checked["passed"] and len(complete)==len(manifest["seeds"])
                and stats["CI_simultaneous_9"][1] is not None
                and stats["CI_simultaneous_9"][1]<0 and method_failures<=base_failures)
            comparison["conflict_run_pairs"] = sum(c["scenario"]==scenario and c["n"]==n
                and c["target"]==target and c["baseline"]==baseline for c in conflicts)
            comparisons.append(comparison)
            if complete:
                means = sorted((float(np.mean(v)),s) for s,v in complete.items())
                for label,(delta,seed) in zip(["best","median","worst"],
                                            [means[0],means[len(means)//2],means[-1]]):
                    cases.append({"scenario":scenario,"n":n,"target":target,"baseline":baseline,
                                  "case":label,"seed":seed,"mean_delta":delta})
    sensitivity = []
    srows = read(root/"bootstrap_sensitivity.jsonl")
    for scenario in manifest["scenarios"]:
        for b in [100,500,1000]:
            differences,times = [],[]
            for seed,draw in itertools.product(manifest["seeds"],manifest["draws"]):
                rs = [r for r in srows if r["scenario"]==scenario and r["seed"]==seed and r["draw"]==draw]
                refs = [r["alpha"] for r in rs if r["B"]==10000]
                tests = [r for r in rs if r["B"]==b]
                if not refs or not tests:continue
                center = np.mean(refs,axis=0)
                differences.extend(float(np.max(np.abs(np.asarray(r["alpha"])-center))) for r in tests)
                times.extend(r["seconds"] for r in tests)
            sensitivity.append({"scenario":scenario,"B":b,"reference":"mean of5 independent B=10000 runs",
                "max_alpha_error":summary(differences),"seconds":summary(times)})
    reuse = []
    for scenario in manifest["scenarios"]:
        rs = [r for r in read(root/"reuse.jsonl") if r["scenario"]==scenario]
        if rs:
            reuse.append({"scenario":scenario,"pairs":len(rs),
                "fresh_initial_seconds":float(np.mean([r["fresh"]["initialization_seconds"] for r in rs])),
                "reuse_initial_seconds":float(np.mean([r["reused"]["initialization_seconds"] for r in rs])),
                "saved_requests_per_run":3,"saved_input_samples_per_run":60})
    result = {"phase":"formal" if formal else "exploratory", "verification":checked,
        "summaries":groups,"comparisons":comparisons,"blind_stability":stability,
        "bootstrap_sensitivity":sensitivity,"prediction_reuse":reuse,
        "conflicts":conflicts,"representative_cases":cases,
        "statistical_unit":"seed; replacement blind draws averaged within seed; no pseudo-replication"}
    dump(root/"analysis.json",result)
    return result


def power_plan(result):
    contrasts = [r for r in result["comparisons"] if r["target"]==0 and r["n"]==20]
    sd = max(r["seed_SD"] for r in contrasts if r["seed_SD"] is not None)
    nominal = int(np.ceil(((norm.ppf(1-.05/18)+norm.ppf(.8))*sd/.005)**2))
    count = max(30,min(120,nominal))
    return {"resolution_delta":.005,"max_pilot_paired_SD":sd,"normal_approx_required_N":nominal,
            "formal_N":count,"capped":nominal>120,"seeds":list(range(4200,4200+count)),
            "limitations":"pilot SD uncertain; normal approximation for planning only; NOT proof of power or business threshold"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--formal",action="store_true")
    parser.add_argument("--power-plan",action="store_true")
    args = parser.parse_args()
    result = analyze(args.root,args.formal)
    if args.power_plan:
        dump(Path(args.root)/"formal_sample_size.json",power_plan(result))
    print(json.dumps({"verification":result["verification"],"comparisons":len(result["comparisons"])},ensure_ascii=False))
    raise SystemExit(0 if result["verification"]["passed"] else 1)

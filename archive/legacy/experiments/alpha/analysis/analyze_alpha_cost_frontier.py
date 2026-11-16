"""Exploratory quality/request frontier from previously solved alpha candidates.

No pooled result is used to choose a candidate within a seed. Saved internal
scores replay the same selector on a fixed, predeclared subset of candidates.
All cohorts are reused and cannot establish a new quality or cost guarantee.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from run_validation_calibrated_alpha import CANDIDATE_METHODS, META_METHOD


PORTFOLIOS = {
    "A_only": ("A_uniform",),
    "B_only": ("B_stage0",),
    "A_B": ("A_uniform", "B_stage0"),
    "A_B_target": ("A_uniform", "B_stage0", "target_local_precision"),
    "A_B_mse": ("A_uniform", "B_stage0", "mse_power_1p5"),
    "four_without_target": (
        "A_uniform", "B_stage0", "mse_power_1p5", "corr_precision_0p5"),
    "original_five": CANDIDATE_METHODS,
}


def _rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _index(rows, expected, kind):
    result = {}
    for row in rows:
        key = (int(row["seed"]), row["method"])
        if key in result:
            raise ValueError(f"Duplicate {kind} key: {key}")
        result[key] = row
    if set(result) != expected:
        raise ValueError(f"{kind} coverage mismatch: missing={expected-set(result)}, "
                         f"unexpected={set(result)-expected}")
    return result


def analyze(root):
    root = Path(root)
    manifest = json.loads((root / "suite_manifest.json").read_text())
    verified = json.loads((root / "selection_verification.json").read_text())
    if (manifest["status"] != "complete" or not verified["passed"]
            or len(manifest["scenarios"]) != 1):
        raise ValueError(f"Incomplete or unverified root: {root}")
    scenario, = manifest["scenarios"]
    folder = root / scenario
    seeds = [int(seed) for seed in manifest["seeds"]]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Duplicate manifest seed: {root}")
    candidates = _index(_rows(folder / "candidate_rows.csv"),
                        {(s, m) for s in seeds for m in
                         manifest["candidate_initializations"]}, "candidate")
    summaries = _index(_rows(folder / "summary.csv"),
                       {(s, m) for s in seeds for m in manifest["methods"]},
                       "summary")
    records = []
    for seed in seeds:
        baseline = summaries[seed, META_METHOD]
        audit = json.loads(baseline["alpha_audit"])
        if (tuple(audit.get("candidate_method_order", CANDIDATE_METHODS))
                != CANDIDATE_METHODS or
                set(audit["internal_scores"]) != set(CANDIDATE_METHODS)):
            raise ValueError(f"Unexpected baseline candidate order: {scenario}/{seed}")
        n_agents = len(json.loads(baseline["agent_ids"]))
        scores = {method: float(audit["internal_scores"][method][
            "internal_target_error"]) for method in CANDIDATE_METHODS}
        pooled = {method: float(candidates[seed, method]["external_global_error"])
                  for method in CANDIDATE_METHODS}
        requests = {method: (float(candidates[seed, method]["initialization_requests"])
                             +float(candidates[seed, method]["stage3_requests"]))
                    for method in CANDIDATE_METHODS}
        if (not all(np.isfinite(x) for x in (*scores.values(), *pooled.values(),
                                            *requests.values()))
                or any(candidates[seed, method]["delivery_status"] != "valid" or
                       candidates[seed, method]["final_box_feasible"] != "True"
                       for method in CANDIDATE_METHODS)):
            raise ValueError(f"Invalid saved candidate: {scenario}/{seed}")
        original_choice = min(CANDIDATE_METHODS, key=lambda method: scores[method])
        if (original_choice != audit["selected_initialization"] or
                abs(pooled[original_choice]-float(
                    baseline["external_global_error"])) > 1e-7):
            raise ValueError(f"Baseline choice mismatch: {scenario}/{seed}")
        full_requests = sum(requests.values()) + n_agents*(1+len(CANDIDATE_METHODS))
        saved_requests = (float(baseline["initialization_requests"])
                          +float(baseline["stage3_requests"]))
        if abs(full_requests-saved_requests) > 1e-7:
            raise ValueError(f"Baseline cost mismatch: {scenario}/{seed}")
        for name, methods in PORTFOLIOS.items():
            choice = min(methods, key=lambda method: scores[method])
            if len(methods) == 1:
                # A or B alone does not fit the multi-candidate residual
                # calibrator or score another candidate.
                cost = requests[choice]
            else:
                # One calibration query and one scoring query per agent and
                # candidate; matched exactly to the saved five-way cost above.
                cost = sum(requests[m] for m in methods) + n_agents*(1+len(methods))
            records.append({
                "scenario": scenario, "seed": seed, "portfolio": name,
                "chosen": choice, "pooled_error": pooled[choice],
                "model_requests": cost,
                "pooled_delta_vs_five": pooled[choice]-pooled[original_choice],
                "request_delta_vs_five": cost-full_requests,
                "choice_differs_from_five": choice != original_choice,
            })
    report = []
    for name in PORTFOLIOS:
        group = [r for r in records if r["portfolio"] == name]
        delta = np.asarray([r["pooled_delta_vs_five"] for r in group])
        report.append({
            "scenario": scenario, "portfolio": name,
            "seed_count": len(group), "candidate_count": len(PORTFOLIOS[name]),
            "mean_pooled_error": float(np.mean([r["pooled_error"] for r in group])),
            "mean_pooled_delta_vs_five": float(delta.mean()),
            "q95_pooled_delta_vs_five": float(np.quantile(delta, .95)),
            "worst_pooled_delta_vs_five": float(delta.max()),
            "mean_model_requests": float(np.mean([r["model_requests"] for r in group])),
            "mean_request_delta_vs_five": float(np.mean(
                [r["request_delta_vs_five"] for r in group])),
            "choice_differs_from_five_count": sum(
                r["choice_differs_from_five"] for r in group),
            "nonzero_pooled_difference_count": int(np.count_nonzero(
                np.abs(delta)>1e-10)),
            "evidence_status": "reused_cohort_exploratory_no_formal_claim",
        })
    return report, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    report, records = [], []
    for root in args.roots:
        summary, rows = analyze(root)
        report.extend(summary)
        records.extend(rows)
    (output / "summary.json").write_text(json.dumps(
        report, ensure_ascii=False, indent=2)+"\n")
    with (output / "seed_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Posthoc economic sensitivity map for four saved alpha strategies.

Pooled errors are evaluation outcomes. This analysis does not choose an alpha
for any saved seed and does not revise preregistered quality conclusions.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from research_analysis import paired_bootstrap_mean_ci, valid_delivery
from run_validation_calibrated_alpha import META_METHOD, TRIO_METHOD


METHODS = {
    "A_uniform": "A_uniform",
    "B_stage0": "B_stage0",
    "three_candidate": TRIO_METHOD,
    "five_candidate": META_METHOD,
}
EDGES = (("A_uniform", "B_stage0"),
         ("B_stage0", "three_candidate"),
         ("three_candidate", "five_candidate"),
         ("B_stage0", "five_candidate"))


def analyze(root):
    root = Path(root)
    manifest = json.loads((root / "suite_manifest.json").read_text())
    verification = json.loads((root / "selection_verification.json").read_text())
    if (manifest["status"] != "complete" or not verification["passed"]
            or manifest["scenarios"] != ["low_quality_25pct"]
            or set(METHODS.values()) != set(manifest["methods"])):
        raise ValueError("Unverified or incompatible study")
    scenario = manifest["scenarios"][0]
    seeds = manifest["seeds"]
    with (root / scenario / "summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    index = {}
    for row in rows:
        key = (int(row["seed"]), row["method"])
        if key in index:
            raise ValueError(f"Duplicate summary key: {key}")
        index[key] = row
    expected = {(seed, method) for seed in seeds for method in METHODS.values()}
    if set(index) != expected:
        raise ValueError("Incomplete or unexpected summary conditions")
    arrays = {}
    stats = {}
    for label, method in METHODS.items():
        group = [index[seed, method] for seed in seeds]
        if not all(valid_delivery(row) for row in group):
            raise ValueError(f"Invalid delivery for {label}")
        error = np.asarray([float(row["external_global_error"]) for row in group])
        request = np.asarray([float(row["initialization_requests"])
                              +float(row["stage3_requests"]) for row in group])
        if not (np.isfinite(error).all() and np.isfinite(request).all()):
            raise ValueError(f"Nonfinite quality or cost for {label}")
        arrays[label] = (error, request)
        stats[label] = {
            "valid": len(group), "mean_pooled_error": float(error.mean()),
            "mean_model_requests": float(request.mean()),
            "median_pooled_error": float(np.median(error)),
            "q1_q3_pooled_error": [float(x) for x in np.quantile(error, [.25, .75])],
            "q95_pooled_error": float(np.quantile(error, .95)),
            "worst_pooled_error": float(error.max()),
        }
    comparisons = []
    for cheaper, dearer in EDGES:
        error_c, request_c = arrays[cheaper]
        error_d, request_d = arrays[dearer]
        gain = error_c-error_d
        extra_requests = request_d-request_c
        mean_gain = float(gain.mean())
        mean_extra = float(extra_requests.mean())
        ci_gain = paired_bootstrap_mean_ci(gain)
        ci_cost = paired_bootstrap_mean_ci(extra_requests)
        comparisons.append({
            "cheaper_label": cheaper, "dearer_label": dearer,
            "mean_pooled_error_gain": mean_gain,
            "descriptive_95pct_ci_error_gain": ci_gain,
            "mean_extra_model_requests": mean_extra,
            "descriptive_95pct_ci_extra_requests": ci_cost,
            "wins_ties_losses_for_dearer": [int((gain>0).sum()),
                                             int((gain==0).sum()),
                                             int((gain<0).sum())],
            "q95_pooled_regression_for_dearer": float(np.quantile(-gain, .95)),
            "worst_pooled_regression_for_dearer": float((-gain).max()),
            "break_even_c_over_v_on_observed_means": (
                mean_gain/mean_extra if mean_gain > 0 and mean_extra > 0
                else None),
            "break_even_v_per_0p001_error_over_c": (
                .001*mean_extra/mean_gain if mean_gain > 0 and mean_extra > 0
                else None),
            "quality_comparison_preregistered": (
                cheaper == "three_candidate" and dearer == "five_candidate"),
        })
    return {
        "scenario": scenario, "seed_count": len(seeds),
        "status": "posthoc_economic_sensitivity_not_formal_advantage",
        "utility_definition": "U = -v * pooled_absolute_error - c * local_model_API_requests; "
                              "v and c are positive but unpriced. Shared costs "
                              "and tail-risk preferences are reported separately.",
        "methods": stats, "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

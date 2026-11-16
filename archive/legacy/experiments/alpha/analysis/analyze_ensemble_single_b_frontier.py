"""Post-hoc quality/request frontier for one Stage 0 B solve under the new objective."""

import argparse
import json
from pathlib import Path

import numpy as np

from research_analysis import paired_bootstrap_mean_ci


STUDIES = {
    "low_quality_25pct":
        "alpha_weight_outputs/alpha_ensemble_target_formal_assessment_2700_2879/seed_rows.jsonl",
    "reference_clean":
        "alpha_weight_outputs/alpha_ensemble_target_transfer_formal_assessment_3000_3179/reference_clean/seed_rows.jsonl",
    "sample_imbalance_4x":
        "alpha_weight_outputs/alpha_ensemble_target_transfer_formal_assessment_3000_3179/sample_imbalance_4x/seed_rows.jsonl",
}


def analyze(paths=STUDIES):
    output = {"status": "posthoc_development_not_preregistered_quality_claim",
              "method": "one B_stage0 alpha, one ensemble-target Stage 3 solve, no five-way selector",
              "scenarios": {}}
    for scenario, source in paths.items():
        rows = [json.loads(line) for line in Path(source).read_text().splitlines()]
        if (not rows or len({row["seed"] for row in rows}) != len(rows)
                or any(row.get("status", "valid") != "valid" or
                       row["candidates"]["B_stage0"]["status"] != "valid"
                       for row in rows)):
            raise ValueError(f"Incomplete or invalid saved candidates: {scenario}")
        b_error = np.asarray([row["candidates"]["B_stage0"]["pooled_error"]
                              for row in rows], dtype=float)
        b_requests = np.asarray([row["candidates"]["B_stage0"]["stage3_requests"]
                                 for row in rows], dtype=float)
        old_error = np.asarray([row["old_pooled_error"] for row in rows], dtype=float)
        new_error = np.asarray([row["new_pooled_error"] for row in rows], dtype=float)
        new_requests = np.asarray([row["new_stage3_requests"] for row in rows], dtype=float)
        if not all(np.isfinite(values).all() for values in
                   (b_error, b_requests, old_error, new_error, new_requests)):
            raise ValueError(f"Nonfinite score or request count: {scenario}")
        versus_old = b_error-old_error
        versus_new = b_error-new_error
        output["scenarios"][scenario] = {
            "seed_count": len(rows),
            "mean_pooled_B_single": float(b_error.mean()),
            "mean_pooled_old_five": float(old_error.mean()),
            "mean_pooled_new_five": float(new_error.mean()),
            "paired_mean_B_minus_old_five": float(versus_old.mean()),
            "paired_95pct_ci_B_minus_old_five_descriptive": paired_bootstrap_mean_ci(versus_old),
            "paired_mean_B_minus_new_five": float(versus_new.mean()),
            "paired_95pct_ci_B_minus_new_five_descriptive": paired_bootstrap_mean_ci(versus_new),
            "mean_B_stage3_requests": float(b_requests.mean()),
            "mean_new_five_stage3_requests": float(new_requests.mean()),
            "stage3_request_saving_vs_new_five_fraction": float(1-b_requests.mean()/new_requests.mean()),
            "candidate_failure_count": 0,
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(target)
    result = analyze()
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                 allow_nan=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

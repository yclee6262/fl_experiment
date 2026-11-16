"""Analyze paid-label initial-alpha development without making a formal claim."""

import argparse
import json
from pathlib import Path

import numpy as np

from research_analysis import paired_bootstrap_mean_ci


def _summary(rows):
    against_b = np.asarray([row["pooled_error"]-row["B_single_pooled_error"]
                            for row in rows], dtype=float)
    against_five = np.asarray([row["pooled_error"]-row["new_five_pooled_error"]
                               for row in rows], dtype=float)
    if not np.isfinite(against_b).all() or not np.isfinite(against_five).all():
        raise ValueError("Nonfinite paired quality")
    return {
        "seeds": len(rows),
        "mean_pooled_error": float(np.mean([row["pooled_error"] for row in rows])),
        "mean_delta_vs_B_single": float(against_b.mean()),
        "descriptive_95pct_ci_vs_B_single": paired_bootstrap_mean_ci(against_b),
        "wins_ties_losses_vs_B_single": [int((against_b < -1e-7).sum()),
                                           int((np.abs(against_b) <= 1e-7).sum()),
                                           int((against_b > 1e-7).sum())],
        "median_delta_vs_B_single": float(np.median(against_b)),
        "worst_regression_vs_B_single": {"seed": rows[int(against_b.argmax())]["seed"],
                                          "delta": float(against_b.max())},
        "mean_delta_vs_new_five": float(against_five.mean()),
        "descriptive_95pct_ci_vs_new_five": paired_bootstrap_mean_ci(against_five),
        "mean_stage3_requests": float(np.mean([row["stage3_requests"] for row in rows])),
        "mean_B_single_stage3_requests": float(np.mean(
            [row["B_single_stage3_requests"] for row in rows])),
        "mean_new_five_stage3_requests": float(np.mean(
            [row["new_five_stage3_requests"] for row in rows])),
        "initialization_requests_per_seed": sorted({row["initialization_requests"]
                                                     for row in rows}),
        "paid_trusted_labels_per_seed": sorted({row["trusted_label_count"] for row in rows}),
        "failure_count": sum(row["status"] != "valid" for row in rows),
    }


def analyze(paths):
    cohorts = {}
    all_rows = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        rows = payload["rows"]
        if (not rows or payload["seeds"] != [row["seed"] for row in rows]
                or any(row["status"] != "valid" for row in rows)):
            raise ValueError(f"Incomplete saved development result: {path}")
        cohorts[Path(path).name] = _summary(rows)
        all_rows.extend(rows)
    if len({row["seed"] for row in all_rows}) != len(all_rows):
        raise ValueError("Overlapping development seeds")
    return {"status": "exploratory_reused_seeds_no_formal_quality_claim",
            "cohorts": cohorts, "combined": _summary(all_rows),
            "decision": "No new formal seed allocation without stable advantage over B and priced label access"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = analyze(args.inputs)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                 allow_nan=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Post-hoc diagnostic of the registered B->Uniform trigger and selector."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from research_analysis import paired_bootstrap_mean_ci


def analyze(adaptive_path, five_path):
    adaptive = json.loads(Path(adaptive_path).read_text())["rows"]
    five = json.loads(Path(five_path).read_text())["rows"]
    by_seed = {row["seed"]: row for row in five}
    if len(by_seed) != len(five) or set(by_seed) != {row["seed"] for row in adaptive}:
        raise ValueError("Candidate and adaptive seeds do not align")
    triggered = []
    always_two_delta_B = []
    always_two_delta_adaptive = []
    always_two_requests = []
    triggered_held_B = 0
    triggered_held_B_but_U_better = 0
    switched_U = switched_U_better = switched_U_worse = 0
    U_better_by_trigger = {False: 0, True: 0}
    count_by_trigger = {False: 0, True: 0}
    for row in adaptive:
        control = by_seed[row["seed"]]
        b = control["candidates"]["B_stage0"]
        u = control["candidates"]["A_uniform"]
        scores = control["internal_scores"]
        b_error = b["pooled_error"]
        u_error = u["pooled_error"]
        trigger = bool(row["trigger_uniform"])
        count_by_trigger[trigger] += 1
        U_better_by_trigger[trigger] += u_error < b_error - 1e-7
        choose_U = (scores["A_uniform"]["internal_target_error"]
                    < scores["B_stage0"]["internal_target_error"])
        always_error = u_error if choose_U else b_error
        always_two_delta_B.append(always_error-b_error)
        always_two_delta_adaptive.append(always_error-row["pooled_error"])
        # Three blind-label batch queries, three B-X queries, three U-X queries.
        always_two_requests.append(b["stage3_requests"]+u["stage3_requests"]+9)
        if trigger:
            triggered.append((
                scores["A_uniform"]["internal_target_error"]
                - scores["B_stage0"]["internal_target_error"],
                u_error-b_error))
            if row["selected_initialization"] == "B_stage0":
                triggered_held_B += 1
                triggered_held_B_but_U_better += u_error < b_error - 1e-7
            else:
                switched_U += 1
                switched_U_better += u_error < b_error - 1e-7
                switched_U_worse += u_error > b_error + 1e-7
    corr = spearmanr([x[0] for x in triggered], [x[1] for x in triggered])
    always_B = np.asarray(always_two_delta_B)
    always_adaptive = np.asarray(always_two_delta_adaptive)
    return {
        "scope": "posthoc_mechanism_diagnostic_not_confirmatory_evidence",
        "valid_seeds": len(adaptive),
        "triggered": len(triggered),
        "triggered_but_kept_B": triggered_held_B,
        "triggered_kept_B_even_though_U_pooled_better":
            triggered_held_B_but_U_better,
        "switched_to_U": switched_U,
        "switched_to_U_and_pooled_better": switched_U_better,
        "switched_to_U_and_pooled_worse": switched_U_worse,
        "U_pooled_better_when_triggered": U_better_by_trigger[True],
        "U_pooled_better_when_not_triggered": U_better_by_trigger[False],
        "triggered_count": count_by_trigger[True],
        "not_triggered_count": count_by_trigger[False],
        "spearman_internal_U_minus_B_vs_pooled_U_minus_B_on_triggered":
            float(corr.statistic),
        "counterfactual_always_two": {
            "definition": "Always solve B and Uniform; B20 scores select the smaller internal error",
            "mean_pooled_delta_vs_B": float(always_B.mean()),
            "descriptive_95pct_ci_vs_B": paired_bootstrap_mean_ci(always_B),
            "mean_pooled_delta_vs_registered_adaptive": float(always_adaptive.mean()),
            "descriptive_95pct_ci_vs_registered_adaptive":
                paired_bootstrap_mean_ci(always_adaptive),
            "mean_stage3_plus_selection_requests": float(np.mean(always_two_requests)),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adaptive_json")
    parser.add_argument("five_json")
    parser.add_argument("--output", required=True)
    options = parser.parse_args()
    path = Path(options.output)
    if path.exists():
        raise FileExistsError(path)
    result = analyze(options.adaptive_json, options.five_json)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

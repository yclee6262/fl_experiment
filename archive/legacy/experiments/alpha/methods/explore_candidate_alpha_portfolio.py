"""Offline development of ex-ante candidate-alpha portfolios.

The portfolio is chosen from alpha vectors available before Stage 3. Saved
pooled outcomes are used only to assess a portfolio after its choice.
"""

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from research_analysis import paired_bootstrap_mean_ci
from validation_calibrated_alpha import CANDIDATE_METHODS


ANCHORS = ("A_uniform", "B_stage0")
EXTRAS = tuple(method for method in CANDIDATE_METHODS if method not in ANCHORS)


def choose_portfolios(alphas):
    """Select candidate names using only their ex-ante alpha vectors."""
    def distance(left, right):
        return float(np.sum(np.abs(alphas[left]-alphas[right])))
    portfolios = {
        "fixed_four_without_target_local": ANCHORS + ("mse_power_1p5", "corr_precision_0p5"),
        "fixed_three_A_B_power": ANCHORS + ("mse_power_1p5",),
    }
    novelty = {method: min(distance(method, anchor) for anchor in ANCHORS)
               for method in EXTRAS}
    by_novelty = sorted(EXTRAS, key=lambda method: (-novelty[method], EXTRAS.index(method)))
    portfolios["novelty_four"] = ANCHORS + tuple(by_novelty[:2])
    portfolios["novelty_three"] = ANCHORS + (by_novelty[0],)
    portfolios["maximin_four"] = max(
        (ANCHORS + pair for pair in itertools.combinations(EXTRAS, 2)),
        key=lambda subset: (min(distance(a, b) for a, b in itertools.combinations(subset, 2)),
                            -sum(EXTRAS.index(m) for m in subset if m in EXTRAS)))
    portfolios["mean_distance_four"] = max(
        (ANCHORS + pair for pair in itertools.combinations(EXTRAS, 2)),
        key=lambda subset: (sum(distance(a, b) for a, b in itertools.combinations(subset, 2)),
                            -sum(EXTRAS.index(m) for m in subset if m in EXTRAS)))
    return portfolios


def evaluate(roots, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for root_arg in roots:
        root = Path(root_arg)/"reference_clean"
        cohort = Path(root_arg).name
        candidates = defaultdict(dict)
        for row in csv.DictReader((root/"candidate_rows.csv").open()):
            candidates[int(row["seed"])][row["method"]] = row
        meta = {int(row["seed"]):row for row in csv.DictReader((root/"summary.csv").open())
                if row["method"] == "validation_calibrated_multistart"}
        for seed, old in meta.items():
            alphas = {method:np.asarray(json.loads(candidates[seed][method]["initial_alpha"]),float)
                      for method in CANDIDATE_METHODS}
            scores = json.loads(old["alpha_audit"])["internal_scores"]
            baseline_error = float(old["external_global_error"])
            baseline_requests = (float(old["initialization_requests"])
                                 + float(old["stage3_requests"]))
            for name,subset in choose_portfolios(alphas).items():
                selected = min(subset,key=lambda method:scores[method]["internal_target_error"])
                error = float(candidates[seed][selected]["external_global_error"])
                # The final runner will separately count one calibration and one
                # candidate-scoring batch, so include those predictable calls.
                initialization = sum(float(candidates[seed][m]["initialization_requests"])
                                     for m in subset)
                stage3 = sum(float(candidates[seed][m]["stage3_requests"])
                             for m in subset)
                selector_requests = len(alphas["A_uniform"])*(1+len(subset))
                requests = initialization + stage3 + selector_requests
                records.append({"cohort":cohort,"seed":seed,"portfolio":name,
                                "methods":json.dumps(subset),"selected":selected,
                                "candidate_count":len(subset),
                                "pooled_error":error,"baseline_error":baseline_error,
                                "pooled_delta":error-baseline_error,
                                "requests":requests,"baseline_requests":baseline_requests,
                                "request_delta":requests-baseline_requests})
        print("completed",cohort,flush=True)
    with output.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    groups=defaultdict(list)
    for row in records:groups[(row["cohort"],row["portfolio"])].append(row)
    for (cohort,portfolio),rows in sorted(groups.items()):
        differences=[row["pooled_delta"] for row in rows]
        print(cohort,portfolio,len(rows),float(np.mean(differences)),
              paired_bootstrap_mean_ci(differences),
              "requests",float(np.mean([row["requests"] for row in rows])),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--roots",nargs="+",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args();evaluate(args.roots,args.output)

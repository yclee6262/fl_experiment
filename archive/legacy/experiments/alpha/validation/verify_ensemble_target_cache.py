"""Verify cached ensemble-target replay against an uncached saved-model replay.

This checks solution equivalence and actual local-model requests. It does not
turn a reused-seed engineering comparison into new pooled-quality evidence.
"""

import argparse
import json
from pathlib import Path

import numpy as np


TOLERANCE = 1e-7


def _close(left, right, label):
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise AssertionError(f"Nonfinite or mismatched {label}")
    difference = float(np.max(np.abs(a - b))) if a.size else 0.0
    if difference > TOLERANCE:
        raise AssertionError(f"{label} changed by {difference}")
    return difference


def verify(reference_jsonl, cached_json):
    reference = {}
    for line in Path(reference_jsonl).read_text().splitlines():
        row = json.loads(line)
        if row["seed"] in reference:
            raise ValueError(f"Duplicate uncached seed: {row['seed']}")
        reference[row["seed"]] = row
    cached_payload = json.loads(Path(cached_json).read_text())
    if cached_payload["stage3_prediction_cache"] != "per_candidate_agentnode_float32":
        raise ValueError("The comparison input did not enable Stage 3 cache")
    cached = cached_payload["rows"]
    if not cached or len({row["seed"] for row in cached}) != len(cached):
        raise ValueError("Cached seeds are empty or duplicated")
    if not set(row["seed"] for row in cached) <= set(reference):
        raise ValueError("Cached seeds are missing from uncached reference")
    maximum_difference = 0.0
    uncached_requests = cached_requests = cache_hits = candidates = 0
    for new in cached:
        old = reference[new["seed"]]
        if (new["scenario"] != old["scenario"]
                or new["new_selected_initialization"] != old["new_selected_initialization"]
                or new["candidate_failures"] != old["candidate_failures"]):
            raise AssertionError(f"Delivery status or selection changed: {new['seed']}")
        for key in ("new_pooled_error", "new_truth_error", "old_pooled_error",
                    "old_truth_error"):
            maximum_difference = max(maximum_difference,
                                     _close(old[key], new[key], f"{new['seed']}/{key}"))
        if set(old["candidates"]) != set(new["candidates"]):
            raise AssertionError(f"Candidate set changed: {new['seed']}")
        for method, current in new["candidates"].items():
            baseline = old["candidates"][method]
            label = f"{new['seed']}/{method}"
            if baseline["status"] != current["status"] or baseline["solver"] != current["solver"]:
                raise AssertionError(f"Candidate status or solver changed: {label}")
            for key in ("alpha", "solution", "pooled_error", "truth_error",
                        "ensemble_target_error", "max_box_violation"):
                maximum_difference = max(maximum_difference,
                                         _close(baseline[key], current[key], f"{label}/{key}"))
            original = baseline["stage3_requests"]
            used = current["stage3_requests"]
            hits = current["stage3_cache_hits"]
            misses = current["stage3_cache_misses"]
            if used != misses or original != hits + misses:
                raise AssertionError(f"Request accounting changed: {label}")
            uncached_requests += original
            cached_requests += used
            cache_hits += hits
            candidates += 1
        if old["internal_scores"].keys() != new["internal_scores"].keys():
            raise AssertionError(f"Internal score candidates changed: {new['seed']}")
        for method, score in new["internal_scores"].items():
            for key, value in score.items():
                maximum_difference = max(maximum_difference,
                    _close(old["internal_scores"][method][key], value,
                           f"{new['seed']}/{method}/{key}"))
        if sum(row["stage3_requests"] for row in new["candidates"].values()) != new["new_stage3_requests"]:
            raise AssertionError(f"Cached seed request total changed: {new['seed']}")
    return {"passed": True, "seeds": len(cached), "candidates": candidates,
            "maximum_solution_alpha_score_difference": maximum_difference,
            "uncached_stage3_requests": uncached_requests,
            "cached_stage3_requests": cached_requests,
            "cache_hits": cache_hits,
            "saved_requests": uncached_requests - cached_requests,
            "saved_fraction": (uncached_requests - cached_requests) / uncached_requests}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("uncached_seed_rows_jsonl")
    parser.add_argument("cached_replay_json")
    args = parser.parse_args()
    print(json.dumps(verify(args.uncached_seed_rows_jsonl, args.cached_replay_json),
                     indent=2))


if __name__ == "__main__":
    main()

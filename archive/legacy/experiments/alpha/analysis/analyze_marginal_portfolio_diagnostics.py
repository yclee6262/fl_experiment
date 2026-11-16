"""Exploratory six-candidate and selector-regret audit on saved alpha cohorts.

These cohorts were already used for method selection or confirmation. Outputs
are diagnostic hypotheses, never new independent superiority evidence. Pooled
scores are read only after replaying the saved internal candidate scores.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from run_validation_calibrated_alpha import (
    CANDIDATE_METHODS, MARGINAL_CANDIDATE, MARGINAL_METHOD, META_METHOD,
)


SIX_METHODS = CANDIDATE_METHODS + (MARGINAL_CANDIDATE,)


def _read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _index(rows, scenario, seeds, methods, kind):
    expected = {(scenario, str(seed), method) for seed in seeds for method in methods}
    keys = [(r["scenario"], r["seed"], r["method"]) for r in rows]
    actual = set(keys)
    if len(keys) != len(actual) or actual != expected:
        raise ValueError(f"{kind}: duplicate/missing/extra keys: "
                         f"duplicates={len(keys)-len(actual)}, "
                         f"missing={len(expected-actual)}, extra={len(actual-expected)}")
    return {(r["seed"], r["method"]): r for r in rows}


def _finite(row, key):
    value = float(row[key])
    if not np.isfinite(value):
        raise ValueError(f"Nonfinite {key} for seed {row['seed']} {row['method']}")
    return value


def _valid(row):
    return row["delivery_status"] == "valid" and row["final_box_feasible"] == "True"


def _audit_choice(row, method_order):
    audit = json.loads(row["alpha_audit"])
    scores = audit["internal_scores"]
    choice = min((m for m in method_order if m in scores),
                 key=lambda m: _finite_score(scores[m]))
    if choice != audit["selected_initialization"]:
        raise ValueError(f"Saved selector mismatch at seed {row['seed']}")
    return audit, choice


def _finite_score(score):
    value = float(score["internal_target_error"])
    if not np.isfinite(value):
        raise ValueError("Nonfinite internal candidate score")
    return value


def analyze(root):
    root = Path(root)
    manifest = json.loads((root / "suite_manifest.json").read_text())
    if manifest["status"] != "complete" or len(manifest["scenarios"]) != 1:
        raise ValueError(f"Incomplete or multi-scenario root: {root}")
    scenario = manifest["scenarios"][0]
    seeds = manifest["seeds"]
    verification = json.loads((root / "selection_verification.json").read_text())
    if not verification["passed"]:
        raise ValueError(f"Selection replay did not pass: {root}")
    folder = root / scenario
    summary = _index(_read_csv(folder / "summary.csv"), scenario, seeds,
                     manifest["methods"], "summary")
    candidates = _index(_read_csv(folder / "candidate_rows.csv"), scenario,
                        seeds, manifest["candidate_initializations"], "candidates")
    records = []
    for seed in seeds:
        key = str(seed)
        old = summary[key, META_METHOD]
        new = summary[key, MARGINAL_METHOD]
        if not (_valid(old) and _valid(new)):
            raise ValueError(f"No valid paired delivery at seed {seed}")
        old_audit, old_choice = _audit_choice(old, CANDIDATE_METHODS)
        new_audit, new_choice = _audit_choice(new, tuple(
            m for m in SIX_METHODS if m != "target_local_precision"))
        scores = {m: _finite_score(old_audit["internal_scores"][m])
                  for m in CANDIDATE_METHODS}
        for method in set(CANDIDATE_METHODS) & set(new_audit["internal_scores"]):
            if abs(scores[method] - _finite_score(
                    new_audit["internal_scores"][method])) > 1e-10:
                raise ValueError(f"Shared score changed at seed {seed}: {method}")
        scores[MARGINAL_CANDIDATE] = _finite_score(
            new_audit["internal_scores"][MARGINAL_CANDIDATE])
        six_choice = min(SIX_METHODS, key=lambda m: scores[m])
        candidate = {m: candidates[key, m] for m in SIX_METHODS}
        if any(not _valid(candidate[m]) for m in SIX_METHODS):
            raise ValueError(f"Six-way diagnostic requires valid candidates: {seed}")
        for meta, chosen in ((old, old_choice), (new, new_choice)):
            if (abs(_finite(meta, "external_global_error") -
                    _finite(candidate[chosen], "external_global_error")) > 1e-7):
                raise ValueError(f"Candidate score mismatch at seed {seed}")
        pooled = {m: _finite(candidate[m], "external_global_error")
                  for m in SIX_METHODS}
        oracle_choice = min(SIX_METHODS, key=lambda m: pooled[m])
        old_error = pooled[old_choice]
        new_error = pooled[new_choice]
        six_error = pooled[six_choice]
        old_requests = (_finite(old, "initialization_requests") +
                        _finite(old, "stage3_requests"))
        # Add the actually observed marginal solve and three scoring calls.
        # Stage 0 calibration is shared once; marginal candidate init is not.
        six_requests = (old_requests +
                        _finite(candidate[MARGINAL_CANDIDATE], "initialization_requests") +
                        _finite(candidate[MARGINAL_CANDIDATE], "stage3_requests") +
                        len(json.loads(old["agent_ids"])))
        records.append({
            "scenario": scenario, "seed": seed,
            "old_choice": old_choice, "new_choice": new_choice,
            "six_choice": six_choice, "oracle_choice_posthoc_only": oracle_choice,
            "old_pooled": old_error, "new_pooled": new_error,
            "six_pooled": six_error, "oracle_pooled_posthoc_only": pooled[oracle_choice],
            "new_minus_old": new_error-old_error,
            "six_minus_old": six_error-old_error,
            "six_minus_new": six_error-new_error,
            "old_selector_regret_posthoc_only": old_error-pooled[oracle_choice],
            "new_selector_regret_posthoc_only": new_error-pooled[oracle_choice],
            "six_selector_regret_posthoc_only": six_error-pooled[oracle_choice],
            "old_requests": old_requests, "six_requests": six_requests,
            "six_request_increase_fraction": (six_requests-old_requests)/old_requests,
            "target_local_internal_score": scores["target_local_precision"],
            "marginal_internal_score": scores[MARGINAL_CANDIDATE],
            "target_local_pooled_posthoc_only": pooled["target_local_precision"],
            "marginal_pooled_posthoc_only": pooled[MARGINAL_CANDIDATE],
        })
    old = np.asarray([r["old_pooled"] for r in records])
    new = np.asarray([r["new_pooled"] for r in records])
    six = np.asarray([r["six_pooled"] for r in records])
    result = {
        "scenario": scenario, "root": str(root), "seeds": seeds,
        "evidence_status": "exploratory_reuse_of_formal_seeds_not_independent_confirmation",
        "six_candidate_order": SIX_METHODS,
        "count": len(records),
        "mean_pooled": {"old_five": float(old.mean()),
                        "replacement_five": float(new.mean()),
                        "six": float(six.mean())},
        "mean_delta": {"replacement_minus_old": float((new-old).mean()),
                       "six_minus_old": float((six-old).mean()),
                       "six_minus_replacement": float((six-new).mean())},
        "six_vs_old_counts": {"wins": int(np.sum(six < old)),
                              "ties": int(np.sum(six == old)),
                              "losses": int(np.sum(six > old))},
        "six_choices": dict(Counter(r["six_choice"] for r in records)),
        "mean_requests": {"old_five": float(np.mean([r["old_requests"] for r in records])),
                          "six": float(np.mean([r["six_requests"] for r in records]))},
        "mean_six_request_increase_fraction": float(np.mean([
            r["six_request_increase_fraction"] for r in records])),
        "worst_replacement_seed": max(records, key=lambda r: r["new_minus_old"])["seed"],
        "worst_six_seed": max(records, key=lambda r: r["six_minus_old"])["seed"],
        "mean_selector_regret_posthoc_only": {
            "old_five": float(np.mean([r["old_selector_regret_posthoc_only"] for r in records])),
            "replacement_five": float(np.mean([r["new_selector_regret_posthoc_only"] for r in records])),
            "six": float(np.mean([r["six_selector_regret_posthoc_only"] for r in records])),
        },
    }
    return result, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    reports, rows = [], []
    for root in args.roots:
        report, records = analyze(root)
        reports.append(report)
        rows.extend(records)
    (output / "summary.json").write_text(json.dumps(
        reports, ensure_ascii=False, indent=2) + "\n")
    with (output / "seed_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Formal seed-level analysis for P3 contribution external validity."""

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def _source_workspace(source_file):
    """Find the flat workspace or a standalone historical source snapshot."""
    start = Path(source_file).resolve().parent
    for candidate in (start, *start.parents):
        if all((candidate / name).is_file() for name in
               ("host_server.py", "run_experiments.py", "dataset.py")):
            return candidate
        if (candidate / "project_support/legacy.py").is_file():
            from project_support.legacy import prepare
            return prepare()
    raise FileNotFoundError(f"Cannot locate source workspace from {source_file}")


import numpy as np
from scipy.stats import spearmanr

from research_analysis import paired_bootstrap_mean_ci


BOOTSTRAP_SEED = 20261006
BOOTSTRAP_REPLICATES = 10_000
SIMULTANEOUS_CONFIDENCE = 0.9875
PRIMARY_DEAD_ZONE = 1e-7
SENSITIVITY_DEAD_ZONES = (1e-6, 1e-4)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def mean_optional(rows, field):
    values = [finite(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def direction(value, dead_zone):
    value = float(value)
    if value > dead_zone:
        return 1
    if value < -dead_zone:
        return -1
    return 0


def safe_spearman(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) < 2 or np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return None
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else None


def scenario_seed_metrics(rows, dead_zone):
    operational = [float(row["operational_internal_contribution"]) for row in rows]
    candidate = [float(row["operational_external_contribution"]) for row in rows]
    full = [float(row["full_removal_external_contribution"]) for row in rows]
    truth = [float(row["full_removal_true_contribution"]) for row in rows]
    op_sign = [direction(value, dead_zone) for value in operational]
    full_sign = [direction(value, dead_zone) for value in full]
    truth_sign = [direction(value, dead_zone) for value in truth]
    nonzero = [index for index in range(len(rows))
               if op_sign[index] != 0 and full_sign[index] != 0]
    predicted_negative = [index for index, sign in enumerate(op_sign) if sign < 0]
    return {
        "spearman_operational_vs_full": safe_spearman(operational, full),
        "spearman_candidate_pooled_vs_full": safe_spearman(candidate, full),
        "direction_agreement": float(np.mean(np.equal(op_sign, full_sign))),
        "nonzero_pairs": len(nonzero),
        "nonzero_direction_agreement": (
            float(np.mean([op_sign[index] == full_sign[index] for index in nonzero]))
            if nonzero else None
        ),
        "predicted_negative": len(predicted_negative),
        "predicted_negative_removal_improved": sum(
            full_sign[index] < 0 for index in predicted_negative
        ),
        "pooled_truth_direction_agreement": float(np.mean(np.equal(full_sign, truth_sign))),
        "pooled_truth_conflicts": sum(a != b for a, b in zip(full_sign, truth_sign)),
    }


def analyze(inputs, output_dir, expected_seeds):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    source_root = _source_workspace(__file__)
    snapshot = output_dir / "source_snapshot"
    snapshot.mkdir()
    analysis_source_hashes = {}
    for name in ("analyze_p3_external_validity.py", "research_analysis.py"):
        shutil.copy2(source_root / name, snapshot / name)
        analysis_source_hashes[name] = hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
    all_effects = []
    all_summaries = []
    manifests = {}
    failures = []
    expected_sizes = {"clean": 3, "controlled_poison": 5}

    for scenario, source in inputs.items():
        source = Path(source)
        manifest = json.loads((source / "run_manifest.json").read_text())
        manifests[scenario] = manifest
        effects = read_csv(source / "participant_effects.csv")
        summaries = read_csv(source / "scenario_summary.csv")
        for row in effects:
            row["scenario"] = scenario
            row["source_dir"] = str(source)
        for row in summaries:
            row["scenario"] = scenario
            row["source_dir"] = str(source)
        all_effects.extend(effects)
        all_summaries.extend(summaries)

        summary_counts = Counter(int(row["seed"]) for row in summaries)
        for seed in expected_seeds:
            if summary_counts[seed] != 1:
                failures.append({"scenario": scenario, "seed": seed,
                                 "reason": f"scenario_summary_count={summary_counts[seed]}"})
        effect_counts = Counter(int(row["seed"]) for row in effects)
        for seed in expected_seeds:
            if effect_counts[seed] != expected_sizes[scenario]:
                failures.append({"scenario": scenario, "seed": seed,
                                 "reason": f"participant_effect_count={effect_counts[seed]}"})

    keys = [(row["scenario"], row["seed"], row["agent_id"]) for row in all_effects]
    for key, count in Counter(keys).items():
        if count != 1:
            failures.append({"scenario": key[0], "seed": key[1],
                             "reason": f"duplicate_agent_key={key};count={count}"})

    required_numeric = (
        "operational_internal_contribution", "operational_external_contribution",
        "full_removal_external_contribution", "full_removal_true_contribution",
    )
    valid_by_scenario_seed = defaultdict(list)
    conflict_rows = []
    for row in all_effects:
        reason = None
        if row.get("delivery_status") != "valid_delivery":
            reason = row.get("failure_reason") or "non_valid_delivery"
        elif any(row.get(field) != "True" for field in (
            "baseline_box_feasible", "operational_box_feasible", "full_removal_box_feasible"
        )):
            reason = "infeasible_solution"
        elif any(finite(row.get(field)) is None for field in required_numeric):
            reason = "non_finite_primary_metric"
        elif not (Path(row["source_dir"]) / row["scenario_artifact_dir"]).is_dir():
            reason = "missing_evaluator_artifact"
        if reason:
            failures.append({"scenario": row["scenario"], "seed": row["seed"],
                             "agent_id": row.get("agent_id", ""), "reason": reason})
            continue
        valid_by_scenario_seed[(row["scenario"], int(row["seed"]))].append(row)
        pooled_sign = direction(row["full_removal_external_contribution"], PRIMARY_DEAD_ZONE)
        truth_sign = direction(row["full_removal_true_contribution"], PRIMARY_DEAD_ZONE)
        if pooled_sign != truth_sign:
            conflict_rows.append({
                "scenario": row["scenario"], "seed": row["seed"],
                "agent_id": row["agent_id"],
                "pooled_contribution": row["full_removal_external_contribution"],
                "truth_contribution": row["full_removal_true_contribution"],
                "pooled_direction": pooled_sign, "truth_direction": truth_sign,
            })

    seed_rows = []
    sensitivity_rows = []
    for dead_zone in (PRIMARY_DEAD_ZONE,) + SENSITIVITY_DEAD_ZONES:
        for scenario in inputs:
            for seed in expected_seeds:
                rows = valid_by_scenario_seed.get((scenario, seed), [])
                expected = expected_sizes[scenario]
                if len(rows) != expected:
                    metrics = {key: None for key in (
                        "spearman_operational_vs_full", "spearman_candidate_pooled_vs_full",
                        "direction_agreement", "nonzero_pairs", "nonzero_direction_agreement",
                        "predicted_negative", "predicted_negative_removal_improved",
                        "pooled_truth_direction_agreement", "pooled_truth_conflicts",
                    )}
                    status = "incomplete"
                else:
                    metrics = scenario_seed_metrics(rows, dead_zone)
                    status = "valid"
                record = {"scenario": scenario, "seed": seed, "dead_zone": dead_zone,
                          "status": status, "participant_effects": len(rows), **metrics}
                sensitivity_rows.append(record)
                if dead_zone == PRIMARY_DEAD_ZONE:
                    seed_rows.append(record)

    aggregate_rows = []
    statistical_pass = True
    practical_pass = True
    coverage_pass = not failures
    for scenario in inputs:
        selected = [row for row in seed_rows if row["scenario"] == scenario]
        valid = [row for row in selected if row["status"] == "valid"]
        scenario_effects = [row for row in all_effects if row["scenario"] == scenario]
        scenario_summaries = [row for row in all_summaries if row["scenario"] == scenario]
        rho = [row["spearman_operational_vs_full"] for row in valid
               if row["spearman_operational_vs_full"] is not None]
        candidate_rho = [row["spearman_candidate_pooled_vs_full"] for row in valid
                         if row["spearman_candidate_pooled_vs_full"] is not None]
        agreement = [row["direction_agreement"] for row in valid]
        negative_count = sum(int(row["predicted_negative"] or 0) for row in valid)
        negative_improved = sum(
            int(row["predicted_negative_removal_improved"] or 0) for row in valid
        )
        poisoned = [row for row in scenario_effects if row.get("is_poisoned") == "True"]
        rho_ci = paired_bootstrap_mean_ci(
            rho, confidence=SIMULTANEOUS_CONFIDENCE, seed=BOOTSTRAP_SEED,
            replicates=BOOTSTRAP_REPLICATES,
        )
        agreement_ci = paired_bootstrap_mean_ci(
            agreement, confidence=SIMULTANEOUS_CONFIDENCE, seed=BOOTSTRAP_SEED + 1,
            replicates=BOOTSTRAP_REPLICATES,
        )
        rho_mean = float(np.mean(rho)) if rho else None
        agreement_mean = float(np.mean(agreement)) if agreement else None
        scenario_statistical = bool(
            len(valid) == len(expected_seeds) and len(rho) == len(expected_seeds)
            and rho_ci[0] is not None and rho_ci[0] > 0
            and agreement_ci[0] is not None and agreement_ci[0] > 0.5
        )
        scenario_practical = bool(
            scenario_statistical and rho_mean >= 0.5 and agreement_mean >= 0.70
        )
        statistical_pass &= scenario_statistical
        practical_pass &= scenario_practical
        aggregate_rows.append({
            "scenario": scenario, "expected_seeds": len(expected_seeds),
            "complete_valid_seeds": len(valid), "valid_spearman_seeds": len(rho),
            "mean_spearman_operational_vs_full": rho_mean,
            "simultaneous_ci_low_spearman": rho_ci[0],
            "simultaneous_ci_high_spearman": rho_ci[1],
            "mean_direction_agreement": agreement_mean,
            "simultaneous_ci_low_direction_agreement": agreement_ci[0],
            "simultaneous_ci_high_direction_agreement": agreement_ci[1],
            "statistical_signal": scenario_statistical,
            "practical_threshold": scenario_practical,
            "mean_spearman_candidate_pooled_vs_full": (
                float(np.mean(candidate_rho)) if candidate_rho else None
            ),
            "predicted_negative_effects": negative_count,
            "predicted_negative_removal_improvement_rate": (
                negative_improved / negative_count if negative_count else None
            ),
            "invalid_q_scenarios": sum(
                row.get("operational_q_valid") != "True" for row in scenario_summaries
            ),
            "poisoned_effects": len(poisoned),
            "poisoned_removal_improvement_rate": (
                sum(direction(row["full_removal_external_contribution"], PRIMARY_DEAD_ZONE) < 0
                    for row in poisoned) / len(poisoned) if poisoned else None
            ),
            "pooled_truth_conflicts": sum(
                int(row["pooled_truth_conflicts"] or 0) for row in valid
            ),
            "mean_candidate_exclusion_requests": (
                mean_optional(scenario_summaries, "candidate_exclusion_requests")
            ),
            "mean_full_removal_requests": (
                mean_optional(scenario_summaries, "full_removal_requests")
            ),
            "mean_candidate_exclusion_seconds": (
                mean_optional(scenario_summaries, "candidate_exclusion_seconds")
            ),
            "mean_full_removal_seconds": (
                mean_optional(scenario_summaries, "full_removal_seconds")
            ),
        })

    if coverage_pass and practical_pass:
        conclusion = "usable_external_validity"
    elif coverage_pass and statistical_pass:
        conclusion = "limited_signal"
    else:
        conclusion = "external_validity_not_established"

    representatives = sorted(
        [row for row in all_effects if finite(row.get("full_removal_external_contribution")) is not None],
        key=lambda row: float(row["full_removal_external_contribution"]),
    )
    representative_rows = []
    if representatives:
        chosen = [representatives[0], representatives[len(representatives) // 2], representatives[-1]]
        labels = ("largest_improvement", "median", "largest_worsening")
        for label, row in zip(labels, chosen):
            representative_rows.append({
                "case": label, "scenario": row["scenario"], "seed": row["seed"],
                "agent_id": row["agent_id"],
                "operational_internal_contribution": row["operational_internal_contribution"],
                "full_removal_external_contribution": row["full_removal_external_contribution"],
            })

    write_csv(output_dir / "seed_metrics.csv", seed_rows)
    write_csv(output_dir / "aggregate_metrics.csv", aggregate_rows)
    write_csv(output_dir / "sensitivity_metrics.csv", sensitivity_rows)
    write_csv(output_dir / "pooled_truth_conflicts.csv", conflict_rows,
              fields=("scenario", "seed", "agent_id", "pooled_contribution",
                      "truth_contribution", "pooled_direction", "truth_direction"))
    write_csv(output_dir / "failure_report.csv", failures,
              fields=("scenario", "seed", "agent_id", "reason"))
    write_csv(output_dir / "representative_cases.csv", representative_rows,
              fields=("case", "scenario", "seed", "agent_id",
                      "operational_internal_contribution",
                      "full_removal_external_contribution"))
    result = {
        "study": "p3_external_validity_formal_v1",
        "expected_seeds": expected_seeds,
        "primary_dead_zone": PRIMARY_DEAD_ZONE,
        "sensitivity_dead_zones": list(SENSITIVITY_DEAD_ZONES),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "simultaneous_confidence": SIMULTANEOUS_CONFIDENCE,
        "coverage_complete": coverage_pass,
        "statistical_signal_all_scenarios": statistical_pass,
        "practical_threshold_all_scenarios": practical_pass,
        "formal_conclusion": conclusion,
        "failure_count": len(failures),
        "analysis_source_snapshot_sha256": analysis_source_hashes,
        "interpretation": "Full removal excludes candidate, engine, and alpha without retraining or replacement recruitment; this is not Shapley value.",
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", required=True)
    parser.add_argument("--controlled-poison", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="3600-3629")
    args = parser.parse_args()
    start, end = (int(value) for value in args.seeds.split("-"))
    result = analyze(
        {"clean": args.clean, "controlled_poison": args.controlled_poison},
        args.output_dir, list(range(start, end + 1)),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

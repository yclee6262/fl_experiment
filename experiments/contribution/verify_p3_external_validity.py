"""Acceptance checks for P3 external-validity pilot and formal runs."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_run(root, scenario, seeds, expected_size):
    root = Path(root)
    errors = []
    manifest = json.loads((root / "run_manifest.json").read_text())
    if manifest.get("status") != "complete":
        errors.append(f"{scenario}: manifest status is not complete")
    args = manifest.get("arguments", {})
    if args.get("constraint_mode") != "box_constrained":
        errors.append(f"{scenario}: constraint mode is not box_constrained")
    if [int(value) for value in args.get("seeds", [])] != seeds:
        errors.append(f"{scenario}: manifest seeds do not match expected seeds")
    for name, expected in manifest.get("source_snapshot_sha256", {}).items():
        path = root / "source_snapshot" / name
        if not path.is_file():
            errors.append(f"{scenario}: missing source snapshot {name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f"{scenario}: source snapshot hash mismatch {name}")
    for name, expected in manifest.get("artifact_sha256", {}).items():
        path = root / name
        if not path.is_file():
            errors.append(f"{scenario}: missing artifact {name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f"{scenario}: artifact hash mismatch {name}")

    summaries = read_csv(root / "scenario_summary.csv")
    effects = read_csv(root / "participant_effects.csv")
    summary_counts = Counter(int(row["seed"]) for row in summaries)
    effect_counts = Counter(int(row["seed"]) for row in effects)
    for seed in seeds:
        if summary_counts[seed] != 1:
            errors.append(f"{scenario}/{seed}: scenario summary count={summary_counts[seed]}")
        if effect_counts[seed] != expected_size:
            errors.append(f"{scenario}/{seed}: participant effect count={effect_counts[seed]}")
    keys = [(row["experiment_id"], row["scenario"], row["seed"], row["agent_id"])
            for row in effects]
    duplicates = [key for key, count in Counter(keys).items() if count != 1]
    if duplicates:
        errors.append(f"{scenario}: duplicate effect keys={duplicates[:3]}")
    for row in effects:
        key = f"{scenario}/{row.get('seed')}/{row.get('agent_id')}"
        if row.get("delivery_status") != "valid_delivery":
            if not row.get("failure_reason"):
                errors.append(f"{key}: failure has no reason")
            continue
        for field in ("baseline_box_feasible", "operational_box_feasible",
                      "full_removal_box_feasible"):
            if row.get(field) != "True":
                errors.append(f"{key}: {field} is not true")
        for field in ("baseline_max_box_violation", "operational_max_box_violation",
                      "full_removal_max_box_violation"):
            try:
                if float(row[field]) > 1e-8:
                    errors.append(f"{key}: {field} exceeds 1e-8")
            except (KeyError, TypeError, ValueError):
                errors.append(f"{key}: {field} missing or invalid")
        artifact = root / row.get("scenario_artifact_dir", "")
        if not artifact.is_dir():
            errors.append(f"{key}: missing evaluator artifact")

    reevaluation_path = root / "reevaluation.csv"
    if not reevaluation_path.is_file():
        errors.append(f"{scenario}: missing reevaluation.csv")
    else:
        reevaluation = read_csv(reevaluation_path)
        expected_rows = len(effects) * 3
        if len(reevaluation) != expected_rows:
            errors.append(
                f"{scenario}: reevaluation rows={len(reevaluation)}, expected={expected_rows}"
            )
        for row in reevaluation:
            if row.get("external_score_matches") != "True":
                errors.append(
                    f"{scenario}/{row.get('seed')}/{row.get('agent_id')}: external reload mismatch"
                )
            if row.get("box_matches") != "True" or row.get("recomputed_box_feasible") != "True":
                errors.append(
                    f"{scenario}/{row.get('seed')}/{row.get('agent_id')}: box reload mismatch"
                )
    return {"scenario": scenario, "scenario_rows": len(summaries),
            "effect_rows": len(effects), "errors": errors, "accepted": not errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", required=True)
    parser.add_argument("--controlled-poison", required=True)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--seeds", required=True, help="Inclusive range such as 0-4")
    args = parser.parse_args()
    start, end = (int(value) for value in args.seeds.split("-"))
    seeds = list(range(start, end + 1))
    results = [
        verify_run(args.clean, "clean", seeds, 3),
        verify_run(args.controlled_poison, "controlled_poison", seeds, 5),
    ]
    analysis = json.loads((Path(args.analysis) / "analysis_manifest.json").read_text())
    errors = [error for result in results for error in result["errors"]]
    if analysis.get("expected_seeds") != seeds:
        errors.append("analysis expected seeds do not match")
    if not analysis.get("coverage_complete"):
        errors.append("analysis reports incomplete coverage")
    output = {"accepted": not errors, "runs": results,
              "analysis_conclusion": analysis.get("formal_conclusion"), "errors": errors}
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

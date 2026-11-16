"""Acceptance verifier for the constrained P4/P5 and budget-aware P6 pilots."""

import argparse
import csv
import json
from pathlib import Path


def rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_legacy_failure(path):
    failures = [row for row in rows(path) if row["final_box_feasible"] != "True"]
    assert failures, "Legacy high-target output lacks the recorded failure."
    return len(failures)


def verify_p5(root):
    root = Path(root)
    manifest = json.loads((root / "suite_manifest.json").read_text())
    assert manifest["status"] == "complete"
    expected_ids = {
        "controlled_poison": [1, 2, 3, 4, 20],
        "controlled_coalition_5": [1, 2, 3, 4, 20],
    }
    total = 0
    for spec in manifest["scenarios"]:
        scenario = spec["name"]
        result_dir = root / scenario
        result_rows = rows(result_dir / "summary.csv")
        assert len(result_rows) == 10
        assert {(row["method_label"], int(row["seed"])) for row in result_rows} == {
            (method, seed) for method in
            ("A_uniform_no_update", "D_uniform_guarded") for seed in range(5)
        }
        assert all(row["delivery_status"] == "valid" for row in result_rows)
        assert all(row["final_box_feasible"] == "True" for row in result_rows)
        assert all(float(row["max_box_violation"]) <= 1e-8 for row in result_rows)
        participants = expected_ids.get(scenario, [1, 2, 20])
        assert all(json.loads(row["selected_agent_ids"]) == participants
                   for row in result_rows)
        assert (result_dir / "source_snapshot" / "host_server.py").exists()
        reevaluated = rows(result_dir / "reevaluation.csv")
        assert len(reevaluated) == 10
        assert all(row["external_score_matches"] == "True" and
                   row["true_score_matches"] == "True" for row in reevaluated)
        total += len(result_rows)
    low_quality = rows(root / "low_quality_25pct" / "summary.csv")[0]
    metadata = json.loads((
        root / "low_quality_25pct" / low_quality["scenario_artifact_dir"] /
        "metadata.json"
    ).read_text())
    agent20 = next(row for row in metadata["agent_data_profile"]
                   if row["agent_id"] == 20)
    assert agent20["noise_std"] == 0.3
    assert agent20["evaluator_noise_std"] == 0.05
    return total


def verify_p4(root):
    root = Path(root)
    manifest = json.loads((root / "suite_manifest.json").read_text())
    assert manifest["status"] == "complete"
    analysis = rows(root / "analysis" / "aggregate_results.csv")
    assert len(analysis) == 2
    assert all(int(row["accepted_state_mismatches"]) == 0 for row in analysis)
    for directory in root.glob("*_patience*"):
        assert all(row["final_box_feasible"] == "True"
                   for row in rows(directory / "summary.csv"))
        reevaluated = rows(directory / "reevaluation.csv")
        assert len(reevaluated) == 5
        assert all(row["external_score_matches"] == "True" and
                   row["true_score_matches"] == "True" for row in reevaluated)
    return sum(int(row["seeds"]) for row in analysis)


def verify_p6(root):
    root = Path(root)
    manifest = json.loads((root / "run_manifest.json").read_text())
    assert manifest["status"] == "complete"
    result_rows = rows(root / "selection_results.csv")
    assert len(result_rows) == 90
    assert len(rows(root / "candidates.csv")) == 50
    assert all(row["delivery_status"] == "valid" for row in result_rows)
    assert all(row["budget_feasible"] == "True" for row in result_rows)
    assert all(row["box_feasible"] == "True" for row in result_rows)
    assert all(int(row["coalition_size"]) <= 3 for row in result_rows)
    assert all(float(row["accounting_total_committed_cost"]) ==
               float(row["recruitment_bid_commitment"]) for row in result_rows)
    grouped = {}
    for row in result_rows:
        key = (row["seed"], row["budget"], row["diversity_eta"])
        grouped.setdefault(key, {})[row["selection_method"]] = row
    assert len(grouped) == 30
    for methods in grouped.values():
        assert float(methods["exact"]["selection_objective"]) + 1e-12 >= float(
            methods["objective_greedy"]["selection_objective"]
        )
    reevaluated = rows(root / "reevaluation.csv")
    assert len(reevaluated) == 90
    assert all(row["external_score_matches"] == "True" and
               row["true_score_matches"] == "True" for row in reevaluated)
    assert (root / "source_snapshot" / "p6_budget_pilot.py").exists()
    return len(result_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-hard-target", required=True)
    parser.add_argument("--p4", required=True)
    parser.add_argument("--p5", required=True)
    parser.add_argument("--p6", required=True)
    args = parser.parse_args()
    result = {
        "legacy_box_failures": verify_legacy_failure(args.legacy_hard_target),
        "p4_paired_seeds": verify_p4(args.p4),
        "p5_results": verify_p5(args.p5),
        "p6_results": verify_p6(args.p6),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

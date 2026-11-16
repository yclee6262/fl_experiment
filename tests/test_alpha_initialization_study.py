import csv
import json

from analyze_alpha_initialization_study import analyze
from run_alpha_initialization_study import FIELDS


def test_analysis_preserves_missing_and_failed_pairs(tmp_path):
    root = tmp_path
    (root / "suite_manifest.json").write_text(json.dumps({
        "status": "complete", "scenarios": ["reference_clean"],
        "seeds": [0, 1], "methods": ["A_uniform", "B_stage0", "quality_shrink",
                                     "ensemble_shrink", "target_peer_shrink"]}))
    scenario = root / "reference_clean"
    scenario.mkdir()
    with (scenario / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for method in ("A_uniform", "B_stage0", "quality_shrink",
                       "ensemble_shrink", "target_peer_shrink"):
            writer.writerow({"scenario": "reference_clean", "seed": 0,
                "method": method, "delivery_status": "valid",
                "status": "valid", "final_box_feasible": True,
                "final_solution": "[0, 0]", "external_global_error": .1,
                "final_target_error": .1})
        writer.writerow({"scenario": "reference_clean", "seed": 1,
            "method": "A_uniform", "delivery_status": "none",
            "status": "solver_failure"})
    report = analyze(root)
    assert len(report["missing_keys"]) == 4
    assert report["complete"] is False
    comparisons = [x for x in report["comparisons"] if x["baseline"] == "A_uniform"]
    assert all(x["paired_n"] == 1 and not x["formal_advantage"] for x in comparisons)
    assert all(x["excluded"]["missing_row"] == 1 for x in comparisons)

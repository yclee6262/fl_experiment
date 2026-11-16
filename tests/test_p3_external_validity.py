import csv
import json
import tempfile
import unittest
from pathlib import Path

from analyze_p3_external_validity import analyze


EFFECT_FIELDS = (
    "experiment_id", "scenario", "seed", "agent_id", "delivery_status",
    "failure_reason", "baseline_box_feasible", "operational_box_feasible",
    "full_removal_box_feasible", "operational_internal_contribution",
    "operational_external_contribution", "full_removal_external_contribution",
    "full_removal_true_contribution", "scenario_artifact_dir",
)


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class P3ExternalValidityAnalysisTests(unittest.TestCase):
    def make_source(self, root, scenario, seeds, size, constant=False):
        source = Path(root) / scenario
        source.mkdir()
        (source / "run_manifest.json").write_text(json.dumps({
            "status": "complete", "arguments": {"seeds": seeds,
            "constraint_mode": "box_constrained"}
        }))
        effects = []
        summaries = []
        for seed in seeds:
            artifact = source / "artifacts" / f"seed_{seed}"
            artifact.mkdir(parents=True)
            summaries.append({"seed": seed})
            for index in range(size):
                value = 1.0 if constant else float(index - 1)
                effects.append({
                    "experiment_id": "test", "scenario": scenario, "seed": seed,
                    "agent_id": index + 1, "delivery_status": "valid_delivery",
                    "failure_reason": "", "baseline_box_feasible": True,
                    "operational_box_feasible": True, "full_removal_box_feasible": True,
                    "operational_internal_contribution": value,
                    "operational_external_contribution": value,
                    "full_removal_external_contribution": value,
                    "full_removal_true_contribution": value,
                    "scenario_artifact_dir": f"artifacts/seed_{seed}",
                })
        write_csv(source / "participant_effects.csv", effects, EFFECT_FIELDS)
        write_csv(source / "scenario_summary.csv", summaries, ("seed",))
        return source

    def test_complete_inputs_can_meet_formal_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            clean = self.make_source(tmp, "clean", [10, 11], 3)
            poison = self.make_source(tmp, "controlled_poison", [10, 11], 5)
            result = analyze(
                {"clean": clean, "controlled_poison": poison},
                Path(tmp) / "analysis", [10, 11],
            )
            self.assertTrue(result["coverage_complete"])
            self.assertEqual(result["formal_conclusion"], "usable_external_validity")

    def test_missing_effect_prevents_formal_conclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            clean = self.make_source(tmp, "clean", [10], 3)
            poison = self.make_source(tmp, "controlled_poison", [10], 5)
            rows = list(csv.DictReader((clean / "participant_effects.csv").open()))[:-1]
            write_csv(clean / "participant_effects.csv", rows, EFFECT_FIELDS)
            result = analyze(
                {"clean": clean, "controlled_poison": poison},
                Path(tmp) / "analysis", [10],
            )
            self.assertFalse(result["coverage_complete"])
            self.assertEqual(result["formal_conclusion"], "external_validity_not_established")

    def test_undefined_spearman_does_not_create_false_ci(self):
        with tempfile.TemporaryDirectory() as tmp:
            clean = self.make_source(tmp, "clean", [10], 3, constant=True)
            poison = self.make_source(tmp, "controlled_poison", [10], 5, constant=True)
            result = analyze(
                {"clean": clean, "controlled_poison": poison},
                Path(tmp) / "analysis", [10],
            )
            self.assertFalse(result["statistical_signal_all_scenarios"])
            aggregate = list(csv.DictReader(
                (Path(tmp) / "analysis" / "aggregate_metrics.csv").open()
            ))
            self.assertEqual(aggregate[0]["simultaneous_ci_low_spearman"], "")


if __name__ == "__main__":
    unittest.main()

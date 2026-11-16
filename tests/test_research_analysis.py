import unittest
import copy
import numpy as np

from research_analysis import audit_coverage, paired_bootstrap_mean_ci, valid_delivery
from alpha_weight_experiments import StageCounter, attach_query_counter


class ResearchAnalysisTests(unittest.TestCase):
    def test_stage_counter_tracks_api_calls_and_batched_input_rows_separately(self):
        class Agent:
            def api_predict(self, values):
                return np.zeros(len(values))
        agent=Agent();counter=attach_query_counter([agent])
        counter["stage"]="qualification"
        agent.api_predict([[1.0,2.0],[2.0,3.0],[3.0,4.0]])
        self.assertEqual(counter["requests"]["qualification"],1)
        self.assertEqual(counter["input_samples"]["qualification"],3)

    def test_audit_reports_joint_missing_duplicate_and_unexpected_keys(self):
        audit = audit_coverage(
            [{"method": "A", "seed": 1}, {"method": "A", "seed": 1},
             {"method": "B", "seed": 3}],
            [("A", 1), ("B", 1)], ("method", "seed"),
        )
        self.assertEqual(audit["duplicate_keys"], [("A", 1)])
        self.assertEqual(audit["missing_keys"], [("B", 1)])
        self.assertEqual(audit["unexpected_keys"], [("B", 3)])
        self.assertFalse(audit["complete"])

    def test_delivery_requires_finite_score_and_feasibility(self):
        valid = {"delivery_status": "valid", "box_feasible": "True",
                 "external_global_error": "0.2"}
        self.assertTrue(valid_delivery(valid))
        self.assertFalse(valid_delivery({**valid, "external_global_error": "NaN"}))
        self.assertFalse(valid_delivery({**valid, "box_feasible": "False"}))
        self.assertFalse(valid_delivery({**valid, "final_solution": ""}))

    def test_seed_level_bootstrap_is_reproducible_and_empty_is_undefined(self):
        first = paired_bootstrap_mean_ci([-1.0, -0.5, 0.2])
        second = paired_bootstrap_mean_ci([-1.0, -0.5, 0.2])
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], first[1])
        self.assertEqual(paired_bootstrap_mean_ci([]), (None, None))

    def test_stage_counter_can_be_deep_copied_for_method_pairing(self):
        counter = StageCounter()
        counter["stage"] = "qualification"
        counter["requests"]["qualification"] = 3
        cloned = copy.deepcopy(counter)
        self.assertEqual(cloned["requests"]["qualification"], 3)
        self.assertEqual(cloned["stage"], "qualification")


if __name__ == "__main__":
    unittest.main()

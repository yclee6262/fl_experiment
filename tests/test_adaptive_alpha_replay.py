import unittest

from run_adaptive_alpha_replay import should_solve_uniform


class AdaptiveAlphaDecisionTests(unittest.TestCase):
    def test_valid_B_at_or_below_threshold_stops_without_second_solve(self):
        self.assertEqual(should_solve_uniform("valid", 0.01),
                         (False, "B_internal_error_within_threshold"))
        self.assertEqual(should_solve_uniform("valid", 0.0),
                         (False, "B_internal_error_within_threshold"))

    def test_bad_or_unscorable_B_triggers_uniform(self):
        self.assertEqual(should_solve_uniform("valid", 0.010001),
                         (True, "B_internal_error_above_threshold"))
        self.assertEqual(should_solve_uniform("solver_failure", None),
                         (True, "B_failed"))
        self.assertEqual(should_solve_uniform("valid", float("nan")),
                         (True, "B_unscorable"))

    def test_invalid_threshold_is_rejected(self):
        for threshold in (-0.1, float("nan"), float("inf")):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    should_solve_uniform("valid", 0.0, threshold)


if __name__ == "__main__":
    unittest.main()

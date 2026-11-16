import csv
import json
import tempfile
import unittest
from pathlib import Path

from analyze_p5_results import analyze as analyze_p5
from analyze_p6_recruitment import analyze as analyze_p6


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader();writer.writerows(rows)


class FailureAnalysisTests(unittest.TestCase):
    def test_p5_retains_paired_qualification_failures_without_quality_estimate(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);scenario=root/"reference_clean";scenario.mkdir()
            (root/"suite_manifest.json").write_text(json.dumps({"seeds":[0],
                "comparison":["A_uniform_no_update","D_uniform_guarded"],
                "scenarios":[{"name":"reference_clean"}]}))
            rows=[{"method_label":method,"seed":"0","execution_status":"completed",
                "delivery_status":"qualification_failure","failure_stage":"qualification",
                "failure_reason":"agent rejected","final_solution":"","final_box_feasible":"",
                "external_global_error":"","final_target_error":""}
                for method in ("A_uniform_no_update","D_uniform_guarded")]
            write_csv(scenario/"summary.csv",rows)
            summaries,pairs,comparisons,coverage,shared=analyze_p5(root)
            self.assertTrue(coverage[0]["complete"])
            self.assertFalse(pairs[0]["valid_quality_pair"])
            self.assertIsNone(comparisons[0]["mean_delta_D_minus_A"])
            self.assertEqual(sum(row.get("record_type")=="aggregate" and row.get("valid_deliveries")==0 for row in summaries),2)

    def test_p5_duplicate_key_is_audited_instead_of_silently_collapsed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);scenario=root/"reference_clean";scenario.mkdir()
            (root/"suite_manifest.json").write_text(json.dumps({"seeds":[0],
                "comparison":["A_uniform_no_update","D_uniform_guarded"],
                "scenarios":[{"name":"reference_clean"}]}))
            rows=[{"method_label":method,"seed":"0","delivery_status":"qualification_failure",
                "failure_reason":"qualification failed"} for method in ("A_uniform_no_update","D_uniform_guarded")]
            write_csv(scenario/"summary.csv",rows+[rows[0]])
            _,_,_,coverage,_=analyze_p5(root)
            self.assertFalse(coverage[0]["complete"])
            self.assertEqual(len(json.loads(coverage[0]["duplicate_keys"])),1)

    def test_p6_all_selection_failures_still_analyze_complete_condition_grid(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/"run_manifest.json").write_text(json.dumps({"seeds":[0],"budgets":[2.5],"etas":[0.0]}))
            rows=[]
            for method in ("exact","objective_greedy","legacy_greedy"):
                rows.append({"seed":"0","budget":"2.5","diversity_eta":"0.0",
                    "selection_method":method,"delivery_status":"no_feasible_selection",
                    "failure_stage":"selection","failure_reason":"no affordable coalition",
                    "final_solution":"","selection_objective":"","external_global_error":"",
                    "budget_feasible":"","box_feasible":"","payment_status":"not_run_by_configuration"})
            write_csv(root/"selection_results.csv",rows)
            summaries,pairs,aggregates,costs,coverage=analyze_p6(root)
            self.assertTrue(coverage["complete"])
            self.assertEqual(len(summaries),3)
            self.assertTrue(all(row["external_error_delta_from_exact"] is None for row in pairs))
            self.assertEqual(len(costs),1)
            self.assertEqual(len(aggregates),3)


if __name__=="__main__":unittest.main()

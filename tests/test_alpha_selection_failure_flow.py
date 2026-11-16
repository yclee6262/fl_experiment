"""An all-qualification-failure suite must remain analyzable without external models."""
import torch
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import run_alpha_initialization_selection as runner

class FailureFlow(unittest.TestCase):
 def test_all_qualification_failures_preserved_and_no_evaluator_loaded(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);source=root/'source'
   for sc in runner.SCENARIOS:
    p=source/sc;p.mkdir(parents=True)
    (p/'shared_costs.jsonl').write_text(json.dumps({'seed':9,'missing_fixed_ids':[20],'artifact_path':None})+'\n')
   reg={'source_sha256':{},'contrasts':[('C','B_stage0')],'failure_test':{'source':str(source),'seeds':[9],'results':str(root/'results')}}
   rp=root/'registration.json';rp.write_text(json.dumps(reg))
   with patch.object(runner,'load_evaluator_artifact',side_effect=AssertionError('Evaluator must not load')),patch.object(runner.shared,'load_base',side_effect=AssertionError('Model must not load')):
    runner.main_run(str(rp),'failure_test')
   rows=runner.lines(root/'results/results.jsonl')
   self.assertEqual(len(rows),30);self.assertTrue(all(r['status']=='qualification_failure' for r in rows))
   self.assertTrue(json.load(open(root/'results/verification.json'))['passed'])
   self.assertEqual(json.load(open(root/'results/analysis.json'))['statuses'],{'qualification_failure':30})

if __name__=='__main__':unittest.main()

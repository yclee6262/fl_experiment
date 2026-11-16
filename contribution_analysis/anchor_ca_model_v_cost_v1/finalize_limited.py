"""Describe a resource-limited run without declaring the full experiment complete."""
import json
from pathlib import Path
O=Path(__file__).resolve().parents[1]/'anchor_ca_model_v_cost_20261104_attempt_01'
failure=json.loads((O/'failure.json').read_text());rows=json.loads((O/'progress.json').read_text())
assert failure['error']=="RuntimeError('physical_sample_cap')"
assert failure['physical_samples']==5000000
value=dict(complete=False,status='resource_limit',completed_jobs=len(rows),planned_jobs=300,
    historical_complete=sum(r['phase']=='historical' for r in rows)==20,
    known_analytic_complete=sum(r['phase']=='known_analytic' for r in rows)==40,
    new_primary_complete=sum(r['phase']=='new_analytic' and r['iterations']==30 for r in rows)==80,
    new_10_complete=sum(r['phase']=='new_analytic' and r['iterations']==10 for r in rows)==80,
    new_60_completed=sum(r['phase']=='new_analytic' and r['iterations']==60 for r in rows),
    physical_samples_all_repeats=failure['physical_samples'],uncompleted_job_artifacts='not analyzed',
    warning='60-round prefix is incomplete and cannot stand for all 80 teams')
(O/'completion.json').write_text(json.dumps(value,indent=2));print(value)

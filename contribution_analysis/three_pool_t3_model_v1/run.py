import argparse
import datetime
from fractions import Fraction
from pathlib import Path
import shutil
import subprocess
import sys
import time
from contribution_analysis.custom_realized_gain_v1.schema import read,save,sha,plain
from .fixtures import cases
from .core import execute
from .reference import verify
from .historical import reanalyze,OLD

WORK=Path(__file__).resolve().parents[2]
DESIGN=WORK/'docs/three_pool_t3_model_design_v1_20261028.md'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=WORK/'contribution_analysis/three_pool_t3_model_experiment_v1_20261028');args=ap.parse_args()
    root=args.output.resolve();root.mkdir(exist_ok=False);(root/'source_snapshot').mkdir();(root/'cases').mkdir();started=time.perf_counter();calls=0
    sources=[DESIGN,WORK/'docs/three_pool_contribution_validation_plan_v1_20261028.md',WORK/'host_server.py',*Path(__file__).parent.glob('*.py'),
        *(WORK/'contribution_analysis/model_step_decision_v1').glob('*.py'),*(OLD/'source_snapshot'/n for n in ['verify.py','study.py','engine.py'])]
    lock={}
    for p in sources:
        rel=p.relative_to(WORK);dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);lock[str(rel)]=sha(p)
    save(root/'source_lock.json',lock);fixed=cases();save(root/'fixtures.json',fixed)
    save(root/'registration.json',dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),design_sha256=sha(DESIGN),fixtures_sha256=sha(root/'fixtures.json'),
        development=30,heldout=30,deadzone=1e-8,historical_deadzone=1e-4,diagnostic_grid_points=21,max_calls=10000,max_seconds=120,max_bytes=100000000,
        expectation='technical correctness plus anticipated self-preference counterexamples',actual_payments=0,pooled_calls=0,new_formal_confirmation=False))
    try:
        historic,hv,inputs=reanalyze();save(root/'historical_reanalysis.json',historic);save(root/'historical_verification.json',hv);save(root/'historical_inputs.json',inputs)
        tests=[]
        commands=[[sys.executable,'-m','unittest','discover','-s','contribution_analysis/model_step_decision_v1','-p','test_engine.py','-v'],
                  [sys.executable,'-m','unittest','contribution_analysis.three_pool_t3_model_v1.test_core','-v']]
        for cmd in commands:
            result=subprocess.run(cmd,cwd=WORK,capture_output=True,text=True);tests.append(dict(command=cmd,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        save(root/'technical_tests.json',tests)
        if any(t['returncode'] for t in tests):raise ValueError('technical_tests_failed')
        results=[]
        for split in ['development','heldout']:
            for c in [c for c in fixed if c['split']==split]:
                doc=execute(c);verification=verify(c,plain(doc));doc['verification']=verification
                from .historical import interpreter,server,same
                import numpy as np
                s=server(dict(candidates=[[float(Fraction(z)) for z in row] for row in c['anchors']],alphas=[float(Fraction(z)) for z in c['anchor_weights']]))
                table={(e['model'],tuple(e['x'])):e['reply'] for e in doc['events']};w=np.asarray(doc['model_weights'])
                def objective(ids):
                    ww=w[ids]/sum(w[ids]);return lambda x:float(np.dot(ww,[abs(table[i,tuple(x)]) for i in ids]))
                allids=list(range(len(c['models'])));same(interpreter()(s,doc['full']['pre'],objective(allids)),doc['full'])
                if doc['minus'] is not None:same(interpreter()(s,doc['minus']['pre'],objective(doc['remaining_model_indices'])),doc['minus'])
                calls+=sum(doc['stage_calls'].values())
                save(root/'cases'/(c['id']+'.json'),doc)
                results.append(dict(id=c['id'],category=c['category'],split=split,score=doc['score'],stage_calls=doc['stage_calls'],
                    full_accepted=doc['full']['accepted'],minus_accepted=None if doc['minus'] is None else doc['minus']['accepted'],
                    full_point=doc['lock']['full'],minus_point=doc['lock']['minus'],verification=verification,seconds=doc['seconds']))
                if calls>10000 or time.perf_counter()-started>120:raise RuntimeError('budget_exceeded')
            print(split,'PASSED',sum(r['split']==split for r in results),flush=True)
        for rel,h in lock.items():
            if sha(WORK/rel)!=h or sha(root/'source_snapshot'/rel)!=h:raise ValueError('source_drift')
        for path,h in inputs.items():
            if sha(path)!=h:raise ValueError('historical_input_drift')
        size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        if size>100000000:raise RuntimeError('size_budget')
        save(root/'summary.json',dict(complete=True,analytic_cases=results,analytic_settings=len(results),defined_comparisons=sum(r['score']['status']=='available' for r in results),
             new_synthetic_searches=120,new_synthetic_model_ablation_decisions=sum(r['score']['status']=='available' for r in results),new_synthetic_api_calls=calls,
             historical_development_comparisons=sum(r['phase']=='development' for r in historic),historical_pilot_comparisons=sum(r['phase']=='pilot' for r in historic),
             trained_model_calls=0,new_training=0,pooled_calls=0,truth_loaded=False,analytic_reference_diagnostic_only=True,actual_payments=0,
             diagnostic_model_polynomial_evaluations=60*2*21,diagnostic_reference_grid_points=60*21,seconds=time.perf_counter()-started,bytes_before_audit=size,
             technical_reference_validated=True,objective_quality_contribution_established=False,low_cost_estimator_established=False,
             payment_candidate_validity_passed=False,new_formal_confirmation=False,notion_modified=False))
        print('COMPLETE',len(results),'analytic;',calls,'synthetic API calls')
    except Exception as exc:
        save(root/'failure.json',dict(complete=False,error=repr(exc),synthetic_calls=calls));raise


if __name__=='__main__':main()

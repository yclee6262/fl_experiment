"""Freeze then rescore 60 archived T3 endpoint pairs; no solver execution."""
import argparse
import copy
import datetime
from fractions import Fraction as F
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from contribution_analysis.custom_realized_gain_v1.schema import read,save,sha,key
from .score import score
from .reference import verify,polynomial

WORK=Path(__file__).resolve().parents[2]
T3=WORK/'contribution_analysis/three_pool_t3_model_experiment_v1_20261028'
DESIGN=WORK/'docs/model_scale_invariance_design_v1_20261028.md'


def original_locks():
    result={}
    for name in ['three_pool_t3_model_experiment_v1_20261028','anchor_pool_end_to_end_experiment_v1_20261028']:
        root=WORK/'contribution_analysis'/name;p=root/'completion_receipt.json';result[str(p)]=sha(p)
        for rel,h in read(p)['file_sha256'].items():
            if sha(root/rel)!=h:raise ValueError('old_artifact_hash')
    return result


def loss(model,z):
    x=np.asarray(z,float);l=np.asarray([float(F(v))for v in model['linear']]);Q=np.asarray([[float(F(v))for v in row]for row in model['quadratic']])
    return abs(float(F(model['offset']))+l@x+x@Q@x)


def converted(p,j,factor,consistent=True):
    q=copy.deepcopy(p)
    if consistent:q['probes'][j]=[factor*v for v in q['probes'][j]]
    q['full'][j]*=factor
    if q['minus']is not None:q['minus'][j]*=factor
    return q


def raw_link(d):
    c=d['fixture'];lock=d['lock'];score=d['score']
    for arm,label in [('full','losses_full'),('minus','losses_minus')]:
        if lock[arm]is None:
            if score[label]is not None:raise ValueError('missing_endpoint')
            continue
        for j,v in enumerate(score[label]):
            es=[e for i,e in enumerate(d['events'])if i>=lock['event_index']and e['stage']=='common'and e['model']==j and key(e['x'])==key(lock[arm])]
            if not es or any(abs(e['reply'])!=v for e in es):raise ValueError('raw_endpoint_link')
    return True


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=WORK/'contribution_analysis/model_scale_invariance_experiment_v1_20261028');args=ap.parse_args()
    root=args.output.resolve();root.mkdir(exist_ok=False);(root/'source_snapshot').mkdir();(root/'cases').mkdir();start=time.perf_counter();calls=0
    lock={}
    for p in [DESIGN,*Path(__file__).parent.glob('*.py'),Path(__file__).parent/'README.md']:
        rel=p.relative_to(WORK);dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);lock[str(rel)]=sha(p)
    save(root/'source_lock.json',lock)
    grid=[[a,b]for a in [-1.,-.5,0.,.5,1.]for b in [-1.,-.5,0.,.5,1.]]
    spec=dict(grid=grid,factors=[.1,10.,100.],quantiles=[.1,.9],quantile_method='linear',scale_floor=1e-12,atol=1e-10,rtol=1e-9,
              archived_cases=[p.name for p in sorted((T3/'cases').glob('*.json'))])
    save(root/'fixtures.json',spec);save(root/'registration.json',dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),design_sha256=sha(DESIGN),fixtures_sha256=sha(root/'fixtures.json'),
        old_data_exploration=True,max_analytic_calls=5000,max_seconds=60,max_bytes=50000000,new_searches=0,new_trained_model_calls=0,actual_payments=0))
    try:
        prior=original_locks();save(root/'historical_receipts.json',prior)
        test=subprocess.run([sys.executable,'-m','unittest','contribution_analysis.model_scale_invariance_v1.test_score','-v'],cwd=WORK,text=True,capture_output=True)
        save(root/'technical_tests.json',dict(returncode=test.returncode,stdout=test.stdout,stderr=test.stderr))
        if test.returncode:raise RuntimeError('technical_tests_failed')
        source={};rows=[]
        for name in spec['archived_cases']:
            path=T3/'cases'/name;source[str(path)]=sha(path);d=read(path);raw_link(d);c=d['fixture']
            probes=[[float(loss(model,z))for z in grid]for model in c['models']];calls+=len(grid)*len(probes)
            for model,values in zip(c['models'],probes):
                for z,v in zip(grid,values):
                    if abs(float(abs(polynomial(model,z)))-v)>1e-10:raise ValueError('probe_polynomial')
            panel=dict(weights=d['model_weights'],probes=probes,full=d['score']['losses_full'],minus=d['score']['losses_minus'],target=c['target'])
            baseline=score(panel);verify(panel,baseline);variants=[]
            for j in range(len(probes)):
                for factor in spec['factors']:
                    p=converted(panel,j,factor);v=score(p);independent=verify(p,v);errors={}
                    if v['status']!=baseline['status']:raise ValueError('invariance_status')
                    for field in ['C','own','others']:
                        if baseline[field]is None:
                            if v[field]is not None:raise ValueError('invariance_null')
                            errors[field]=None
                        else:
                            error=abs(v[field]-baseline[field]);errors[field]=error
                            if error>1e-10+1e-9*abs(baseline[field]):raise ValueError('invariance_numeric')
                            if abs(baseline[field])>1e-10 and(v[field]>0)!=(baseline[field]>0):raise ValueError('invariance_sign')
                    variants.append(dict(model=j,factor=factor,panel=p,score=v,absolute_errors=errors,independent=independent))
            control=None
            if baseline['status']=='available':
                p=converted(panel,c['target'][0],10.,False);v=score(p);verify(p,v);control=dict(panel=p,score=v,delta_C=v['C']-baseline['C'])
                if c['id']=='useful_information_0'and abs(control['delta_C'])<=1e-8:raise ValueError('negative_control_missing')
            doc=dict(case_id=c['id'],fixture=c,endpoint_lock=d['lock'],raw_score=d['score'],panel=panel,baseline=baseline,variants=variants,
                     inconsistent_control=control,analytic_probe_calls=len(grid)*len(probes),new_searches=0,source_path=str(path))
            save(root/'cases'/name,doc)
            rows.append(dict(id=c['id'],category=c['category'],raw_C=d['score']['C'],normalized_C=baseline['C'],status=baseline['status'],
                             max_invariance_error=max([e for v in variants for e in v['absolute_errors'].values()if e is not None],default=0),
                             control_delta=None if control is None else control['delta_C'],D_analytic=d['score']['D_analytic']))
            if calls>5000 or time.perf_counter()-start>60:raise RuntimeError('budget_exceeded')
        save(root/'source_cases.json',source)
        if original_locks()!=prior:raise ValueError('old_changed')
        for rel,h in lock.items():
            if sha(WORK/rel)!=h:raise ValueError('source_changed')
        size=sum(p.stat().st_size for p in root.rglob('*')if p.is_file())
        if size>50000000:raise RuntimeError('size_budget')
        save(root/'summary.json',dict(complete=True,cases=rows,baseline_cases=len(rows),transformed_panels=6*len(rows),
            available_baselines=sum(r['status']=='available'for r in rows),undefined_baselines=sum(r['status']!='available'for r in rows),
            max_invariance_error=max(r['max_invariance_error']for r in rows),analytic_probe_calls=calls,new_trained_model_calls=0,new_searches=0,new_training=0,
            pooled_calls=0,actual_payments=0,seconds=time.perf_counter()-start,bytes_before_audit=size,new_formal_confirmation=False,
            fixed_endpoint_scale_invariance_validated=True,decision_or_quality_validity_established=False))
        print('COMPLETE',len(rows),'baselines',6*len(rows),'scale variants',calls,'analytic probe calls')
    except Exception as exc:
        save(root/'failure.json',dict(complete=False,error=repr(exc),analytic_probe_calls=calls));raise


if __name__=='__main__':main()

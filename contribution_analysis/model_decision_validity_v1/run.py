"""Freeze probes before new decisions; keep quality failures as complete evidence."""
import argparse
import copy
import datetime
from pathlib import Path
import shutil
import subprocess
import sys
import time
from contribution_analysis.custom_realized_gain_v1.schema import read,save,sha,plain
from contribution_analysis.three_pool_t3_model_v1.core import execute
from contribution_analysis.three_pool_t3_model_v1.reference import verify as branch_verify
from contribution_analysis.model_scale_invariance_v1.score import score
from contribution_analysis.model_scale_invariance_v1.reference import verify as score_verify
from contribution_analysis.model_scale_invariance_v1.run import loss,converted,original_locks,raw_link
from .fixtures import cases
from .analysis import summarize,matched_pairs

WORK=Path(__file__).resolve().parents[2]
DESIGN=WORK/'docs/model_decision_validity_design_v1_20261028.md'


def previous():
    locks=original_locks()
    p=WORK/'contribution_analysis/model_scale_invariance_experiment_v1_20261028/completion_receipt.json'
    locks[str(p)]=sha(p)
    for rel,h in read(p)['file_sha256'].items():
        if sha(p.parent/rel)!=h:raise ValueError('old_scale_artifact_changed')
    return locks


def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=WORK/'contribution_analysis/model_decision_validity_experiment_v1_20261028')
    root=ap.parse_args().output.resolve();root.mkdir(exist_ok=False);(root/'cases').mkdir();started=time.perf_counter();calls=0
    try:
        deps=[WORK/'host_server.py',WORK/'contribution_analysis/custom_realized_gain_v1/schema.py',DESIGN]
        for folder in ['model_decision_validity_v1','three_pool_t3_model_v1','model_scale_invariance_v1','model_step_decision_v1']:
            deps+=list((WORK/'contribution_analysis'/folder).glob('*.py'))
        deps.append(Path(__file__).parent/'README.md');locked={}
        for p in deps:
            rel=p.relative_to(WORK);dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,dest);locked[str(rel)]=sha(p)
        save(root/'source_lock.json',locked);save(root/'historical_receipts.json',previous())
        fixed=cases();grid=[[x,y]for x in [-1.,-.5,0.,.5,1.]for y in [-1.,-.5,0.,.5,1.]]
        spec=dict(cases=fixed,grid=grid,factors=[.1,10.,100.],deadzone=1e-10,scale_floor=1e-12,
                  max_calls=10000,max_seconds=120,max_bytes=100000000,expected_searches=90)
        save(root/'fixtures.json',spec)
        save(root/'registration.json',dict(created_at=stamp(),design_sha256=sha(DESIGN),fixtures_sha256=sha(root/'fixtures.json'),
             development=15,heldout_parameter=15,new_formal_confirmation=False,analytic_truth_diagnostic_only=True,
             new_trained_model_calls=0,pooled_calls=0,actual_payments=0))
        cmd=[sys.executable,'-m','unittest','contribution_analysis.model_scale_invariance_v1.test_score',
             'contribution_analysis.model_decision_validity_v1.test_analysis','-v']
        t=subprocess.run(cmd,cwd=WORK,capture_output=True,text=True)
        save(root/'technical_tests.json',dict(command=cmd,returncode=t.returncode,stdout=t.stdout,stderr=t.stderr))
        if t.returncode:raise ValueError('technical_tests_failed')
        # Scales must exist for all fixtures before even the first new solver call.
        calibration={c['id']:[[float(loss(model,z))for z in grid]for model in c['models']]for c in fixed}
        save(root/'calibration_lock.json',dict(locked_at=stamp(),probes=calibration,analytic_calls=1500,
                                              solver_calls_so_far=0,fixtures_sha256=sha(root/'fixtures.json')))
        rows=[];docs=[];execution=[]
        for split in ['development','heldout_parameter']:
            for c in [c for c in fixed if c['split']==split]:
                began=stamp();d=plain(execute(c));bv=branch_verify(c,d);raw_link(d)
                panel=dict(weights=d['model_weights'],probes=calibration[c['id']],full=d['score']['losses_full'],
                           minus=d['score']['losses_minus'],target=c['target'])
                norm=score(panel);sv=score_verify(panel,norm);variants=[]
                for j in range(2):
                    for factor in spec['factors']:
                        p=converted(panel,j,factor);v=score(p);score_verify(p,v)
                        errors={k:abs(v[k]-norm[k])for k in ['C','own','others']}
                        if any(errors[k]>1e-10+1e-9*abs(norm[k])for k in errors):raise ValueError('scale_invariance')
                        variants.append(dict(model=j,factor=factor,panel=p,score=v,errors=errors))
                remaining=sum(d['model_weights'][i]for i in d['remaining_model_indices'])
                D=d['score']['D_analytic'];doc=dict(fixture=c,decision=d,panel=panel,normalized=norm,D=D,
                    variants=variants,verification=dict(branch=bv,score=sv),started_at=began,ended_at=stamp(),
                    calibration_sha256=sha(root/'calibration_lock.json'))
                save(root/'cases'/(c['id']+'.json'),doc);docs.append(doc)
                row=dict(id=c['id'],category=c['category'],split=split,raw_C=d['score']['C'],C=norm['C'],own=norm['own'],
                    others=norm['others'],peer=norm['others']/remaining,D=D,full=d['lock']['full'],minus=d['lock']['minus'],
                    full_accepted=d['full']['accepted'],minus_accepted=d['minus']['accepted'],stage_calls=d['stage_calls'],
                    max_invariance_error=max(e for v in variants for e in v['errors'].values()))
                rows.append(row);calls+=sum(d['stage_calls'].values());execution.append(dict(id=c['id'],started_at=began,ended_at=doc['ended_at']))
                if calls>10000 or time.perf_counter()-started>120:raise RuntimeError('budget_exceeded')
            stage=dict(split=split,completed_at=stamp(),rows=[r for r in rows if r['split']==split],
                       analysis=summarize([r for r in rows if r['split']==split]))
            save(root/(split+'_checkpoint.json'),stage)
            print(split,'complete',len(stage['rows']),'quality gate',stage['analysis']['quality_gate_passed'],flush=True)
        pairs=matched_pairs(docs);save(root/'matched_pairs.json',pairs);save(root/'execution_order.json',execution)
        for rel,h in locked.items():
            if sha(WORK/rel)!=h or sha(root/'source_snapshot'/rel)!=h:raise ValueError('source_drift')
        if previous()!=read(root/'historical_receipts.json'):raise ValueError('historical_changed')
        size=sum(p.stat().st_size for p in root.rglob('*')if p.is_file())
        if size>100000000:raise RuntimeError('size_budget')
        save(root/'summary.json',dict(complete=True,rows=rows,analysis=summarize(rows),matched_pairs=len(pairs),
            cases=30,new_synthetic_one_round_solves=90,control_full_solves=30,observed_full_solves=30,minus_solves=30,
            synthetic_search_and_common_api=calls,stage_api={s:sum(r['stage_calls'][s]for r in rows)for s in ['control','full','minus','common']},
            analytic_probe_calls=1500,adapter_diagnostic_model_evaluations=1260,adapter_diagnostic_reference_evaluations=1320,
            numeric_scale_variants=180,max_invariance_error=max(r['max_invariance_error']for r in rows),
            new_trained_model_calls=0,new_training=0,pooled_calls=0,actual_payments=0,seconds=time.perf_counter()-started,
            bytes_before_audit=size,new_formal_confirmation=False,quality_validity_established=False,notion_modified=False))
        print('COMPLETE',len(rows),'cases;',calls,'synthetic API;',len(pairs),'matched pairs')
    except Exception as exc:
        save(root/'failure.json',dict(complete=False,error=repr(exc),synthetic_calls=calls));raise


if __name__=='__main__':main()

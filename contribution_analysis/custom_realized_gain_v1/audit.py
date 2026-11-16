"""Re-read saved evidence and diagnose fixed local algebraic perturbations."""
import argparse
import math
from pathlib import Path
import sys
import time
import numpy as np
from .schema import read,lines,save,sha,plain
from .verify import verify_run


def sensitivity(manifest,run):
    """Freeze G/u/h/weights; perturb one source-model loss at a time.

    This diagnoses the allocation map, NOT an alternate feasible solver trace.
    No forward model/evaluator call and no modification to the delivered score.
    """
    v=np.array(manifest['model_weights']);a=np.array(manifest['direction_weights'])
    owners=list(dict.fromkeys(manifest['anchor_owners']));oi=[owners.index(o) for o in manifest['anchor_owners']]
    K,M=len(a),len(v);ev=run['evaluations'];rows=[];worst=None;undefined=0;tests=0
    for it in run['iterations']:
        if it['accepted_eval_id'] is None:continue
        x=np.asarray(it['x_before']);before=ev[it['current_eval_id']];H=np.zeros((K,len(x)))
        for d in it['directions']:
            if d['skip_reason']:continue
            k=d['index'];loss=np.array(ev[d['source_eval_id']]['individual_losses']);base=np.array(before['individual_losses'])
            H[k]=a[k]*float(np.dot(v,(loss-base)/d['actual_h']))*np.asarray(d['u'])
        def alloc(h):
            g=h.sum(axis=0);n=float(np.linalg.norm(g))
            if n<=1e-8:return None
            p=h@(g/n);po=np.zeros(len(owners))
            for k in range(K):po[oi[k]]+=p[k]
            pos=np.maximum(po,0);den=pos.sum()
            return pos/den if den>0 else None
        baseline=alloc(H)
        for d in it['directions']:
            if d['skip_reason']:continue
            k=d['index'];loss=ev[d['source_eval_id']]['individual_losses']
            for m in range(M):
                eps=1e-6*max(1.,abs(loss[m]),abs(before['individual_losses'][m]))
                for sign in [-1,1]:
                    tests+=1;hh=H.copy();hh[k]+=a[k]*v[m]*sign*eps/d['actual_h']*np.asarray(d['u'])
                    perturbed=alloc(hh)
                    if perturbed is None:undefined+=1;continue
                    delta=float(np.max(abs(perturbed-baseline)))
                    row=dict(t=it['t'],anchor_index=k,model_index=m,sign=sign,source_loss_shift=sign*eps,max_owner_share_change=delta,
                             baseline_share=baseline,perturbed_share=perturbed)
                    if worst is None or delta>worst['max_owner_share_change']:worst=row
                    rows.append(delta)
    return dict(scope='fixed accepted step allocation-map sensitivity; not solver/quality counterfactual',
                epsilon_rule='1e-6 * max(1, abs(source model loss), abs(current model loss))',signs=[-1,1],
                attempted=tests,undefined=undefined,defined=len(rows),max_owner_share_change=max(rows,default=0),
                median_owner_share_change=float(np.median(rows)) if rows else None,
                changes_above_one_percentage_point=sum(z>.01 for z in rows),worst=worst,
                pooled_calls=0,new_model_predictions=0,method_tuned=False)


def audit(root):
    for name,h in read(root/'inputs.json').items():
        if sha(name)!=h:raise ValueError('audit_input_drift')
    for name,h in read(root/'source_lock.json').items():
        if sha(root/name)!=h:raise ValueError('audit_source_drift')
    reports=[]
    for case in read(root/'cases.json'):
        out=root/case['key'];manifest=read(out/'manifest.json');run=read(out/'run_metadata.json')
        for field in ['evaluations','queries','iterations']:run[field]=lines(out/(field+'.jsonl'))
        accounts=read(out/'accounts.json');accounts['steps']=lines(out/'diagnostics.jsonl')
        verification=verify_run(manifest,run,accounts)
        control=read(out/'control.json')
        if control['result']!=run['result']:raise ValueError('audit_control_result')
        for field in ['api_calls','input_samples','query_sha256']:
            if control['costs'][field]!=run['costs'][field]:raise ValueError('audit_control_queries')
        t=time.perf_counter();perturb=sensitivity(manifest,run);perturb['seconds']=time.perf_counter()-t
        save(out/'sensitivity.json',perturb)
        reports.append(dict(case=case['key'],verification=verification,sensitivity=perturb))
    result=dict(accepted=True,complete=True,cases=reports,source_and_input_hashes_passed=True,
                new_model_predictions=0,pooled_loaded=False,truth_loaded=False)
    save(root/'independent_audit.json',result)
    print('AUDIT PASSED',plain(result),flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    root=args.root.resolve();sys.path.insert(0,str(root/'source_snapshot'))
    audit(root)

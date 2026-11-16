"""Persisted T2 audit: independent Decimal and raw historical reply checks.

Does not import core, accounting or the experiment runner.
"""
import argparse
import copy
import math
from pathlib import Path
from contribution_analysis.custom_realized_gain_v1.schema import read,lines,sha,save,key
from contribution_analysis.custom_realized_gain_v1.verify import verify_run
from .reference import verify,decimal

WORK=Path(__file__).resolve().parents[2]
OLD=WORK/'contribution_analysis/a_e_b_validation_v1_20261027'
PILOT=WORK/'contribution_analysis/custom_realized_gain_pilot_v1_20261027'


def near(a,b,tol=1e-10):
    if isinstance(a,list):
        if len(a)!=len(b):raise ValueError('length_mismatch')
        for aa,bb in zip(a,b):near(aa,bb,tol)
    elif not math.isfinite(float(a)) or not math.isfinite(float(b)) or abs(a-b)>tol:raise ValueError('numeric_mismatch')


def bounded(x,d,eta):return max(0.,min([eta]+[(1-v)/z if z>0 else (-1-v)/z for v,z in zip(x,d) if z]))


def verify_historical(reanalysis):
    for folder in [OLD,PILOT]:
        for rel,h in read(folder/'completion_receipt.json')['file_sha256'].items():
            if sha(folder/rel)!=h:raise ValueError('historical_hash_mismatch')
    count=events=0;replays=[]
    for path in sorted(OLD.glob('*_branches.json')):
        doc=read(path);case=doc['case'];m=doc['manifest'];root=PILOT/case;run=read(root/'run_metadata.json')
        for f in ['queries','evaluations','iterations']:run[f]=lines(root/(f+'.jsonl'))
        a=read(root/'accounts.json');a['steps']=lines(root/'diagnostics.jsonl');replays.append(verify_run(m,run,a))
        owners=list(dict.fromkeys(m['anchor_owners']));lookup={}
        for q in doc['queries']:
            expected=math.fsum(w*abs(z-m['target']) for w,z in zip(m['model_weights'],q['replies']));near(expected,q['loss'])
            if key(q['x']) in lookup:near(lookup[key(q['x'])],q['loss'])
            lookup[key(q['x'])]=q['loss'];events+=1
        def validate_branch(branch,x,q,eta,before):
            ts=branch['trials']
            if not ts or len(ts)>10:raise ValueError('historical_trial_count')
            for j,tr in enumerate(ts):
                requested=eta/2**j;step=bounded(x,[-v for v in q],requested)
                near(tr['requested_eta'],requested);near(tr['x'],[v-step*d for v,d in zip(x,q)]);near(tr['loss'],lookup[key(tr['x'])])
                if tr['loss']<before and j<len(ts)-1:raise ValueError('historical_early_accept')
            ok=ts[-1]['loss']<before
            if ok!=branch['accepted'] or (not ok and len(ts)!=10):raise ValueError('historical_acceptance')
            near(branch['x'],ts[-1]['x'] if ok else x);near(branch['loss'],ts[-1]['loss'] if ok else before)
        for row in doc['rows']:
            it=run['iterations'][row['t']];x=it['x_before'];D=len(x);before=run['evaluations'][it['current_eval_id']]['individual_losses'];H=[]
            for owner in owners:
                terms=[[] for _ in range(D)]
                for d in it['directions']:
                    k=d['index']
                    if d['skip_reason'] or m['anchor_owners'][k]!=owner:continue
                    source=run['evaluations'][d['source_eval_id']]['individual_losses']
                    for j in range(D):
                        for v,aa,bb in zip(m['model_weights'],before,source):terms[j].append(m['direction_weights'][k]*v*(bb-aa)/d['actual_h']*d['u'][j])
                H.append([math.fsum(t) for t in terms])
            g=it['g_solver'];norm=math.sqrt(math.fsum(v*v for v in g));q=[v/norm for v in g];idx=owners.index(row['owner'])
            near(H,row['H']);near([math.fsum(h[j] for h in H) for j in range(D)],g)
            ps=[math.fsum(v*d for v,d in zip(h,q)) for h in H];B=max(-ps[idx],0)/math.fsum(abs(v) for v in ps)
            alt=[v-h for v,h in zip(g,H[idx])];alt_norm=math.sqrt(math.fsum(v*v for v in alt));aq=[v/alt_norm for v in alt]
            validate_branch(row['full_round'],x,q,row['original_requested_eta'],it['L_before'])
            validate_branch(row['without_round'],x,aq,row['original_requested_eta'],it['L_before'])
            near(row['full_round']['x'],it['x_after']);near(row['full_round']['loss'],it['L_after'])
            C=row['without_round']['loss']-row['full_round']['loss'];near(C,row['round_D'])
            gain=it['L_before']-it['L_after'];A=gain*max(ps[idx],0)/math.fsum(max(v,0) for v in ps) if it['accepted_eval_id'] is not None else 0.
            target=next(r for r in reanalysis if r['case']==case and r['t']==row['t'] and r['owner']==row['owner'])
            near(target['A'],A);near(target['B'],B);near(target['p'],ps[idx]);near(target['C'],C)
            if target['full_accepted']!=row['full_round']['accepted'] or target['without_accepted']!=row['without_round']['accepted']:raise ValueError('historical_status_mismatch')
            count+=1
    if count!=len(reanalysis) or count!=69:raise ValueError('historical_coverage')
    return dict(passed=True,owner_rounds=count,objective_events=events,saved_objective_solver_replays=replays,new_model_calls=0)


def audit(root):
    for rel,h in read(root/'source_lock.json').items():
        if sha(root/'source_snapshot'/rel)!=h or sha(WORK/rel)!=h:raise ValueError('source_hash')
    reg=read(root/'registration.json')
    if sha(root/'fixtures.json')!=reg['fixtures_sha256']:raise ValueError('fixture_hash')
    for path,h in read(root/'historical_inputs.json').items():
        if sha(path)!=h:raise ValueError('historical_input_drift')
    old=verify_historical(read(root/'historical_reanalysis.json'));reports=[];reply_checks=0;mutations=0
    for c in read(root/'fixtures.json'):
        doc=read(root/'cases'/(c['id']+'.json'));r=doc['run'];m=doc['manifest'];control=doc['control']
        v=verify(c,doc['result']);verify_run(m,r,doc['accounts'])
        if control['result']!=r['result'] or control['costs']['query_sha256']!=r['costs']['query_sha256']:raise ValueError('control_mismatch')
        # Independent exact-rational polynomial evaluation of all raw API replies.
        from decimal import localcontext
        with localcontext() as ctx:
            ctx.prec=60
            def expected(z):
                zz=[decimal(str(v)) for v in z]
                return float(decimal(c['offset'])+sum(decimal(a)*b for a,b in zip(c['linear'],zz))+
                    sum(zz[i]*decimal(c['quadratic'][i][j])*zz[j] for i in range(2) for j in range(2)))
            for q in r['queries']:near(expected(q['x']),q['output'][0],1e-9);reply_checks+=1
            for q in doc['branch_queries']:near(expected(q['x']),q['reply'],1e-9);near(abs(q['reply']),q['loss']);reply_checks+=1
        # Finite C must stay defined when either direction is zero, not forced to 0.
        altered=copy.deepcopy(doc['result']);altered['C']+=1
        try:verify(c,altered)
        except ValueError:mutations+=1
        else:raise ValueError('corrupted_C_not_detected')
        reports.append(dict(id=c['id'],split=c['split'],verification=v))
    if len(reports)!=48 or mutations!=48:raise ValueError('analytic_coverage')
    summary=read(root/'summary.json')
    if summary['new_synthetic_api_calls']!=sum(s['original_search_calls']+s['full_branch_calls']+s['ablation_branch_calls'] for s in summary['analytic_cases']):raise ValueError('cost_count')
    save(root/'independent_audit.json',dict(accepted=True,historical=old,analytic_cases=reports,raw_polynomial_reply_checks=reply_checks,
             corrupted_C_detected=mutations,core_imported=False,new_model_predictions=0,actual_payments=0,pooled_calls=0))
    print('AUDIT PASSED',len(reports),'analytic;',old['owner_rounds'],'historical;',mutations,'corruptions')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);audit(ap.parse_args().root.resolve())

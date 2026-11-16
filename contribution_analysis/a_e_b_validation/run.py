"""Bounded mechanism experiments and saved-round diagnostic interventions."""
import argparse
import copy
import datetime
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tracemalloc
import numpy as np
from contribution_analysis.custom_realized_gain_v1.schema import plain, read, lines, save, sha, check_close
from contribution_analysis.custom_realized_gain_v1.runtime import manifest_for, load_server, equivalent
from contribution_analysis.custom_realized_gain_v1.observer import record_custom_run
from contribution_analysis.custom_realized_gain_v1.accounting import compute_accounts
from contribution_analysis.custom_realized_gain_v1.ledger import build_balance_ledger
from contribution_analysis.custom_realized_gain_v1.verify import verify_run
from contribution_analysis.custom_realized_gain_v1.verify_balance import verify_balance

WORK=Path(__file__).resolve().parents[2]
OLD=WORK/'contribution_analysis/custom_realized_gain_pilot_v1_20261027'
DESIGN=WORK/'docs/a_e_b_validation_design_v1_20261027.md'


class Agent:
    def __init__(self,i,fn): self.agent_id=i; self.fn=fn
    def api_predict(self,x): return np.array([self.fn(np.asarray(x,float))],dtype=np.float64)


def server(functions,anchors,v=None,a=None):
    from host_server import HostServer
    s=HostServer.__new__(HostServer);s.target_T=0.;s.n_features=len(anchors[0])
    s.constraint_mode='box_constrained';s.feasible_lower=-1.;s.feasible_upper=1.;s.feasibility_tolerance=1e-8
    s.I_list=np.asarray(anchors,float);s.alphas=np.asarray(v if v is not None else np.ones(len(functions))/len(functions))
    s.direction_weights=np.asarray(a if a is not None else np.ones(len(anchors))/len(anchors))
    s.trusted_agents=[Agent(i,f) for i,f in enumerate(functions)];return s


def generated(seed,K,M,D):
    rng=np.random.default_rng(seed)
    return dict(seed=seed,anchors=rng.uniform(-.8,.8,(K,D)).tolist(),start=rng.uniform(-.25,.25,D).tolist(),
                centers=rng.uniform(-.5,.5,(M,D)).tolist(),diagonals=rng.uniform(.2,1.5,(M,D)).tolist(),
                offsets=rng.uniform(1,2,M).tolist(),v=(np.arange(M)+1).tolist(),a=(np.arange(K)+1).tolist())


def from_spec(p,scale=1.,perturb=0.,split=False):
    centers=np.asarray(p['centers']);diag=np.asarray(p['diagonals'])
    fs=[lambda x,j=j:scale*(p['offsets'][j]+np.sum(diag[j]*(x-centers[j])**2))+(perturb*x[0] if j==0 else 0.) for j in range(len(centers))]
    anchors=copy.deepcopy(p['anchors']);a=list(p['a']);owners=[f'owner_{k}' for k in range(len(a))]
    if split:
        anchors.insert(0,anchors[0]);a[0]/=2;a.insert(0,a[0]);owners.insert(0,owners[0])
    return server(fs,anchors,p['v'],a),owners


def score(m,r):
    ac=compute_accounts(m,r['evaluations'],r['iterations'],r['initial_eval_id'],r['final_eval_id'])
    b=build_balance_ledger(ac['anchor_owners'],r['iterations'],ac['steps'])
    return ac,b


class Study:
    def __init__(self,root):self.root=root;self.n=0;self.calls=0;self.started=time.perf_counter();self.index=[]
    def count(self,calls):
        self.calls+=calls
        if self.calls>300000 or time.perf_counter()-self.started>300:raise RuntimeError('global_budget')
    def solve(self,name,s,start,n=12,owners=None,control=False):
        m=manifest_for(s,start,dict(num_iterations=n,use_annealing=True,allow_tangent=True),anchor_owners=owners)
        r=record_custom_run(s,start,m,not control);self.n+=1;self.count(r['costs']['api_calls'])
        if control:
            save(self.root/'controls'/f'{name}.json',dict(manifest=m,run=r))
            return m,r,None,None
        ac,b=score(m,r);validation=dict(accounts=verify_run(m,plain(r),plain(ac)),balance=verify_balance(m,plain(r),plain(b)))
        path=self.root/'runs'/f'{name}.json';save(path,dict(manifest=m,run=r,accounts=ac,balance=b,verification=validation))
        self.index.append(str(path.relative_to(self.root)));return m,r,ac,b


def support(H):
    H=np.asarray(H,float);g=H.sum(axis=0);norm=np.linalg.norm(g)
    if norm<=1e-8:return dict(q=None,p=None,Ashare=None,B=None,norm=float(norm))
    q=g/norm;p=H@q;pos=np.maximum(p,0)
    return dict(q=q,p=p,Ashare=pos/pos.sum(),B=np.maximum(-p,0)/np.abs(p).sum(),norm=float(norm))


def mechanisms(st):
    rows=[]
    for name,anchors in [('near',[ [.1],[.2] ]),('far',[ [.1],[.9] ])]:
        _,r,a,b=st.solve('A_'+name,server([lambda x:10+x[0]+4*x[0]**2],anchors),[0.],1)
        rows.append(dict(case=name,anchors=anchors,endpoint=r['result']['solution'],G=a['G_total'],share=a['owner_share']))
    check_close(rows[0]['endpoint'],rows[1]['endpoint'],'A_counterexample_endpoint')
    check_close(rows[0]['G'],rows[1]['G'],'A_counterexample_gain')
    if np.allclose(rows[0]['share'],rows[1]['share']):raise ValueError('A_counterexample_missing')
    fs=[lambda x:10+x[0],lambda x:2*(10+x[0]),lambda x:15+x[0],lambda x:10-x[0]]
    _,r,a,_=st.solve('E_quality',server(fs,[[.8],[.3]]),[0.],1)
    grid=np.linspace(-.5,.5,21);rmse=[float(np.sqrt(np.mean([(f(np.array([x]))-(10+x))**2 for x in grid]))) for f in fs]
    if not (a['E_model'][1]>a['E_model'][0] and rmse[1]>rmse[0]):raise ValueError('E_counterexample_missing')
    hand=[]
    for name,H,gradient in [('collinear',[[3.,0.],[-1.,0.]],[1.,0.]),
                            ('helpful',[[2.,0.],[-1.,.8]],[1.,2.]),
                            ('harmful',[[2.,0.],[-1.,.8]],[1.,0.]),
                            ('cancel',[[1.,0.],[-1.,0.]],[1.,0.]),
                            ('near_cancel',[[1.,0.],[-1.,1e-7]],[1.,0.])]:
        base=support(H);other=support([H[0]]);D=None
        if base['q'] is not None:D=float(np.dot(gradient,-.1*other['q'])-np.dot(gradient,-.1*base['q']))
        hand.append(dict(case=name,H=H,gradient=gradient,full=base,without_second=other,fixed_step_D=D))
    ds={r['case']:r['fixed_step_D'] for r in hand}
    if not (abs(ds['collinear'])<1e-12 and ds['helpful']>0 and ds['harmful']<0 and ds['cancel'] is None):raise ValueError('B_hand_sign')
    return dict(A=rows,E=dict(E=a['E_model'],RMSE=rmse,endpoint=r['result']['solution'],G=a['G_total'],grid=grid,
                              diagnostic_synthetic_reference_evaluations=21,diagnostic_synthetic_model_evaluations=84),B=hand)


def holdouts(st):
    result=[]
    for seed in range(27101,27113):
        p=generated(seed,4,3,2 if seed<=27106 else 8);s,owners=from_spec(p)
        _,control,_,_=st.solve(f'{seed}_control',s,p['start'],owners=owners,control=True)
        m,base,ac,b=st.solve(f'{seed}_base',s,p['start'],owners=owners);equivalent(control,base)
        mods={}
        for label,kwargs in [('split',dict(split=True)),('scale7',dict(scale=7.)),('minus',dict(perturb=-1e-6)),('plus',dict(perturb=1e-6)),('start',{})]:
            ss,oo=from_spec(p,**kwargs);start=list(p['start'])
            if label=='start':start[0]+=.01
            mm,rr,aa,bb=st.solve(f'{seed}_{label}',ss,start,owners=oo)
            delta=None if ac['owner_share'] is None or aa['owner_share'] is None else float(np.max(abs(np.asarray(ac['owner_share'])-aa['owner_share'])))
            mods[label]=dict(share_max_change=delta,endpoint_distance=float(np.linalg.norm(np.asarray(base['result']['solution'])-rr['result']['solution'])),
                G=aa['G_total'],E=aa['E_model'],B=[z['mean_b'] for z in bb['owner_summary']],accepted=aa['accepted_count'],
                rounds=len(rr['iterations']),undefined_share=aa['owner_share'] is None,
                E_sign_changes=int(np.sum(np.sign(ac['E_model'])!=np.sign(aa['E_model']))),
                B_max_change=max((abs(x['mean_b']-y['mean_b']) for x,y in zip(b['owner_summary'],bb['owner_summary']) if x['mean_b'] is not None and y['mean_b'] is not None),default=None))
            if label=='split':
                check_close(ac['A_support_owner'],aa['A_support_owner'],'holdout_split_A')
                check_close(base['result']['solution'],rr['result']['solution'],'holdout_split_endpoint')
                check_close(ac['E_model'],aa['E_model'],'holdout_split_E')
                check_close([z['mean_b'] for z in b['owner_summary']],[z['mean_b'] for z in bb['owner_summary']],'holdout_split_B')
            if label=='scale7':
                check_close(7*np.asarray(ac['A_support_owner']),aa['A_support_owner'],'holdout_scale_A')
                check_close(7*np.asarray(ac['E_model']),aa['E_model'],'holdout_scale_E')
                if delta is not None and delta>1e-8:raise ValueError('holdout_scale_share')
        result.append(dict(seed=seed,spec=p,base=dict(G=ac['G_total'],A=ac['A_support_owner'],E=ac['E_model'],share=ac['owner_share'],
            B=[z['mean_b'] for z in b['owner_summary']],rounds=len(base['iterations']),accepted=ac['accepted_count']),mods=mods,paired_equivalent=True))
        print('HOLDOUT',seed,flush=True)
    return result


def owner_vectors(m,run,it):
    owners=list(dict.fromkeys(m['anchor_owners']));H=np.zeros((len(owners),len(it['x_before'])))
    before=run['evaluations'][it['current_eval_id']]['individual_losses']
    for d in it['directions']:
        if d['skip_reason']:continue
        k=d['index'];src=run['evaluations'][d['source_eval_id']]['individual_losses']
        slope=math.fsum(m['model_weights'][j]*(src[j]-before[j])/d['actual_h'] for j in range(len(before)))
        H[owners.index(m['anchor_owners'][k])]+=m['direction_weights'][k]*slope*np.asarray(d['u'])
    check_close(H.sum(axis=0),it['g_solver'],'branch_source_gradient');return owners,H


def branch_cases(st):
    receipt=read(OLD/'completion_receipt.json')
    for name,h in receipt['file_sha256'].items():
        if sha(OLD/name)!=h:raise ValueError('old_receipt_hash:'+name)
    for name in ['host_server.py','agent_client.py','models.py']:
        if sha(WORK/name)!=sha(OLD/'source_snapshot'/name):raise ValueError('old_solver_or_model_code_drift')
    out=[]
    for c in read(OLD/'cases.json'):
        root=OLD/c['key'];m=read(root/'manifest.json');run=read(root/'run_metadata.json')
        for field in ['queries','evaluations','iterations']:run[field]=lines(root/(field+'.jsonl'))
        oldac=read(root/'accounts.json');oldac['steps']=lines(root/'diagnostics.jsonl')
        verify_run(m,run,oldac)
        if sha(c['models'])!=m['model_artifact_sha256']:raise ValueError('model_hash')
        s=load_server(c);queries=[]
        def objective(x):
            replies=[float(ag.api_predict(x)[0]) for ag in s.trusted_agents];st.count(len(replies))
            val=float(np.sum(np.asarray(m['model_weights'])*np.abs(np.array(replies)-m['target'])))
            queries.append(dict(x=np.asarray(x),replies=replies,loss=val));return val
        rows=[]
        for it in run['iterations']:
            owners,H=owner_vectors(m,run,it);g=np.asarray(it['g_solver']);norm=np.linalg.norm(g)
            if norm<=1e-8:
                rows.append(dict(t=it['t'],status='original_zero_direction',owners=owners));continue
            x=np.asarray(it['x_before']);q=g/norm;eta=it['trials'][0]['requested_eta'];origstep=it['trials'][0]['actual_bounded_step'];before=it['L_before']
            original_first=x-s._bounded_step(x,-q,eta)*q
            check_close(original_first,run['evaluations'][it['trials'][0]['eval_id']]['x'],'branch_control_first')
            check_close(objective(original_first),run['evaluations'][it['trials'][0]['eval_id']]['objective_value'],'branch_control_loss')
            def accept(qx):
                trials=[];step=eta
                for _ in range(10):
                    z=x-s._bounded_step(x,-qx,step)*qx;loss=objective(z);trials.append(dict(x=z,loss=loss,requested_eta=step))
                    if loss<before:return dict(x=z,loss=loss,accepted=True,trials=trials)
                    step/=2
                return dict(x=x,loss=before,accepted=False,trials=trials)
            full=accept(q)
            check_close(full['x'],it['x_after'],'branch_control_accepted_endpoint');check_close(full['loss'],it['L_after'],'branch_control_accepted_loss')
            for oi,o in enumerate(owners):
                altg=g-H[oi];anorm=float(np.linalg.norm(altg));p=float(H[oi]@q);den=float(np.abs(H@q).sum())
                row=dict(t=it['t'],owner=o,p=p,B=max(-p,0)/den,opposing=p<0,x=x,H=H,g=g,owner_index=oi,
                         original_requested_eta=eta,original_first_actual_step=origstep,full_round=full,
                         original_loss=before,status='alternative_zero_direction' if anorm<=1e-8 else 'available')
                if anorm>1e-8:
                    aq=altg/anorm;common=min(origstep,s._bounded_step(x,-aq,origstep))
                    fullx=x-common*q;altx=x-common*aq;fl=objective(fullx);al=objective(altx);alt=accept(aq)
                    row.update(common_step=common,fixed_full_x=fullx,fixed_without_x=altx,fixed_full_loss=fl,fixed_without_loss=al,
                               fixed_D=al-fl,fixed_status='available' if common>0 else 'no_common_motion',without_round=alt,
                               round_D=alt['loss']-full['loss'],direction_distance=float(np.linalg.norm(q-aq)))
                rows.append(row)
        save(st.root/(c['key']+'_branches.json'),dict(case=c['key'],model_sha256=m['model_artifact_sha256'],manifest=m,rows=rows,
             queries=queries,new_model_calls=len(queries)*len(m['model_ids']),old_data_exploration=True))
        out.append(dict(case=c['key'],rows=rows,new_model_calls=len(queries)*len(m['model_ids'])))
        print('BRANCH',c['key'],len(rows),flush=True)
    return out


def costs(st):
    result=[]
    for j,(K,M,D,N) in enumerate([(3,3,2,10),(8,8,8,20),(16,16,16,30),(8,3,8,20)]):
        spec=generated(28101+j,K,M,D);s,oo=from_spec(spec);m=manifest_for(s,spec['start'],dict(num_iterations=N,use_annealing=True,allow_tangent=True),anchor_owners=oo)
        samples=[];last=None
        for rep in range(8):
            pair={}
            for flag in ([False,True] if rep%2==0 else [True,False]):
                r=record_custom_run(s,spec['start'],m,flag);st.n+=1;st.count(r['costs']['api_calls']);pair[flag]=r
            equivalent(pair[False],pair[True]);last=pair[True]
            if rep:samples.append(dict(control_seconds=pair[False]['costs']['search_seconds'],record_seconds=last['costs']['search_seconds'],
                                      control_calls=pair[False]['costs']['api_calls'],record_calls=last['costs']['api_calls'],
                                      control_digest=pair[False]['costs']['query_sha256'],record_digest=last['costs']['query_sha256'],
                                      control_result=pair[False]['result'],record_result=last['result']))
        offline=[];serialization=[]
        for _ in range(7):
            t=time.perf_counter();ac,b=score(m,last);offline.append(time.perf_counter()-t)
            t=time.perf_counter();payload=json.dumps(plain(dict(manifest=m,run=last,accounts=ac,balance=b)),allow_nan=False).encode();serialization.append(time.perf_counter()-t)
        # Separate memory probe, no search or model calls.
        tracemalloc.start();score(m,last);_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        verify_run(m,plain(last),plain(ac));verify_balance(m,plain(last),plain(b))
        path=st.root/'runs'/f'cost_{j}.json';save(path,dict(manifest=m,run=last,accounts=ac,balance=b));st.index.append(str(path.relative_to(st.root)))
        ctrl=float(np.median([v['control_seconds'] for v in samples]));rec=float(np.median([v['record_seconds'] for v in samples]))
        result.append(dict(K=K,M=M,D=D,budget=N,spec=spec,rounds=len(last['iterations']),accepted=ac['accepted_count'],samples=samples,
                           control_median_seconds=ctrl,record_median_seconds=rec,ratio=rec/ctrl,target_passed=rec/ctrl<=1.15,
                           offline_samples=offline,offline_median_seconds=float(np.median(offline)),serialization_samples=serialization,
                           serialization_median_seconds=float(np.median(serialization)),serialized_bytes=len(payload),
                           offline_python_peak_bytes=peak,search_memory_measured=False,native_memory_measured=False))
        print('COST',K,M,D,rec/ctrl,flush=True)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=WORK/'contribution_analysis/a_e_b_validation_v1_20261027');args=ap.parse_args()
    root=args.output.resolve();root.mkdir(exist_ok=False);(root/'runs').mkdir();(root/'controls').mkdir();(root/'source_snapshot').mkdir()
    sources=[DESIGN,*Path(__file__).parent.glob('*.py'),* (WORK/'contribution_analysis/custom_realized_gain_v1').glob('*.py'),
             * (WORK/'contribution_analysis/custom_realized_gain_v1/tests').glob('*.py'),
             * (WORK/'contribution_analysis/custom_realized_gain_v1/tests_balance').glob('*.py'),
             WORK/'host_server.py',WORK/'agent_client.py',WORK/'models.py']
    lock={}
    for p in sources:
        rel=p.relative_to(WORK);dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);lock[str(rel)]=sha(p)
    save(root/'source_lock.json',lock)
    save(root/'registration.json',dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),design_sha256=sha(DESIGN),
         synthetic_seeds=list(range(27101,27113)),cost_seeds=list(range(28101,28105)),original_formulas_unchanged=True,
         historical_cases='development only',synthetic_holdout='fixed mechanism validation, not real task confirmation',
         pooled_loaded=False,truth_loaded=False,synthetic_reference_used_for_diagnosis=True,new_training=False,new_formal_confirmation=False))
    st=Study(root)
    try:
        tests=[]
        for folder in ['tests','tests_balance']:
            run=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(WORK/'contribution_analysis/custom_realized_gain_v1'/folder),'-v'],capture_output=True,text=True,cwd=WORK)
            tests.append(dict(folder=folder,returncode=run.returncode,stdout=run.stdout,stderr=run.stderr))
        save(root/'technical_tests.json',tests)
        if any(r['returncode'] for r in tests):raise RuntimeError('existing_tests_failed')
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        mech=mechanisms(st);save(root/'mechanisms.json',mech)
        hs=holdouts(st);save(root/'holdouts.json',hs)
        bs=branch_cases(st);cs=costs(st)
        save(root/'costs.json',cs);save(root/'run_index.json',st.index)
        for rel,h in lock.items():
            if sha(WORK/rel)!=h or sha(root/'source_snapshot'/rel)!=h:raise ValueError('source_drift')
        size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        if size>250000000:raise RuntimeError('size_budget')
        save(root/'summary.json',dict(complete=True,synthetic_solves=st.n,actual_api_calls=st.calls,
            saved_recorded_solves=len(st.index),trained_model_branch_calls=sum(c['new_model_calls'] for c in bs),
            bytes_before_audit=size,seconds=time.perf_counter()-st.started,pooled_calls=0,new_training=0,new_formal_confirmation=False,
            economic_validity_established=False,notion_modified=False))
        print('COMPLETE',root,flush=True)
    except Exception as error:
        save(root/'failure.json',dict(error=repr(error),complete=False,solves=st.n,calls=st.calls));raise


if __name__=='__main__':main()

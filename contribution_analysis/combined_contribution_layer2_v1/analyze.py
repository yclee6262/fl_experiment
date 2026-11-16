"""Frozen-rule local additivity, offline endpoint quality, and full-path checks."""
import csv
import time
import numpy as np
from scipy.stats import spearmanr
from study import ROOT,WORK,read,save,sha,setup,check,key

def sign(x):return int(x>1e-4)-int(x< -1e-4)
def rho(x,y):
    if len(x)<3 or np.ptp(x)<=1e-4 or np.ptp(y)<=1e-4:return None
    r=float(spearmanr(x,y).statistic);return r if np.isfinite(r) else None
def boot(v):
    if not v:return dict(mean=None,interval=None,n=0)
    a=np.asarray(v,float);rng=np.random.default_rng(20261021)
    means=a[rng.integers(0,len(a),(10000,len(a)))].mean(axis=1)
    return dict(mean=float(a.mean()),interval=np.percentile(means,[.1,99.9]).tolist(),n=len(a))
def agreement(rows,x,y):
    ds=[r for r in rows if sign(r[x])]
    return None if not ds else float(np.mean([sign(r[x])==sign(r[y]) for r in ds]))
def csvsave(p,rows):
    with p.open('x',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)

def analyze():
    import torch
    from models import PretrainedModel
    check();assert read(ROOT/'verification.json')['accepted'];lock=read(ROOT/'scores_and_points_locked.json')['hashes']
    for n,h in lock.items():assert sha(ROOT/n)==h
    start=time.perf_counter();rows=[];teams=[];parts=[];pooled=[];pool_count=0;reload_error=0.
    prior=read(WORK/'contribution_analysis/combined_contribution_layer1_experiment_v1_20261021/comparisons.json')
    refs={(r['key'],r['agent_id']):r for r in prior}
    for c in read(ROOT/'cases.json')['development']:
        print('OFFLINE',c['key'],flush=True);record=read(ROOT/'development'/(c['key']+'.json'))
        net=PretrainedModel(len(c['fixed_start']));net.load_state_dict(torch.load(c['evaluator'],map_location='cpu',weights_only=True)['state_dict']);net.eval();table={}
        def E(x):
            nonlocal pool_count
            k=key(x)
            if k not in table:
                if (pool_count+1)*2>12000:raise RuntimeError('pooled_budget')
                with torch.no_grad():y=abs(float(net(torch.tensor(x,dtype=torch.float32)[None,:]).item()))
                table[k]=dict(x=x,E=y);pool_count+=1
            return table[k]['E']
        local=[]
        for g in record['groups']:
            fullx=g['full']['post']['S_current'];ef=E(fullx);e={role:E(b['endpoint'])-ef for role,b in g['roles'].items()}
            r=dict(key=c['key'],scenario=c['scenario'],seed=c['seed'],iteration=g['t'],agent_id=g['agent_id'],
                a2=g['a2'],m2=g['m2'],c2=g['c2'],j=g['j'],residual=g['residual'],
                e_anchor=e['anchor'],e_model=e['model'],e_joint=e['joint'],full_accepted=g['full']['accepted'],
                mode=g['full']['pre']['current_method'],joint_accepted=g['roles']['joint']['decision']['accepted'],
                full_x=fullx,joint_x=g['roles']['joint']['endpoint'])
            local.append(r)
        rows.extend(local)
        for i in c['ids']:
            rr=[r for r in local if r['agent_id']==i];ref=refs[c['key'],i]
            assert len(rr)==len(record['groups'])//len(c['ids'])
            p=dict(key=c['key'],scenario=c['scenario'],seed=c['seed'],agent_id=i,
                A2=sum(r['a2'] for r in rr),M2=sum(r['m2'] for r in rr),C2=sum(r['c2'] for r in rr),J=sum(r['j'] for r in rr),
                R=sum(r['residual'] for r in rr),D_E=ref['D_E'],D_L=ref['D_L'],old_A=ref['A'],old_C=ref['C'])
            assert abs(p['M2']-ref['M'])<1e-10;parts.append(p)
        T=len(record['groups'])//len(c['ids'])
        for x,y,purpose in [('c2','j','additivity'),('c2','e_joint','combined_external'),('j','e_joint','joint_external')]:
            rs=[]
            for t in range(T):
                rr=[r for r in local if r['iteration']==t];v=rho([r[x] for r in rr],[r[y] for r in rr])
                if v is not None:rs.append(v)
            numerator=sum(abs(r['residual']) for r in local)
            denominator=sum(abs(r['a2'])+abs(r['m2'])+abs(r['j']) for r in local)
            teams.append(dict(key=c['key'],scenario=c['scenario'],seed=c['seed'],scope='local',purpose=purpose,
                score=x,reference=y,rho=None if not rs else float(np.mean(rs)),valid_rounds=len(rs),rounds=T,
                agreement=agreement(local,x,y),residual_ratio=None if denominator==0 else numerator/denominator))
        pr=[p for p in parts if p['key']==c['key']]
        for ref in ['D_E','D_L']:
            for score in ['A2','M2','C2','J','old_A','old_C']:
                teams.append(dict(key=c['key'],scenario=c['scenario'],seed=c['seed'],scope='full_path',purpose='full_path',
                    score=score,reference=ref,rho=rho([p[score] for p in pr],[p[ref] for p in pr]),
                    agreement=agreement(pr,score,ref)))
        # Separate reload traversal; no scores or endpoints are changed.
        second=PretrainedModel(len(c['fixed_start']));second.load_state_dict(torch.load(c['evaluator'],map_location='cpu',weights_only=True)['state_dict']);second.eval()
        for value in table.values():
            with torch.no_grad():y=abs(float(second(torch.tensor(value['x'],dtype=torch.float32)[None,:]).item()))
            reload_error=max(reload_error,abs(y-value['E']));assert y==value['E']
        pooled.append(dict(key=c['key'],evaluator_sha256=sha(c['evaluator']),values=table))
    summary=[];pairs=[];examples={};cost={}
    for scenario in ['clean','controlled_poison']:
        keys=sorted({(t['scope'],t['purpose'],t['score'],t['reference']) for t in teams if t['scenario']==scenario})
        for scope,purpose,x,y in keys:
            tt=[t for t in teams if t['scenario']==scenario and (t['scope'],t['purpose'],t['score'],t['reference'])==(scope,purpose,x,y)]
            rank=boot([t['rho'] for t in tt if t['rho'] is not None]);sg=boot([t['agreement'] for t in tt if t['agreement'] is not None])
            cov=None if scope!='local' else sum(t['valid_rounds'] for t in tt)/sum(t['rounds'] for t in tt)
            ratio=None if scope!='local' else float(np.mean([t['residual_ratio'] for t in tt if t['residual_ratio'] is not None]))
            rg=.8 if purpose=='additivity' else .5;gs=.9 if purpose=='additivity' else .7
            rank_pass=rank['n']>=8 and rank['mean']>=rg and rank['interval'][0]>0 and (cov is None or cov>=.5)
            sign_pass=sg['n']>=8 and sg['mean']>=gs and sg['interval'][0]>.5
            passed=rank_pass and sign_pass and (purpose!='additivity' or ratio<=.1)
            summary.append(dict(scenario=scenario,scope=scope,purpose=purpose,score=x,reference=y,
                rank=rank,sign=sg,rank_coverage=cov,residual_ratio=ratio,rank_pass=rank_pass,sign_pass=sign_pass,pass_all=passed))
        for ref in ['D_E','D_L']:
            for other in ['A2','M2']:
                ct={t['key']:t for t in teams if t['scenario']==scenario and t['scope']=='full_path' and t['score']=='C2' and t['reference']==ref}
                ot={t['key']:t for t in teams if t['scenario']==scenario and t['scope']=='full_path' and t['score']==other and t['reference']==ref}
                rr=[dict(key=k,delta=ct[k]['rho']-ot[k]['rho']) for k in ct if ct[k]['rho'] is not None and ot[k]['rho'] is not None]
                b=boot([r['delta'] for r in rr]);pairs.append(dict(scenario=scenario,scope='full_path',reference=ref,comparison='C2-'+other,stat=b,pairs=rr,incremental_gain=b['n']>=8 and b['interval'][0]>0))
        # Local external incremental comparison is diagnostic, not selection of a replacement method.
        ct={t['key']:t for t in teams if t['scenario']==scenario and t['purpose']=='combined_external'}
        jt={t['key']:t for t in teams if t['scenario']==scenario and t['purpose']=='joint_external'}
        rr=[dict(key=k,delta=ct[k]['rho']-jt[k]['rho']) for k in ct if ct[k]['rho'] is not None and jt[k]['rho'] is not None]
        pairs.append(dict(scenario=scenario,scope='local',reference='e_joint',comparison='c2-j',stat=boot([r['delta'] for r in rr]),pairs=rr,diagnostic_only=True))
        rr=[r for r in rows if r['scenario']==scenario];mis=[r for r in rr if sign(r['j'])*sign(r['e_joint'])==-1]
        trans=[r for r in rr if r['full_accepted']!=r['joint_accepted']]
        examples[scenario]=dict(max_residual=max(rr,key=lambda r:abs(r['residual'])),
            max_internal_external_disagreement=None if not mis else max(mis,key=lambda r:abs(r['j']-r['e_joint'])),
            max_transition=None if not trans else max(trans,key=lambda r:abs(r['j'])),
            residual_median_abs=float(np.median([abs(r['residual']) for r in rr])),
            residual_p90_abs=float(np.percentile([abs(r['residual']) for r in rr],90)))
    for phase in ['pilot','development']:
        cost[phase]={}
        for p in (ROOT/phase).glob('*.json'):
            record=read(p)
            for stage,cs in record['cost'].items():
                dest=cost[phase].setdefault(stage,dict(requests=0,samples=0,seconds=0.,cache_hits=0))
                for k,v in cs.items():dest[k]+=v
    save(ROOT/'local_comparisons.json',rows);csvsave(ROOT/'local_comparisons.csv',rows)
    save(ROOT/'participant_totals.json',parts);save(ROOT/'team_metrics.json',teams)
    save(ROOT/'pooled_evaluation.json',pooled)
    save(ROOT/'summary.json',dict(metrics=summary,paired=pairs,examples=examples,cost=cost,
        development_groups=len(rows),development_role_branches=len(rows)*3,participants=len(parts),
        pooled_unique_points=pool_count,pooled_recheck_points=pool_count,max_pooled_reload_error=reload_error,
        evaluation_wall_seconds=time.perf_counter()-start,new_confirmation=False))
    save(ROOT/'evaluation_lock.json',{p.name:sha(p) for p in ROOT.glob('*.json') if p.name!='evaluation_lock.json'})
    for m in summary:
        if m['scope']=='local' or (m['reference']=='D_E' and m['score'] not in ['old_A','old_C']):
            print(m['scenario'],m['scope'],m['score'],m['reference'],m['rank'],m['sign'],'residual',m['residual_ratio'],'pass',m['pass_all'],flush=True)

if __name__=='__main__':setup();analyze()

"""Preregistered path diagnostics. Pooled is loaded only after decision/score locks."""
import csv,shutil,time
import numpy as np
from scipy.stats import spearmanr
from common import *
def sign(x):return int(x>1e-4)-int(x< -1e-4)
def rho(x,y):
    if len(x)<3 or np.ptp(x)<=1e-4 or np.ptp(y)<=1e-4:return None
    return float(spearmanr(x,y).statistic)
def boot(values):
    if not values:return dict(mean=None,interval=None,n=0)
    x=np.asarray(values);rng=np.random.default_rng(20261024);ix=rng.integers(0,len(x),(10000,len(x)))
    return dict(mean=float(x.mean()),interval=np.percentile(x[ix].mean(axis=1),[.1,99.9]).tolist(),n=len(x))
def avg(values):return None if not values else float(np.mean(values))
def classify(rr,field):
    nonzero=[r for r in rr if sign(r['j'])]
    return dict(nonzero=len(nonzero),retained=avg([sign(r[field])==sign(r['j']) for r in nonzero]),
        reversed=avg([sign(r[field])==-sign(r['j']) for r in nonzero]),vanished=avg([sign(r[field])==0 for r in nonzero]),
        attenuated=avg([abs(r[field])<=.5*abs(r['j']) for r in nonzero]),amplified=avg([abs(r[field])>=2*abs(r['j']) for r in nonzero]),
        rho=rho([r['j'] for r in rr],[r[field] for r in rr]))
def rq2(rr):
    out=dict(material=avg([abs(r['u'])>1e-4 for r in rr]),mean_abs_u=avg([abs(r['u']) for r in rr]),
        sign_change=avg([sign(r['pi'])!=sign(r['tau']) for r in rr]),rho=rho([r['pi'] for r in rr],[r['tau'] for r in rr]),
        restart=avg([r['restart'] for r in rr]),distance=avg([r['final_distance'] for r in rr]))
    for field in ['pi','tau','u']:
        for label,sgn in [('positive',1),('negative',-1),('nearzero',0)]:out[field+'_'+label]=avg([sign(r[field])==sgn for r in rr])
    return out
def replacement(rr,x='pi',y='D_L'):
    decisive=[r for r in rr if sign(r[y])];pos=[r for r in rr if sign(r[y])==1];neg=[r for r in rr if sign(r[y])==-1];zero=[r for r in rr if sign(r[y])==0]
    denom=sum(abs(r[y]) for r in rr)
    return dict(rho=rho([r[x] for r in rr],[r[y] for r in rr]),
        sign=avg([sign(r[x])==sign(r[y]) for r in decisive]),positive_recall=avg([sign(r[x])==1 for r in pos]),
        negative_recall=avg([sign(r[x])==-1 for r in neg]),nearzero_false_positive=avg([sign(r[x])!=0 for r in zero]),
        nmae=None if denom<=len(rr)*1e-4 else sum(abs(r[x]-r[y]) for r in rr)/denom)

def analyze():
    check();assert read(ROOT/'verification.json')['accepted'];tic=time.perf_counter()
    snap=ROOT/'analysis_source_snapshot';snap.mkdir()
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,snap/p.name)
    save(ROOT/'analysis_source_lock.json',{p.name:sha(p) for p in snap.glob('*.py')})
    refs={(r['key'],r['agent_id']):r for r in read(CA/'contribution_four_way_experiment_v2_20261022/comparisons.json')}
    rows=[];records=[];allcost=read(ROOT/'deployment_ledgers_verified.json')
    for c in read(ROOT/'cases.json')['development']:
        r=read(ROOT/'development'/(c['key']+'.json'));records.append(r)
        for g in r['groups']:
            for q in g['q_labels']:
                row=dict(key=c['key'],seed=c['seed'],scenario=c['scenario'],q=q,t=g['t'],agent_id=g['agent_id'],
                    j=g['j'],tau=g['tau'],pi=g['pi'],u=g['u'],tau_abs=g['tau_abs'],restart=g['restart'],
                    full_accepted=g['full_accepted'],mode=g['mode'],D_L=refs[c['key'],g['agent_id']]['D_L'],
                    D_E=refs[c['key'],g['agent_id']]['D_E'],
                    final_distance=float(np.linalg.norm(np.asarray(g['P']['result']['solution'])-np.asarray(g['T']['result']['solution']))),
                    P_steps=len(g['first']['steps'])+len(g['P']['steps']),T_steps=len(g['first']['steps'])+len(g['T']['steps']),
                    P_halted=g['P']['final']['halted'],T_halted=g['T']['final']['halted'])
                for h in [1,3,5]:row['tau_'+str(h)]=g['observations'][str(h)]['T']['difference'];row['pi_'+str(h)]=g['observations'][str(h)]['P']['difference']
                if q==0:near(row['pi'],row['D_L'])
                rows.append(row)
    save(ROOT/'internal_comparisons.json',rows)
    with (ROOT/'internal_comparisons.csv').open('x',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    teams=[];strata=[]
    for c in read(ROOT/'cases.json')['development']:
        for q in [0,.25,.5,.75]:
            rr=[r for r in rows if r['key']==c['key'] and r['q']==q]
            head=dict(key=c['key'],seed=c['seed'],scenario=c['scenario'],q=q,n=len(rr))
            for h,field in [('1','tau_1'),('3','tau_3'),('5','tau_5'),('final','tau'),('abs','tau_abs')]:
                teams.append(dict(**head,rq='RQ1',h=h,**classify(rr,field)))
            teams.append(dict(**head,rq='RQ2',**rq2(rr)))
            if q>0:teams.append(dict(**head,rq='RQ3',**replacement(rr)))
            for label,subset in [('restart',[v for v in rr if v['restart']]),('not_restart',[v for v in rr if not v['restart']]),
                ('accepted',[v for v in rr if v['full_accepted']]),('not_accepted',[v for v in rr if not v['full_accepted']]),
                ('secant',[v for v in rr if v['mode']=='secant']),('dynamic',[v for v in rr if v['mode']=='dynamic']),
                ('positive_j',[v for v in rr if sign(v['j'])==1]),('negative_j',[v for v in rr if sign(v['j'])==-1])]:
                if subset:strata.append(dict(**{k:v for k,v in head.items() if k!='n'},stratum=label,n=len(subset),rq1=classify(subset,'tau'),rq2=rq2(subset)))
    save(ROOT/'team_metrics.json',teams);save(ROOT/'strata_team_metrics.json',strata)
    summaries=[];costs=[]
    for sc in ['clean','controlled_poison']:
        for q in [0,.25,.5,.75]:
            ls=[v for v in allcost if v['phase']=='development' and v['key'].startswith('budget_'+sc+'_') and v['q']==q]
            zero=[v for v in allcost if v['phase']=='development' and v['key'].startswith('budget_'+sc+'_') and v['q']==0]
            total=sum(v['total_samples'] for v in ls);reference=sum(v['total_samples'] for v in zero)
            costs.append(dict(scenario=sc,q=q,baseline=sum(v['baseline_samples'] for v in ls),new=sum(v['new_samples'] for v in ls),
                total=total,full_loo_total=reference,ratio=total/reference,cost_pass=total<=.7*reference))
            for rq in ['RQ1','RQ2','RQ3']:
                if rq=='RQ3' and q==0:continue
                horizons=['1','3','5','final','abs'] if rq=='RQ1' else [None]
                for h in horizons:
                    tt=[t for t in teams if t['scenario']==sc and t['q']==q and t['rq']==rq and (h is None or t['h']==h)]
                    ignored={'key','seed','scenario','q','n','rq','h','nonzero'}
                    fields=[f for f in tt[0] if f not in ignored]
                    metrics={f:boot([t[f] for t in tt if t[f] is not None]) for f in fields}
                    result=dict(scenario=sc,q=q,rq=rq,h=h,metrics=metrics)
                    if rq=='RQ1':
                        number=sum(t['nonzero'] for t in tt);a=metrics['retained'];b=metrics['reversed']
                        result.update(nonzero=number,pass_gate=bool(number>=20 and a['n']>=8 and a['mean']>=.8 and a['interval'][0]>=.7 and b['mean']<=.1 and b['interval'][1]<=.2))
                    elif rq=='RQ2':
                        a=metrics['material'];result['pass_gate']=bool(a['n']>=8 and a['mean']>=.3 and a['interval'][0]>.1)
                    else:
                        a=metrics['rho'];b=metrics['sign'];d=metrics['nmae']
                        ok=(a['n']>=8 and b['n']>=8 and d['n']>=8 and a['mean']>=.8 and a['interval'][0]>.5 and b['mean']>=.85 and b['interval'][0]>.7 and d['mean']<=.25 and d['interval'][1]<.5)
                        result['effect_pass']=bool(ok);result['pass_gate']=bool(ok and costs[-1]['cost_pass'])
                    summaries.append(result)
    candidates=[q for q in [.25,.5,.75] if all(v['pass_gate'] for v in summaries if v['rq']=='RQ3' and v['q']==q)]
    examples={}
    for sc in ['clean','controlled_poison']:
        rr=sorted([v for v in rows if v['scenario']==sc],key=lambda r:(r['seed'],r['q'],r['agent_id']))
        examples[sc]=dict(max_temporal_change=max(rr,key=lambda r:abs(r['tau']-r['j'])),max_restoration_gap=max(rr,key=lambda r:abs(r['pi']-r['tau'])))
    grouped=[]
    for sc in ['clean','controlled_poison']:
        for q in [0,.25,.5,.75]:
            for label in sorted({t['stratum'] for t in strata}):
                ts=[t for t in strata if t['scenario']==sc and t['q']==q and t['stratum']==label]
                grouped.append(dict(scenario=sc,q=q,stratum=label,teams=len(ts),records=sum(t['n'] for t in ts),
                    interpret=bool(len(ts)>=3 and sum(t['rq1']['nonzero'] for t in ts)>=20),
                    rq1_retained=boot([t['rq1']['retained'] for t in ts if t['rq1']['retained'] is not None]),
                    rq2_material=boot([t['rq2']['material'] for t in ts])))
    save(ROOT/'internal_summary.json',dict(metrics=summaries,costs=costs,strata=grouped,examples=examples,candidates=candidates,
        groups=len(rows),teams=20,old_data_exploration=True,new_confirmation=False))
    save(ROOT/'internal_analysis_locked.json',dict(comparisons_sha256=sha(ROOT/'internal_comparisons.json'),summary_sha256=sha(ROOT/'internal_summary.json'),pooled_loaded=False))
    # Offline evaluator after all endpoint and internal-statistic locks.
    import torch
    from models import PretrainedModel
    pooled=[];external=[];counter=0;reloaderror=0.;pooltic=time.perf_counter()
    for c,r in zip(read(ROOT/'cases.json')['development'],records):
        print('OFFLINE',c['key'],flush=True)
        net=PretrainedModel(len(c['fixed_start']));net.load_state_dict(torch.load(c['evaluator'],map_location='cpu',weights_only=True)['state_dict']);net.eval();cache={}
        def E(x):
            nonlocal counter
            k=np.asarray(x,np.float64).tobytes().hex()
            if k not in cache:
                if counter+2>4000 or time.perf_counter()-pooltic>300:raise RuntimeError('pooled_cap')
                with torch.no_grad():v=abs(float(net(torch.tensor(x,dtype=torch.float32)[None,:]).item()))
                cache[k]=dict(x=x,E=v);counter+=1
            return cache[k]['E']
        ef=E(r['baseline']['solution']);D={g['agent_id']:E(g['P']['result']['solution'])-ef for g in r['groups'] if 0 in g['q_labels']}
        for g in r['groups']:
            one=g['first']['result']['solution'];fx=r['controls'][str(g['t'])]['steps'][0]['post']['global_best_S']
            je=E(one)-E(fx);te=E(g['T']['result']['solution'])-ef;pe=E(g['P']['result']['solution'])-ef
            for q in g['q_labels']:
                original=next(v for v in rows if v['key']==c['key'] and v['q']==q and v['agent_id']==g['agent_id'])
                near(D[g['agent_id']],original['D_E'])
                external.append(dict(**original,j_E=je,tau_E=te,pi_E=pe,u_E=pe-te,tau_abs_E=E(g['T_abs_x'])-ef))
        other=PretrainedModel(len(c['fixed_start']));other.load_state_dict(torch.load(c['evaluator'],map_location='cpu',weights_only=True)['state_dict']);other.eval()
        for item in cache.values():
            if counter>=4000 or time.perf_counter()-pooltic>300:raise RuntimeError('pooled_recheck_cap')
            with torch.no_grad():v=abs(float(other(torch.tensor(item['x'],dtype=torch.float32)[None,:]).item()))
            assert v==item['E'];counter+=1;reloaderror=max(reloaderror,abs(v-item['E']))
        pooled.append(dict(key=c['key'],evaluator_sha256=sha(c['evaluator']),points=list(cache.values())))
    save(ROOT/'pooled_locked.json',pooled);save(ROOT/'external_comparisons.json',external)
    ext=[]
    for sc in ['clean','controlled_poison']:
        for q in [0,.25,.5,.75]:
            for x,y in [('j','j_E'),('tau','tau_E'),('pi','pi_E'),('u','u_E'),('pi','D_E'),('pi_E','D_E')]:
                tt=[]
                for c in read(ROOT/'cases.json')['development']:
                    if c['scenario']!=sc:continue
                    rr=[v for v in external if v['key']==c['key'] and v['q']==q]
                    decisive=[v for v in rr if sign(v[x])]
                    tt.append(dict(key=c['key'],rho=rho([v[x] for v in rr],[v[y] for v in rr]),agreement=avg([sign(v[x])==sign(v[y]) for v in decisive])))
                ext.append(dict(scenario=sc,q=q,comparison=x+':'+y,rank=boot([v['rho'] for v in tt if v['rho'] is not None]),
                    agreement=boot([v['agreement'] for v in tt if v['agreement'] is not None]),teams=tt))
    save(ROOT/'external_summary.json',ext)
    save(ROOT/'analysis_completion.json',dict(complete=True,pooled_samples=counter,pooled_raw_reload_error=reloaderror,wall_seconds=time.perf_counter()-tic,
        internal_candidates=candidates,truth_loaded=False));check()
    for m in summaries:
        if (m['rq']=='RQ1' and m['h']=='final') or m['rq'] in ['RQ2','RQ3']:print(m,flush=True)
    print('COSTS',costs,flush=True)
if __name__=='__main__':setup();analyze()

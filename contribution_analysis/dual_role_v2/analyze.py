import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from common import read,save,sha,csvsave,check_sources


def rank(x,y):
    x=np.asarray(x);y=np.asarray(y)
    raw=None if np.ptp(x)==0 or np.ptp(y)==0 else float(spearmanr(x,y).statistic)
    reason='near_flat_reference' if np.ptp(y)<=1e-4 else 'constant_score' if np.ptp(x)==0 else None
    return dict(spearman=None if reason else raw,raw_spearman=raw,reason=reason)


def boot(values,percentiles=(.625,99.375)):
    v=np.asarray([x for x in values if x is not None],float)
    if not len(v):return dict(mean=None,interval=None,n=0)
    rng=np.random.default_rng(20261020);means=v[rng.integers(0,len(v),(10000,len(v)))].mean(axis=1)
    return dict(mean=float(v.mean()),interval=np.percentile(means,percentiles).tolist(),n=len(v))


def category(v,deadzone):return np.where(np.asarray(v)>deadzone,1,np.where(np.asarray(v)<-deadzone,-1,0))


def sign_summary(teams,deadzone=1e-4):
    matrices=[]
    for score,ref in teams:
        a=category(score,deadzone);b=category(ref,deadzone);mat=np.zeros((3,3),int)
        for x,y in zip(a,b):mat[x+1,y+1]+=1
        matrices.append(mat)
    matrices=np.asarray(matrices);total=matrices.sum(axis=0)
    def compute(mat):
        positive=mat[...,2,:].sum(axis=-1);negative=mat[...,0,:].sum(axis=-1);n=mat.sum(axis=(-1,-2))
        def ratio(a,b):return np.divide(a,b,out=np.full(np.shape(a),np.nan),where=b!=0)
        return dict(positive_precision=ratio(mat[...,2,2],positive),negative_precision=ratio(mat[...,0,0],negative),
                    negative_harmful_false_flag_rate=ratio(mat[...,0,2],negative),
                    negative_reference_neutral_rate=ratio(mat[...,0,1],negative),
                    agreement=ratio(np.trace(mat,axis1=-2,axis2=-1),n))
    rng=np.random.default_rng(20261020);sample=matrices[rng.integers(0,len(matrices),(10000,len(matrices)))].sum(axis=1)
    point=compute(total);draws=compute(sample);metrics={}
    for name,value in point.items():
        valid=draws[name][np.isfinite(draws[name])]
        metrics[name]=dict(value=float(value) if np.isfinite(value) else None,
                           interval95=np.percentile(valid,[2.5,97.5]).tolist() if len(valid) else None,
                           valid_bootstrap_fraction=len(valid)/10000)
    return dict(deadzone=deadzone,order=['negative','neutral','positive'],rows='estimated',columns='reference',
                confusion=total.tolist(),n=int(total.sum()),predicted_positive=int(total[2].sum()),
                predicted_negative=int(total[0].sum()),predicted_neutral=int(total[1].sum()),metrics=metrics,
                inference='descriptive, no payment gate')


def analyze(root):
    import torch
    from models import PretrainedModel
    cfg=check_sources(root)
    for folder,lock in [(None,'scores_lock.json'),('references','references_lock.json')]:
        for name,h in read(root/lock)['record_sha256'].items():assert sha((root/folder/name) if folder else root/name)==h
    old=Path(cfg['previous_custom']);rows=[];teams=[];endpoint_rows=[]
    for c in read(root/'cases.json')['development']:
        print('OFFLINE_QUALITY',c['key'],flush=True);sc=read(root/'scores'/(c['key']+'.json'))
        ep=torch.load(c['evaluator'],map_location='cpu',weights_only=True);f=PretrainedModel(2);f.load_state_dict(ep['state_dict']);f.eval()
        def evaluate(x):
            assert np.isfinite(x).all() and np.max(np.abs(x))<=1+1e-8
            with torch.no_grad():return abs(float(f(torch.tensor(x,dtype=torch.float32)[None,:]).item()))
        E0=evaluate(sc['baseline_solution']);local=[]
        for j,i in enumerate(c['ids']):
            anchor=read(root/'references'/(c['key']+'_anchor_'+str(i)+'.json'))
            model=read(root/'references'/(c['key']+'_model_'+str(i)+'.json'))
            full=read(old/'removals'/(c['key']+'_'+str(i)+'.json'))
            Ea=evaluate(anchor['result']['solution']);Em=evaluate(model['result']['solution']);Ef=evaluate(full['result']['solution'])
            row=dict(key=c['key'],scenario=c['scenario'],seed=c['seed'],agent_id=i,A=sc['A'][j],B=sc['B'][j],M=sc['M'][j],
                q=sc['q'][j],N=sc['per_agent_samples'][j],qN=sc['q'][j]*sc['per_agent_samples'][j],
                old_model_S=sc['old_model_S'][j],old_anchor_S=sc['old_anchor_S'][j],
                C_anchor=Ea-E0,C_model=Em-E0,C_full=Ef-E0,C_anchor_internal=anchor['final_fixed_loss']-sc['baseline_loss'])
            rows.append(row);local.append(row)
            endpoint_rows.append(dict(key=c['key'],agent_id=i,baseline_E=E0,anchor_E=Ea,model_E=Em,full_E=Ef,
                anchor_solution=anchor['result']['solution'],model_solution=model['result']['solution'],full_solution=full['result']['solution']))
        pairs=[('A','C_anchor'),('M','C_model'),('B','C_model'),('q','C_model'),('qN','C_model'),
               ('old_model_S','C_model'),('old_anchor_S','C_anchor'),('A','C_full'),('M','C_full'),('A','C_anchor_internal')]
        for score,ref in pairs:
            result=rank([r[score] for r in local],[r[ref] for r in local])
            if sc['status']!='available' or (score=='M' and sc['model_status']!='available'):
                result.update(spearman=None,reason=sc['status'])
            teams.append(dict(key=c['key'],scenario=c['scenario'],seed=c['seed'],score=score,reference=ref,**result))
    summaries=[];contrasts=[];signs=[]
    pairs=list(dict.fromkeys((t['score'],t['reference']) for t in teams))
    for scenario in ['clean','controlled_poison']:
        for score,ref in pairs:
            ts=[t for t in teams if t['scenario']==scenario and (t['score'],t['reference'])==(score,ref)]
            summary=dict(scenario=scenario,score=score,reference=ref,total_teams=len(ts),
                         coverage=sum(t['spearman'] is not None for t in ts)/len(ts),**boot([t['spearman'] for t in ts]))
            summary['rank_gate']=bool(summary['coverage']>=.9 and summary['mean'] is not None and summary['mean']>.5 and summary['interval'][0]>0)
            summaries.append(summary)
        for comparator in ['q','B']:
            diffs=[];keys=[]
            for c in read(root/'cases.json')['development']:
                if c['scenario']!=scenario:continue
                left=next(t['spearman'] for t in teams if t['key']==c['key'] and t['score']=='M' and t['reference']=='C_model')
                right=next(t['spearman'] for t in teams if t['key']==c['key'] and t['score']==comparator and t['reference']=='C_model')
                if left is not None and right is not None:diffs.append(left-right);keys.append(c['key'])
            b=boot(diffs);contrasts.append(dict(scenario=scenario,contrast='M_minus_'+comparator,case_keys=keys,
                                                increment_established=bool(b['interval'] and b['interval'][0]>0),**b))
        for score,ref in [('A','C_anchor'),('M','C_model')]:
            vectors=[]
            for c in read(root/'cases.json')['development']:
                if c['scenario']!=scenario:continue
                rs=[r for r in rows if r['key']==c['key']];vectors.append(([r[score] for r in rs],[r[ref] for r in rs]))
            for dz in [1e-4,0.]: signs.append(dict(scenario=scenario,score=score,reference=ref,**sign_summary(vectors,dz)))
    examples=[]
    for scenario in ['clean','controlled_poison']:
        for score,ref in [('A','C_anchor'),('M','C_model')]:
            valid=[t for t in teams if t['scenario']==scenario and t['score']==score and t['reference']==ref and t['spearman'] is not None]
            for tag,direction in [('best',-1),('worst',1)]:
                if valid:
                    t=sorted(valid,key=lambda x:(direction*x['spearman'],x['seed']))[0]
                    examples.append(dict(category=tag,team=t,participants=[r for r in rows if r['key']==t['key']]))
    primary=[s for s in summaries if (s['score'],s['reference']) in [('A','C_anchor'),('M','C_model')]]
    gate=dict(exploration_completed=True,new_confirmation=False,rank_all_pass=all(s['rank_gate'] for s in primary),
              cost_gate=read(root/'technical.json')['cost_gate'],primary=primary,
              payment_validity_established=False,negative_sign_validity_established=False,
              adoption='research_only',next_action='stop and report; no formula changes or additional seeds')
    save(root/'summary.json',summaries);save(root/'contrasts.json',contrasts);save(root/'sign_metrics.json',signs)
    save(root/'representative_cases.json',examples);save(root/'gate.json',gate)
    csvsave(root/'participant_comparison.csv',rows);csvsave(root/'team_metrics.csv',teams);csvsave(root/'endpoint_evaluations.csv',endpoint_rows)
    save(root/'evaluation_ledger.json',dict(pooled_points=260,baseline=20,anchor=80,model=80,full=80,
        quality_read_only_after_scores_and_references_locked=True))
    print(gate)


if __name__=='__main__':
    root=Path(sys.argv[1]).resolve();sys.path.insert(0,str(root/'source_snapshot'))
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);analyze(root)
